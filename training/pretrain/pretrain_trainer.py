from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer

IGNORE_TOKEN_ID = -100

TASK_VOCAB = [
    "image_caption",
    "speech_to_text",
    "text_to_image",
    "text_to_speech",
    "image_to_text",
    "speech_to_image",
    "text_to_text",
]
ID2TASK = {i: n for i, n in enumerate(TASK_VOCAB)}
PACK_BASE = 2048


def _unpack_meta(meta_col: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    task_ids = meta_col // PACK_BASE - 1
    resp_lens = meta_col % PACK_BASE
    return task_ids.to(torch.long), resp_lens.to(torch.long)


def extract_image_parts_for_tasks(
    input_ids: torch.Tensor,
    hidden_states,
    labels: torch.Tensor,
    tasks: list[str],
    allowed_tasks: set[str] | None = None,
    image_start_id: int = 8197,
    image_end_id: int = 8196,  # kept for API compatibility; not used in the original logic
    seq_length: int = 1024,
):
    """
    Extract image-token spans (token ids + last-layer hidden states) from a batch.

    Logic is identical to the original implementation:
    - Find all positions where input_ids == image_start_id.
    - For each (b, s), if:
        - tasks[b] is in allowed_tasks,
        - labels[b, s] is not IGNORE_TOKEN_ID,
        - and s + seq_length + 1 <= T,
      then collect input_ids[b, s+1 : s+seq_length+1] and last_hidden[b, s+1 : s+seq_length+1, :].
    """
    if allowed_tasks is None:
        allowed_tasks = {"text_to_image"}

    device = input_ids.device
    last_hidden = hidden_states[-1]
    B, T = input_ids.shape

    start_positions = torch.nonzero(input_ids == image_start_id, as_tuple=False)
    if start_positions.numel() == 0:
        return None, None

    image_ids, image_hidden = [], []
    for b, s in start_positions:
        b = int(b.item())
        s = int(s.item())

        if tasks[b] not in allowed_tasks:
            continue
        if labels[b, s].item() == IGNORE_TOKEN_ID:
            continue

        e = s + seq_length + 1
        if e > T:
            continue

        image_ids.append(input_ids[b, s + 1 : e])
        image_hidden.append(last_hidden[b, s + 1 : e, :])

    if not image_ids:
        return None, None

    return torch.stack(image_ids, 0).to(device), torch.stack(image_hidden, 0).to(device)


class PretrainWeightedAndPerceptionTrainer(Trainer):
    def __init__(
        self,
        *args,
        response_weighted_tasks: set[str] = frozenset({"image_caption", "speech_to_text"}),
        response_seg_weight: float = 2.0,
        loss_net=None,
        perception_weight: float = 1.0,
        image_start_id: int = 8197,
        image_end_id: int = 8196,
        image_seq_length: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response_weighted_tasks = set(response_weighted_tasks)
        self.response_seg_weight = float(response_seg_weight)

        self.loss_net = loss_net
        self.perception_weight = float(perception_weight)
        self.image_start_id = int(image_start_id)
        self.image_end_id = int(image_end_id)
        self.image_seq_len = int(image_seq_length)

        if self.loss_net is not None:
            self.loss_net.to(self.args.device)

        self._ce = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_TOKEN_ID)

        self.eos_token_id = getattr(self.model.config, "eos_token_id", None)

    @torch.no_grad()
    def _build_tail_mask_shifted_without_eos(
        self,
        shift_labels: torch.Tensor,
        tasks: list[str],
        resp_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a tail-segment mask on the *shifted* label axis.

        Rules (identical to the original behavior):
        - Only enabled for tasks in `self.response_weighted_tasks`.
        - Only positions where labels != IGNORE_TOKEN_ID are eligible.
        - Explicitly exclude positions where label == eos_token_id (if eos_token_id is available).
        - For each sample, select the last `resp_len` eligible indices.
        """
        B, S = shift_labels.size()
        mask = torch.zeros((B, S), dtype=torch.bool, device=shift_labels.device)

        has_eos = self.eos_token_id is not None
        for i in range(B):
            if tasks[i] not in self.response_weighted_tasks:
                continue
            k = int(resp_lens[i].item())
            if k <= 0:
                continue

            valid_i = shift_labels[i] != IGNORE_TOKEN_ID
            if has_eos:
                valid_i = valid_i & (shift_labels[i] != self.eos_token_id)

            idx = torch.nonzero(valid_i, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            k = min(k, idx.numel())
            tail_idx = idx[-k:]
            mask[i, tail_idx] = True

        return mask

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attn_mask = inputs.get("attention_mask", None)

        meta_col = labels[:, 0]
        task_ids_from_meta, resp_lens_from_meta = _unpack_meta(meta_col)

        if attn_mask is None:
            raise ValueError("attention_mask is required to recover seq_len when dispatch_batches=True")
        seq_lens = attn_mask.sum(dim=1).to(torch.long)  # kept (even if unused) to preserve behavior/inputs

        tasks = inputs.get("tasks", None)
        if tasks is None:
            tasks = [ID2TASK.get(int(i.item()), "unknown") for i in task_ids_from_meta]

        resp_lens = inputs.get("resp_lens", None)
        if resp_lens is None:
            resp_lens = resp_lens_from_meta

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_tok = self._ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        valid = shift_labels != IGNORE_TOKEN_ID
        weights = valid.float()

        tail_mask_shifted = None
        if self.response_seg_weight != 1.0:
            tail_mask_shifted = self._build_tail_mask_shifted_without_eos(
                shift_labels=shift_labels,
                tasks=tasks,
                resp_lens=resp_lens,
            )
            weights = weights + (self.response_seg_weight - 1.0) * (tail_mask_shifted & valid).float()

        denom = weights.sum().clamp_min(1.0)
        local_weighted_loss = (per_tok * weights).sum() / denom

        perception_loss = torch.zeros((), device=last_hidden.device, dtype=last_hidden.dtype)
        if self.loss_net is not None:
            allowed = {"text_to_image"}
            image_ids, image_hidden = extract_image_parts_for_tasks(
                input_ids=input_ids,
                hidden_states=outputs.hidden_states,
                labels=labels,
                tasks=tasks,
                allowed_tasks=allowed,
                image_start_id=self.image_start_id,
                image_end_id=self.image_end_id,
                seq_length=self.image_seq_len,
            )
            if image_ids is not None:
                image_hidden = image_hidden.to(dtype=last_hidden.dtype)
                out = self.loss_net(input_ids=image_ids, generated_hidden_states=image_hidden)
                if not torch.is_tensor(out):
                    out = torch.tensor(out, device=last_hidden.device, dtype=last_hidden.dtype)
                else:
                    out = out.to(device=last_hidden.device, dtype=last_hidden.dtype)
                if out.dim() > 0:
                    out = out.mean()
                perception_loss = out

        if int(valid.sum().item()) == 0 and (self.loss_net is None or "image_ids" not in locals() or image_ids is None):
            noop = shift_logits.sum() * 0.0
            return (noop, outputs) if return_outputs else noop

        total_loss = local_weighted_loss + self.perception_weight * perception_loss

        if self.model.training and (self.state is None or (self.state.global_step % self.args.logging_steps == 0)):
            tail_ratio = 0.0
            if tail_mask_shifted is not None:
                denom2 = valid.sum().item()
                if denom2 > 0:
                    tail_ratio = float((tail_mask_shifted & valid).sum().item() / denom2)
            self.log(
                {
                    "loss_local_weighted": float(local_weighted_loss.detach().cpu()),
                    "loss_perception": float(perception_loss.detach().cpu()),
                    "loss_total": float(total_loss.detach().cpu()),
                    "resp_seg_weight": float(self.response_seg_weight),
                    "perception_weight": float(self.perception_weight),
                    "tail_token_ratio": tail_ratio,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss
