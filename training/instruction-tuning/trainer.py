from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer

PAD_TOKEN_ID = 2
IGNORE_TOKEN_ID = -100


def extract_image_parts(
    input_ids: torch.Tensor,
    hidden_states,  # HF outputs.hidden_states: list[torch.Tensor]
    labels: torch.Tensor,
    image_start_id: int = 8197,
    image_end_id: int = 8196,  # kept for API compatibility; not used in the fixed-length slicing logic
    seq_length: int = 1024,
):
    """
    Extract image-token spans (token ids + last-layer hidden states) from a batch.

    Behavior is identical to the original:
    - Use image_start_id as the anchor.
    - Slice a fixed-length span of `seq_length` tokens following the start marker.
    - Only include spans where labels[b, start_pos] != IGNORE_TOKEN_ID.
    - Drop spans that would go out of bounds.
    Returns:
      image_ids:           (N, seq_length)
      image_hidden_states: (N, seq_length, hidden_size)
    If no valid spans exist, returns (None, None).
    """
    device = input_ids.device
    last_hidden = hidden_states[-1]  # (B, T, H)
    B, T = input_ids.shape

    start_mask = input_ids == image_start_id
    start_positions = torch.nonzero(start_mask, as_tuple=False)  # (N, 2) -> [b, pos]

    if start_positions.numel() == 0:
        return None, None

    image_ids = []
    image_hidden = []

    for b, s in start_positions:
        if labels[b, s].item() == IGNORE_TOKEN_ID:
            continue

        e = s + seq_length + 1  # skip the start marker itself
        if e > T:
            continue

        image_ids.append(input_ids[b, s + 1 : e])
        image_hidden.append(last_hidden[b, s + 1 : e, :])

    if len(image_ids) == 0:
        return None, None

    image_ids = torch.stack(image_ids, dim=0).to(device)
    image_hidden = torch.stack(image_hidden, dim=0).to(device)
    return image_ids, image_hidden


class WeightedTrainer(Trainer):
    """
    Supports:
      1) Segment-weighted language modeling loss:
         tokens inside paired (seg_start_id, seg_end_id) spans are up-weighted by seg_weight.
      2) Optional global image perception loss:
         extracts image token spans and applies `loss_net`.
    total_loss = local_weighted_loss + global_weight * global_loss
    """

    def __init__(
        self,
        *args,
        seg_start_id: int = 12840,
        seg_end_id: int = 12841,
        seg_weight: float = 1.0,
        loss_net=None,
        global_weight: float = 1.0,
        image_start_id: int = 8197,
        image_end_id: int = 8196,  # kept for future extension; not used in current logic
        image_seq_length: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seg_start_id = seg_start_id
        self.seg_end_id = seg_end_id
        self.seg_weight = seg_weight

        self.loss_net = loss_net
        self.global_weight = global_weight
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id
        self.image_seq_len = image_seq_length

        if self.loss_net is not None:
            self.loss_net.to(self.args.device)

        self._ce = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_TOKEN_ID)

    @torch.no_grad()
    def _build_inside_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns a boolean tensor of shape (B, T) marking tokens strictly inside paired segments.

        Pairing is greedy:
        - each start is paired with the first unused end that appears after it.
        - incomplete or out-of-order markers are ignored.
        """
        B, T = input_ids.size()
        mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)

        for b in range(B):
            ids = input_ids[b]
            starts = (ids == self.seg_start_id).nonzero(as_tuple=False).flatten().tolist()
            ends = (ids == self.seg_end_id).nonzero(as_tuple=False).flatten().tolist()

            e_ptr = 0
            for s in starts:
                while e_ptr < len(ends) and ends[e_ptr] <= s:
                    e_ptr += 1
                if e_ptr >= len(ends):
                    break
                e = ends[e_ptr]
                if e - s > 1:
                    mask[b, s + 1 : e] = True
                e_ptr += 1

        return mask

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        1) Forward to get logits & hidden_states (do not pass labels to avoid model-internal loss).
        2) Compute token-level CE with custom weights (local loss).
        3) Optionally compute global/perception loss from extracted image spans.
        4) Combine and return.
        """
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attn_mask = inputs.get("attention_mask", None)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]

        # Local: segment-weighted CE
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_tok = self._ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        raw_seg_mask = self._build_inside_mask(input_ids)  # (B, T)
        seg_mask = raw_seg_mask[:, 1:]  # align to shift

        valid = shift_labels != IGNORE_TOKEN_ID
        weights = valid.float()
        if self.seg_weight != 1.0:
            weights = weights + (self.seg_weight - 1.0) * (seg_mask & valid).float()

        denom = weights.sum().clamp_min(1.0)
        local_weighted_loss = (per_tok * weights).sum() / denom

        # Global: optional perception loss
        global_loss = torch.zeros((), device=last_hidden.device, dtype=last_hidden.dtype)
        image_ids = None

        if self.loss_net is not None:
            image_ids, image_hidden = extract_image_parts(
                input_ids=input_ids,
                hidden_states=outputs.hidden_states,
                labels=labels,
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

                global_loss = out

            # ZeRO-3 safety patch: always touch loss_net params so all ranks include them in the graph
            dummy_touch = 0.0
            for p in self.loss_net.parameters():
                dummy_touch = dummy_touch + (p.to(dtype=last_hidden.dtype).sum() * 0.0)

            global_loss = global_loss + dummy_touch.to(device=last_hidden.device, dtype=last_hidden.dtype)

        # Ensure total_loss always depends on the main model graph
        dummy_main = shift_logits.sum() * 0.0

        total_loss = local_weighted_loss + self.global_weight * global_loss + dummy_main

        if self.model.training and (self.state is None or (self.state.global_step % self.args.logging_steps == 0)):
            self.log(
                {
                    "loss_local_weighted": float(local_weighted_loss.detach().cpu()),
                    "loss_global": float(global_loss.detach().cpu()),
                    "loss_total": float(total_loss.detach().cpu()),
                    "global_weight": float(self.global_weight),
                    "valid_tok_count": int((shift_labels != IGNORE_TOKEN_ID).sum().item()),
                    "has_image_segments": 1 if image_ids is not None else 0,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss
