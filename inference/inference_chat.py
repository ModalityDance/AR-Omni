#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import uuid
import wave
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile

from transformers import AutoTokenizer
from chameleon.inference.aromni import AROmniInferenceModel, Options, DistributedMode

# New converter: text -> speech -> tokens (CosyVoice2 + WavTokenizer)
from speech2tokens import Text2TokenConverter

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- special tokens ----------
SPECIAL_TOKENS = {
    "<boa>": 12821,
    "<eoa>": 12820,
    "<bos>": 0,
    "<eos>": 2,
    "<eoh>": 12830,
    "<eom>": 12831,
}
BOI = 8197
EOI = 8196

DEFAULT_USER_APPEND_TEXT = "Please acknowledge the user's vocal input, create a textual response."


def encode_special(token_text: str) -> int:
    return SPECIAL_TOKENS[token_text]


def normalize_device(dev: Union[int, str, torch.device]) -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{dev}")
        return torch.device("cpu")
    s = str(dev).strip().lower()
    if s.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(s)


# -----------------------------
# Image tokenization
# -----------------------------
def tokenize_image_from_path(token_manager, image_path: Union[str, Path], **tokenize_kwargs) -> List[int]:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        toks = token_manager.tokenize_image(img, **tokenize_kwargs)
    if torch.is_tensor(toks):
        return toks.view(-1).tolist()
    if isinstance(toks, list):
        return toks
    return list(toks)


# -----------------------------
# Output segmentation + decoding
# -----------------------------
def split_segments(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Robust segment splitter.

    - Image: BOI ... EOI (EOI required; if missing, raise)
    - Audio: either:
        (a) BOA + consecutive audio tokens + optional EOA
        (b) consecutive audio tokens without BOA/EOA
      Here, EOA is NOT required.
    - Text: everything else.
    """
    BOA = SPECIAL_TOKENS["<boa>"]
    EOA = SPECIAL_TOKENS["<eoa>"]
    tokens = tensor.tolist()[0] if tensor.dim() == 2 else tensor.tolist()

    AUDIO_MIN, AUDIO_MAX = 8724, 12819

    segments: List[Dict[str, Any]] = []
    i = 0
    n = len(tokens)

    def is_audio_tok(x: int) -> bool:
        return AUDIO_MIN <= x <= AUDIO_MAX

    while i < n:
        t = tokens[i]

        # ---- image segment: BOI ... EOI ----
        if t == BOI:
            j = i + 1
            while j < n and tokens[j] != EOI:
                j += 1
            if j >= n:
                raise ValueError(f"EOI token not found after BOI at index {i}")
            segments.append({"type": "image", "tokens": tokens[i : j + 1]})
            i = j + 1
            continue

        # ---- audio segment with BOA (EOA optional) ----
        if t == BOA:
            j = i + 1

            # collect consecutive audio tokens; stop at EOA/BOI/BOA or first non-audio token
            while j < n:
                if tokens[j] == EOA:
                    # include EOA and stop
                    j += 1
                    break
                if tokens[j] in (BOI, BOA):
                    break
                if not is_audio_tok(tokens[j]):
                    break
                j += 1

            # Avoid creating empty audio segments like [BOA] only
            seg = tokens[i:j]
            if any(is_audio_tok(x) for x in seg):
                segments.append({"type": "audio", "tokens": seg})
            # if it's empty/no audio codes, just ignore BOA token
            i = j
            continue

        # ---- audio segment without BOA/EOA: run of audio tokens ----
        if is_audio_tok(t):
            j = i + 1
            while j < n and is_audio_tok(tokens[j]):
                j += 1
            segments.append({"type": "audio", "tokens": tokens[i:j]})
            i = j
            continue

        # ---- text segment: until next boundary ----
        j = i + 1
        while j < n and tokens[j] not in (BOI, BOA) and not is_audio_tok(tokens[j]):
            j += 1
        segments.append({"type": "text", "tokens": tokens[i:j]})
        i = j

    return segments


def seg_decode(
    tensor: torch.Tensor,
    model: AROmniInferenceModel,
    wavtokenizer,
    device: Union[int, str, torch.device],
) -> List[Dict[str, Any]]:
    device = normalize_device(device)
    segments = split_segments(tensor)
    decoded_results: List[Dict[str, Any]] = []

    for segment in segments:
        seg_type = segment["type"]
        tokens = segment["tokens"]

        if seg_type == "text":
            decoded = model.decode_text([tokens])

        elif seg_type == "image":
            trans = model.token_manager.translation
            bpe_tok = trans.bpe2img_search_tensors[0]
            dev = bpe_tok.device

            t = torch.as_tensor(tokens, dtype=torch.long, device=dev)

            # strip BOI/EOI
            t = t[(t != BOI) & (t != EOI)]
            # keep only image BPE tokens
            t = t[torch.isin(t, bpe_tok)]

            # align to 1024
            n = t.numel()
            if n == 0:
                raise ValueError("image segment has 0 valid tokens after stripping boi/eoi")
            if n < 1024:
                pad = t.new_full((1024 - n,), t[0].item())
                t = torch.cat([t, pad], dim=0)
            elif n > 1024:
                t = t[:1024]

            decoded = model.decode_image(t.unsqueeze(0))

        elif seg_type == "audio":
            # audio tokens live in [8724..12819] after shift; ignore <boa>/<eoa>
            audio_tokens = [tok - 8724 for tok in tokens if 8724 <= tok <= 12819]
            if audio_tokens:
                t = torch.tensor(audio_tokens, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
                feats = wavtokenizer.codes_to_features(t)
                wav = wavtokenizer.decode(feats, bandwidth_id=torch.tensor([0]).to(device)).cpu()
                decoded = wav
            else:
                decoded = None
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

        decoded_results.append({"type": seg_type, "decode": decoded})

    return decoded_results


# -----------------------------
# Saving utilities
# -----------------------------
def _to_pil_image(im: Any) -> Optional[Image.Image]:
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, np.ndarray):
        arr = im
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return Image.fromarray(arr)
    if torch.is_tensor(im):
        t = im.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        if t.dtype != torch.uint8:
            t = (t.float().clamp(0, 1) * 255).byte()
        return Image.fromarray(t.numpy())
    return None


def save_images_from_decoded(
    decoded_segments: List[Dict[str, Any]],
    out_dir: Path,
    run_uid: str,
    image_format: str = "png",
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_format = image_format.lower()
    saved: List[str] = []

    for seg_idx, item in enumerate(decoded_segments, start=1):
        if item["type"] != "image":
            continue
        content = item["decode"]
        imgs = content if isinstance(content, (list, tuple)) else [content]
        if not imgs:
            continue

        for im_idx, im in enumerate(imgs, start=1):
            pil = _to_pil_image(im)
            if pil is None:
                continue

            if image_format in ("jpg", "jpeg") and pil.mode in ("RGBA", "LA"):
                pil = pil.convert("RGB")

            img_uid = uuid.uuid4().hex[:8]
            p = out_dir / f"{run_uid}_seg{seg_idx}_img{im_idx}_{img_uid}.{image_format}"
            pil.save(p.as_posix(), format=image_format.upper())
            saved.append(p.as_posix())

    return saved


def save_audio_tensor_to_wav(audio_tensor: torch.Tensor, wav_path: Union[str, Path], sample_rate: int) -> str:
    wav_path = Path(wav_path)
    x = audio_tensor.detach().cpu().float().numpy()
    x = np.squeeze(x)
    if x.ndim != 1:
        x = x.reshape(-1)

    maxv = np.max(np.abs(x)) if x.size else 1.0
    if maxv < 1e-9:
        maxv = 1.0
    x = x / maxv
    pcm = (x * 32767.0).astype(np.int16)

    with wave.open(wav_path.as_posix(), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

    return wav_path.as_posix()


def save_audio_from_decoded(
    decoded_segments: List[Dict[str, Any]],
    out_dir: Path,
    run_uid: str,
    sample_rate: int,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for seg_idx, item in enumerate(decoded_segments, start=1):
        if item["type"] != "audio":
            continue
        wav = item["decode"]
        if not torch.is_tensor(wav):
            continue
        audio_uid = uuid.uuid4().hex[:8]
        p = out_dir / f"{run_uid}_seg{seg_idx}_audio_{audio_uid}.wav"
        saved.append(save_audio_tensor_to_wav(wav, p, sample_rate))
    return saved


# -----------------------------
# Prompt builder
# -----------------------------
def _strip_history_bos_eos(
    history_ids: Optional[List[int]],
    bos_id: int = SPECIAL_TOKENS["<bos>"],
    eos_id: int = SPECIAL_TOKENS["<eos>"],
) -> List[int]:
    if not history_ids:
        return []
    left, right = 0, len(history_ids) - 1
    while left <= right and history_ids[left] in (bos_id, eos_id):
        left += 1
    while right >= left and history_ids[right] in (bos_id, eos_id):
        right -= 1
    return history_ids[left : right + 1]


def build_prompt_input_ids(
    audio_tokens: List[int],
    tokenizer: AutoTokenizer,
    user_role: str = "USER",
    assistant_role: str = "GPT",
    user_append_text: Optional[str] = None,
    image_tokens: Optional[List[int]] = None,
    history: Optional[List[int]] = None,
) -> List[int]:
    history_ids = _strip_history_bos_eos(history)
    if history_ids:
        eom_id = SPECIAL_TOKENS["<eom>"]
        if history_ids[-1] != eom_id:
            history_ids.append(eom_id)

    if user_append_text is None:
        user_append_text = DEFAULT_USER_APPEND_TEXT
    append_ids = tokenizer.encode(user_append_text, add_special_tokens=False) if user_append_text != "" else []

    user_prefix_ids = tokenizer.encode(f"{user_role}:", add_special_tokens=False)
    assistant_prefix_ids = tokenizer.encode(f"{assistant_role}:", add_special_tokens=False)

    payload: List[int] = []
    if image_tokens:
        payload.extend(image_tokens)
    payload.extend(audio_tokens)
    payload.extend(append_ids)

    input_ids = (
        [encode_special("<bos>")]
        + history_ids
        + user_prefix_ids
        + payload
        + [encode_special("<eoh>")]
        + assistant_prefix_ids
    )
    return input_ids


# -----------------------------
# Data schema
# -----------------------------
@dataclass
class Turn:
    # New primary input:
    text: Optional[str] = None  # Preferred: text -> speech -> tokens

    # Optional legacy input (kept for backward compatibility):
    wav_path: Optional[str] = None  # WAV -> tokens (only used if text is None)

    image_paths: Optional[List[str]] = None
    user_append_text: str = DEFAULT_USER_APPEND_TEXT

    # CosyVoice2 options
    speaker_wav: Optional[str] = None
    prompt_text: Optional[str] = None

    # Silence padding only
    silence_head_sec: float = 0.0
    silence_tail_sec: float = 0.0
    silence_head_tokens: Optional[List[int]] = None
    silence_tail_tokens: Optional[List[int]] = None

    reset: bool = False
    dialog_id: Optional[str] = None


# -----------------------------
# JSON loader (JSON or JSONL)
# -----------------------------
def load_json_or_jsonl(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.as_posix())

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Empty input file")

    # Try JSON first
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Fallback to JSONL
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_to_dialogs(data: Any) -> List[Dict[str, Any]]:
    """
    Returns list of dialogs:
      [{"dialog_id": str, "turns": [turn_dict,...]}, ...]
    """
    dialogs: List[Dict[str, Any]] = []

    if isinstance(data, dict) and "turns" in data:
        did = str(data.get("dialog_id", "dialog_0000"))
        dialogs.append({"dialog_id": did, "turns": list(data["turns"])})
        return dialogs

    if isinstance(data, list):
        # list could be list of turns OR list of dialogs
        if data and isinstance(data[0], dict) and "turns" in data[0]:
            for d in data:
                did = str(d.get("dialog_id", f"dialog_{len(dialogs):04d}"))
                dialogs.append({"dialog_id": did, "turns": list(d.get("turns", []))})
            return dialogs
        else:
            dialogs.append({"dialog_id": "dialog_0000", "turns": data})
            return dialogs

    raise ValueError("Unsupported JSON structure. Expect list/dict with turns.")


def as_turn(turn_dict: Dict[str, Any]) -> Turn:
    # New preferred input
    text = turn_dict.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"'text' must be a string if provided: {turn_dict}")

    # Legacy input
    wav_path = turn_dict.get("wav_path") or turn_dict.get("audio_path")
    if wav_path is not None and not isinstance(wav_path, str):
        raise ValueError(f"'wav_path' must be a string if provided: {turn_dict}")

    if not text and not wav_path:
        raise ValueError(f"Turn must provide either 'text' or 'wav_path': {turn_dict}")

    silence_head_tokens = turn_dict.get("silence_head_tokens")
    silence_tail_tokens = turn_dict.get("silence_tail_tokens")
    if silence_head_tokens is not None and not isinstance(silence_head_tokens, list):
        raise ValueError("'silence_head_tokens' must be a list[int] if provided")
    if silence_tail_tokens is not None and not isinstance(silence_tail_tokens, list):
        raise ValueError("'silence_tail_tokens' must be a list[int] if provided")

    return Turn(
        text=text,
        wav_path=wav_path,
        image_paths=turn_dict.get("image_paths"),
        user_append_text=turn_dict.get("user_append_text", DEFAULT_USER_APPEND_TEXT),
        speaker_wav=turn_dict.get("speaker_wav"),
        prompt_text=turn_dict.get("prompt_text"),
        silence_head_sec=float(turn_dict.get("silence_head_sec", 0.0) or 0.0),
        silence_tail_sec=float(turn_dict.get("silence_tail_sec", 0.0) or 0.0),
        silence_head_tokens=silence_head_tokens,
        silence_tail_tokens=silence_tail_tokens,
        reset=bool(turn_dict.get("reset", False)),
        dialog_id=turn_dict.get("dialog_id"),
    )


# -----------------------------
# Main runner
# -----------------------------
class BatchRunner:
    def __init__(
        self,
        model_root: Path,
        hf_tokenizer_path: str,
        wavtok_device_id: int,
        decode_device: Union[int, str],
        output_dir: Path,
        base_dir: Optional[Path] = None,
        # speech2tokens.py required args
        cosyvoice_model_dir: Union[str, Path] = "",
        cosyvoice_pythonpath: Optional[Union[str, Path]] = None,
        wavtokenizer_cfg_path: Union[str, Path] = "",
        wavtokenizer_ckpt_path: Union[str, Path] = "",
        wavtokenizer_root: Optional[Union[str, Path]] = None,
        distributed_mode: DistributedMode = DistributedMode.AUTO
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir

        model_root = Path(model_root)
        model_7b = (model_root / "models" / "7b").as_posix()
        tok_text = (model_root / "tokenizer" / "text_tokenizer.json").as_posix()
        tok_img_ckpt = (model_root / "tokenizer" / "vqgan.ckpt").as_posix()
        tok_img_cfg = (model_root / "tokenizer" / "vqgan.yaml").as_posix()

        self.model = AROmniInferenceModel(model_7b, tok_text, tok_img_cfg, tok_img_ckpt, distributed_mode=distributed_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path, use_fast=True)

        if not cosyvoice_model_dir:
            raise ValueError("Missing required argument: cosyvoice_model_dir")
        if not wavtokenizer_cfg_path or not wavtokenizer_ckpt_path:
            raise ValueError("Missing required arguments: wavtokenizer_cfg_path / wavtokenizer_ckpt_path")

        # Text -> speech -> tokens converter
        self.converter = Text2TokenConverter(
            cosyvoice_model_dir=cosyvoice_model_dir,
            cosyvoice_pythonpath=cosyvoice_pythonpath,
            device_id=wavtok_device_id,
            wavtokenizer_cfg_path=wavtokenizer_cfg_path,
            wavtokenizer_ckpt_path=wavtokenizer_ckpt_path,
            wavtokenizer_root=wavtokenizer_root,
        )

        self.decode_device = decode_device

        self.history: List[int] = []
        self.current_dialog_id: Optional[str] = None

    def _resolve_path(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        pp = Path(p)
        if not pp.is_absolute() and self.base_dir is not None:
            pp = self.base_dir / pp
        return pp.as_posix()

    def _resolve_paths(self, paths: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not paths:
            return None
        resolved = []
        for p in paths:
            rp = self._resolve_path(p)
            if rp:
                resolved.append(rp)
        return resolved

    def reset_history(self):
        self.history = []

    def text_to_audio_tokens(self, text: str, bandwidth_id: int, turn: Turn) -> List[int]:
        speaker_wav = self._resolve_path(turn.speaker_wav) if turn.speaker_wav else None
        toks = self.converter(
            text,
            bandwidth_id=bandwidth_id,
            speaker_wav=speaker_wav,
            prompt_text=turn.prompt_text,
            silence_head_sec=turn.silence_head_sec,
            silence_tail_sec=turn.silence_tail_sec,
            silence_head_tokens=turn.silence_head_tokens,
            silence_tail_tokens=turn.silence_tail_tokens,
        )
        return list(toks)

    def wav_to_audio_tokens_legacy(self, wav_path: str, bandwidth_id: int, turn: Turn) -> List[int]:
        wav_path = self._resolve_path(wav_path)
        if not wav_path:
            raise ValueError("Empty wav_path after resolve")
        toks = self.converter.wav_file_to_tokens(
            wav_path,
            bandwidth_id=bandwidth_id,
            silence_head_sec=turn.silence_head_sec,
            silence_tail_sec=turn.silence_tail_sec,
            silence_head_tokens=turn.silence_head_tokens,
            silence_tail_tokens=turn.silence_tail_tokens,
        )
        return list(toks)

    def images_to_tokens(self, image_paths: Optional[Sequence[str]]) -> Optional[List[int]]:
        image_paths = self._resolve_paths(image_paths)
        if not image_paths:
            return None
        all_toks: List[int] = []
        for p in image_paths:
            toks = tokenize_image_from_path(self.model.token_manager, p)
            all_toks.extend(toks)
        return all_toks if all_toks else None

    def run_one_turn(
        self,
        turn: Turn,
        *,
        greedy: bool,
        extra_eos_token: int,
        txt_temp: Optional[float],
        txt_top_p: Optional[float],
        img_temp: Optional[float],
        save_images: bool,
        save_audio: bool,
        save_tokens: bool,
        bandwidth_id: int,
    ) -> Dict[str, Any]:
        # dialog switching / reset logic
        did = turn.dialog_id or self.current_dialog_id or "dialog_0000"
        if self.current_dialog_id is None:
            self.current_dialog_id = did

        if did != self.current_dialog_id:
            self.current_dialog_id = did
            self.reset_history()

        if turn.reset:
            self.reset_history()

        dialog_dir = self.output_dir / did

        # compute turn_index WITHOUT creating directories
        if dialog_dir.exists():
            turn_index = len([p for p in dialog_dir.iterdir() if p.is_dir() and p.name.startswith("turn_")]) + 1
        else:
            turn_index = 1

        run_uid = uuid.uuid4().hex[:8]
        turn_dir = dialog_dir / f"turn_{turn_index:04d}_{run_uid}"

        # -------------------------
        # 1) prepare inputs (NO filesystem write)
        # -------------------------
        if turn.text:
            audio_tokens = self.text_to_audio_tokens(turn.text, bandwidth_id=bandwidth_id, turn=turn)
        else:
            if not turn.wav_path:
                raise ValueError("Turn has neither text nor wav_path (unexpected).")
            audio_tokens = self.wav_to_audio_tokens_legacy(turn.wav_path, bandwidth_id=bandwidth_id, turn=turn)

        image_tokens = self.images_to_tokens(turn.image_paths)

        # 2) prompt
        prompt_ids = build_prompt_input_ids(
            audio_tokens=audio_tokens,
            tokenizer=self.tokenizer,
            user_role="USER",
            assistant_role="GPT",
            user_append_text=turn.user_append_text,
            image_tokens=image_tokens,
            history=self.history,
        )

        # 3) options
        opt = Options()
        opt.txt.greedy = bool(greedy)
        opt.extra_eos_tokens = [int(extra_eos_token)]
        if txt_temp is not None:
            opt.txt.temp = float(txt_temp)
        if txt_top_p is not None:
            opt.txt.top_p = float(txt_top_p)
        if img_temp is not None:
            opt.img.temp = float(img_temp)

        # 4) generate (NO filesystem write)
        out_tokens = self.model.generate(input_ids=prompt_ids, options=opt)
        decoded_texts = self.model.decode_text(out_tokens)

        # 5) decode segments (NO filesystem write)
        decoded_segments = seg_decode(out_tokens, self.model, self.converter.wav_tok, device=self.decode_device)

        # -------------------------
        # Only NOW create output dirs (avoid empty dirs on failure)
        # -------------------------
        dialog_dir.mkdir(parents=True, exist_ok=True)
        turn_dir.mkdir(parents=True, exist_ok=True)

        # 6) save artifacts
        saved_images_list: List[str] = []
        saved_audios_list: List[str] = []

        if save_images:
            saved_images_list = save_images_from_decoded(decoded_segments, turn_dir, run_uid, image_format="png")

        if save_audio:
            saved_audios_list = save_audio_from_decoded(
                decoded_segments,
                turn_dir,
                run_uid,
                sample_rate=int(self.converter.sample_rate),
            )

        # Save decoded text
        text_path = turn_dir / "decoded_text.txt"
        with text_path.open("w", encoding="utf-8") as f:
            if isinstance(decoded_texts, list):
                for t in decoded_texts:
                    f.write(str(t))
                    f.write("\n")
            else:
                f.write(str(decoded_texts))

        # Save meta
        meta = {
            "dialog_id": did,
            "turn_index": turn_index,
            "run_uid": run_uid,
            "text": turn.text,
            "wav_path": self._resolve_path(turn.wav_path) if turn.wav_path else None,
            "image_paths": self._resolve_paths(turn.image_paths),
            "user_append_text": turn.user_append_text,
            "tts": {
                "speaker_wav": self._resolve_path(turn.speaker_wav) if turn.speaker_wav else None,
                "prompt_text": turn.prompt_text,
                "silence_head_sec": turn.silence_head_sec,
                "silence_tail_sec": turn.silence_tail_sec,
                "silence_head_tokens": turn.silence_head_tokens,
                "silence_tail_tokens": turn.silence_tail_tokens,
            },
            "decoded_text_file": text_path.as_posix(),
            "saved_images": saved_images_list,
            "saved_audios": saved_audios_list,
        }
        (turn_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if save_tokens:
            out_ids = out_tokens.tolist()[0] if torch.is_tensor(out_tokens) and out_tokens.dim() == 2 else (
                out_tokens.tolist() if torch.is_tensor(out_tokens) else list(out_tokens)
            )
            (turn_dir / "output_tokens.json").write_text(json.dumps(out_ids), encoding="utf-8")

        # 7) update history
        user_prefix_ids = self.tokenizer.encode("USER:", add_special_tokens=False)
        assistant_prefix_ids = self.tokenizer.encode("GPT:", add_special_tokens=False)
        append_ids = self.tokenizer.encode(turn.user_append_text or "", add_special_tokens=False)
        eoh_id = SPECIAL_TOKENS["<eoh>"]

        user_turn_ids: List[int] = []
        user_turn_ids += user_prefix_ids
        if image_tokens:
            user_turn_ids += image_tokens
        user_turn_ids += audio_tokens + append_ids + [eoh_id]

        assistant_out = out_tokens.tolist()[0] if torch.is_tensor(out_tokens) and out_tokens.dim() == 2 else (
            out_tokens.tolist() if torch.is_tensor(out_tokens) else list(out_tokens)
        )
        assistant_turn_ids = assistant_prefix_ids + assistant_out

        self.history.extend(user_turn_ids + assistant_turn_ids)

        return meta

    def run_dialogs(
        self,
        dialogs: List[Dict[str, Any]],
        *,
        greedy: bool,
        extra_eos_token: int,
        txt_temp: Optional[float],
        txt_top_p: Optional[float],
        img_temp: Optional[float],
        save_images: bool,
        save_audio: bool,
        save_tokens: bool,
        bandwidth_id: int,
    ) -> List[Dict[str, Any]]:
        all_meta: List[Dict[str, Any]] = []

        for d in dialogs:
            did = str(d.get("dialog_id", "dialog_0000"))
            turns = d.get("turns", [])
            if not isinstance(turns, list):
                continue

            self.current_dialog_id = did
            self.reset_history()

            for idx, td in enumerate(turns, start=1):
                if not isinstance(td, dict):
                    continue
                td = dict(td)
                td.setdefault("dialog_id", did)

                turn = as_turn(td)

                # try:
                meta = self.run_one_turn(
                    turn,
                    greedy=greedy,
                    extra_eos_token=extra_eos_token,
                    txt_temp=txt_temp,
                    txt_top_p=txt_top_p,
                    img_temp=img_temp,
                    save_images=save_images,
                    save_audio=save_audio,
                    save_tokens=save_tokens,
                    bandwidth_id=bandwidth_id,
                )
                all_meta.append(meta)
                print(f"[OK] {did} turn#{meta['turn_index']} -> {meta['decoded_text_file']}")

        log_path = self.output_dir / "batch_log.json"
        log_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] log -> {log_path.as_posix()}")
        return all_meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON/JSONL input file")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--base_dir", default=None, help="Base dir for resolving relative paths (default: input file's folder)")

    ap.add_argument("--model_root", required=True, help="Converted model root (contains models/7b and tokenizer/)")
    ap.add_argument("--hf_tokenizer", required=True, help="HF tokenizer path")

    # converter devices
    ap.add_argument("--wavtok_device_id", type=int, default=0, help="WavTokenizer encode device id")
    ap.add_argument("--decode_device", default="cuda:0", help="Decode device (e.g. cuda:0, cpu, or int)")

    # speech2tokens.py required args
    ap.add_argument("--cosyvoice_model_dir", required=True, help="CosyVoice2 model dir")
    ap.add_argument("--cosyvoice_pythonpath", default=None, help="Optional path to CosyVoice repo if not installed")
    ap.add_argument("--wavtokenizer_cfg_path", required=True, help="WavTokenizer cfg path")
    ap.add_argument("--wavtokenizer_ckpt_path", required=True, help="WavTokenizer ckpt path")
    ap.add_argument("--wavtokenizer_root", default=None, help="Optional path to WavTokenizer repo if not installed")

    # generation params
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding for text")
    ap.add_argument("--extra_eos_token", type=int, default=12831, help="extra eos token id (default 12831)")
    ap.add_argument("--txt_temp", type=float, default=None)
    ap.add_argument("--txt_top_p", type=float, default=None)
    ap.add_argument("--img_temp", type=float, default=None)

    # wavtokenizer bandwidth id for encode/decode
    ap.add_argument("--bandwidth_id", type=int, default=0)

    # saving
    ap.add_argument("--save_images", action="store_true", help="Save decoded images")
    ap.add_argument("--save_audio", action="store_true", help="Save decoded audios to wav")
    ap.add_argument("--save_tokens", action="store_true", help="Save raw output token ids (can be large)")
    return ap.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.base_dir) if args.base_dir else input_path.parent

    data = load_json_or_jsonl(input_path)
    dialogs = normalize_to_dialogs(data)

    runner = BatchRunner(
        model_root=Path(args.model_root),
        hf_tokenizer_path=args.hf_tokenizer,
        wavtok_device_id=args.wavtok_device_id,
        decode_device=args.decode_device if not str(args.decode_device).isdigit() else int(args.decode_device),
        output_dir=output_dir,
        base_dir=base_dir,
        cosyvoice_model_dir=args.cosyvoice_model_dir,
        cosyvoice_pythonpath=args.cosyvoice_pythonpath,
        wavtokenizer_cfg_path=args.wavtokenizer_cfg_path,
        wavtokenizer_ckpt_path=args.wavtokenizer_ckpt_path,
        wavtokenizer_root=args.wavtokenizer_root,
    )

    runner.run_dialogs(
        dialogs,
        greedy=args.greedy,
        extra_eos_token=args.extra_eos_token,
        txt_temp=args.txt_temp,
        txt_top_p=args.txt_top_p,
        img_temp=args.img_temp,
        save_images=args.save_images,
        save_audio=args.save_audio,
        save_tokens=args.save_tokens,
        bandwidth_id=int(args.bandwidth_id),
    )


if __name__ == "__main__":
    main()
