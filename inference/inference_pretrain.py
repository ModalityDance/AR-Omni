# infer_all.py
"""
Single-input inference for 4 tasks (no batching / no parallelism):
  - t2i:  one text prompt -> one image
  - caption: one image -> one caption
  - asr:  one audio path -> one transcript
  - tts:  one text prompt -> one wav

Change note:
- Removed the literal "<reserved12826>" from T2I prompts.
- We now insert the real boundary token explicitly: <eoh> (id=12830).
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer
from chameleon.inference.aromni import AROmniInferenceModel, Options

# ---- Model-specific token IDs ----
SPECIAL = {
    "<bos>": 0,
    "<eos>": 2,
    "<boa>": 12821,  # begin-of-audio
    "<eoa>": 12820,  # end-of-audio
    "<eoh>": 12830,  # end-of-human turn
}
EXTRA_EOS = 12831

# ---- Audio token protocol (for ASR input / TTS output post-processing) ----
AUDIO_TOKEN_MIN = 8724
AUDIO_TOKEN_MAX = 12819
AUDIO_OFFSET = 8724
TTS_SAMPLE_RATE = 24000

# ---- Default single-instruction strings ----
DEFAULT_CAPTION_INSTR = "Describe this image."
DEFAULT_T2I_INSTR = "Create a picture that matches the essence of this text."
DEFAULT_ASR_INSTR = "Can you please convert this speech into written text?"
DEFAULT_TTS_INSTR = "Convert this text into speech."


# =========================
# Utilities
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_torch_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def load_chameleon(ckpt_path: str) -> AROmniInferenceModel:
    ckpt = Path(ckpt_path)
    return AROmniInferenceModel(
        (ckpt / "models" / "7b").as_posix(),
        (ckpt / "tokenizer" / "text_tokenizer.json").as_posix(),
        (ckpt / "tokenizer" / "vqgan.yaml").as_posix(),
        (ckpt / "tokenizer" / "vqgan.ckpt").as_posix(),
    )

def enc(tokenizer, s: str) -> List[int]:
    return tokenizer.encode(s, add_special_tokens=False)


# =========================
# Options factories
# =========================
def make_opts_text(max_gen_len: int) -> Options:
    opts = Options()
    opts.img = False
    opts.txt.greedy = True
    opts.extra_eos_tokens = [EXTRA_EOS]
    opts.max_gen_len = max_gen_len
    return opts

def make_opts_asr(max_seq_len: int) -> Options:
    opts = Options()
    opts.img = False
    opts.txt.greedy = True
    opts.extra_eos_tokens = [EXTRA_EOS]
    opts.max_seq_len = max_seq_len
    return opts

def make_opts_t2i(temp: float, guidance_scale_image: float) -> Options:
    opts = Options()
    opts.txt = False  # only image tokens
    opts.extra_eos_tokens = [EXTRA_EOS]
    opts.img.temp = temp
    opts.img.cfg.guidance_scale_image = guidance_scale_image
    return opts


# =========================
# Prompt builders (single input)
# =========================
def build_caption_prompt_ids(tokenizer, instruction: str, image_tokens: List[int]) -> List[int]:
    # Assumes image_tokens are already in the correct format for direct insertion.
    return (
        [SPECIAL["<bos>"]]
        + enc(tokenizer, "USER:")
        + image_tokens
        + enc(tokenizer, instruction)
        + [SPECIAL["<eoh>"]]
        + enc(tokenizer, "GPT:")
    )

def build_t2i_prompt_ids(tokenizer, instruction: str, text: str, add_bos: bool = True) -> List[int]:
    """
    Tokenized T2I prompt (no "<reserved12826>" literal).
    Equivalent structure to your original string:
      USER:{instruction} This is input: {text} <eoh> GPT:
    """
    ids = []
    if add_bos:
        ids.append(SPECIAL["<bos>"])
    ids += enc(tokenizer, "USER:")
    ids += enc(tokenizer, instruction)
    ids += enc(tokenizer, " This is input: ")  # keep the space like your original t2i string
    ids += enc(tokenizer, text)
    ids += [SPECIAL["<eoh>"]]
    ids += enc(tokenizer, " GPT:")             # keep the space like your original t2i string
    return ids

def build_asr_prompt_ids(tokenizer, instruction: str, model_audio_tokens: List[int]) -> List[int]:
    return (
        [SPECIAL["<bos>"]]
        + enc(tokenizer, "USER:")
        + enc(tokenizer, instruction)
        + [SPECIAL["<boa>"]]
        + model_audio_tokens
        + [SPECIAL["<eoa>"], SPECIAL["<eoh>"]]
        + enc(tokenizer, "GPT:")
    )

def build_tts_prompt_ids(tokenizer, instruction: str, text: str) -> List[int]:
    return (
        [SPECIAL["<bos>"]]
        + enc(tokenizer, "USER:")
        + enc(tokenizer, instruction)
        + enc(tokenizer, "This is input: ")
        + enc(tokenizer, text)
        + [SPECIAL["<eoh>"]]
        + enc(tokenizer, "GPT:")
    )


# =========================
# Image/audio token helpers (best-effort, with explicit fallbacks)
# =========================
def load_image_tokens_from_args(args) -> List[int]:
    if args.image_tokens_json:
        tok = load_json(args.image_tokens_json)
        if not isinstance(tok, list):
            raise ValueError("--image_tokens_json must be a JSON list of ints.")
        return [int(x) for x in tok]

    if args.image_path:
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "caption with --image_path requires Pillow (pip install Pillow), "
                "or provide --image_tokens_json."
            ) from e

        img = Image.open(args.image_path).convert("RGB")
        model = args._model  # injected in main()

        for fn_name in ("encode_image", "tokenize_image", "image_to_tokens"):
            if hasattr(model, fn_name):
                out = getattr(model, fn_name)(img)
                if hasattr(out, "tolist"):
                    out = out.tolist()
                if not isinstance(out, list):
                    raise RuntimeError(f"{fn_name} returned unsupported type: {type(out)}")
                return [int(x) for x in out]

        raise RuntimeError(
            "AROmniInferenceModel does not expose an image encoder method "
            "(encode_image/tokenize_image/image_to_tokens). "
            "Please provide --image_tokens_json instead."
        )

    raise ValueError("caption requires either --image_path or --image_tokens_json.")

def load_asr_model_audio_tokens(args, device: torch.device) -> List[int]:
    """
    Returns model-vocab audio tokens (already shifted by AUDIO_OFFSET).
    Priority:
      1) --audio_tokens_json : already model tokens
      2) --audio_codes_json  : wavtokenizer codes (0-based), will be shifted by AUDIO_OFFSET
      3) --audio_path        : encode with WavTokenizer, then shift by AUDIO_OFFSET
    """
    if args.audio_tokens_json:
        tok = load_json(args.audio_tokens_json)
        if not isinstance(tok, list):
            raise ValueError("--audio_tokens_json must be a JSON list of ints.")
        return [int(x) for x in tok]

    if args.audio_codes_json:
        codes = load_json(args.audio_codes_json)
        if not isinstance(codes, list):
            raise ValueError("--audio_codes_json must be a JSON list of ints.")
        return [int(x) + AUDIO_OFFSET for x in codes]

    if not args.audio_path:
        raise ValueError("asr requires one of: --audio_path / --audio_tokens_json / --audio_codes_json")

    if args.wavtokenizer_root and args.wavtokenizer_root not in sys.path:
        sys.path.append(args.wavtokenizer_root)

    try:
        from decoder.pretrained import WavTokenizer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "asr with --audio_path requires WavTokenizer importable as `decoder.pretrained`, "
            "or provide --audio_tokens_json / --audio_codes_json. "
            "You can also pass --wavtokenizer_root /path/to/WavTokenizer."
        ) from e

    try:
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "asr with --audio_path requires torchaudio (pip install torchaudio), "
            "or provide --audio_tokens_json / --audio_codes_json."
        ) from e

    if not args.wavtokenizer_config or not args.wavtokenizer_ckpt:
        raise ValueError("asr with --audio_path also requires --wavtokenizer_config and --wavtokenizer_ckpt")

    wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtokenizer_config, args.wavtokenizer_ckpt).to(device)

    wav, sr = torchaudio.load(args.audio_path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TTS_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TTS_SAMPLE_RATE)

    codes = None
    if hasattr(wavtokenizer, "encode"):
        try:
            codes = wavtokenizer.encode(wav.to(device))
        except TypeError:
            codes = wavtokenizer.encode(wav.to(device), sample_rate=TTS_SAMPLE_RATE)
    elif hasattr(wavtokenizer, "wav_to_codes"):
        codes = wavtokenizer.wav_to_codes(wav.to(device))
    else:
        raise RuntimeError(
            "WavTokenizer has no supported encode API (encode / wav_to_codes). "
            "Please provide --audio_tokens_json / --audio_codes_json."
        )

    if hasattr(codes, "detach"):
        codes = codes.detach()
    if hasattr(codes, "cpu"):
        codes = codes.cpu()
    if hasattr(codes, "tolist"):
        codes = codes.tolist()

    while isinstance(codes, list) and len(codes) > 0 and isinstance(codes[0], list):
        codes = codes[0]
    if not isinstance(codes, list):
        raise RuntimeError(f"Unexpected WavTokenizer codes type: {type(codes)}")

    return [int(x) + AUDIO_OFFSET for x in codes]


# =========================
# TTS helpers
# =========================
def extract_audio_codes_from_output(output_tokens: List[int]) -> List[int]:
    return [t - AUDIO_OFFSET for t in output_tokens if AUDIO_TOKEN_MIN <= t <= AUDIO_TOKEN_MAX]

def save_wav_from_codes(wavtokenizer, codes: List[int], out_path: Path, device: torch.device):
    tokens_tensor = torch.tensor(codes, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
    features = wavtokenizer.codes_to_features(tokens_tensor)
    bandwidth_id = torch.tensor([0]).to(device)
    audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).cpu()
    import torchaudio
    torchaudio.save(out_path.as_posix(), audio, sample_rate=TTS_SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)


# =========================
# Task runners (single input)
# =========================
def run_caption(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = args._model

    image_tokens = load_image_tokens_from_args(args)
    prompt = build_caption_prompt_ids(tokenizer, args.instruction, image_tokens)

    opts = make_opts_text(max_gen_len=args.max_gen_len)

    with torch.inference_mode():
        outputs = model.generate(batch_input_ids=[prompt], options=opts)
        decoded = model.decode_text(outputs)[0].strip()

    out_dir = Path(args.out_dir) / "caption"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "result.json", {
        "image_path": args.image_path,
        "image_tokens_json": args.image_tokens_json,
        "instruction": args.instruction,
        "generated_text": decoded,
    })
    print(decoded)

def run_t2i(args, device: torch.device):
    try:
        from PIL import Image  # noqa: F401
    except ImportError as e:
        raise ImportError("t2i requires Pillow. Install with: pip install Pillow") from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = args._model

    prompt_ids = build_t2i_prompt_ids(tokenizer, args.instruction, args.text, add_bos=(not args.no_bos))
    opts = make_opts_t2i(temp=args.temp, guidance_scale_image=args.guidance_scale_image)

    with torch.inference_mode():
        image_tokens = model.generate(batch_input_ids=[prompt_ids], options=opts)
        images = model.decode_image(image_tokens)

    out_dir = Path(args.out_dir) / "t2i"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_img = img_dir / (args.out_name if args.out_name else "result.png")
    images[0].save(out_img)

    dump_json(out_dir / "result.json", {
        "text": args.text,
        "instruction": args.instruction,
        "temp": args.temp,
        "guidance_scale_image": args.guidance_scale_image,
        "image_path": str(out_img.relative_to(out_dir)),
    })
    print(str(out_img))

def run_asr(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = args._model

    model_audio_tokens = load_asr_model_audio_tokens(args, device=device)
    prompt = build_asr_prompt_ids(tokenizer, args.instruction, model_audio_tokens)

    opts = make_opts_asr(max_seq_len=args.max_seq_len)

    with torch.inference_mode():
        outputs = model.generate(batch_input_ids=[prompt], options=opts)
        decoded = model.decode_text(outputs)[0].strip()

    out_dir = Path(args.out_dir) / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "result.json", {
        "audio_path": args.audio_path,
        "audio_tokens_json": args.audio_tokens_json,
        "audio_codes_json": args.audio_codes_json,
        "instruction": args.instruction,
        "decoded_text": decoded,
    })
    print(decoded)

def run_tts(args, device: torch.device):
    if args.wavtokenizer_root and args.wavtokenizer_root not in sys.path:
        sys.path.append(args.wavtokenizer_root)

    try:
        from decoder.pretrained import WavTokenizer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "tts requires WavTokenizer importable as `decoder.pretrained`.\n"
            "Either install it properly, or pass --wavtokenizer_root /path/to/WavTokenizer."
        ) from e

    try:
        import torchaudio  # noqa: F401
    except ImportError as e:
        raise ImportError("tts requires torchaudio. Install with: pip install torchaudio") from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = args._model
    wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtokenizer_config, args.wavtokenizer_ckpt).to(device)

    prompt = build_tts_prompt_ids(tokenizer, args.instruction, args.text)
    opts = make_opts_text(max_gen_len=args.max_gen_len)

    with torch.inference_mode():
        outputs = model.generate(batch_input_ids=[prompt], options=opts)
    out_tokens = outputs[0].tolist()

    codes = extract_audio_codes_from_output(out_tokens)
    if not codes:
        raise RuntimeError("No audio tokens found in model output (check token range / prompt formatting).")

    out_dir = Path(args.out_dir) / "tts"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_wav = wav_dir / (args.out_name if args.out_name else "result.wav")
    save_wav_from_codes(wavtokenizer, codes, out_wav, device=device)

    dump_json(out_dir / "result.json", {
        "text": args.text,
        "instruction": args.instruction,
        "wav_path": str(out_wav.relative_to(out_dir)),
    })
    print(str(out_wav))


# =========================
# CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser("Single-input inference: caption / t2i / asr / tts")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--tokenizer_path", type=str, required=True)

    sp = p.add_subparsers(dest="task", required=True)

    cap = sp.add_parser("caption", help="one image -> one caption")
    cap.add_argument("--instruction", type=str, default=DEFAULT_CAPTION_INSTR)
    cap.add_argument("--max_gen_len", type=int, default=128)
    cap.add_argument("--image_path", type=str, default="", help="optional: raw image path (requires model image encoder)")
    cap.add_argument("--image_tokens_json", type=str, default="", help="optional: JSON list[int] image tokens")

    t2i = sp.add_parser("t2i", help="one prompt -> one image")
    t2i.add_argument("--text", type=str, required=True)
    t2i.add_argument("--instruction", type=str, default=DEFAULT_T2I_INSTR)
    t2i.add_argument("--temp", type=float, default=1.0)
    t2i.add_argument("--guidance_scale_image", type=float, default=1.32)
    t2i.add_argument("--no_bos", action="store_true", help="do not prepend <bos> for t2i")
    t2i.add_argument("--out_name", type=str, default="", help="output filename (default: result.png)")

    asr = sp.add_parser("asr", help="one audio path -> one transcript")
    asr.add_argument("--instruction", type=str, default=DEFAULT_ASR_INSTR)
    asr.add_argument("--max_seq_len", type=int, default=1024)
    asr.add_argument("--audio_path", type=str, default="", help="optional: raw audio path (requires WavTokenizer+torchaudio)")
    asr.add_argument("--audio_tokens_json", type=str, default="", help="optional: JSON list[int] model audio tokens")
    asr.add_argument("--audio_codes_json", type=str, default="", help="optional: JSON list[int] wavtokenizer codes (0-based)")
    asr.add_argument("--wavtokenizer_root", type=str, default="", help="optional: path to WavTokenizer repo root")
    asr.add_argument("--wavtokenizer_config", type=str, default="")
    asr.add_argument("--wavtokenizer_ckpt", type=str, default="")

    tts = sp.add_parser("tts", help="one prompt -> one wav")
    tts.add_argument("--text", type=str, required=True)
    tts.add_argument("--instruction", type=str, default=DEFAULT_TTS_INSTR)
    tts.add_argument("--max_gen_len", type=int, default=256)
    tts.add_argument("--wavtokenizer_root", type=str, default="", help="optional: path to WavTokenizer repo root")
    tts.add_argument("--wavtokenizer_config", type=str, required=True)
    tts.add_argument("--wavtokenizer_ckpt", type=str, required=True)
    tts.add_argument("--out_name", type=str, default="", help="output filename (default: result.wav)")

    return p

def main():
    args = build_parser().parse_args()
    device = get_torch_device(args.device)

    model = load_chameleon(args.ckpt_path)
    args._model = model  # inject for helpers

    if args.task == "caption":
        run_caption(args, device=device)
    elif args.task == "t2i":
        run_t2i(args, device=device)
    elif args.task == "asr":
        run_asr(args, device=device)
    elif args.task == "tts":
        run_tts(args, device=device)
    else:
        raise ValueError(args.task)

if __name__ == "__main__":
    main()
