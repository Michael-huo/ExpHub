import argparse
import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 禁用 HuggingFace 原生的模型下载/加载进度条，避免污染日志系统
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from _common import ensure_dir, list_frames_sorted, log_info, log_prog, log_warn, write_json_atomic

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IDX_RE = re.compile(r"(\d+)")

def _resolve_frames_dir(p: Path) -> Path:
    p = p.resolve()
    if p.is_dir() and p.name == "frames":
        return p
    if (p / "frames").is_dir():
        return (p / "frames").resolve()
    return p


def _build_idx_map(frames_dir: Path) -> Tuple[Dict[int, Path], int]:
    """
    Build idx->path map based on numeric stem.
    Prefer png > jpg > jpeg > webp if duplicates.
    frames_avail = max_idx + 1
    """
    idx2path: Dict[int, Path] = {}
    max_idx = -1
    ext_rank = {".png": 0, ".jpg": 1, ".jpeg": 2, ".webp": 3}

    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMG_EXTS:
            continue

        stem = p.stem
        if stem.isdigit():
            idx = int(stem)
        else:
            m = IDX_RE.search(stem)
            if not m:
                continue
            idx = int(m.group(1))

        max_idx = max(max_idx, idx)

        if idx in idx2path:
            old = idx2path[idx]
            if ext_rank.get(ext, 99) < ext_rank.get(old.suffix.lower(), 99):
                idx2path[idx] = p
        else:
            idx2path[idx] = p

    frames_avail = max_idx + 1 if max_idx >= 0 else 0
    return idx2path, frames_avail


def _auto_kf_gap(fps: int) -> int:
    g = fps - (fps % 4)
    return g if g > 0 else fps


def _compute_num_clips(frames_avail: int, base_idx: int, kf_gap: int, num_segments: int) -> int:
    # same as ExpHub/Wan: (frames_avail - 1 - base_idx) // kf_gap
    max_segments = (frames_avail - 1 - base_idx) // kf_gap
    if max_segments <= 0:
        return 0
    return min(max_segments, num_segments) if num_segments > 0 else max_segments


def _rep_indices_for_clip(start_idx: int, end_idx: int) -> List[int]:
    gap = end_idx - start_idx
    if gap <= 0:
        return [start_idx]
    candidates = [
        start_idx,
        start_idx + int(round(gap * 0.25)),
        start_idx + int(round(gap * 0.50)),
        start_idx + int(round(gap * 0.75)),
        end_idx,
    ]
    out, seen = [], set()
    for x in candidates:
        x = max(start_idx, min(end_idx, x))
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


DEFAULT_BASE_PROMPT = (
    "First-person camera moving forward along an outdoor park walkway. Photorealistic. Stable exposure and white balance. Consistent perspective and geometry, level horizon. Sharp, stable textures on pavement, grass, and trees. No flicker, no warping, no artifacts. "

)
DEFAULT_BASE_NEG = (
    "blurry, flickering, warping, wobble, rolling shutter artifacts, ghosting, double edges, inconsistent geometry, wrong perspective, texture swimming, repeating patterns, oversharpening halos, heavy motion blur, text, watermark, jpeg artifacts, excessive noise, color shift, low quality, crowds, many people, fast moving objects "

)
DEFAULT_INSTR = (
    "You will be given multiple frames sampled from a short video segment.\n"
    "Write a concise prompt for image generation in 1–2 sentences, <= 60 tokens.\n"
    "Focus on scene type, main structures, main objects, lighting.\n"
    "Do NOT describe camera motion (no turning, accelerating, shaking, panning).\n"
    "Do NOT list too many details.\n"
    "Output ONLY the prompt text."
)


def _clean_prompt(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = " ".join([x.strip() for x in s.splitlines() if x.strip()])
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--segment_dir", default="", help="segment dir (contains frames/) OR frames dir")
    ap.add_argument("--frames_dir", default="", help="direct frames dir override")
    ap.add_argument("--exp_dir", default="", help="if set, outputs go under <exp_dir>/segment and <exp_dir>/prompts")

    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--kf_gap", type=int, default=0)
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--num_segments", type=int, default=0)

    ap.add_argument("--model_dir", default="/data/hx/Qwen2-VL-Prompt/models/Qwen2-VL-7B-Instruct")
    ap.add_argument("--use_fast", action="store_true", help="use fast processor (default False)")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)

    ap.add_argument("--base_prompt", default=DEFAULT_BASE_PROMPT)
    ap.add_argument("--base_neg_prompt", default=DEFAULT_BASE_NEG)
    ap.add_argument("--instr", default=DEFAULT_INSTR)
    ap.add_argument("--max_new_tokens", type=int, default=80)

    ap.add_argument("--out_json", default="", help="output clip_prompts.json")
    ap.add_argument("--out_manifest", default="", help="output prompt_manifest.json")

    args = ap.parse_args()

    if args.frames_dir.strip():
        frames_dir = Path(args.frames_dir).resolve()
    elif args.segment_dir.strip():
        frames_dir = _resolve_frames_dir(Path(args.segment_dir))
    else:
        raise SystemExit("[ERR] must provide --frames_dir or --segment_dir")

    frames_dir = ensure_dir(frames_dir, "frames_dir")

    model_dir = Path(args.model_dir).resolve()
    try:
        model_dir = ensure_dir(model_dir, "model_dir")
    except SystemExit:
        raise SystemExit(
            "[ERR] model_dir not found or not a directory: {}. "
            "Please set --model_dir to a valid Qwen2-VL model directory.".format(model_dir)
        )

    # Readability smoke check on one frame, fail early with clear message.
    frame_files = list_frames_sorted(frames_dir)
    if not frame_files:
        raise SystemExit(f"[ERR] no image files found in frames_dir: {frames_dir}")
    first_img = frame_files[0]
    try:
        with first_img.open("rb") as fh:
            fh.read(32)
    except Exception as e:
        raise SystemExit(f"[ERR] cannot read sample frame: {first_img} ({e})")

    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir.strip() else None
    if exp_dir:
        out_json = Path(args.out_json).resolve() if args.out_json else (exp_dir / "segment" / "clip_prompts.json")
        out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else (exp_dir / "prompts" / "manifest.json")
    else:
        if not args.out_json or not args.out_manifest:
            raise SystemExit("[ERR] without --exp_dir, you must provide --out_json and --out_manifest")
        out_json = Path(args.out_json).resolve()
        out_manifest = Path(args.out_manifest).resolve()

    fps = int(args.fps)
    if fps <= 0:
        raise SystemExit("[ERR] --fps must be > 0")

    kf_gap = int(args.kf_gap) if int(args.kf_gap) > 0 else _auto_kf_gap(fps)
    base_idx = int(args.base_idx)
    if base_idx < 0:
        raise SystemExit("[ERR] --base_idx must be >= 0")

    idx2path, frames_avail = _build_idx_map(frames_dir)
    if frames_avail <= 0:
        raise SystemExit(f"[ERR] no frames found in {frames_dir}")
    if base_idx >= frames_avail:
        raise SystemExit(f"[ERR] base_idx={base_idx} out of range (frames_avail={frames_avail})")

    nclips = _compute_num_clips(frames_avail, base_idx, kf_gap, int(args.num_segments))
    if nclips <= 0:
        raise SystemExit(
            f"[ERR] not enough frames for 1 clip: frames_avail={frames_avail} base_idx={base_idx} kf_gap={kf_gap}"
        )

    # Load model once
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        use_fast=bool(args.use_fast),
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    # ---- sanitize generation_config (avoid ignored-flags warning & avoid deprecation) ----
    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.num_beams = 1
    gen_cfg.temperature = 1.0
    gen_cfg.top_p = 1.0
    gen_cfg.top_k = 50  # keep HF default to avoid advisory warning in greedy mode
    # Put max_new_tokens into generation_config so we DON'T pass both (fix deprecation warning)
    setattr(gen_cfg, "max_new_tokens", int(args.max_new_tokens))

    log_info(
        "loaded model+processor in {:.2f}s | frames_avail={} | clips={} | kf_gap={}".format(
            time.time() - t0, frames_avail, nclips, kf_gap
        )
    )

    clips: List[dict] = []
    errors: List[str] = []

    with torch.inference_mode():
        for clip_id in tqdm(
            range(nclips),
            desc="Prompt Gen",
            bar_format="[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            start_idx = base_idx + clip_id * kf_gap
            end_idx = start_idx + kf_gap
            clip_seconds = float(end_idx - start_idx) / float(fps)

            rep_idx = _rep_indices_for_clip(start_idx, end_idx)

            rep_paths: List[Path] = []
            missing: List[int] = []
            for idx in rep_idx:
                p = idx2path.get(idx)
                if p is None:
                    missing.append(idx)
                else:
                    rep_paths.append(p)

            if missing:
                msg = f"missing frames for clip {clip_id}: idx={missing} (start_idx={start_idx}, end_idx={end_idx})"
                errors.append(msg)
                clips.append(
                    {
                        "clip_id": int(clip_id),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "clip_seconds": clip_seconds,
                        "rep_frames": [f"{i:06d}" for i in rep_idx],
                        "prompt": "",
                        "error": msg,
                    }
                )
                continue

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": f"file://{str(p)}"} for p in rep_paths]
                    + [{"type": "text", "text": args.instr}],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Only pass generation_config (no max_new_tokens here) -> removes deprecation warning
            out_ids = model.generate(
                **inputs,
                generation_config=gen_cfg,
            )

            gen = out_ids[:, inputs["input_ids"].shape[1] :]
            prompt = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            prompt = _clean_prompt(prompt)

            clips.append(
                {
                    "clip_id": int(clip_id),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "clip_seconds": clip_seconds,
                    "rep_frames": [p.name for p in rep_paths],
                    "prompt": prompt,
                }
            )

    clip_prompts = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(model_dir),
        "frames_dir": str(frames_dir),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "base_idx": int(base_idx),
        "num_clips": int(nclips),
        "rep_policy": "start+quartiles+end",
        "max_new_tokens": int(args.max_new_tokens),
        "use_fast": bool(args.use_fast),
        "clips": clips,
        "errors": errors,
    }
    write_json_atomic(out_json, clip_prompts, indent=2)

    # Wan manifest (Wan-compatible)
    seg_items = []
    for it in clips:
        seg_items.append(
            {
                "seg": int(it["clip_id"]),
                "clip_id": int(it["clip_id"]),  # redundant alias
                "delta_prompt": str(it.get("prompt", "") or "").strip(),
            }
        )

    manifest = {
        "version": 1,
        "base_prompt": (str(args.base_prompt).strip() or DEFAULT_BASE_PROMPT),
        "base_neg_prompt": str(args.base_neg_prompt).strip(),
        "segments": seg_items,
    }
    write_json_atomic(out_manifest, manifest, indent=2)

    # Compact step metadata (additional, non-breaking).
    manifest_bytes = out_manifest.read_bytes()
    manifest_size = int(len(manifest_bytes))
    clip_prompts_size = 0
    if out_json.is_file():
        try:
            clip_prompts_size = int(os.path.getsize(str(out_json)))
        except Exception:
            clip_prompts_size = 0
    outputs_bytes_sum = int(manifest_size + clip_prompts_size)

    step_meta = {
        "step": "prompt",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(model_dir),
        "prompt_style": "qwen2vl_delta",
        "frames_count": int(len(frame_files)),
        "clips_count": int(nclips),
        "manifest_path": str(out_manifest),
        "manifest_size": int(manifest_size),
        "manifest_sha1": hashlib.sha1(manifest_bytes).hexdigest(),
        "outputs": {
            "bytes_sum": int(outputs_bytes_sum),
            "manifest_bytes_sum": int(manifest_size),
            "clip_prompts_bytes_sum": int(clip_prompts_size),
            "manifest_file_count": 1,
            "clip_prompts_file_count": 1 if out_json.is_file() else 0,
        },
    }
    write_json_atomic(out_manifest.parent / "step_meta.json", step_meta, indent=2)

    log_prog("prompt clips generated: {}/{}".format(int(len(clips)), int(nclips)))
    log_info("wrote: {}".format(out_json))
    log_info("wrote: {}".format(out_manifest))
    log_info("wrote: {}".format(out_manifest.parent / "step_meta.json"))
    if errors:
        log_warn(f"{len(errors)} clips had missing frames; see errors in clip_prompts.json")


if __name__ == "__main__":
    main()
