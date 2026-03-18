import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 禁用 HuggingFace 原生的模型下载/加载进度条，避免污染日志系统
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from tqdm import tqdm

from _common import ensure_dir, get_platform_config, log_info, log_prog, log_warn, write_json_atomic
from _prompt.api import create_backend
from _prompt.backends.smolvlm2_backend import DEFAULT_SMOLVLM2_MODEL_ID
from _prompt.manifest_v2 import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    build_global_invariants,
    build_intent_instruction,
    build_manifest_v2,
    compile_legacy_prompts,
    parse_intent_response,
)
from _prompt.sampling import list_images, sample_images
from _schedule import (
    build_execution_segments_from_deploy_schedule,
    build_legacy_execution_segments,
    load_deploy_schedule,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
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
    Prefer png > jpg > jpeg > webp > bmp if duplicates.
    frames_avail = max_idx + 1
    """
    idx2path = {}  # type: Dict[int, Path]
    max_idx = -1
    ext_rank = {".png": 0, ".jpg": 1, ".jpeg": 2, ".webp": 3, ".bmp": 4}

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
    max_segments = (frames_avail - 1 - base_idx) // kf_gap
    if max_segments <= 0:
        return 0
    return min(max_segments, num_segments) if num_segments > 0 else max_segments


def _find_deploy_schedule(frames_dir: Path, exp_dir: Optional[Path]) -> Optional[Path]:
    candidates = []  # type: List[Path]
    if exp_dir is not None:
        candidates.append((exp_dir / "segment" / "deploy_schedule.json").resolve())
    if frames_dir.name == "frames":
        candidates.append((frames_dir.parent / "deploy_schedule.json").resolve())

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


DEFAULT_BASE_PROMPT = (
    "First-person camera moving forward along an outdoor park walkway. Photorealistic. Stable exposure and white balance. Consistent perspective and geometry, level horizon. Sharp, stable textures on pavement, grass, and trees. No flicker, no warping, no artifacts. "
)
DEFAULT_BASE_NEG = (
    "blurry, flickering, warping, wobble, rolling shutter artifacts, ghosting, double edges, inconsistent geometry, wrong perspective, texture swimming, repeating patterns, oversharpening halos, heavy motion blur, text, watermark, jpeg artifacts, excessive noise, color shift, low quality, crowds, many people, fast moving objects "
)
DEFAULT_INSTR = (
    "You will be given multiple frames sampled from a short video segment.\n"
    "Extract a structured intent card for video generation.\n"
    "Capture stable scene identity, motion intent, geometry constraints, appearance constraints, and suppressions.\n"
    "Keep the result conservative and compact.\n"
    "Output JSON only."
)


def _clean_prompt(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    text = " ".join([line.strip() for line in text.splitlines() if line.strip()])
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def _resolve_default_qwen_model() -> str:
    try:
        cfg = get_platform_config()
        return str(cfg.get("models", {}).get("qwen2_vl", {}).get("path", "") or "").strip()
    except Exception:
        return ""


def _resolve_model_ref(args, default_qwen_model):  # type: (argparse.Namespace, str) -> str
    raw_model = str(args.model_dir or "").strip()
    if raw_model:
        return raw_model
    backend = str(args.backend or "smolvlm2").strip().lower()
    if backend == "qwen":
        return str(default_qwen_model or "").strip()
    if backend == "smolvlm2":
        return DEFAULT_SMOLVLM2_MODEL_ID
    return raw_model


def _resolve_max_new_tokens(args):  # type: (argparse.Namespace) -> int
    raw_value = int(args.max_new_tokens)
    if raw_value > 0:
        return raw_value
    if str(args.backend or "smolvlm2").strip().lower() == "smolvlm2":
        return 48
    return 80


def _rep_policy_name(sample_mode: str, num_images: int) -> str:
    mode = str(sample_mode or "even").strip().lower()
    if mode == "quartiles" and int(num_images) == 5:
        return "start+quartiles+end"
    return mode


def _build_instruction(base_instruction: str, backend_name: str, structured: bool) -> str:
    return build_intent_instruction(base_instruction or DEFAULT_INSTR, backend_name, strict_json=bool(structured))


def _collect_clip_files(idx2path, start_idx, end_idx):  # type: (Dict[int, Path], int, int) -> List[str]
    out = []  # type: List[str]
    for idx in range(int(start_idx), int(end_idx) + 1):
        path_obj = idx2path.get(idx)
        if path_obj is None:
            continue
        out.append(str(path_obj.resolve()))
    return out


def _extract_rep_indices(selected_paths):  # type: (List[str]) -> List[int]
    out = []  # type: List[int]
    for path_text in selected_paths:
        stem = Path(path_text).stem
        if stem.isdigit():
            out.append(int(stem))
            continue
        match = IDX_RE.search(stem)
        if match:
            out.append(int(match.group(1)))
    return out


def main():
    default_qwen_model = _resolve_default_qwen_model()

    ap = argparse.ArgumentParser()

    ap.add_argument("--segment_dir", default="", help="segment dir (contains frames/) OR frames dir")
    ap.add_argument("--frames_dir", default="", help="direct frames dir override")
    ap.add_argument("--exp_dir", default="", help="if set, outputs go under <exp_dir>/segment and <exp_dir>/prompt")

    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--kf_gap", type=int, default=0)
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--num_segments", type=int, default=0)

    ap.add_argument("--backend", "--prompt_backend", dest="backend", default="smolvlm2", choices=["qwen", "smolvlm2"])
    ap.add_argument("--model_dir", default="", help="backend model dir or HF model id")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--sample_mode", default="even", choices=["quartiles", "even", "first", "last", "all"])
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--structured", action="store_true")
    ap.add_argument("--use_fast", action="store_true", help="use fast processor (default False)")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)

    ap.add_argument("--base_prompt", default=DEFAULT_BASE_PROMPT)
    ap.add_argument("--base_neg_prompt", default=DEFAULT_BASE_NEG)
    ap.add_argument("--instr", default=DEFAULT_INSTR)
    ap.add_argument("--max_new_tokens", type=int, default=0)

    ap.add_argument("--out_json", default="", help="output clip_prompts.json")
    ap.add_argument("--out_manifest", default="", help="output prompt_manifest.json")
    ap.add_argument("--backend_python_phase", default="prompt", help="effective python phase used by cli")

    args = ap.parse_args()
    total_t0 = time.time()

    if args.frames_dir.strip():
        frames_dir = Path(args.frames_dir).resolve()
    elif args.segment_dir.strip():
        frames_dir = _resolve_frames_dir(Path(args.segment_dir))
    else:
        raise SystemExit("[ERR] must provide --frames_dir or --segment_dir")

    frames_dir = ensure_dir(frames_dir, "frames_dir")
    try:
        frame_files = list_images(str(frames_dir))
    except Exception as exc:
        raise SystemExit("[ERR] failed to enumerate frames_dir {}: {}".format(frames_dir, exc))
    if not frame_files:
        raise SystemExit("[ERR] no image files found in frames_dir: {}".format(frames_dir))

    first_img = Path(frame_files[0])
    try:
        with first_img.open("rb") as fh:
            fh.read(32)
    except Exception as exc:
        raise SystemExit("[ERR] cannot read sample frame: {} ({})".format(first_img, exc))

    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir.strip() else None
    if exp_dir:
        out_json = Path(args.out_json).resolve() if args.out_json else (exp_dir / "segment" / "clip_prompts.json")
        out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else (exp_dir / "prompt" / "manifest.json")
    else:
        if not args.out_json or not args.out_manifest:
            raise SystemExit("[ERR] without --exp_dir, you must provide --out_json and --out_manifest")
        out_json = Path(args.out_json).resolve()
        out_manifest = Path(args.out_manifest).resolve()

    fps = int(args.fps)
    if fps <= 0:
        raise SystemExit("[ERR] --fps must be > 0")

    if str(args.sample_mode) != "all" and int(args.num_images) <= 0:
        raise SystemExit("[ERR] --num_images must be > 0 when --sample_mode is not all")

    kf_gap = int(args.kf_gap) if int(args.kf_gap) > 0 else _auto_kf_gap(fps)
    base_idx = int(args.base_idx)
    if base_idx < 0:
        raise SystemExit("[ERR] --base_idx must be >= 0")

    idx2path, frames_avail = _build_idx_map(frames_dir)
    if frames_avail <= 0:
        raise SystemExit("[ERR] no frames found in {}".format(frames_dir))
    if base_idx >= frames_avail:
        raise SystemExit("[ERR] base_idx={} out of range (frames_avail={})".format(base_idx, frames_avail))

    deploy_schedule_path = _find_deploy_schedule(frames_dir, exp_dir)
    deploy_schedule = None
    execution_segments = []  # type: List[dict]
    execution_source = "legacy_kf_gap"
    execution_backend = "legacy_uniform"
    if deploy_schedule_path is not None:
        deploy_schedule = load_deploy_schedule(deploy_schedule_path)
        if deploy_schedule:
            execution_segments = build_execution_segments_from_deploy_schedule(deploy_schedule)
            execution_source = "deploy_schedule"
            execution_backend = str(deploy_schedule.get("backend", "") or "wan_r4")
            log_info(
                "execution schedule resolved from deploy_schedule: backend={} segments={}".format(
                    execution_backend, len(execution_segments)
                )
            )

    if not execution_segments:
        nclips = _compute_num_clips(frames_avail, base_idx, kf_gap, int(args.num_segments))
        if nclips <= 0:
            raise SystemExit(
                "[ERR] not enough frames for 1 clip: frames_avail={} base_idx={} kf_gap={}".format(
                    frames_avail, base_idx, kf_gap
                )
            )
        execution_segments = build_legacy_execution_segments(frames_avail, base_idx, kf_gap, nclips)
        log_warn("execution schedule fallback: deploy_schedule.json missing, using legacy kf_gap slicing")
    nclips = int(len(execution_segments))

    model_ref = _resolve_model_ref(args, default_qwen_model)
    max_new_tokens = _resolve_max_new_tokens(args)
    instruction = _build_instruction(args.instr, args.backend, bool(args.structured))
    base_prompt_text = str(args.base_prompt).strip() or DEFAULT_BASE_PROMPT
    base_neg_prompt_text = str(args.base_neg_prompt).strip()
    global_invariants = build_global_invariants(base_prompt_text, base_neg_prompt_text)

    backend = create_backend(
        backend_name=args.backend,
        model_ref=model_ref,
        dtype=args.dtype,
        use_fast=bool(args.use_fast),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        max_new_tokens=max_new_tokens,
    )
    log_info(
        "Initializing prompt backend={} model={} sample_mode={} num_images={}".format(
            str(args.backend), model_ref or "<default>", str(args.sample_mode), int(args.num_images)
        )
    )
    log_info("Loading prompt backend resources...")
    backend.load()
    backend_meta = backend.meta()
    log_info("Processor loaded in {:.2f}s".format(float(backend_meta.get("processor_load_sec", 0.0) or 0.0)))
    log_info("Model weights loaded in {:.2f}s".format(float(backend_meta.get("model_load_sec", 0.0) or 0.0)))
    log_info(
        "Initialization completed in {:.2f}s | frames_avail={} | clips={} | schedule_source={}".format(
            time.time() - total_t0, frames_avail, nclips, execution_source
        )
    )

    clips = []  # type: List[dict]
    errors = []  # type: List[str]
    prompt_times = []  # type: List[float]
    parse_mode_counts = {}  # type: Dict[str, int]
    structured_ok_count = 0
    fallback_parse_count = 0

    for clip_id in tqdm(
        range(nclips),
        desc="Prompt Gen",
        bar_format="[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        seg_spec = dict(execution_segments[int(clip_id)])
        start_idx = int(seg_spec["start_idx"])
        end_idx = int(seg_spec["end_idx"])
        clip_seconds = float(end_idx - start_idx) / float(fps)
        clip_files = _collect_clip_files(idx2path, start_idx, end_idx)
        selected_paths = []  # type: List[str]
        error_msg = ""
        try:
            selected_paths = sample_images(clip_files, args.sample_mode, int(args.num_images))
        except Exception as exc:
            error_msg = "sample_images failed for clip {}: {}".format(clip_id, exc)

        if not selected_paths and not error_msg:
            error_msg = "no frames available for clip {} in range [{}-{}]".format(clip_id, start_idx, end_idx)

        if error_msg:
            errors.append(error_msg)
            clips.append(
                {
                    "clip_id": int(clip_id),
                    "segment_id": int(seg_spec.get("segment_id", clip_id)),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "raw_start_idx": int(seg_spec.get("raw_start_idx", start_idx)),
                    "raw_end_idx": int(seg_spec.get("raw_end_idx", end_idx)),
                    "deploy_start_idx": int(seg_spec.get("deploy_start_idx", start_idx)),
                    "deploy_end_idx": int(seg_spec.get("deploy_end_idx", end_idx)),
                    "raw_gap": int(seg_spec.get("raw_gap", end_idx - start_idx)),
                    "deploy_gap": int(seg_spec.get("deploy_gap", end_idx - start_idx)),
                    "num_frames": int(seg_spec.get("num_frames", end_idx - start_idx + 1)),
                    "schedule_source": str(seg_spec.get("schedule_source", execution_source)),
                    "execution_backend": str(seg_spec.get("execution_backend", execution_backend)),
                    "clip_seconds": clip_seconds,
                    "rep_frames": [Path(path_text).name for path_text in selected_paths],
                    "prompt": "",
                    "error": error_msg,
                }
            )
            continue

        prompt_t0 = time.time()
        prompt_raw = backend.generate(selected_paths, instruction)
        prompt_sec = float(time.time() - prompt_t0)
        prompt_times.append(prompt_sec)
        prompt_raw = _clean_prompt(prompt_raw)
        parsed_prompt = parse_intent_response(prompt_raw)
        parse_mode = str(parsed_prompt.get("parse_mode", "") or "unknown")
        parse_mode_counts[parse_mode] = int(parse_mode_counts.get(parse_mode, 0)) + 1
        if bool(parsed_prompt.get("structured_ok")):
            structured_ok_count += 1
        else:
            fallback_parse_count += 1

        compiled_prompt = compile_legacy_prompts(
            base_prompt=base_prompt_text,
            base_neg_prompt=base_neg_prompt_text,
            global_invariants=global_invariants,
            intent_card=parsed_prompt.get("intent_card", {}),
            control_hints=parsed_prompt.get("control_hints", {}),
        )
        if not prompt_raw:
            compiled_prompt["delta_prompt"] = ""
            compiled_prompt["delta_neg_prompt"] = ""
            compiled_prompt["final_prompt_preview"] = base_prompt_text
            compiled_prompt["final_neg_prompt_preview"] = base_neg_prompt_text

        clips.append(
            {
                "clip_id": int(clip_id),
                "segment_id": int(seg_spec.get("segment_id", clip_id)),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_start_idx": int(seg_spec.get("raw_start_idx", start_idx)),
                "raw_end_idx": int(seg_spec.get("raw_end_idx", end_idx)),
                "deploy_start_idx": int(seg_spec.get("deploy_start_idx", start_idx)),
                "deploy_end_idx": int(seg_spec.get("deploy_end_idx", end_idx)),
                "raw_gap": int(seg_spec.get("raw_gap", end_idx - start_idx)),
                "deploy_gap": int(seg_spec.get("deploy_gap", end_idx - start_idx)),
                "num_frames": int(seg_spec.get("num_frames", end_idx - start_idx + 1)),
                "schedule_source": str(seg_spec.get("schedule_source", execution_source)),
                "execution_backend": str(seg_spec.get("execution_backend", execution_backend)),
                "clip_seconds": clip_seconds,
                "rep_frames": [Path(path_text).name for path_text in selected_paths],
                "rep_indices": _extract_rep_indices(selected_paths),
                "prompt": str(compiled_prompt.get("delta_prompt", "") or "").strip(),
                "prompt_raw": prompt_raw,
                "delta_neg_prompt": str(compiled_prompt.get("delta_neg_prompt", "") or "").strip(),
                "prompt_parse_mode": parse_mode,
                "intent_card": dict(parsed_prompt.get("intent_card", {}) or {}),
                "control_hints": dict(parsed_prompt.get("control_hints", {}) or {}),
                "compiled": {
                    "final_prompt_preview": str(compiled_prompt.get("final_prompt_preview", "") or "").strip(),
                    "final_neg_prompt_preview": str(compiled_prompt.get("final_neg_prompt_preview", "") or "").strip(),
                },
            }
        )

    model_record = str(backend_meta.get("model_dir", "") or backend_meta.get("model_id", "") or model_ref)
    clip_prompts = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_record,
        "frames_dir": str(frames_dir),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "base_idx": int(base_idx),
        "num_clips": int(nclips),
        "schedule_source": str(execution_source),
        "execution_backend": str(execution_backend),
        "deploy_schedule_path": str(deploy_schedule_path) if deploy_schedule_path is not None else "",
        "rep_policy": _rep_policy_name(str(args.sample_mode), int(args.num_images)),
        "max_new_tokens": int(max_new_tokens),
        "use_fast": bool(args.use_fast),
        "clips": clips,
        "errors": errors,
    }
    write_json_atomic(out_json, clip_prompts, indent=2)

    seg_items = []
    for it in clips:
        intent_card = dict(it.get("intent_card", {}) or {})
        control_hints = dict(it.get("control_hints", {}) or {})
        compiled = dict(it.get("compiled", {}) or {})
        raw_prompt = str(it.get("prompt_raw", "") or "").strip()
        seg_items.append(
            {
                "seg": int(it["clip_id"]),
                "seg_id": int(it["clip_id"]),
                "clip_id": int(it["clip_id"]),
                "segment_id": int(it.get("segment_id", it["clip_id"])),
                "schedule_source": str(it.get("schedule_source", execution_source)),
                "execution_backend": str(it.get("execution_backend", execution_backend)),
                "start_idx": int(it["start_idx"]),
                "end_idx": int(it["end_idx"]),
                "raw_start_idx": int(it.get("raw_start_idx", it["start_idx"])),
                "raw_end_idx": int(it.get("raw_end_idx", it["end_idx"])),
                "deploy_start_idx": int(it.get("deploy_start_idx", it["start_idx"])),
                "deploy_end_idx": int(it.get("deploy_end_idx", it["end_idx"])),
                "raw_gap": int(it.get("raw_gap", it["end_idx"] - it["start_idx"])),
                "deploy_gap": int(it.get("deploy_gap", it["end_idx"] - it["start_idx"])),
                "num_frames": int(it.get("num_frames", it["end_idx"] - it["start_idx"] + 1)),
                "rep_indices": list(it.get("rep_indices", []) or []),
                "boundary_meta": {
                    "raw_start_idx": int(it.get("raw_start_idx", it["start_idx"])),
                    "raw_end_idx": int(it.get("raw_end_idx", it["end_idx"])),
                    "deploy_start_idx": int(it.get("deploy_start_idx", it["start_idx"])),
                    "deploy_end_idx": int(it.get("deploy_end_idx", it["end_idx"])),
                    "boundary_shift": int(it.get("boundary_shift", 0)),
                    "gap_error": int(it.get("gap_error", 0)),
                },
                "intent_card": {
                    "scene_anchor": str(intent_card.get("scene_anchor", "") or ""),
                    "motion_intent": str(intent_card.get("motion_intent", "") or ""),
                    "geometry_constraints": list(intent_card.get("geometry_constraints", []) or []),
                    "appearance_constraints": list(intent_card.get("appearance_constraints", []) or []),
                    "suppressions": list(intent_card.get("suppressions", []) or []),
                },
                "control_hints": {
                    "motion_intensity": str(control_hints.get("motion_intensity", "low") or "low"),
                    "geometry_priority": str(control_hints.get("geometry_priority", "high") or "high"),
                    "risk_level": str(control_hints.get("risk_level", "low") or "low"),
                },
                "legacy": {
                    "raw_response": raw_prompt,
                    "parse_mode": str(it.get("prompt_parse_mode", "") or ""),
                    "delta_prompt": str(it.get("prompt", "") or "").strip(),
                    "delta_neg_prompt": str(it.get("delta_neg_prompt", "") or "").strip(),
                },
                "compiled": {
                    "final_prompt_preview": str(compiled.get("final_prompt_preview", "") or "").strip(),
                    "final_neg_prompt_preview": str(compiled.get("final_neg_prompt_preview", "") or "").strip(),
                },
                "delta_prompt": str(it.get("prompt", "") or "").strip(),
                "delta_neg_prompt": str(it.get("delta_neg_prompt", "") or "").strip(),
            }
        )

    sequence_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_record,
        "frames_dir": str(frames_dir),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "base_idx": int(base_idx),
        "num_clips": int(nclips),
        "rep_policy": _rep_policy_name(str(args.sample_mode), int(args.num_images)),
        "sample_mode": str(args.sample_mode),
        "num_images": int(args.num_images),
        "backend": str(args.backend),
        "schedule_source": str(execution_source),
        "execution_backend": str(execution_backend),
        "deploy_schedule_path": str(deploy_schedule_path) if deploy_schedule_path is not None else "",
    }
    compiler_meta = {
        "name": "legacy_prompt_compiler",
        "version": 1,
        "compat_mode": "delta_prompt_legacy_fields",
        "parse_strategy": [
            "json",
            "json_repaired",
            "kv_pairs",
            "fallback_from_json",
            "fallback_from_json_repaired",
            "fallback_from_kv_pairs",
            "raw_text_fallback",
        ],
        "strict_json_requested": bool(args.structured),
        "structured_ok_segments": int(structured_ok_count),
        "fallback_segments": int(fallback_parse_count),
        "parse_mode_counts": parse_mode_counts,
    }
    manifest = build_manifest_v2(
        base_prompt=base_prompt_text,
        base_neg_prompt=base_neg_prompt_text,
        sequence_meta=sequence_meta,
        global_invariants=global_invariants,
        compiler=compiler_meta,
        segments=seg_items,
    )
    manifest["schedule_source"] = str(execution_source)
    manifest["execution_backend"] = str(execution_backend)
    manifest["deploy_schedule_path"] = str(deploy_schedule_path) if deploy_schedule_path is not None else ""
    write_json_atomic(out_manifest, manifest, indent=2)

    manifest_bytes = out_manifest.read_bytes()
    manifest_size = int(len(manifest_bytes))
    clip_prompts_size = 0
    if out_json.is_file():
        try:
            clip_prompts_size = int(os.path.getsize(str(out_json)))
        except Exception:
            clip_prompts_size = 0
    outputs_bytes_sum = int(manifest_size + clip_prompts_size)
    avg_prompt_sec = float(sum(prompt_times) / float(len(prompt_times))) if prompt_times else 0.0
    prompt_style = "intent_card_v2_legacy_delta"

    step_meta = {
        "step": "prompt",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": str(backend_meta.get("model_dir", "") or ""),
        "model_id": str(backend_meta.get("model_id", "") or ""),
        "backend": str(args.backend),
        "attn_impl": "sdpa",
        "dtype": str(backend_meta.get("dtype", "") or ""),
        "sample_mode": str(args.sample_mode),
        "num_images": int(args.num_images),
        "structured": True,
        "manifest_version": int(MANIFEST_VERSION),
        "manifest_schema": MANIFEST_SCHEMA,
        "processor_load_sec": float(backend_meta.get("processor_load_sec", 0.0) or 0.0),
        "model_load_sec": float(backend_meta.get("model_load_sec", 0.0) or 0.0),
        "prompt_gen_total_sec": float(time.time() - total_t0),
        "avg_prompt_sec_per_clip": avg_prompt_sec,
        "selected_rep_policy": str(args.sample_mode),
        "backend_python_phase": str(args.backend_python_phase or ""),
        "prompt_style": prompt_style,
        "compiler_name": str(compiler_meta.get("name", "")),
        "structured_ok_segments": int(structured_ok_count),
        "fallback_segments": int(fallback_parse_count),
        "parse_mode_counts": parse_mode_counts,
        "frames_count": int(len(frame_files)),
        "clips_count": int(nclips),
        "schedule_source": str(execution_source),
        "execution_backend": str(execution_backend),
        "deploy_schedule_path": str(deploy_schedule_path) if deploy_schedule_path is not None else "",
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
        log_warn("{} clips had sampling/input issues; see errors in clip_prompts.json".format(len(errors)))
    if fallback_parse_count > 0:
        log_warn("{} clips used raw-text fallback while compiling prompt manifest v2".format(int(fallback_parse_count)))


if __name__ == "__main__":
    main()
