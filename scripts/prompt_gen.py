import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable HF progress bars to keep terminal/log output clean.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from tqdm import tqdm

from _common import ensure_dir, get_platform_config, log_info, log_prog, log_prompt, log_warn, write_json_atomic
from _prompt.api import create_backend
from _prompt.backends.smolvlm2_backend import DEFAULT_SMOLVLM2_MODEL_ID
from _prompt.generator import build_final_prompt_payload
from _prompt.profile import (
    PROMPT_PROFILE_VERSION,
    aggregate_prompt_profiles,
    build_profile_instruction,
    default_prompt_profile,
    parse_profile_response,
)
from _prompt.reporting import build_prompt_report, cleanup_legacy_prompt_outputs, write_prompt_report
from _prompt.sampling import list_images, sample_images
from _prompt.state_prompt import build_state_prompt_artifacts


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IDX_RE = re.compile(r"(\d+)")


def _resolve_frames_dir(path_obj):
    # type: (Path) -> Path
    candidate = path_obj.resolve()
    if candidate.is_dir() and candidate.name == "frames":
        return candidate
    if (candidate / "frames").is_dir():
        return (candidate / "frames").resolve()
    return candidate


def _resolve_default_qwen_model():
    # type: () -> str
    try:
        cfg = get_platform_config()
        return str(cfg.get("models", {}).get("qwen2_vl", {}).get("path", "") or "").strip()
    except Exception:
        return ""


def _resolve_model_ref(args, default_qwen_model):
    # type: (argparse.Namespace, str) -> str
    raw_model = str(args.model_dir or "").strip()
    if raw_model:
        return raw_model
    backend = str(args.backend or "smolvlm2").strip().lower()
    if backend == "qwen":
        return str(default_qwen_model or "").strip()
    if backend == "smolvlm2":
        return DEFAULT_SMOLVLM2_MODEL_ID
    return ""


def _resolve_max_new_tokens(args):
    # type: (argparse.Namespace) -> int
    raw_value = int(args.max_new_tokens)
    if raw_value > 0:
        return raw_value
    if str(args.backend or "smolvlm2").strip().lower() == "smolvlm2":
        return 48
    return 80


def _extract_frame_index(path_text):
    # type: (str) -> Optional[int]
    stem = Path(path_text).stem
    if stem.isdigit():
        return int(stem)
    match = IDX_RE.search(stem)
    if match is not None:
        return int(match.group(1))
    return None


def _clamp_num_images(value):
    # type: (int) -> int
    return max(3, min(5, int(value)))


def main():
    total_t0 = time.time()
    default_qwen_model = _resolve_default_qwen_model()

    ap = argparse.ArgumentParser()
    ap.add_argument("--segment_dir", default="", help="segment dir (contains frames/) OR frames dir")
    ap.add_argument("--frames_dir", default="", help="direct frames dir override")
    ap.add_argument("--exp_dir", default="", help="if set, outputs go under <exp_dir>/prompt")
    ap.add_argument("--fps", type=int, default=0, help="dataset fps, used only for metadata")
    ap.add_argument("--backend", "--prompt_backend", dest="backend", default="smolvlm2", choices=["qwen", "smolvlm2"])
    ap.add_argument("--model_dir", default="", help="backend model dir or HF model id")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--sample_mode", default="even", choices=["quartiles", "even", "first", "last"])
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--use_fast", action="store_true", help="use fast processor (default False)")
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=0)
    ap.add_argument("--out_profile", default="", help="output profile.json")
    ap.add_argument("--out_final_prompt", default="", help="output final_prompt.json")
    ap.add_argument("--backend_python_phase", default="prompt", help="effective python phase used by cli")
    args = ap.parse_args()

    if args.frames_dir.strip():
        frames_dir = Path(args.frames_dir).resolve()
    elif args.segment_dir.strip():
        frames_dir = _resolve_frames_dir(Path(args.segment_dir))
    else:
        raise SystemExit("[ERR] must provide --frames_dir or --segment_dir")
    frames_dir = ensure_dir(frames_dir, "frames_dir")

    frame_files = list_images(str(frames_dir))
    if not frame_files:
        raise SystemExit("[ERR] no image files found in frames_dir: {}".format(frames_dir))

    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir.strip() else None
    if exp_dir is not None:
        prompt_dir = exp_dir / "prompt"
        legacy_profile_export_path = Path(args.out_profile).resolve() if args.out_profile else None
        out_final_prompt = (
            Path(args.out_final_prompt).resolve() if args.out_final_prompt else (prompt_dir / "final_prompt.json")
        )
    else:
        if not args.out_final_prompt:
            raise SystemExit("[ERR] without --exp_dir, you must provide --out_final_prompt")
        legacy_profile_export_path = Path(args.out_profile).resolve() if args.out_profile else None
        out_final_prompt = Path(args.out_final_prompt).resolve()

    rep_count = _clamp_num_images(int(args.num_images))
    if int(args.num_images) != rep_count:
        log_warn("prompt representative frame count clamped from {} to {}".format(int(args.num_images), rep_count))

    selected_paths = sample_images(frame_files, str(args.sample_mode), rep_count)
    if len(selected_paths) < 3:
        log_warn(
            "only {} representative frames available; prompt profile will use all available samples".format(
                len(selected_paths)
            )
        )

    model_ref = _resolve_model_ref(args, default_qwen_model)
    max_new_tokens = _resolve_max_new_tokens(args)
    instruction = build_profile_instruction()

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
        "initializing prompt backend={} model={} sample_mode={} rep_frames={}".format(
            str(args.backend),
            model_ref or "<default>",
            str(args.sample_mode),
            len(selected_paths),
        )
    )
    log_info("loading prompt backend resources...")
    backend.load()
    backend_meta = dict(backend.meta() or {})
    log_info("processor loaded in {:.2f}s".format(float(backend_meta.get("processor_load_sec", 0.0) or 0.0)))
    log_info("model weights loaded in {:.2f}s".format(float(backend_meta.get("model_load_sec", 0.0) or 0.0)))

    candidates = []  # type: List[Dict[str, object]]
    frame_records = []  # type: List[Dict[str, object]]
    errors = []  # type: List[str]
    prompt_times = []  # type: List[float]

    for idx, frame_path in enumerate(
        tqdm(
            selected_paths,
            desc="Prompt Profile",
            bar_format="[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    ):
        frame_t0 = time.time()
        frame_name = Path(frame_path).name
        try:
            raw_output = str(backend.generate([frame_path], instruction) or "").strip()
            candidate = parse_profile_response(raw_output)
        except Exception as exc:
            raw_output = ""
            candidate = default_prompt_profile()
            errors.append("frame {} classification failed: {}".format(frame_name, exc))
            log_warn("frame classification failed: {} ({})".format(frame_name, exc))

        elapsed = float(time.time() - frame_t0)
        prompt_times.append(elapsed)
        candidates.append(candidate)
        frame_records.append(
            {
                "frame": frame_name,
                "frame_idx": _extract_frame_index(frame_path),
                "profile": dict(candidate),
            }
        )
        log_prompt(
            "prompt detail: frame {}/{} classified in {:.2f}s: scene={} surface={} lighting={} risk={}".format(
                int(idx + 1),
                int(len(selected_paths)),
                elapsed,
                str(candidate.get("scene_type", "")),
                str(candidate.get("surface_type", "")),
                str(candidate.get("lighting_type", "")),
                str(candidate.get("dynamic_risk", "")),
            )
        )

    aggregated_profile = aggregate_prompt_profiles(candidates)
    aggregated_profile["version"] = int(PROMPT_PROFILE_VERSION)
    final_prompt_payload = build_final_prompt_payload(aggregated_profile)
    state_prompt_manifest_path = out_final_prompt.parent / "state_prompt_manifest.json"
    deploy_to_state_prompt_map_path = out_final_prompt.parent / "deploy_to_state_prompt_map.json"
    state_prompt_result = build_state_prompt_artifacts(
        frames_dir=frames_dir,
        frames_count=len(frame_files),
        prompt_dir=out_final_prompt.parent,
        global_prompt_path=out_final_prompt,
        exp_dir=exp_dir,
        segment_dir=Path(args.segment_dir).resolve() if args.segment_dir.strip() else None,
    )
    state_prompt_manifest = dict(state_prompt_result.get("state_prompt_manifest") or {})
    deploy_to_state_prompt_map = dict(state_prompt_result.get("deploy_to_state_prompt_map") or {})
    state_prompt_summary = dict(state_prompt_result.get("summary") or {})

    write_json_atomic(out_final_prompt, final_prompt_payload, indent=2)
    write_json_atomic(state_prompt_manifest_path, state_prompt_manifest, indent=2)
    write_json_atomic(deploy_to_state_prompt_map_path, deploy_to_state_prompt_map, indent=2)
    if legacy_profile_export_path is not None:
        write_json_atomic(legacy_profile_export_path, aggregated_profile, indent=2)

    avg_prompt_sec = float(sum(prompt_times) / float(len(prompt_times))) if prompt_times else 0.0

    model_record = str(backend_meta.get("model_dir", "") or backend_meta.get("model_id", "") or model_ref)
    prompt_report = build_prompt_report(
        prompt_dir=out_final_prompt.parent,
        aggregated_profile=aggregated_profile,
        final_prompt_payload=final_prompt_payload,
        state_prompt_manifest=state_prompt_manifest,
        deploy_to_state_prompt_map=deploy_to_state_prompt_map,
        state_prompt_summary=state_prompt_summary,
        backend_meta=backend_meta,
        backend_name=str(args.backend),
        backend_python_phase=str(args.backend_python_phase or ""),
        model_record=model_record,
        dtype=str(args.dtype),
        sample_mode=str(args.sample_mode),
        num_images_requested=int(args.num_images),
        selected_paths=selected_paths,
        frame_files_count=len(frame_files),
        fps=(int(args.fps) if int(args.fps) > 0 else None),
        frame_records=frame_records,
        errors=errors,
        total_sec=float(time.time() - total_t0),
        avg_prompt_sec=avg_prompt_sec,
    )
    prompt_report["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    prompt_report["profile_version"] = int(PROMPT_PROFILE_VERSION)
    prompt_report["frames_dir"] = str(frames_dir)
    if legacy_profile_export_path is not None:
        profile_bytes = legacy_profile_export_path.read_bytes()
        prompt_report["legacy_profile_export"] = {
            "profile_path": str(legacy_profile_export_path),
            "profile_size": int(len(profile_bytes)),
            "profile_sha1": hashlib.sha1(profile_bytes).hexdigest(),
        }
    report_path = write_prompt_report(out_final_prompt.parent, prompt_report)
    cleanup_legacy_prompt_outputs(out_final_prompt.parent, preserve_paths=[legacy_profile_export_path] if legacy_profile_export_path is not None else [])

    log_prog("prompt profile generated from {} representative frames".format(int(len(selected_paths))))
    log_info("state prompt detected state_segments={}".format(bool(state_prompt_summary.get("has_state_segments", False))))
    log_info("state prompt manifest generated: count={}".format(int(state_prompt_summary.get("state_segment_count", 0) or 0)))
    log_info(
        "deploy to state prompt map generated: count={}".format(
            int(state_prompt_summary.get("deploy_segment_count", 0) or 0)
        )
    )
    log_info("wrote: {}".format(out_final_prompt))
    log_info("wrote: {}".format(state_prompt_manifest_path))
    log_info("wrote: {}".format(deploy_to_state_prompt_map_path))
    log_info("wrote: {}".format(report_path))
    if legacy_profile_export_path is not None:
        log_info("wrote legacy profile export: {}".format(legacy_profile_export_path))
    if errors:
        log_warn("{} representative frames fell back to the safe default profile".format(int(len(errors))))


if __name__ == "__main__":
    main()
