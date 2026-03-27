#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ExpHub infer frontend with pluggable Wan backends."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from _common import (
    ensure_dir,
    ensure_file,
    get_platform_config,
    list_frames_sorted,
    log_info,
    log_prog,
    log_warn,
    write_json_atomic,
)
from _infer.api import create_backend
from _infer.prompt_resolver import build_prompt_resolution_for_infer
from _infer.reporting import build_infer_report, cleanup_legacy_infer_outputs, write_infer_report
from _infer.request import InferRequest
from _schedule import (
    build_execution_segments_from_deploy_schedule,
    build_legacy_execution_segments,
    load_deploy_schedule,
)


def _samefile_safe(src, dst):
    # type: (Path, Path) -> bool
    src_s = os.path.abspath(str(src))
    dst_s = os.path.abspath(str(dst))
    try:
        return os.path.samefile(src_s, dst_s)
    except Exception:
        return src_s == dst_s


def _resolve_frames_dir(segment_dir):
    # type: (Path) -> Path
    if segment_dir.name == "frames":
        return segment_dir
    frames = segment_dir / "frames"
    if frames.is_dir():
        return frames
    return segment_dir


def _mean(values):
    # type: (list) -> float
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _normalize_extra(extra_args):
    # type: (list) -> list
    extra = list(extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _count_by_key(items, key):
    # type: (list, str) -> dict
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get(key, "") or "").strip() or "unknown"
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _write_runtime_execution_plan(infer_dir, schedule_source, schedule_backend, execution_segments):
    # type: (Path, str, str, list) -> Path
    infer_dir = Path(infer_dir).resolve()
    infer_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="execution_plan_", suffix=".json", dir=str(infer_dir))
    os.close(fd)
    runtime_plan_path = Path(tmp_path).resolve()
    write_json_atomic(
        runtime_plan_path,
        {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "schedule_source": str(schedule_source),
            "execution_backend": str(schedule_backend),
            "segments": list(execution_segments or []),
        },
        indent=2,
    )
    return runtime_plan_path


def _merge_prompt_resolution_into_plan(plan_obj, segment_resolutions):
    # type: (dict, list) -> dict
    plan = dict(plan_obj or {})
    plan_segments = list(plan.get("segments", []) or [])
    resolution_map = {}
    for raw_item in list(segment_resolutions or []):
        if not isinstance(raw_item, dict):
            continue
        seg = raw_item.get("seg")
        if seg is None:
            continue
        try:
            resolution_map[int(seg)] = dict(raw_item)
        except Exception:
            continue

    for idx, raw_segment in enumerate(plan_segments):
        if not isinstance(raw_segment, dict):
            continue
        seg_item = raw_segment
        seg_key = seg_item.get("seg", idx)
        try:
            resolved = resolution_map.get(int(seg_key), {})
        except Exception:
            resolved = {}
        if not isinstance(resolved, dict) or not resolved:
            continue
        seg_item["prompt_source"] = str(resolved.get("prompt_source", seg_item.get("prompt_source", "")) or "")
        seg_item["deploy_segment_id"] = int(resolved.get("deploy_segment_id", seg_item.get("segment_id", idx)) or 0)
        seg_item["state_segment_id"] = resolved.get("state_segment_id")
        seg_item["state_label"] = resolved.get("state_label")
        seg_item["motion_trend"] = resolved.get("motion_trend")
        seg_item["prompt_preview"] = resolved.get("prompt_preview")
    plan["segments"] = plan_segments
    return plan


def main():
    cfg = get_platform_config()
    default_videox_repo = cfg.get("repos", {}).get("videox_fun", "")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segment_dir",
        required=True,
        help="ExpHub segment dir (contains frames/ and preferred deploy_schedule.json) or frames dir",
    )
    ap.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (will create infer/ subdir inside)")
    ap.add_argument("--videox_root", default=default_videox_repo, help="VideoX-Fun repo root")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--fps", type=int, default=24, help="Target FPS (also used as dataset_fps)")
    ap.add_argument(
        "--kf_gap",
        type=int,
        default=0,
        help=(
            "Keyframe gap in frames. If 0, choose the largest multiple of 4 <= fps (r=4 safe default). "
            "Example: fps=24 -> kf_gap=24 (1s)."
        ),
    )
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--num_segments", type=int, default=0, help="If >0, cap segments to this value")
    ap.add_argument("--seed_base", type=int, default=43)
    ap.add_argument(
        "--prompt_file",
        default="",
        help="Optional base final_prompt.json; archived under <exp_dir>/prompt/final_prompt.json",
    )
    ap.add_argument(
        "--infer_backend",
        default="wan_fun_5b_inp",
        choices=["wan_fun_a14b_inp", "wan_fun_5b_inp"],
        help="infer backend name",
    )
    ap.add_argument(
        "--infer_model_dir",
        default="",
        help="override infer backend model dir or model id",
    )
    ap.add_argument(
        "--backend_python_phase",
        default="infer",
        help="effective python phase selected by cli",
    )
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    segment_dir = Path(args.segment_dir).resolve()
    ensure_dir(segment_dir, "segment_dir")
    frames_dir = _resolve_frames_dir(segment_dir)
    ensure_dir(frames_dir, "segment/frames")
    schedule_dir = frames_dir.parent if frames_dir.name == "frames" else segment_dir

    frames_avail = len(list_frames_sorted(frames_dir))
    if frames_avail <= 0:
        raise SystemExit("[ERR] segment has no frames: {}".format(frames_dir))

    fps = int(args.fps)
    if fps <= 0:
        raise SystemExit("[ERR] --fps must be > 0")

    kf_gap = int(args.kf_gap)
    if kf_gap <= 0:
        kf_gap = fps - (fps % 4)
        if kf_gap <= 0:
            raise SystemExit("[ERR] cannot auto-pick kf_gap; provide --kf_gap")
    if kf_gap <= 0:
        raise SystemExit("[ERR] --kf_gap must be > 0")
    if kf_gap % 4 != 0:
        log_warn("kf_gap={} is not divisible by 4 (r=4); video_length may be truncated by the model".format(kf_gap))

    base_idx = int(args.base_idx)
    if base_idx < 0:
        raise SystemExit("[ERR] --base_idx must be >= 0")
    if base_idx >= frames_avail:
        raise SystemExit("[ERR] base_idx={} out of range (frames_avail={})".format(base_idx, frames_avail))

    exp_dir = Path(args.exp_dir).resolve()
    prompt_dir = exp_dir / "prompt"
    prompt_file_std = prompt_dir / "final_prompt.json"
    infer_dir = exp_dir / "infer"
    runs_root = infer_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if not str(args.videox_root).strip():
        raise SystemExit(
            "[ERR] --videox_root is empty. Set repos.videox_fun in config/platform.yaml or pass --videox_root."
        )
    videox_root = Path(args.videox_root).resolve()
    ensure_dir(videox_root, "videox_root")

    if args.prompt_file:
        src_prompt_file = Path(args.prompt_file).resolve()
        ensure_file(src_prompt_file, "prompt_file")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        if _samefile_safe(src_prompt_file, prompt_file_std):
            log_info("prompt_file already at standard path, skip copy: {}".format(prompt_file_std))
        else:
            shutil.copy2(str(src_prompt_file), str(prompt_file_std))
            log_info("prompt_file copied to standard path: {} -> {}".format(src_prompt_file, prompt_file_std))

    if not prompt_file_std.is_file():
        raise SystemExit(
            "[ERR] missing prompt file: {}. Run prompt step first or provide --prompt_file.".format(
                prompt_file_std
            )
        )

    execution_segments = []
    schedule_source = ""
    schedule_backend = ""

    if not execution_segments:
        deploy_schedule = load_deploy_schedule(schedule_dir / "deploy_schedule.json")
        if deploy_schedule:
            execution_segments = build_execution_segments_from_deploy_schedule(deploy_schedule)
            schedule_source = "deploy_schedule"
            schedule_backend = str(deploy_schedule.get("backend", "") or "wan_r4")

    if not execution_segments:
        execution_segments = build_legacy_execution_segments(frames_avail, base_idx, kf_gap, int(args.num_segments))
        schedule_source = "legacy_kf_gap"
        schedule_backend = "legacy_uniform"
        log_warn("execution schedule fallback: segment/deploy_schedule.json missing, using legacy kf_gap slicing")

    segments = int(len(execution_segments))
    if segments <= 0:
        raise SystemExit("[ERR] execution schedule resolved to 0 segments")

    used_start_idx = int(execution_segments[0]["start_idx"])
    used_end_idx = int(execution_segments[-1]["end_idx"])
    used_frames = int(used_end_idx - used_start_idx + 1)
    tail_drop = int(max(0, frames_avail - (used_end_idx + 1)))
    mean_deploy_gap = float(_mean([int(seg["deploy_gap"]) for seg in execution_segments]))
    if tail_drop > 0:
        log_warn(
            "segment has {} frames (plan {}/{}); uses {} frames [{}->{}], tail_drop={}".format(
                frames_avail,
                schedule_source,
                schedule_backend,
                used_frames,
                used_start_idx,
                used_end_idx,
                tail_drop,
            )
        )
    else:
        log_info(
            "segment has {} frames (plan {}/{}); uses {} frames [{}->{}]".format(
                frames_avail,
                schedule_source,
                schedule_backend,
                used_frames,
                used_start_idx,
                used_end_idx,
            )
        )

    try:
        prompt_resolution = dict(
            build_prompt_resolution_for_infer(
                prompt_dir=prompt_dir,
                infer_dir=infer_dir,
                execution_segments=execution_segments,
            )
            or {}
        )
    except Exception as exc:
        raise SystemExit("[ERR] failed to resolve infer prompt manifest: {}".format(exc))

    for msg in list(prompt_resolution.get("warnings", []) or []):
        log_warn("state prompt fallback: {}".format(msg))

    log_info(
        "state prompt files: detected={} state_segments={} mapped_segments={}".format(
            bool(prompt_resolution.get("state_prompt_files_detected", False)),
            int(prompt_resolution.get("state_prompt_segment_count", 0) or 0),
            int(prompt_resolution.get("mapped_execution_segment_count", 0) or 0),
        )
    )
    log_info(
        "prompt manifest mode={} state_enabled={} runtime_prompt_file={}".format(
            str(prompt_resolution.get("prompt_manifest_mode", "global_only")),
            bool(prompt_resolution.get("state_prompt_enabled", False)),
            prompt_resolution.get("prompt_file_path", prompt_file_std),
        )
    )
    for preview_item in list(prompt_resolution.get("segment_resolutions", []) or [])[:3]:
        log_info(
            "prompt seg {}: source={} state_seg={} motion={}".format(
                int(preview_item.get("seg", 0) or 0),
                str(preview_item.get("prompt_source", "") or "unknown"),
                preview_item.get("state_segment_id"),
                str(preview_item.get("motion_trend", "") or "n/a"),
            )
        )

    prompt_file_for_runtime = Path(str(prompt_resolution.get("prompt_file_path", prompt_file_std))).resolve()
    ensure_file(prompt_file_for_runtime, "infer prompt manifest")
    runtime_execution_plan_path = _write_runtime_execution_plan(
        infer_dir=infer_dir,
        schedule_source=str(schedule_source),
        schedule_backend=str(schedule_backend),
        execution_segments=execution_segments,
    )

    request = InferRequest(
        frames_dir=frames_dir,
        exp_dir=exp_dir,
        prompt_file_path=prompt_file_for_runtime,
        execution_plan_path=runtime_execution_plan_path,
        fps=int(fps),
        kf_gap=int(kf_gap),
        base_idx=int(used_start_idx),
        num_segments=int(segments),
        seed_base=int(args.seed_base),
        gpus=int(args.gpus),
        schedule_source=str(schedule_source),
        execution_backend=str(schedule_backend),
        execution_segments=list(execution_segments),
        infer_extra=_normalize_extra(args.extra),
    )

    backend = create_backend(
        backend_name=str(args.infer_backend),
        videox_root=str(videox_root),
        model_ref=str(args.infer_model_dir or ""),
        backend_python_phase=str(args.backend_python_phase or "infer"),
    )
    backend.load()
    backend_meta = dict(backend.meta() or {})

    log_prog(
        "infer config: backend={} segments={} fps={} gpus={}".format(
            backend_meta.get("infer_backend", args.infer_backend),
            segments,
            fps,
            args.gpus,
        )
    )
    log_info(
        "infer detail: schedule_source={} execution_backend={} used_frames={}".format(
            schedule_source,
            schedule_backend,
            used_frames,
        )
    )

    t0 = time.time()
    try:
        backend_result = dict(backend.run(request) or {})
    finally:
        try:
            if runtime_execution_plan_path.is_file():
                runtime_execution_plan_path.unlink()
        except Exception:
            pass
    dt = time.time() - t0

    runs_plan = ensure_file(infer_dir / "runs_plan.json", "runs_plan")
    plan_obj = json.loads(runs_plan.read_text(encoding="utf-8"))
    if not isinstance(plan_obj, dict):
        raise SystemExit("[ERR] invalid runs_plan.json: {}".format(runs_plan))
    plan_obj = _merge_prompt_resolution_into_plan(plan_obj, prompt_resolution.get("segment_resolutions", []))
    plan_obj["prompt_manifest_mode"] = str(prompt_resolution.get("prompt_manifest_mode", "global_only"))
    plan_obj["state_prompt_enabled"] = bool(prompt_resolution.get("state_prompt_enabled", False))
    plan_obj["state_prompt_segment_count"] = int(prompt_resolution.get("state_prompt_segment_count", 0) or 0)
    plan_obj["mapped_execution_segment_count"] = int(prompt_resolution.get("mapped_execution_segment_count", 0) or 0)
    plan_obj["prompt_source_counts"] = dict(prompt_resolution.get("prompt_source_counts", {}) or {})
    plan_obj["state_motion_trend_counts"] = dict(prompt_resolution.get("state_motion_trend_counts", {}) or {})
    write_json_atomic(runs_plan, plan_obj, indent=2)
    plan_segment_items = list(plan_obj.get("segments", []) or [])
    plan_segments = int(len(plan_segment_items))
    ensure_file(prompt_file_for_runtime, "infer prompt manifest")
    infer_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": int(args.gpus),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "frames_avail": int(frames_avail),
        "segments": int(plan_segments),
        "used_frames": int(used_frames),
        "used_start_idx": int(used_start_idx),
        "used_end_idx": int(used_end_idx),
        "schedule_source": str(schedule_source),
        "execution_backend": str(schedule_backend),
        "mean_deploy_gap": float(mean_deploy_gap),
        "prompt_manifest_mode": str(prompt_resolution.get("prompt_manifest_mode", "global_only")),
        "state_prompt_enabled": bool(prompt_resolution.get("state_prompt_enabled", False)),
        "state_prompt_files_detected": bool(prompt_resolution.get("state_prompt_files_detected", False)),
        "state_prompt_segment_count": int(prompt_resolution.get("state_prompt_segment_count", 0) or 0),
        "mapped_execution_segment_count": int(prompt_resolution.get("mapped_execution_segment_count", 0) or 0),
        "prompt_file_version": int(plan_obj.get("prompt_file_version", 1)),
        "prompt_file_source": str(plan_obj.get("prompt_file_source", "")),
        "prompt_source_counts": _count_by_key(plan_segment_items, "prompt_source"),
        "state_motion_trend_counts": dict(prompt_resolution.get("state_motion_trend_counts", {}) or {}),
    }
    infer_report = build_infer_report(
        infer_dir=infer_dir,
        runs_plan_obj=plan_obj,
        prompt_resolution=prompt_resolution,
        backend_meta=backend_meta,
        backend_result=backend_result,
        runtime_summary=infer_report,
    )
    report_path = write_infer_report(infer_dir, infer_report)
    cleanup_legacy_infer_outputs(infer_dir)

    log_info("infer finished: {:.2f}s".format(dt))
    log_info("report written: {}".format(report_path))


if __name__ == "__main__":
    main()
