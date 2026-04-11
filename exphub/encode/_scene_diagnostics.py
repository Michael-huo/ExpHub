from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from exphub.common.io import write_json_atomic


@dataclass(frozen=True)
class SceneSplitArtifactPaths:
    exp_dir: Path
    root: Path
    frames_dir: Path
    keyframes_dir: Path
    manifest_path: Path
    report_path: Path
    visuals_dir: Path
    state_overview_path: Path
    calib_path: Path
    timestamps_path: Path


def build_paths(exp_dir):
    exp_dir_path = Path(exp_dir).resolve()
    root = (exp_dir_path / "segment").resolve()
    return SceneSplitArtifactPaths(
        exp_dir=exp_dir_path,
        root=root,
        frames_dir=(root / "frames").resolve(),
        keyframes_dir=(root / "keyframes").resolve(),
        manifest_path=(root / "segment_manifest.json").resolve(),
        report_path=(root / "report.json").resolve(),
        visuals_dir=(root / "visuals").resolve(),
        state_overview_path=(root / "visuals" / "state_overview.png").resolve(),
        calib_path=(root / "calib.txt").resolve(),
        timestamps_path=(root / "timestamps.txt").resolve(),
    )


def relative_to_exp(exp_dir, target_path):
    exp_dir_path = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_dir_path))
    except Exception:
        return str(target)


def ensure_layout(paths):
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    paths.keyframes_dir.mkdir(parents=True, exist_ok=True)
    paths.visuals_dir.mkdir(parents=True, exist_ok=True)


def remove_stale_scene_split_outputs(paths):
    stale_paths = [
        paths.root / "state_segmentation",
        paths.root / "state_segments.json",
        paths.root / "state_report.json",
        paths.root / "signal_extraction",
        paths.root / ".segment_cache",
        paths.root / "deploy_schedule.json",
        paths.keyframes_dir / "keyframes_meta.json",
    ]
    for stale_path in stale_paths:
        try:
            if stale_path.is_symlink() or stale_path.is_file():
                stale_path.unlink()
            elif stale_path.is_dir():
                shutil.rmtree(str(stale_path), ignore_errors=True)
        except FileNotFoundError:
            continue
        except Exception:
            continue


def materialize_keyframes(frames_dir, keyframes_dir, keyframe_indices, mode_requested):
    frames_dir_path = Path(frames_dir).resolve()
    keyframes_dir_path = Path(keyframes_dir).resolve()
    keyframes_dir_path.mkdir(parents=True, exist_ok=True)

    actual_mode = str(mode_requested or "symlink")

    def _make_one(src_path, dst_path):
        nonlocal actual_mode
        try:
            if os.path.lexists(str(dst_path)):
                os.remove(str(dst_path))
        except Exception:
            pass

        if actual_mode == "symlink":
            try:
                rel = os.path.relpath(str(src_path), start=str(dst_path.parent))
                os.symlink(rel, str(dst_path))
                return
            except Exception:
                actual_mode = "hardlink"

        if actual_mode == "hardlink":
            try:
                os.link(str(src_path), str(dst_path))
                return
            except Exception:
                actual_mode = "copy"

        shutil.copy2(str(src_path), str(dst_path))

    bytes_sum = 0
    for frame_idx in list(keyframe_indices or []):
        src_path = frames_dir_path / "{:06d}.png".format(int(frame_idx))
        dst_path = keyframes_dir_path / "{:06d}.png".format(int(frame_idx))
        if not src_path.is_file():
            continue
        _make_one(src_path, dst_path)
        try:
            bytes_sum += int(src_path.stat().st_size)
        except Exception:
            pass
    return actual_mode, int(bytes_sum)


def materialize_scene_split_visuals(paths, detector_result):
    from ._state_visualize import materialize_formal_visuals

    return dict(materialize_formal_visuals(paths, detector_result) or {})


def _dir_file_stats(dir_path):
    path = Path(dir_path).resolve()
    if not path.is_dir():
        return 0, 0
    file_count = 0
    bytes_sum = 0
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        file_count += 1
        try:
            bytes_sum += int(child.stat().st_size)
        except Exception:
            pass
    return int(file_count), int(bytes_sum)


def build_quality_diagnostics(paths, state_segments_payload, state_report_payload, extraction_meta):
    state_summary = dict(state_segments_payload.get("summary") or {})
    state_report = dict(state_report_payload.get("state", {}) or {})
    frame_files, frame_bytes = _dir_file_stats(paths.frames_dir)
    keyframe_files, keyframe_bytes = _dir_file_stats(paths.keyframes_dir)
    return {
        "frames": {
            "file_count": int(frame_files),
            "bytes_sum": int(frame_bytes),
            "timestamps_count": int(extraction_meta.get("timestamps_count", 0) or 0),
        },
        "keyframes": {
            "file_count": int(keyframe_files),
            "bytes_sum": int(keyframe_bytes),
        },
        "state_summary": {
            "segment_count": int(state_summary.get("segment_count", 0) or 0),
            "high_state_frame_ratio": float(state_summary.get("high_state_frame_ratio", 0.0) or 0.0),
            "high_risk_interval_count": int(state_summary.get("high_risk_interval_count", 0) or 0),
        },
        "quality_notes": {
            "overview_available": bool(paths.state_overview_path.is_file()),
            "state_report_embedded": bool(state_report_payload),
            "diagnostics_mode": str(state_report.get("report_schema_version", "") or "state_report"),
        },
    }


def build_segment_manifest(
    paths,
    policy_name,
    keyframes_meta,
    state_segments_payload,
    state_report_payload,
    quality_diagnostics,
):
    state_summary = dict(state_segments_payload.get("summary") or {})
    return {
        "version": 2,
        "schema": "segment_manifest.v2",
        "stage": "encode",
        "substage": "scene_split",
        "policy": str(policy_name),
        "contract": "encode_scene_split_mainline",
        "root": relative_to_exp(paths.exp_dir, paths.root),
        "artifacts": {
            "report": relative_to_exp(paths.exp_dir, paths.report_path),
            "visuals_dir": relative_to_exp(paths.exp_dir, paths.visuals_dir),
            "state_overview": relative_to_exp(paths.exp_dir, paths.state_overview_path),
        },
        "frames": {
            "dir": relative_to_exp(paths.exp_dir, paths.frames_dir),
            "frame_count": int(keyframes_meta.get("frame_count_total", 0) or 0),
            "frame_count_used": int(keyframes_meta.get("frame_count_used", 0) or 0),
            "tail_drop": int(keyframes_meta.get("tail_drop", 0) or 0),
        },
        "keyframes": {
            "count": int(keyframes_meta.get("keyframe_count", 0) or 0),
            "indices": list(keyframes_meta.get("keyframe_indices") or []),
            "uniform_base_indices": list(keyframes_meta.get("uniform_base_indices") or []),
            "summary": dict(keyframes_meta.get("summary") or {}),
        },
        "state_segments": dict(state_segments_payload),
        "planner": {
            "planner": "generation_units",
            "prompt_strategy": "prompt_spans",
            "planned_artifacts": {
                "motion_score": "segment/motion_score.json",
                "semantic_shift": "segment/semantic_shift.json",
                "generation_risk": "segment/generation_risk.json",
                "candidate_boundaries": "segment/candidate_boundaries.json",
                "generation_units": "segment/generation_units.json",
                "prompt_spans": "prompt/prompt_spans.json",
            },
        },
        "state_report": dict(state_report_payload),
        "quality_diagnostics": dict(quality_diagnostics or {}),
        "summary": {
            "scene_segment_count": int(state_summary.get("segment_count", 0) or 0),
            "state_segment_count": int(state_summary.get("segment_count", 0) or 0),
            "high_state_frame_ratio": float(state_summary.get("high_state_frame_ratio", 0.0) or 0.0),
            "high_risk_interval_count": int(state_summary.get("high_risk_interval_count", 0) or 0),
        },
    }


def build_segment_report(
    paths,
    inputs_meta,
    keyframes_meta,
    state_segments_payload,
    state_report_payload,
    quality_diagnostics,
    timings,
):
    state_summary = dict(state_segments_payload.get("summary") or {})
    return {
        "version": 2,
        "schema": "segment_report.v2",
        "stage": "encode",
        "substage": "scene_split",
        "policy": str(keyframes_meta.get("policy_name", "") or ""),
        "inputs": dict(inputs_meta),
        "artifacts": {
            "segment_manifest": relative_to_exp(paths.exp_dir, paths.manifest_path),
            "report": relative_to_exp(paths.exp_dir, paths.report_path),
            "state_overview": relative_to_exp(paths.exp_dir, paths.state_overview_path),
        },
        "frames": {
            "frame_count": int(keyframes_meta.get("frame_count_total", 0) or 0),
            "frame_count_used": int(keyframes_meta.get("frame_count_used", 0) or 0),
            "tail_drop": int(keyframes_meta.get("tail_drop", 0) or 0),
        },
        "keyframes": {
            "count": int(keyframes_meta.get("keyframe_count", 0) or 0),
            "bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
            "summary": dict(keyframes_meta.get("summary") or {}),
        },
        "state": {
            "summary": dict(state_summary),
            "report": dict(state_report_payload),
        },
        "planner": {
            "planner": "generation_units",
            "prompt_strategy": "prompt_spans",
        },
        "quality_diagnostics": dict(quality_diagnostics or {}),
        "timings_sec": dict(timings or {}),
    }


def write_segment_manifest(paths, manifest):
    write_json_atomic(paths.manifest_path, manifest, indent=2)
    return paths.manifest_path


def write_segment_report(paths, report):
    write_json_atomic(paths.report_path, report, indent=2)
    return paths.report_path
