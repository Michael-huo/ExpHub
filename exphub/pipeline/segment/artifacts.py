from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from exphub.common.io import read_json_dict, write_json_atomic
from exphub.contracts import segment as segment_contract


@dataclass(frozen=True)
class SegmentArtifactPaths:
    exp_dir: Path
    root: Path
    frames_dir: Path
    keyframes_dir: Path
    keyframes_meta_path: Path
    deploy_schedule_path: Path
    manifest_path: Path
    report_path: Path
    visuals_dir: Path
    state_overview_path: Path
    compat_state_dir: Path
    compat_state_segments_path: Path
    compat_state_report_path: Path
    compat_state_overview_path: Path
    step_meta_path: Path
    preprocess_meta_path: Path
    calib_path: Path
    timestamps_path: Path


def build_paths(exp_dir):
    exp_dir_path = Path(exp_dir).resolve()
    root = (exp_dir_path / "segment").resolve()
    return SegmentArtifactPaths(
        exp_dir=exp_dir_path,
        root=root,
        frames_dir=(root / "frames").resolve(),
        keyframes_dir=(root / "keyframes").resolve(),
        keyframes_meta_path=(root / "keyframes" / "keyframes_meta.json").resolve(),
        deploy_schedule_path=(root / "deploy_schedule.json").resolve(),
        manifest_path=(root / segment_contract.SEGMENT_MANIFEST_NAME).resolve(),
        report_path=(root / segment_contract.SEGMENT_REPORT_NAME).resolve(),
        visuals_dir=(root / segment_contract.SEGMENT_VISUALS_DIRNAME).resolve(),
        state_overview_path=(
            root / segment_contract.SEGMENT_VISUALS_DIRNAME / segment_contract.SEGMENT_OVERVIEW_NAME
        ).resolve(),
        compat_state_dir=(root / segment_contract.SEGMENT_COMPAT_STATE_DIRNAME).resolve(),
        compat_state_segments_path=(
            root
            / segment_contract.SEGMENT_COMPAT_STATE_DIRNAME
            / segment_contract.SEGMENT_COMPAT_STATE_SEGMENTS_NAME
        ).resolve(),
        compat_state_report_path=(
            root
            / segment_contract.SEGMENT_COMPAT_STATE_DIRNAME
            / segment_contract.SEGMENT_COMPAT_STATE_REPORT_NAME
        ).resolve(),
        compat_state_overview_path=(
            root
            / segment_contract.SEGMENT_COMPAT_STATE_DIRNAME
            / segment_contract.SEGMENT_OVERVIEW_NAME
        ).resolve(),
        step_meta_path=(root / "step_meta.json").resolve(),
        preprocess_meta_path=(root / "preprocess_meta.json").resolve(),
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


def write_keyframes_meta(paths, keyframes_meta):
    write_json_atomic(paths.keyframes_meta_path, keyframes_meta, indent=2)
    return paths.keyframes_meta_path


def write_deploy_schedule(paths, deploy_schedule):
    write_json_atomic(paths.deploy_schedule_path, deploy_schedule, indent=2)
    return paths.deploy_schedule_path


def write_segment_manifest(paths, manifest):
    write_json_atomic(paths.manifest_path, manifest, indent=2)
    return paths.manifest_path


def write_segment_report(paths, report):
    write_json_atomic(paths.report_path, report, indent=2)
    return paths.report_path


def write_step_meta(paths, step_meta):
    write_json_atomic(paths.step_meta_path, step_meta, indent=2)
    return paths.step_meta_path


def load_compat_state_payloads(paths):
    return {
        "state_segments": read_json_dict(paths.compat_state_segments_path),
        "state_report": read_json_dict(paths.compat_state_report_path),
    }


def copy_overview_to_formal_visuals(paths, source_path):
    source = Path(source_path).resolve()
    if not source.is_file():
        return None
    paths.visuals_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(paths.state_overview_path))
    return paths.state_overview_path


def build_manifest(
    paths,
    policy_name,
    keyframes_meta,
    deploy_schedule,
    state_segments_payload,
    state_report_payload,
):
    projection_stats = dict(deploy_schedule.get("projection_stats") or {})
    state_summary = dict(state_segments_payload.get("summary") or {})
    return {
        "version": 1,
        "schema": "segment_manifest.v1",
        "stage": "segment",
        "policy": str(policy_name),
        "contract": "formal_segment_mainline",
        "root": relative_to_exp(paths.exp_dir, paths.root),
        "artifacts": {
            "report": relative_to_exp(paths.exp_dir, paths.report_path),
            "visuals_dir": relative_to_exp(paths.exp_dir, paths.visuals_dir),
            "state_overview": relative_to_exp(paths.exp_dir, paths.state_overview_path),
            "keyframes_meta": relative_to_exp(paths.exp_dir, paths.keyframes_meta_path),
            "deploy_schedule": relative_to_exp(paths.exp_dir, paths.deploy_schedule_path),
            "state_segments_compat": relative_to_exp(paths.exp_dir, paths.compat_state_segments_path),
            "state_report_compat": relative_to_exp(paths.exp_dir, paths.compat_state_report_path),
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
        "deploy_schedule": dict(deploy_schedule),
        "state_segments": dict(state_segments_payload),
        "state_report": dict(state_report_payload),
        "summary": {
            "deploy_segment_count": int(projection_stats.get("segment_count", 0) or 0),
            "state_segment_count": int(state_summary.get("segment_count", 0) or 0),
            "high_state_frame_ratio": float(state_summary.get("high_state_frame_ratio", 0.0) or 0.0),
        },
    }


def build_report(
    paths,
    inputs_meta,
    keyframes_meta,
    deploy_schedule,
    state_segments_payload,
    state_report_payload,
    timings,
):
    projection_stats = dict(deploy_schedule.get("projection_stats") or {})
    state_summary = dict(state_segments_payload.get("summary") or {})
    return {
        "version": 1,
        "schema": "segment_report.v1",
        "stage": "segment",
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
        "deploy_schedule": {
            "backend": str(deploy_schedule.get("backend", "") or ""),
            "segment_count": int(projection_stats.get("segment_count", 0) or 0),
            "mean_abs_boundary_shift": float(projection_stats.get("mean_abs_boundary_shift", 0.0) or 0.0),
            "max_abs_gap_error": int(projection_stats.get("max_abs_gap_error", 0) or 0),
        },
        "state": {
            "summary": dict(state_summary),
            "report_summary": dict(state_report_payload.get("summary") or {}),
        },
        "timings_sec": dict(timings or {}),
    }

