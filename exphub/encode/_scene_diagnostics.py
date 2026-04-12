from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from exphub.common.io import write_json_atomic


@dataclass(frozen=True)
class SceneSplitArtifactPaths:
    exp_dir: Path
    root: Path
    frames_dir: Path
    report_path: Path
    calib_path: Path
    timestamps_path: Path


def build_paths(exp_dir):
    exp_dir_path = Path(exp_dir).resolve()
    root = (exp_dir_path / "input").resolve()
    return SceneSplitArtifactPaths(
        exp_dir=exp_dir_path,
        root=root,
        frames_dir=(root / "frames").resolve(),
        report_path=(root / "input_report.json").resolve(),
        calib_path=(root / ".calib_runtime.txt").resolve(),
        timestamps_path=(root / ".timestamps_runtime.txt").resolve(),
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


def remove_stale_scene_split_outputs(paths):
    stale_paths = [
        paths.root / "keyframes",
        paths.root / "visuals",
        paths.root / "state_overview.png",
        paths.root / "state_segmentation",
        paths.root / "signal_extraction",
        paths.root / "state_segments.json",
        paths.root / "state_report.json",
        paths.root / "deploy_schedule.json",
        paths.calib_path,
        paths.timestamps_path,
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


def summarize_keyframes(frames_dir, keyframe_indices, mode_requested):
    frames_dir_path = Path(frames_dir).resolve()
    bytes_sum = 0
    for frame_idx in list(keyframe_indices or []):
        src_path = frames_dir_path / "{:06d}.png".format(int(frame_idx))
        if not src_path.is_file():
            continue
        try:
            bytes_sum += int(src_path.stat().st_size)
        except Exception:
            pass
    return str(mode_requested or "metadata_only"), int(bytes_sum)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _segment_row_map(rows):
    mapped = {}
    for idx, raw_item in enumerate(list(rows or [])):
        row = _as_dict(raw_item)
        segment_id = _safe_int(row.get("segment_id"), idx)
        mapped[int(segment_id)] = row
    return mapped


def _build_encode_overview_frame_rows(input_report, encode_plan):
    input_obj = _as_dict(input_report)
    encode_obj = _as_dict(encode_plan)
    state_segments = list(_as_dict(input_obj.get("state_segments")).get("segments") or [])
    if not state_segments:
        raise RuntimeError("input report missing state_segments.segments for encode overview")

    signals = _as_dict(encode_obj.get("signals"))
    motion_rows = _segment_row_map(_as_dict(signals.get("motion_score")).get("segments"))
    semantic_rows = _segment_row_map(_as_dict(signals.get("semantic_shift")).get("segments"))
    risk_rows = _segment_row_map(_as_dict(signals.get("generation_risk")).get("segments"))

    frame_rows = []
    for idx, raw_segment in enumerate(state_segments):
        segment = _as_dict(raw_segment)
        segment_id = _safe_int(segment.get("segment_id"), idx)
        start_frame = _safe_int(segment.get("start_frame"), 0)
        end_frame = max(start_frame, _safe_int(segment.get("end_frame"), start_frame))
        motion_row = _as_dict(motion_rows.get(segment_id))
        semantic_row = _as_dict(semantic_rows.get(segment_id))
        risk_row = _as_dict(risk_rows.get(segment_id))
        motion_value = _safe_float(
            motion_row.get("motion_score"),
            _safe_float(segment.get("state_score_mean"), _safe_float(segment.get("state_score_peak"))),
        )
        semantic_value = _safe_float(semantic_row.get("semantic_shift"))
        state_value = _safe_float(segment.get("state_score_mean"), _safe_float(segment.get("state_score_peak")))
        detector_value = _safe_float(
            segment.get("detector_score_mean"),
            _safe_float(risk_row.get("generation_risk"), _safe_float(segment.get("detector_score_peak"))),
        )
        timestamp_start = _safe_float(segment.get("start_time"))
        timestamp_end = _safe_float(segment.get("end_time"), timestamp_start)
        duration_frames = max(1, int(end_frame - start_frame + 1))
        duration_sec = max(0.0, float(timestamp_end - timestamp_start))
        dt_sec = float(duration_sec / float(duration_frames - 1)) if duration_frames > 1 else 0.0

        for offset, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            frame_rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp": float(timestamp_start + (float(offset) * dt_sec)),
                    "motion_velocity_state_signal": float(motion_value),
                    "semantic_velocity_state_signal": float(semantic_value),
                    "state_score": float(state_value),
                    "detector_score": float(detector_value),
                    "state_label": str(segment.get("state_label", "low_state") or "low_state"),
                }
            )
    return frame_rows, state_segments


def _keyframe_indices(input_report):
    raw_indices = list(_as_dict(_as_dict(input_report).get("keyframes")).get("indices") or [])
    indices = []
    for value in raw_indices:
        try:
            indices.append(int(value))
        except Exception:
            continue
    return sorted(set(indices))


def _boundary_indices(encode_plan):
    raw_indices = list(_as_dict(_as_dict(encode_plan).get("boundaries")).get("selected") or [])
    indices = []
    for value in raw_indices:
        try:
            indices.append(int(value))
        except Exception:
            continue
    return sorted(set(indices))


def write_encode_segmentation_overview(output_path, input_report, encode_plan, source_path=None):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None:
        source_path = Path(source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError("encode overview source not found: {}".format(source_path))
        if source_path != output_path:
            shutil.copy2(str(source_path), str(output_path))
        else:
            output_path.touch()
        return output_path

    raise RuntimeError(
        "encode overview source path missing from detector_result; "
        "the overview must be produced from the real frame_rows upstream"
    )
    return output_path


def materialize_scene_split_visuals(paths, detector_result):
    paths = Path(paths.root).resolve() if hasattr(paths, "root") else None
    raw_source_path = None
    if isinstance(detector_result, dict):
        raw_source_path = detector_result.get("state_overview_path")
    if paths is None or not raw_source_path:
        return {}
    source_path = Path(raw_source_path).resolve()
    if not source_path.is_file():
        return {}
    handoff_path = Path(paths) / "state_overview.png"
    try:
        if source_path != handoff_path:
            shutil.copy2(str(source_path), str(handoff_path))
        else:
            handoff_path.touch()
    except Exception:
        return {}
    return {
        "state_overview_path": handoff_path,
        "source_path": source_path,
        "handoff_path": handoff_path,
    }


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


def build_quality_diagnostics(paths, state_segments_payload, state_report_payload, extraction_meta, keyframes_meta):
    state_summary = dict(state_segments_payload.get("summary") or {})
    state_report = dict(state_report_payload.get("state", {}) or {})
    frame_files, frame_bytes = _dir_file_stats(paths.frames_dir)
    return {
        "frames": {
            "file_count": int(frame_files),
            "bytes_sum": int(frame_bytes),
            "timestamps_count": int(extraction_meta.get("timestamps_count", 0) or 0),
        },
        "keyframes": {
            "count": int(keyframes_meta.get("keyframe_count", 0) or 0),
            "bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
            "mode": str(keyframes_meta.get("mode_actual", "") or "metadata_only"),
        },
        "state_summary": {
            "segment_count": int(state_summary.get("segment_count", 0) or 0),
            "high_state_frame_ratio": float(state_summary.get("high_state_frame_ratio", 0.0) or 0.0),
            "high_risk_interval_count": int(state_summary.get("high_risk_interval_count", 0) or 0),
        },
        "quality_notes": {
            "state_report_embedded": bool(state_report_payload),
            "diagnostics_mode": str(state_report.get("report_schema_version", "") or "state_report"),
        },
    }


def build_input_report(
    paths,
    inputs_meta,
    extraction_meta,
    keyframes_meta,
    state_segments_payload,
    state_report_payload,
    quality_diagnostics,
    timings,
):
    timestamps = list(extraction_meta.get("timestamps") or [])
    calib = list(extraction_meta.get("calib") or [])
    state_summary = dict(state_segments_payload.get("summary") or {})
    return {
        "version": 1,
        "schema": "input_report.v1",
        "stage": "input",
        "substage": "frames_prepare",
        "policy": str(keyframes_meta.get("policy_name", "") or ""),
        "inputs": dict(inputs_meta),
        "artifacts": {
            "input_report": relative_to_exp(paths.exp_dir, paths.report_path),
            "frames_dir": relative_to_exp(paths.exp_dir, paths.frames_dir),
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
            "bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
            "summary": dict(keyframes_meta.get("summary") or {}),
        },
        "state_segments": dict(state_segments_payload),
        "state_report": dict(state_report_payload),
        "camera": {
            "calib": list(calib),
            "timestamps": list(timestamps),
        },
        "extraction": {
            "timestamps_count": int(extraction_meta.get("timestamps_count", 0) or 0),
            "frame_count": int(extraction_meta.get("frame_count", 0) or 0),
        },
        "quality_diagnostics": dict(quality_diagnostics or {}),
        "summary": {
            "frame_count": int(keyframes_meta.get("frame_count_total", 0) or 0),
            "frame_count_used": int(keyframes_meta.get("frame_count_used", 0) or 0),
            "keyframe_count": int(keyframes_meta.get("keyframe_count", 0) or 0),
            "state_segment_count": int(state_summary.get("segment_count", 0) or 0),
            "high_state_frame_ratio": float(state_summary.get("high_state_frame_ratio", 0.0) or 0.0),
            "high_risk_interval_count": int(state_summary.get("high_risk_interval_count", 0) or 0),
        },
        "timings_sec": dict(timings or {}),
    }


def write_input_report(paths, report):
    write_json_atomic(paths.report_path, report, indent=2)
    return paths.report_path
