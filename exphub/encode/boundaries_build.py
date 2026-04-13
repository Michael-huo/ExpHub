from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import list_frames_sorted, write_json_atomic
from exphub.common.logging import log_err, log_info, log_prog, log_warn

diagnostics = sys.modules[__name__]


@dataclass(frozen=True)
class SceneSplitArtifactPaths:
    exp_dir: Path
    root: Path
    frames_dir: Path
    report_path: Path
    calib_path: Path
    timestamps_path: Path


_SEMANTIC_PEAK_THRESHOLD = 0.45
_MOTION_JUMP_THRESHOLD = 0.20
_RISK_JUMP_THRESHOLD = 0.18
_BOUNDARY_STRENGTH_FLOOR = 0.25


_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
FORMAL_SEGMENT_POLICY = "state"


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


def build_candidate_boundaries_payload(motion_score_payload, semantic_shift_payload, generation_risk_payload):
    motion_rows = list(_as_dict(motion_score_payload).get("segments") or [])
    semantic_rows = list(_as_dict(semantic_shift_payload).get("segments") or [])
    risk_rows = list(_as_dict(generation_risk_payload).get("segments") or [])
    if not motion_rows or not semantic_rows or not risk_rows:
        raise RuntimeError("candidate boundary extraction requires motion, semantic, and risk segments")
    if not (len(motion_rows) == len(semantic_rows) == len(risk_rows)):
        raise RuntimeError("candidate boundary extraction requires matched signal segment counts")

    boundaries = []
    for idx in range(1, len(risk_rows)):
        prev_motion = _as_dict(motion_rows[idx - 1])
        curr_motion = _as_dict(motion_rows[idx])
        prev_semantic = _as_dict(semantic_rows[idx - 1])
        curr_semantic = _as_dict(semantic_rows[idx])
        prev_risk = _as_dict(risk_rows[idx - 1])
        curr_risk = _as_dict(risk_rows[idx])

        frame_idx = _safe_int(curr_risk.get("start_frame"), 0)
        motion_jump = abs(_safe_float(curr_motion.get("motion_score")) - _safe_float(prev_motion.get("motion_score")))
        semantic_peak = _safe_float(curr_semantic.get("semantic_shift"))
        risk_jump = abs(_safe_float(curr_risk.get("generation_risk")) - _safe_float(prev_risk.get("generation_risk")))
        scene_change = bool(curr_semantic.get("scene_label") != prev_semantic.get("scene_label"))
        risk_level_change = bool(curr_risk.get("risk_level") != prev_risk.get("risk_level"))
        motion_label_change = bool(curr_motion.get("motion_label") != prev_motion.get("motion_label"))
        stability_change = 1.0 if (scene_change or risk_level_change or motion_label_change) else 0.0
        boundary_strength = min(
            1.0,
            (0.35 * semantic_peak) + (0.30 * risk_jump) + (0.20 * motion_jump) + (0.15 * stability_change),
        )

        reasons = []
        if semantic_peak >= _SEMANTIC_PEAK_THRESHOLD:
            reasons.append("semantic_peak")
        if motion_jump >= _MOTION_JUMP_THRESHOLD:
            reasons.append("motion_jump")
        if risk_jump >= _RISK_JUMP_THRESHOLD:
            reasons.append("risk_jump")
        if scene_change:
            reasons.append("scene_group_change")
        if risk_level_change:
            reasons.append("risk_level_change")
        if motion_label_change:
            reasons.append("motion_label_change")

        if not reasons and boundary_strength < _BOUNDARY_STRENGTH_FLOOR:
            continue

        boundaries.append(
            {
                "candidate_id": int(len(boundaries)),
                "frame_idx": int(frame_idx),
                "previous_segment_id": _safe_int(prev_risk.get("segment_id"), idx - 1),
                "next_segment_id": _safe_int(curr_risk.get("segment_id"), idx),
                "strength": float(boundary_strength),
                "reasons": reasons,
                "source_scores": {
                    "motion_jump": float(motion_jump),
                    "semantic_peak": float(semantic_peak),
                    "risk_jump": float(risk_jump),
                    "stability_change": float(stability_change),
                },
            }
        )

    return {
        "version": 1,
        "schema": "candidate_boundaries.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "candidate_boundaries",
        "source": "encode.candidate_boundaries.extract",
        "extractor": "signal_transition_peaks_v1",
        "criteria": {
            "semantic_peak_threshold": float(_SEMANTIC_PEAK_THRESHOLD),
            "motion_jump_threshold": float(_MOTION_JUMP_THRESHOLD),
            "risk_jump_threshold": float(_RISK_JUMP_THRESHOLD),
            "boundary_strength_floor": float(_BOUNDARY_STRENGTH_FLOOR),
        },
        "boundaries": boundaries,
        "summary": {
            "candidate_count": int(len(boundaries)),
            "sequence_start_idx": _safe_int(_as_dict(risk_rows[0]).get("start_frame"), 0),
            "sequence_end_idx": _safe_int(_as_dict(risk_rows[-1]).get("end_frame"), 0),
            "max_strength": float(max([item.get("strength", 0.0) for item in boundaries]) if boundaries else 0.0),
        },
    }


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


def materialize_scene_split_visuals(paths, detector_result):
    raw_source_path = None
    if isinstance(detector_result, dict):
        raw_source_path = detector_result.get("state_overview_path")
    if not raw_source_path:
        return {}
    source_path = Path(raw_source_path).resolve()
    if not source_path.is_file():
        return {}
    handoff_path = Path(paths.root).resolve() / "state_overview.png"
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


def require_formal_segment_policy(policy_name):
    name = str(policy_name or FORMAL_SEGMENT_POLICY).strip().lower() or FORMAL_SEGMENT_POLICY
    if name != FORMAL_SEGMENT_POLICY:
        raise ValueError(
            "formal segment workflow only supports policy '{}' in the current mainline (got '{}')".format(
                FORMAL_SEGMENT_POLICY,
                str(policy_name or "").strip() or "<empty>",
            )
        )
    return name


def _maybe_progress(iterable):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc="Extract", bar_format=_BAR_FORMAT)


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


def _load_calib(fx, fy, cx, cy, dist):
    if fx is None or fy is None or cx is None or cy is None:
        raise RuntimeError("encode.scene_split requires fx/fy/cx/cy")
    dist_arr = None
    if dist:
        import numpy as np

        dist_arr = np.array([float(item) for item in dist], dtype=np.float64)
    return float(fx), float(fy), float(cx), float(cy), dist_arr


def _compute_center_crop_to_aspect(width, height, target_ratio):
    in_ratio = float(width) / float(height)
    if in_ratio > target_ratio:
        crop_h = int(height)
        crop_w = int(round(float(height) * float(target_ratio)))
        x0 = int((int(width) - crop_w) // 2)
        y0 = 0
    else:
        crop_w = int(width)
        crop_h = int(round(float(width) / float(target_ratio)))
        x0 = 0
        y0 = int((int(height) - crop_h) // 2)

    crop_w = max(1, min(int(crop_w), int(width)))
    crop_h = max(1, min(int(crop_h), int(height)))
    x0 = max(0, min(int(x0), int(width) - crop_w))
    y0 = max(0, min(int(y0), int(height) - crop_h))
    return x0, y0, crop_w, crop_h


def _build_spatial_transform(input_w, input_h, target_w, target_h):
    target_ratio = float(target_w) / float(target_h)
    x0, y0, crop_w, crop_h = _compute_center_crop_to_aspect(input_w, input_h, target_ratio)

    scale_x = float(target_w) / float(crop_w)
    scale_y = float(target_h) / float(crop_h)
    scale = max(scale_x, scale_y)

    resized_w = max(int(target_w), int(round(float(crop_w) * float(scale))))
    resized_h = max(int(target_h), int(round(float(crop_h) * float(scale))))
    crop2_x0 = int((resized_w - int(target_w)) // 2)
    crop2_y0 = int((resized_h - int(target_h)) // 2)

    return {
        "input_w": int(input_w),
        "input_h": int(input_h),
        "target_w": int(target_w),
        "target_h": int(target_h),
        "crop1_x0": int(x0),
        "crop1_y0": int(y0),
        "crop1_w": int(crop_w),
        "crop1_h": int(crop_h),
        "scale": float(scale),
        "resized_w": int(resized_w),
        "resized_h": int(resized_h),
        "crop2_x0": int(crop2_x0),
        "crop2_y0": int(crop2_y0),
    }


def _apply_spatial_transform(img_bgr, transform, interpolation):
    import cv2

    x0 = int(transform["crop1_x0"])
    y0 = int(transform["crop1_y0"])
    crop_w = int(transform["crop1_w"])
    crop_h = int(transform["crop1_h"])
    scale = float(transform["scale"])
    resized_w = int(transform["resized_w"])
    resized_h = int(transform["resized_h"])
    crop2_x0 = int(transform["crop2_x0"])
    crop2_y0 = int(transform["crop2_y0"])
    target_w = int(transform["target_w"])
    target_h = int(transform["target_h"])

    crop1 = img_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if interpolation == "auto":
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    elif interpolation == "area":
        interp = cv2.INTER_AREA
    elif interpolation == "linear":
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_LANCZOS4
    resized = cv2.resize(crop1, (resized_w, resized_h), interpolation=interp)
    return resized[crop2_y0 : crop2_y0 + target_h, crop2_x0 : crop2_x0 + target_w]


def _transform_intrinsics(fx, fy, cx, cy, transform):
    crop_x = int(transform["crop1_x0"])
    crop_y = int(transform["crop1_y0"])
    scale = float(transform["scale"])
    crop2_x = int(transform["crop2_x0"])
    crop2_y = int(transform["crop2_y0"])

    cx_stage1 = (float(cx) - float(crop_x)) * scale
    cy_stage1 = (float(cy) - float(crop_y)) * scale
    fx_stage1 = float(fx) * scale
    fy_stage1 = float(fy) * scale
    return (
        float(fx_stage1),
        float(fy_stage1),
        float(cx_stage1 - float(crop2_x)),
        float(cy_stage1 - float(crop2_y)),
    )


def _decode_compressed_image(msg):
    import cv2
    import numpy as np

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _decode_raw_image(msg):
    import cv2
    import numpy as np

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step) if hasattr(msg, "step") else width
    encoding = str(msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < int(step) * int(height):
        raise ValueError("raw image buffer too small: {} < {}*{}".format(buf.size, step, height))
    buf = buf[: int(step) * int(height)].reshape((int(height), int(step)))

    def _take(bytes_per_pixel, channels):
        need = int(width) * int(bytes_per_pixel)
        if int(step) < int(need):
            raise ValueError("step too small for encoding {}: step={} need={}".format(encoding, step, need))
        return buf[:, :need].reshape((int(height), int(width), int(channels)))

    if encoding == "bgr8":
        return _take(3, 3).copy()
    if encoding == "rgb8":
        return _take(3, 3)[:, :, ::-1].copy()
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(_take(1, 1).reshape((int(height), int(width))), cv2.COLOR_GRAY2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(_take(4, 4), cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(_take(4, 4), cv2.COLOR_RGBA2BGR)
    raise ValueError("unsupported sensor_msgs/Image encoding: {}".format(msg.encoding))


def _decode_ros_image_to_bgr(msg):
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "encoding"):
        return _decode_compressed_image(msg)
    if hasattr(msg, "encoding") and hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
        return _decode_raw_image(msg)
    raise TypeError("unknown ROS image message type: {}".format(getattr(msg, "_type", type(msg))))


def _extract_frames(args, paths):
    try:
        import cv2
        import numpy as np
        import rosbag
        from genpy import Time
    except Exception as exc:
        log_err("encode.scene_split import failed: {}".format(exc))
        raise SystemExit(2)

    if not Path(args.bag).is_file():
        log_err("bag not found: {}".format(args.bag))
        raise SystemExit(2)

    fx, fy, cx, cy, dist = _load_calib(args.fx, args.fy, args.cx, args.cy, args.dist)
    bag = rosbag.Bag(args.bag, "r")
    first_time = None
    first_msg = None
    for _topic, msg, stamp in bag.read_messages(topics=[args.topic]):
        first_time = float(stamp.to_sec())
        first_msg = msg
        break
    if first_time is None:
        log_err("No messages found for topic: {}".format(args.topic))
        raise SystemExit(2)

    if int(args.start_idx) >= 0:
        start_time_abs = None
        idx = 0
        for _topic, _msg, stamp in bag.read_messages(topics=[args.topic]):
            if idx == int(args.start_idx):
                start_time_abs = float(stamp.to_sec())
                break
            idx += 1
        if start_time_abs is None:
            log_err("start_idx {} out of range for topic {}".format(int(args.start_idx), args.topic))
            raise SystemExit(2)
    else:
        start_time_abs = float(first_time) + float(args.start_sec)

    duration = float(args.duration)
    fps = float(args.fps)
    out_count = int(math.floor(float(duration) * float(fps) + 1e-9)) + 1
    dt = 1.0 / float(fps)
    target_abs_times = [float(start_time_abs) + float(idx) * float(dt) for idx in range(out_count)]
    target_rel_times = [float(idx) * float(dt) for idx in range(out_count)]
    end_time_abs = float(start_time_abs) + float(out_count - 1) * float(dt)

    log_info("encode.scene_split start: exp_dir={}".format(paths.exp_dir))
    log_info("scene split state analysis: policy=state kf_gap={}".format(int(args.kf_gap)))
    log_info("out_count: {} fps={:.3f}".format(int(out_count), float(fps)))

    st = Time.from_sec(float(start_time_abs) - 2.0)
    et = Time.from_sec(float(end_time_abs) + 2.0)

    transform = None
    last_src_stamp = None
    last_src_img = None
    calib_out = []
    prev_t = None
    prev_msg = None
    k = 0
    timestamps_lines = []
    for _topic, msg, stamp in _maybe_progress(bag.read_messages(topics=[args.topic], start_time=st, end_time=et)):
        cur_t = float(stamp.to_sec())
        if prev_t is None:
            prev_t = cur_t
            prev_msg = msg
            continue

        while k < out_count and float(target_abs_times[k]) <= cur_t:
            target_time = float(target_abs_times[k])
            if prev_msg is None:
                chosen_t = cur_t
                chosen_msg = msg
            elif abs(float(target_time) - float(prev_t)) <= abs(float(cur_t) - float(target_time)):
                chosen_t = float(prev_t)
                chosen_msg = prev_msg
            else:
                chosen_t = float(cur_t)
                chosen_msg = msg

            if last_src_stamp is not None and abs(float(chosen_t) - float(last_src_stamp)) < 1e-12:
                img_bgr = last_src_img
            else:
                img_bgr = _decode_ros_image_to_bgr(chosen_msg)
                last_src_stamp = float(chosen_t)
                last_src_img = img_bgr

            if img_bgr is None:
                log_warn("failed to decode image at t={:.6f}".format(float(chosen_t)))
                k += 1
                continue

            if transform is None:
                input_h, input_w = img_bgr.shape[:2]
                transform = _build_spatial_transform(input_w, input_h, int(args.width), int(args.height))
                fx2, fy2, cx2, cy2 = _transform_intrinsics(fx, fy, cx, cy, transform)
                calib_out = [fx2, fy2, cx2, cy2]
                if dist is not None and dist.size > 0:
                    calib_out.extend([float(item) for item in dist])
                np.savetxt(
                    str(paths.calib_path),
                    np.array(calib_out, dtype=np.float64).reshape(1, -1),
                    fmt="%.10f",
                )

            out_img = _apply_spatial_transform(img_bgr, transform, "auto")
            out_path = paths.frames_dir / "{:06d}.png".format(int(k))
            cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            timestamps_lines.append("{:.9f}\n".format(float(target_rel_times[k])))
            k += 1

        prev_t = cur_t
        prev_msg = msg
        if k >= out_count:
            break

    if k < out_count and prev_msg is not None:
        log_warn("padding remaining frames with last msg: {}/{}".format(int(k), int(out_count)))
        img_bgr = _decode_ros_image_to_bgr(prev_msg)
        if transform is None and img_bgr is not None:
            input_h, input_w = img_bgr.shape[:2]
            transform = _build_spatial_transform(input_w, input_h, int(args.width), int(args.height))
            fx2, fy2, cx2, cy2 = _transform_intrinsics(fx, fy, cx, cy, transform)
            calib_out = [fx2, fy2, cx2, cy2]
            if dist is not None and dist.size > 0:
                calib_out.extend([float(item) for item in dist])
            np.savetxt(
                str(paths.calib_path),
                np.array(calib_out, dtype=np.float64).reshape(1, -1),
                fmt="%.10f",
            )
        while k < out_count and img_bgr is not None:
            out_img = _apply_spatial_transform(img_bgr, transform, "auto")
            out_path = paths.frames_dir / "{:06d}.png".format(int(k))
            cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            timestamps_lines.append("{:.9f}\n".format(float(target_rel_times[k])))
            k += 1

    bag.close()
    with paths.timestamps_path.open("w", encoding="utf-8") as handle:
        handle.writelines(timestamps_lines)

    actual_frame_count = len(list_frames_sorted(paths.frames_dir))
    return {
        "frame_count": int(actual_frame_count),
        "timestamps_count": int(len(timestamps_lines)),
        "timestamps": [float(item) for item in target_rel_times],
        "calib": [float(item) for item in calib_out],
    }


def _build_keyframes_meta(plan, kf_gap, keyframes_mode, actual_mode, keyframe_bytes_sum):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "state",
        "kf_gap": int(kf_gap),
        "mode_requested": str(keyframes_mode),
        "mode_actual": str(actual_mode),
        "frame_count_total": int(plan["frame_count_total"]),
        "frame_count_used": int(plan["frame_count_used"]),
        "tail_drop": int(plan["tail_drop"]),
        "keyframe_count": int(len(plan["keyframe_indices"])),
        "keyframe_indices": list(plan["keyframe_indices"]),
        "keyframe_bytes_sum": int(keyframe_bytes_sum),
        "policy_name": "state",
        "uniform_base_indices": list(plan.get("uniform_base_indices", [])),
        "keyframes": list(plan["keyframe_items"]),
        "summary": dict(plan["summary"]),
        "policy_meta": dict(plan.get("policy_meta") or {}),
        "note": "encode.scene_split keeps state segmentation as the single mainline keyframe policy.",
    }


def run_formal_mainline(args):
    from ._state_detector import run_state_mainline

    paths = diagnostics.build_paths(args.exp_dir)
    diagnostics.remove_stale_scene_split_outputs(paths)
    diagnostics.ensure_layout(paths)
    total_started = time.time()

    extract_started = time.time()
    extraction_meta = _extract_frames(args, paths)
    extract_sec = float(time.time() - extract_started)
    log_info("scene split frame extraction completed in {:.2f}s".format(extract_sec))

    detector_started = time.time()
    detector_result = run_state_mainline(
        segment_dir=paths.root,
        frames_dir=paths.frames_dir,
        timestamps_path=paths.timestamps_path,
        kf_gap=int(args.kf_gap),
    )
    detect_sec = float(time.time() - detector_started)
    log_info("scene split state analysis completed in {:.2f}s".format(detect_sec))

    materialize_started = time.time()
    actual_mode, keyframe_bytes_sum = diagnostics.summarize_keyframes(
        frames_dir=paths.frames_dir,
        keyframe_indices=detector_result["plan"]["keyframe_indices"],
        mode_requested=args.keyframes_mode,
    )
    keyframes_meta = _build_keyframes_meta(
        detector_result["plan"],
        kf_gap=int(args.kf_gap),
        keyframes_mode=args.keyframes_mode,
        actual_mode=actual_mode,
        keyframe_bytes_sum=keyframe_bytes_sum,
    )
    visual_result = diagnostics.materialize_scene_split_visuals(paths, detector_result)
    materialize_sec = float(time.time() - materialize_started)
    log_info("scene split artifacts completed in {:.2f}s".format(materialize_sec))

    state_segments_payload = dict(detector_result["state_segments_payload"])
    state_report_payload = dict(detector_result["state_report_payload"])
    quality_diagnostics = diagnostics.build_quality_diagnostics(
        paths=paths,
        state_segments_payload=state_segments_payload,
        state_report_payload=state_report_payload,
        extraction_meta=extraction_meta,
        keyframes_meta=keyframes_meta,
    )

    inputs_meta = {
        "bag": str(Path(args.bag).resolve()),
        "topic": str(args.topic),
        "fps": float(args.fps),
        "duration": float(args.duration),
        "start_sec": float(args.start_sec),
        "start_idx": int(args.start_idx),
        "width": int(args.width),
        "height": int(args.height),
    }
    timings = {
        "extract": float(extract_sec),
        "state_mainline": float(detect_sec),
        "materialize": float(materialize_sec),
        "total": float(time.time() - total_started),
    }

    report = diagnostics.build_input_report(
        paths=paths,
        inputs_meta=inputs_meta,
        keyframes_meta=keyframes_meta,
        extraction_meta=extraction_meta,
        state_segments_payload=state_segments_payload,
        state_report_payload=state_report_payload,
        quality_diagnostics=quality_diagnostics,
        timings=timings,
    )
    diagnostics.write_input_report(paths, report)
    for temp_path in (paths.calib_path, paths.timestamps_path):
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
    try:
        shutil.rmtree(str(paths.root / "visuals"), ignore_errors=True)
    except Exception:
        pass

    log_prog(
        "scene split summary: frames={} keyframes={} state_segments={}".format(
            int(extraction_meta.get("frame_count", 0) or 0),
            int(keyframes_meta.get("keyframe_count", 0) or 0),
            int((state_segments_payload.get("summary") or {}).get("segment_count", 0) or 0),
        )
    )
    log_info("scene split report: {}".format(paths.report_path))
    return paths.report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub encode.scene_split mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--bag", required=True, help="rosbag path")
    parser.add_argument("--topic", required=True, help="image topic")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--dist", nargs="*", default=None)
    parser.add_argument("--kf_gap", type=int, required=True)
    parser.add_argument("--keyframes_mode", default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--segment_policy", default="state")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    require_formal_segment_policy(args.segment_policy)
    run_formal_mainline(args)


if __name__ == "__main__":
    main()
