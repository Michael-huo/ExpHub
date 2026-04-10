from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import list_frames_sorted
from exphub.common.logging import log_err, log_info, log_prog, log_warn
from exphub.contracts import segment as segment_contract
from exphub.pipeline.encode.scene_split import diagnostics


_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
_DECODE_ALIGNMENT_MODULUS = 4
_EXPORT_TARGET_FRAMES = 73


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


def _decode_legal_num_frames(desired_num_frames):
    desired = max(1, int(desired_num_frames))
    if desired == 1:
        return 1
    return int(((desired - 1) // _DECODE_ALIGNMENT_MODULUS) * _DECODE_ALIGNMENT_MODULUS + 1)


def _decode_length_units(num_frames):
    frames = int(num_frames)
    if frames <= 0:
        raise ValueError("decode length must be > 0")
    if (frames - 1) % int(_DECODE_ALIGNMENT_MODULUS) != 0:
        raise ValueError("decode length is not legal: {}".format(frames))
    return int((frames - 1) // int(_DECODE_ALIGNMENT_MODULUS))


def _allocate_decode_units(target_units, total_units):
    targets = [max(0.0, float(item)) for item in list(target_units or [])]
    units = [max(0, int(math.floor(item))) for item in targets]
    remain = int(total_units) - int(sum(units))

    while remain > 0:
        best_idx = 0
        best_delta = None
        for idx, target in enumerate(targets):
            cur = float(units[idx])
            delta = ((cur + 1.0) - target) ** 2 - (cur - target) ** 2
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        units[best_idx] += 1
        remain -= 1

    while remain < 0:
        best_idx = None
        best_delta = None
        for idx, target in enumerate(targets):
            cur = int(units[idx])
            if cur <= 0:
                continue
            delta = ((float(cur - 1) - target) ** 2) - ((float(cur) - target) ** 2)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None:
            raise ValueError("decode unit projection underflow: total_units={}".format(int(total_units)))
        units[best_idx] -= 1
        remain += 1

    return units, targets


def _build_raw_boundary_indices(raw_segments):
    rows = list(raw_segments or [])
    if not rows:
        return []
    boundaries = [int(_safe_int(rows[0].get("start_frame"), 0))]
    for item in rows:
        boundaries.append(int(_safe_int(item.get("end_frame"), boundaries[-1])))
    return boundaries


def build_aligned_segment_plan(state_segments_payload):
    payload = dict(state_segments_payload or {})
    raw_segments = list(payload.get("segments") or [])
    if not raw_segments:
        return {
            "version": 2,
            "schema": "aligned_segment_plan.v2",
            "stage": "encode",
            "substage": "scene_split",
            "source": "encode.scene_split.aligned_segments",
            "policy": {
                "projection_mode": "global_boundary_snap",
                "decode_length_rule": "length_is_1_mod_{}".format(int(_DECODE_ALIGNMENT_MODULUS)),
                "boundary_model": "shared_snapped_boundaries",
                "first_boundary": "fixed_to_raw_start",
                "last_boundary": "fixed_to_raw_end",
            },
            "raw_boundary_indices": [],
            "aligned_boundary_indices": [],
            "segments": [],
            "summary": {
                "segment_count": 0,
                "boundary_count": 0,
                "decode_valid_segment_count": 0,
                "export_ready_segment_count": 0,
                "shifted_segment_count": 0,
                "shared_boundary_count": 0,
                "mean_abs_boundary_shift": 0.0,
                "max_abs_boundary_shift": 0,
            },
        }

    prev_raw_end = None
    for idx, raw_item in enumerate(raw_segments):
        row = dict(raw_item or {})
        raw_start_idx = int(_safe_int(row.get("start_frame"), 0))
        raw_end_idx = int(_safe_int(row.get("end_frame"), raw_start_idx))
        if raw_end_idx < raw_start_idx:
            raise ValueError("state segment {} has invalid raw range".format(idx))
        if prev_raw_end is not None and raw_start_idx != int(prev_raw_end) + 1:
            raise ValueError(
                "state segments must stay contiguous for global aligned projection: prev_end={} current_start={}".format(
                    int(prev_raw_end),
                    int(raw_start_idx),
                )
            )
        prev_raw_end = int(raw_end_idx)

    raw_boundary_indices = _build_raw_boundary_indices(raw_segments)
    total_span = int(raw_boundary_indices[-1] - raw_boundary_indices[0])
    if total_span < 0:
        raise ValueError("aligned segment plan received negative raw span")
    if total_span % int(_DECODE_ALIGNMENT_MODULUS) != 0:
        raise ValueError(
            "global aligned projection impossible: raw span {} is not divisible by {}".format(
                int(total_span),
                int(_DECODE_ALIGNMENT_MODULUS),
            )
        )

    total_units = int(total_span // int(_DECODE_ALIGNMENT_MODULUS))
    target_units = []
    desired_num_frames_list = []
    for raw_item in raw_segments:
        row = dict(raw_item or {})
        raw_start_idx = int(_safe_int(row.get("start_frame"), 0))
        raw_end_idx = int(_safe_int(row.get("end_frame"), raw_start_idx))
        desired_num_frames = int(max(0, raw_end_idx - raw_start_idx + 1))
        desired_num_frames_list.append(desired_num_frames)
        target_units.append(max(0.0, float(desired_num_frames - 1) / float(_DECODE_ALIGNMENT_MODULUS)))

    projected_units, projected_targets = _allocate_decode_units(target_units, total_units)
    aligned_boundary_indices = [int(raw_boundary_indices[0])]
    for unit in projected_units:
        aligned_boundary_indices.append(
            int(aligned_boundary_indices[-1] + int(unit) * int(_DECODE_ALIGNMENT_MODULUS))
        )
    if int(aligned_boundary_indices[-1]) != int(raw_boundary_indices[-1]):
        raise ValueError(
            "global aligned projection internal error: aligned_end={} raw_end={}".format(
                int(aligned_boundary_indices[-1]),
                int(raw_boundary_indices[-1]),
            )
        )

    aligned_segments = []
    decode_valid_count = 0
    export_ready_count = 0
    shifted_segment_count = 0
    boundary_shifts = [int(aligned) - int(raw) for raw, aligned in zip(raw_boundary_indices, aligned_boundary_indices)]

    for idx, raw_item in enumerate(raw_segments):
        row = dict(raw_item or {})
        segment_id = _safe_int(row.get("segment_id"), idx)
        raw_start_idx = _safe_int(row.get("start_frame"), 0)
        raw_end_idx = _safe_int(row.get("end_frame"), raw_start_idx)
        desired_start_idx = int(raw_start_idx)
        desired_end_idx = int(raw_end_idx)
        desired_num_frames = int(max(0, desired_end_idx - desired_start_idx + 1))
        aligned_boundary_start_idx = int(aligned_boundary_indices[idx])
        aligned_boundary_end_idx = int(aligned_boundary_indices[idx + 1])
        aligned_start_idx = int(aligned_boundary_start_idx)
        aligned_end_idx = int(aligned_boundary_end_idx)
        aligned_num_frames = int(aligned_end_idx - aligned_start_idx + 1)
        left_shift = int(aligned_start_idx - desired_start_idx)
        right_shift = int(aligned_end_idx - desired_end_idx)
        align_reason = "already_decode_legal"
        if (
            int(left_shift) != 0
            or int(right_shift) != 0
            or int(aligned_num_frames) != int(desired_num_frames)
        ):
            align_reason = "global_boundary_snap_projection"
            shifted_segment_count += 1

        is_valid_for_decode = bool(
            aligned_num_frames > 0 and (int(aligned_num_frames) - 1) % int(_DECODE_ALIGNMENT_MODULUS) == 0
        )
        is_valid_for_export = bool(aligned_num_frames >= int(_EXPORT_TARGET_FRAMES))
        if is_valid_for_decode:
            decode_valid_count += 1
        if is_valid_for_export:
            export_ready_count += 1

        aligned_segments.append(
            {
                "segment_id": int(segment_id),
                "state_label": str(row.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "risk_level": str(row.get("risk_level", "") or ""),
                "raw_boundary_indices": [
                    int(raw_boundary_indices[idx]),
                    int(raw_boundary_indices[idx + 1]),
                ],
                "aligned_boundary_indices": [
                    int(aligned_boundary_indices[idx]),
                    int(aligned_boundary_indices[idx + 1]),
                ],
                "raw_start_idx": int(raw_start_idx),
                "raw_end_idx": int(raw_end_idx),
                "desired_start_idx": int(desired_start_idx),
                "desired_end_idx": int(desired_end_idx),
                "desired_num_frames": int(desired_num_frames),
                "aligned_start_idx": int(aligned_start_idx),
                "aligned_end_idx": int(aligned_end_idx),
                "aligned_num_frames": int(aligned_num_frames),
                "left_shift": int(left_shift),
                "right_shift": int(right_shift),
                "align_reason": str(align_reason),
                "is_valid_for_decode": bool(is_valid_for_decode),
                "is_valid_for_export": bool(is_valid_for_export),
                "state_score_peak": _safe_float(row.get("state_score_peak"), 0.0),
                "detector_score_peak": _safe_float(row.get("detector_score_peak"), 0.0),
            }
        )

    return {
        "version": 2,
        "schema": "aligned_segment_plan.v2",
        "stage": "encode",
        "substage": "scene_split",
        "source": "encode.scene_split.aligned_segments",
        "policy": {
            "projection_mode": "global_boundary_snap",
            "decode_length_rule": "length_is_1_mod_{}".format(int(_DECODE_ALIGNMENT_MODULUS)),
            "export_target_num_frames": int(_EXPORT_TARGET_FRAMES),
            "boundary_model": "shared_snapped_boundaries",
            "first_boundary": "fixed_to_raw_start",
            "last_boundary": "fixed_to_raw_end",
            "boundary_policy": "global_projection_nearest_legal_snap",
            "center_policy": "preserve_semantic_distribution_via_global_unit_allocation",
            "shift_semantics": "signed_offset_vs_raw_semantic_edge",
        },
        "raw_boundary_indices": [int(item) for item in raw_boundary_indices],
        "aligned_boundary_indices": [int(item) for item in aligned_boundary_indices],
        "segments": aligned_segments,
        "summary": {
            "segment_count": int(len(aligned_segments)),
            "boundary_count": int(len(aligned_boundary_indices)),
            "decode_valid_segment_count": int(decode_valid_count),
            "export_ready_segment_count": int(export_ready_count),
            "shifted_segment_count": int(shifted_segment_count),
            "shared_boundary_count": int(max(0, len(aligned_boundary_indices) - 2)),
            "mean_abs_boundary_shift": float(
                sum([abs(int(item)) for item in boundary_shifts]) / float(len(boundary_shifts))
            )
            if boundary_shifts
            else 0.0,
            "max_abs_boundary_shift": int(max([abs(int(item)) for item in boundary_shifts]) if boundary_shifts else 0),
            "projection_stats": {
                "solver": "greedy_global_l2_rounding",
                "total_span": int(total_span),
                "total_units": int(total_units),
                "target_units": [float(item) for item in projected_targets],
                "projected_units": [int(item) for item in projected_units],
                "desired_num_frames": [int(item) for item in desired_num_frames_list],
                "aligned_num_frames": [int(item.get("aligned_num_frames", 0) or 0) for item in aligned_segments],
                "boundary_shifts": [int(item) for item in boundary_shifts],
            },
        },
    }


def run_formal_mainline(args):
    from exphub.pipeline.segment.state.detector import run_state_mainline

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
    actual_mode, keyframe_bytes_sum = diagnostics.materialize_keyframes(
        frames_dir=paths.frames_dir,
        keyframes_dir=paths.keyframes_dir,
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
    aligned_segment_plan_payload = build_aligned_segment_plan(state_segments_payload)
    state_report_payload = dict(detector_result["state_report_payload"])
    quality_diagnostics = diagnostics.build_quality_diagnostics(
        paths=paths,
        state_segments_payload=state_segments_payload,
        state_report_payload=state_report_payload,
        extraction_meta=extraction_meta,
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

    manifest = diagnostics.build_segment_manifest(
        paths=paths,
        policy_name="state",
        keyframes_meta=keyframes_meta,
        state_segments_payload=state_segments_payload,
        aligned_segment_plan_payload=aligned_segment_plan_payload,
        state_report_payload=state_report_payload,
        quality_diagnostics=quality_diagnostics,
    )
    report = diagnostics.build_segment_report(
        paths=paths,
        inputs_meta=inputs_meta,
        keyframes_meta=keyframes_meta,
        state_segments_payload=state_segments_payload,
        aligned_segment_plan_payload=aligned_segment_plan_payload,
        state_report_payload=state_report_payload,
        quality_diagnostics=quality_diagnostics,
        timings=timings,
    )
    diagnostics.write_aligned_segment_plan(paths, aligned_segment_plan_payload)
    diagnostics.write_segment_manifest(paths, manifest)
    diagnostics.write_segment_report(paths, report)

    log_prog(
        "scene split summary: frames={} keyframes={} state_segments={} aligned_segments={}".format(
            int(extraction_meta.get("frame_count", 0) or 0),
            int(keyframes_meta.get("keyframe_count", 0) or 0),
            int((state_segments_payload.get("summary") or {}).get("segment_count", 0) or 0),
            int((aligned_segment_plan_payload.get("summary") or {}).get("segment_count", 0) or 0),
        )
    )
    log_info("scene split manifest: {}".format(paths.manifest_path))
    log_info("scene split aligned plan: {}".format(paths.aligned_plan_path))
    if visual_result.get("state_overview_path") is not None:
        log_info("scene split overview: {}".format(visual_result["state_overview_path"]))
    return paths.manifest_path


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
    segment_contract.require_formal_segment_policy(args.segment_policy)
    run_formal_mainline(args)


if __name__ == "__main__":
    main()
