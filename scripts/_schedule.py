#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pathlib import Path


WAN_R4_BACKEND = "wan_r4"
WAN_R4_GAP_UNIT = 4


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _read_json(path):
    with Path(path).resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def load_deploy_schedule(schedule_path):
    path = Path(schedule_path).resolve()
    if not path.is_file():
        return None
    obj = _read_json(path)
    return obj if isinstance(obj, dict) else None


def load_prompt_manifest(path):
    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise ValueError("prompt manifest must be a JSON object: {}".format(path))
    return obj


def _validate_keyframes(raw_keyframes):
    if len(raw_keyframes) < 2:
        raise ValueError("raw keyframe schedule must contain at least 2 indices")
    last = None
    for idx in raw_keyframes:
        cur = int(idx)
        if cur < 0:
            raise ValueError("raw keyframe index must be >= 0: {}".format(cur))
        if last is not None and cur <= last:
            raise ValueError("raw keyframe schedule must be strictly increasing: {}".format(raw_keyframes))
        last = cur


def _allocate_wan_r4_units(raw_gaps, total_units):
    segment_count = len(raw_gaps)
    if total_units < segment_count:
        raise ValueError(
            "wan_r4 projection impossible: total_units={} < segment_count={} (need at least 4 frames per segment)".format(
                int(total_units), int(segment_count)
            )
        )

    targets = [float(gap) / float(WAN_R4_GAP_UNIT) for gap in raw_gaps]
    units = [1 for _ in raw_gaps]
    remain = int(total_units - segment_count)

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

    return units, targets


def build_wan_r4_deploy_schedule(keyframes_meta):
    if not isinstance(keyframes_meta, dict):
        raise ValueError("keyframes_meta must be a JSON object")

    raw_keyframes = [int(v) for v in list(keyframes_meta.get("keyframe_indices") or [])]
    _validate_keyframes(raw_keyframes)

    raw_start = int(raw_keyframes[0])
    raw_end = int(raw_keyframes[-1])
    total_span = int(raw_end - raw_start)
    if total_span <= 0:
        raise ValueError("raw keyframe schedule span must be > 0")
    if total_span % WAN_R4_GAP_UNIT != 0:
        raise ValueError(
            "wan_r4 projection impossible: total span {} is not divisible by {}".format(
                int(total_span), int(WAN_R4_GAP_UNIT)
            )
        )

    raw_gaps = []
    for idx in range(len(raw_keyframes) - 1):
        gap = int(raw_keyframes[idx + 1] - raw_keyframes[idx])
        if gap <= 0:
            raise ValueError("raw gaps must be > 0")
        raw_gaps.append(gap)

    total_units = int(total_span // WAN_R4_GAP_UNIT)
    deploy_units, targets = _allocate_wan_r4_units(raw_gaps, total_units)
    deploy_gaps = [int(unit * WAN_R4_GAP_UNIT) for unit in deploy_units]

    deploy_keyframes = [int(raw_start)]
    for gap in deploy_gaps:
        deploy_keyframes.append(int(deploy_keyframes[-1] + gap))

    if deploy_keyframes[-1] != raw_end:
        raise ValueError(
            "wan_r4 projection internal error: projected end {} != raw end {}".format(
                int(deploy_keyframes[-1]), int(raw_end)
            )
        )

    segments = []
    boundary_shifts = []
    gap_errors = []
    for seg_idx in range(len(raw_gaps)):
        raw_seg_start = int(raw_keyframes[seg_idx])
        raw_seg_end = int(raw_keyframes[seg_idx + 1])
        deploy_seg_start = int(deploy_keyframes[seg_idx])
        deploy_seg_end = int(deploy_keyframes[seg_idx + 1])
        raw_gap = int(raw_seg_end - raw_seg_start)
        deploy_gap = int(deploy_seg_end - deploy_seg_start)
        gap_error = int(deploy_gap - raw_gap)
        boundary_shift = int(deploy_seg_end - raw_seg_end)
        gap_errors.append(abs(gap_error))
        boundary_shifts.append(abs(boundary_shift))
        segments.append(
            {
                "segment_id": int(seg_idx),
                "raw_start_idx": int(raw_seg_start),
                "raw_end_idx": int(raw_seg_end),
                "deploy_start_idx": int(deploy_seg_start),
                "deploy_end_idx": int(deploy_seg_end),
                "raw_gap": int(raw_gap),
                "deploy_gap": int(deploy_gap),
                "num_frames": int(deploy_gap + 1),
                "boundary_shift": int(boundary_shift),
                "gap_error": int(gap_error),
                "raw_gap_units": float(targets[seg_idx]),
                "deploy_gap_units": int(deploy_units[seg_idx]),
            }
        )

    all_boundary_shifts = [abs(int(deploy_keyframes[idx] - raw_keyframes[idx])) for idx in range(len(raw_keyframes))]

    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": WAN_R4_BACKEND,
        "source_policy": str(keyframes_meta.get("policy_name", "") or "uniform"),
        "raw_keyframe_indices": list(raw_keyframes),
        "deploy_keyframe_indices": list(deploy_keyframes),
        "rules": {
            "first_boundary": "fixed_to_raw",
            "last_boundary": "fixed_to_raw",
            "segment_count": "preserved",
            "deploy_gap_multiple": int(WAN_R4_GAP_UNIT),
            "total_span": "preserved",
            "min_gap_units": 1,
        },
        "projection_stats": {
            "solver": "greedy_marginal_l2_rounding",
            "segment_count": int(len(raw_gaps)),
            "total_span": int(total_span),
            "total_gap_units": int(total_units),
            "mean_abs_boundary_shift": float(_mean(all_boundary_shifts)),
            "max_abs_boundary_shift": int(max(all_boundary_shifts) if all_boundary_shifts else 0),
            "mean_abs_gap_error": float(_mean(gap_errors)),
            "max_abs_gap_error": int(max(gap_errors) if gap_errors else 0),
        },
        "segments": segments,
    }


def build_execution_segments_from_deploy_schedule(schedule_obj):
    if not isinstance(schedule_obj, dict):
        raise ValueError("deploy schedule must be a JSON object")

    raw_keyframes = list(schedule_obj.get("raw_keyframe_indices") or [])
    deploy_keyframes = list(schedule_obj.get("deploy_keyframe_indices") or [])
    segments = list(schedule_obj.get("segments") or [])
    if len(raw_keyframes) < 2 or len(deploy_keyframes) < 2 or not segments:
        raise ValueError("deploy schedule is missing required keyframe data")

    out = []
    for idx, item in enumerate(segments):
        out.append(
            {
                "seg": int(item.get("segment_id", idx)),
                "segment_id": int(item.get("segment_id", idx)),
                "schedule_source": "deploy_schedule",
                "execution_backend": str(schedule_obj.get("backend", WAN_R4_BACKEND)),
                "raw_start_idx": int(item.get("raw_start_idx")),
                "raw_end_idx": int(item.get("raw_end_idx")),
                "deploy_start_idx": int(item.get("deploy_start_idx")),
                "deploy_end_idx": int(item.get("deploy_end_idx")),
                "start_idx": int(item.get("deploy_start_idx")),
                "end_idx": int(item.get("deploy_end_idx")),
                "raw_gap": int(item.get("raw_gap")),
                "deploy_gap": int(item.get("deploy_gap")),
                "num_frames": int(item.get("num_frames")),
                "boundary_shift": int(item.get("boundary_shift", 0)),
                "gap_error": int(item.get("gap_error", 0)),
            }
        )
    return out


def build_legacy_execution_segments(frames_avail, base_idx, kf_gap, num_segments):
    frames_avail_i = int(frames_avail)
    base_idx_i = int(base_idx)
    kf_gap_i = int(kf_gap)
    requested = int(num_segments)

    if frames_avail_i <= 0:
        raise ValueError("frames_avail must be > 0")
    if kf_gap_i <= 0:
        raise ValueError("kf_gap must be > 0 for legacy execution segments")
    if base_idx_i < 0 or base_idx_i >= frames_avail_i:
        raise ValueError("base_idx out of range: {}".format(base_idx_i))

    max_segments = (frames_avail_i - 1 - base_idx_i) // kf_gap_i
    if max_segments <= 0:
        raise ValueError(
            "not enough frames for even 1 legacy segment: frames_avail={} base_idx={} kf_gap={}".format(
                int(frames_avail_i), int(base_idx_i), int(kf_gap_i)
            )
        )

    if requested > 0:
        max_segments = min(max_segments, requested)

    out = []
    for seg in range(max_segments):
        start_idx = int(base_idx_i + seg * kf_gap_i)
        end_idx = int(start_idx + kf_gap_i)
        out.append(
            {
                "seg": int(seg),
                "segment_id": int(seg),
                "schedule_source": "legacy_kf_gap",
                "execution_backend": "legacy_uniform",
                "raw_start_idx": int(start_idx),
                "raw_end_idx": int(end_idx),
                "deploy_start_idx": int(start_idx),
                "deploy_end_idx": int(end_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_gap": int(kf_gap_i),
                "deploy_gap": int(kf_gap_i),
                "num_frames": int(kf_gap_i + 1),
                "boundary_shift": 0,
                "gap_error": 0,
            }
        )
    return out


def extract_execution_segments_from_manifest(manifest_obj):
    if not isinstance(manifest_obj, dict):
        return []

    raw_items = manifest_obj.get("segments")
    if not isinstance(raw_items, list):
        return []

    out = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        if "start_idx" not in item or "end_idx" not in item:
            continue
        start_idx = _safe_int(item.get("start_idx"), None)
        end_idx = _safe_int(item.get("end_idx"), None)
        num_frames = _safe_int(item.get("num_frames"), None)
        deploy_gap = _safe_int(item.get("deploy_gap"), None)
        if start_idx is None or end_idx is None:
            continue
        if num_frames is None:
            num_frames = int(end_idx - start_idx + 1)
        if deploy_gap is None:
            deploy_gap = int(end_idx - start_idx)
        out.append(
            {
                "seg": _safe_int(item.get("seg"), idx),
                "segment_id": _safe_int(item.get("segment_id"), idx),
                "schedule_source": str(item.get("schedule_source", "")),
                "execution_backend": str(item.get("execution_backend", "")),
                "raw_start_idx": _safe_int(item.get("raw_start_idx"), start_idx),
                "raw_end_idx": _safe_int(item.get("raw_end_idx"), end_idx),
                "deploy_start_idx": _safe_int(item.get("deploy_start_idx"), start_idx),
                "deploy_end_idx": _safe_int(item.get("deploy_end_idx"), end_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_gap": _safe_int(item.get("raw_gap"), end_idx - start_idx),
                "deploy_gap": int(deploy_gap),
                "num_frames": int(num_frames),
                "boundary_shift": _safe_int(item.get("boundary_shift"), 0),
                "gap_error": _safe_int(item.get("gap_error"), 0),
            }
        )

    out.sort(key=lambda item: (int(item.get("seg", 0)), int(item.get("start_idx", 0))))
    return out
