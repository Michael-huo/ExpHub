#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _common import log_info


def _row_map(rows):
    return {int(row.get("frame_idx", 0)): row for row in rows}


def _used_rows(rows, used_last_idx, used_frame_count):
    filtered = [row for row in rows if int(row.get("frame_idx", 0)) <= int(used_last_idx)]
    if len(filtered) != int(used_frame_count):
        filtered = list(rows[:used_frame_count])
    return filtered


def _series(signal_map, used_frame_count, key):
    values = []
    for idx in range(int(used_frame_count)):
        values.append(float(signal_map[idx].get(key, 0.0) or 0.0))
    return values


def _build_bundle(used_rows, meta, prefix):
    signal_map = _row_map(used_rows)
    frame_count = len(used_rows)
    return {
        "prefix": str(prefix),
        "rows": list(used_rows),
        "meta": dict(meta),
        "displacement": _series(signal_map, frame_count, "{}_displacement".format(prefix)),
        "velocity": _series(signal_map, frame_count, "{}_velocity".format(prefix)),
        "velocity_smooth": _series(signal_map, frame_count, "{}_velocity_smooth".format(prefix)),
        "acceleration": _series(signal_map, frame_count, "{}_acceleration".format(prefix)),
        "acceleration_smooth": _series(signal_map, frame_count, "{}_acceleration_smooth".format(prefix)),
        "density": _series(signal_map, frame_count, "{}_density".format(prefix)),
        "action": _series(signal_map, frame_count, "{}_action".format(prefix)),
    }


def _resolve_used_bounds(context, used_last_idx=None, used_frame_count=None):
    if used_last_idx is None:
        used_last_idx = int(context["used_last_idx"])
    else:
        used_last_idx = int(used_last_idx)
    if used_frame_count is None:
        used_frame_count = int(context["frame_count_used"])
    else:
        used_frame_count = int(used_frame_count)
    return int(used_last_idx), int(used_frame_count)


def load_semantic_signal_bundle(context, smooth_window, used_last_idx=None, used_frame_count=None):
    try:
        from _segment.research import compute_semantic_rows
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        if missing_name in ("torch", "open_clip"):
            raise RuntimeError(
                "semantic policy requires torch/open_clip in the segment python environment. "
                "Configure environments.phases.segment.python with an interpreter that already has these dependencies, or keep --segment_policy uniform."
            )
        raise

    frame_paths = context["frame_paths"]
    timestamps = context["timestamps"]
    cache_dir = context["policy_cache_dir"]
    used_last_idx, used_frame_count = _resolve_used_bounds(context, used_last_idx, used_frame_count)
    log_info("semantic observe start: frames={}".format(len(frame_paths)))
    try:
        rows, meta = compute_semantic_rows(
            frame_paths,
            cache_dir,
            smooth_window=int(smooth_window),
            timestamps=timestamps,
        )
    except RuntimeError as e:
        if "open_clip" in str(e):
            raise RuntimeError(
                "semantic policy failed to initialize OpenCLIP in the segment python environment. "
                "Configure environments.phases.segment.python with an interpreter that already has torch/open_clip, or keep --segment_policy uniform."
            )
        raise

    used_rows = _used_rows(rows, used_last_idx, used_frame_count)
    return _build_bundle(used_rows, meta, "semantic")


def load_motion_signal_bundle(context, smooth_window, used_last_idx=None, used_frame_count=None):
    from _segment.research import compute_motion_rows

    used_last_idx, used_frame_count = _resolve_used_bounds(context, used_last_idx, used_frame_count)
    rows, meta = compute_motion_rows(
        context["frame_paths"],
        smooth_window=int(smooth_window),
        timestamps=context["timestamps"],
    )
    used_rows = _used_rows(rows, used_last_idx, used_frame_count)
    return _build_bundle(used_rows, meta, "motion")


def load_risk_signal_bundle(context, smooth_window, used_last_idx=None, used_frame_count=None):
    from _segment.research import compute_frame_signal_rows

    used_last_idx, used_frame_count = _resolve_used_bounds(context, used_last_idx, used_frame_count)
    semantic_bundle = load_semantic_signal_bundle(
        context,
        smooth_window,
        used_last_idx=used_last_idx,
        used_frame_count=used_frame_count,
    )
    motion_bundle = load_motion_signal_bundle(
        context,
        smooth_window,
        used_last_idx=used_last_idx,
        used_frame_count=used_frame_count,
    )
    rows, frame_signal_meta = compute_frame_signal_rows(
        context["frame_paths"],
        context["timestamps"],
        semantic_rows=semantic_bundle["rows"],
        motion_rows=motion_bundle["rows"],
    )
    used_rows = _used_rows(rows, used_last_idx, used_frame_count)
    return {
        "rows": list(used_rows),
        "meta": {
            "frame_signals": dict(frame_signal_meta),
            "semantic": dict(semantic_bundle.get("meta", {}) or {}),
            "motion": dict(motion_bundle.get("meta", {}) or {}),
        },
        "semantic": semantic_bundle,
        "motion": motion_bundle,
    }
