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


def load_semantic_signal_bundle(context, smooth_window):
    try:
        from _segment.research import compute_semantic_rows
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        if missing_name in ("torch", "open_clip"):
            raise RuntimeError(
                "sks_v1 requires torch/open_clip in the segment python environment. "
                "Configure environments.phases.segment.python with an interpreter that already has these dependencies, or keep --segment_policy uniform."
            )
        raise

    frame_paths = context["frame_paths"]
    timestamps = context["timestamps"]
    cache_dir = context["policy_cache_dir"]
    log_info("sks_v1 semantic observe start: frames={}".format(len(frame_paths)))
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
                "sks_v1 failed to initialize OpenCLIP in the segment python environment. "
                "Configure environments.phases.segment.python with an interpreter that already has torch/open_clip, or keep --segment_policy uniform."
            )
        raise

    used_rows = _used_rows(rows, context["used_last_idx"], context["frame_count_used"])
    return _build_bundle(used_rows, meta, "semantic")


def load_motion_signal_bundle(context, smooth_window):
    from _segment.research import compute_motion_rows

    rows, meta = compute_motion_rows(
        context["frame_paths"],
        smooth_window=int(smooth_window),
        timestamps=context["timestamps"],
    )
    used_rows = _used_rows(rows, context["used_last_idx"], context["frame_count_used"])
    return _build_bundle(used_rows, meta, "motion")
