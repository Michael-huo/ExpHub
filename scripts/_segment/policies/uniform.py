#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def build_uniform_keyframe_plan(frame_count_total, kf_gap):
    """Return the legacy uniform keyframe sampling plan."""
    frame_count_total = int(frame_count_total)
    kf_gap = int(kf_gap)
    if kf_gap <= 0:
        raise ValueError("kf_gap must be > 0")

    used_last_idx = ((frame_count_total - 1) // kf_gap) * kf_gap
    used_count = used_last_idx + 1
    tail_drop = int(frame_count_total - used_count)

    indices = list(range(0, used_count, kf_gap))
    if len(indices) == 0:
        indices = [0]

    return {
        "frame_count_total": int(frame_count_total),
        "frame_count_used": int(used_count),
        "tail_drop": int(tail_drop),
        "keyframe_indices": indices,
        "keyframe_count": int(len(indices)),
    }
