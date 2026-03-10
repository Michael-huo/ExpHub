#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def compute_uniform_base(frame_count_total, kf_gap):
    frame_count_total = int(frame_count_total)
    kf_gap = max(1, int(kf_gap))
    used_last_idx = ((frame_count_total - 1) // kf_gap) * kf_gap
    used_count = int(used_last_idx + 1)
    tail_drop = int(frame_count_total - used_count)
    indices = list(range(0, used_count, kf_gap))
    if not indices:
        indices = [0]
    return {
        "indices": indices,
        "used_last_idx": int(used_last_idx),
        "used_count": int(used_count),
        "tail_drop": int(tail_drop),
    }


def build_policy_plan(context):
    build_item = context["build_item"]
    indices = list(context["uniform_base_indices"])
    items = []
    for frame_idx in indices:
        items.append(
            build_item(
                frame_idx,
                source_type="uniform",
                source_role="uniform",
                candidate_role="uniform",
                promotion_source="uniform",
            )
        )

    return {
        "frame_count_used": int(context["frame_count_used"]),
        "tail_drop": int(context["tail_drop"]),
        "uniform_base_indices": list(indices),
        "keyframe_indices": list(indices),
        "keyframe_items": items,
        "summary": {
            "policy_name": "uniform",
            "num_uniform_base": int(len(indices)),
            "num_boundary_selected": 0,
            "num_support_selected": 0,
            "num_boundary_relocated": 0,
            "num_boundary_inserted": 0,
            "num_support_inserted": 0,
            "num_promoted_support_inserted": 0,
            "num_burst_windows_triggered": 0,
            "num_final_keyframes": int(len(indices)),
            "extra_kf_ratio": 0.0,
        },
        "policy_meta": {
            "rules": {
                "description": "legacy uniform anchors sampled every kf_gap frames",
            },
        },
    }
