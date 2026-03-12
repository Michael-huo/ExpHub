#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _common import log_info

from _segment.policies.fixed_budget import (
    allocate_fixed_budget,
    build_fixed_budget_item,
    build_fixed_budget_rules,
    build_short_plan,
    build_signal_summary,
)
from _segment.policies.signal_builders import load_motion_signal_bundle


def build_policy_plan(context):
    build_item = context["build_item"]
    base_indices = list(context["uniform_base_indices"])
    rules = build_fixed_budget_rules(context["kf_gap"])

    if len(base_indices) <= 2:
        return build_short_plan(context, "motion", rules)

    signal_bundle = load_motion_signal_bundle(context, rules["smooth_window"])
    allocation = allocate_fixed_budget(
        base_indices=base_indices,
        used_last_idx=context["used_last_idx"],
        density_values=signal_bundle["density"],
        action_values=signal_bundle["action"],
        rules=rules,
    )

    keyframe_items = []
    for pos, base_idx in enumerate(base_indices):
        final_idx = int(allocation["final_indices"][pos])
        keyframe_items.append(
            build_fixed_budget_item(
                build_item=build_item,
                position_idx=pos,
                final_idx=final_idx,
                base_idx=base_idx,
                density_value=signal_bundle["density"][final_idx] if final_idx < len(signal_bundle["density"]) else 0.0,
                source_type="motion",
                source_role="motion_candidate",
            )
        )

    log_info(
        "motion selected: uniform_base={} relocated={} final={}".format(
            len(base_indices),
            len(allocation["relocated_shifts"]),
            len(allocation["final_indices"]),
        )
    )

    signal_meta = dict(signal_bundle["meta"])
    return {
        "frame_count_used": int(context["frame_count_used"]),
        "tail_drop": int(context["tail_drop"]),
        "uniform_base_indices": list(base_indices),
        "keyframe_indices": list(allocation["final_indices"]),
        "keyframe_items": keyframe_items,
        "summary": build_signal_summary(
            policy_name="motion",
            base_indices=base_indices,
            final_indices=allocation["final_indices"],
            relocated_shifts=allocation["relocated_shifts"],
            signal_prefix="motion",
            signal_bundle=signal_bundle,
        ),
        "policy_meta": {
            "rules": dict(rules),
            "motion_backend": str(signal_meta.get("backend", "") or ""),
            "motion_dt_sec": float(signal_meta.get("dt_sec", 0.0) or 0.0),
            "motion_dt_source": str(signal_meta.get("dt_source", "") or ""),
            "motion_preprocess": dict(signal_meta.get("preprocess", {}) or {}),
            "signal_stats": dict(signal_meta.get("signal_stats", {}) or {}),
            "uniform_base_indices": list(base_indices),
            "raw_candidates": list(allocation["raw_candidates"]),
            "snapped_candidates": list(allocation["snapped_candidates"]),
            "relocations": list(allocation["relocations"]),
        },
    }
