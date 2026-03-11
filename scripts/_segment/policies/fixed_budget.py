#!/usr/bin/env python3
# -*- coding: utf-8 -*-


DEFAULT_DENSITY_EPS = 0.03
DEFAULT_DENSITY_ALPHA = 0.7
DEFAULT_DENSITY_BETA = 0.3
DEFAULT_SMOOTH_WINDOW = 5


def scaled_gap(kf_gap, ratio, min_value=1):
    return max(int(min_value), int(round(float(kf_gap) * float(ratio))))


def clip_int(value, low, high):
    value = int(value)
    low = int(low)
    high = int(high)
    if low > high:
        return int(low)
    return int(min(max(value, low), high))


def mean_value(values):
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def max_value(values):
    if not values:
        return 0.0
    return float(max(float(v) for v in values))


def build_fixed_budget_rules(kf_gap):
    return {
        "relocate_radius": int(scaled_gap(kf_gap, 0.75)),
        "min_gap": int(max(2, scaled_gap(kf_gap, 0.33))),
        "snap_radius": int(max(1, scaled_gap(kf_gap, 0.20))),
        "density_eps": float(DEFAULT_DENSITY_EPS),
        "density_alpha": float(DEFAULT_DENSITY_ALPHA),
        "density_beta": float(DEFAULT_DENSITY_BETA),
        "smooth_window": int(DEFAULT_SMOOTH_WINDOW),
    }


def pick_action_candidate(action_values, target_value, low_idx, high_idx, base_idx):
    low_idx = int(low_idx)
    high_idx = int(high_idx)
    base_idx = int(base_idx)
    if low_idx > high_idx:
        return int(low_idx)

    for frame_idx in range(low_idx, high_idx + 1):
        if float(action_values[frame_idx]) >= float(target_value):
            return int(frame_idx)

    best_idx = low_idx
    best_key = None
    for frame_idx in range(low_idx, high_idx + 1):
        key = (
            abs(float(action_values[frame_idx]) - float(target_value)),
            abs(int(frame_idx) - int(base_idx)),
            int(frame_idx),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(frame_idx)
    return int(best_idx)


def snap_to_density_peak(density_values, anchor_idx, low_idx, high_idx, snap_radius, base_idx):
    low_idx = int(low_idx)
    high_idx = int(high_idx)
    anchor_idx = int(anchor_idx)
    base_idx = int(base_idx)
    if low_idx > high_idx:
        return int(anchor_idx)

    snap_low = max(low_idx, int(anchor_idx) - int(snap_radius))
    snap_high = min(high_idx, int(anchor_idx) + int(snap_radius))
    if snap_low > snap_high:
        return int(anchor_idx)

    best_idx = int(anchor_idx)
    best_key = None
    for frame_idx in range(snap_low, snap_high + 1):
        key = (
            -float(density_values[frame_idx]),
            abs(int(frame_idx) - int(anchor_idx)),
            abs(int(frame_idx) - int(base_idx)),
            int(frame_idx),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(frame_idx)
    return int(best_idx)


def allocate_fixed_budget(base_indices, used_last_idx, density_values, action_values, rules):
    final_indices = [int(base_indices[0])]
    raw_candidates = [int(base_indices[0])]
    snapped_candidates = [int(base_indices[0])]
    relocation_records = []
    shift_values = []

    action_end = float(action_values[-1]) if action_values else 0.0
    for pos in range(1, len(base_indices) - 1):
        base_idx = int(base_indices[pos])
        prev_idx = int(final_indices[-1])
        remaining_slots = int((len(base_indices) - 1) - pos)
        order_high = int(base_indices[-1] - remaining_slots * rules["min_gap"])
        window_low = int(base_idx - rules["relocate_radius"])
        window_high = int(base_idx + rules["relocate_radius"])
        low_idx = max(prev_idx + rules["min_gap"], 0, window_low)
        high_idx = min(order_high, used_last_idx, window_high)

        if low_idx > high_idx:
            fallback_idx = clip_int(base_idx, prev_idx + rules["min_gap"], min(order_high, used_last_idx))
            raw_idx = int(fallback_idx)
            final_idx = int(fallback_idx)
        else:
            target_value = float(pos) / float(len(base_indices) - 1) * float(action_end)
            raw_idx = pick_action_candidate(action_values, target_value, low_idx, high_idx, base_idx)
            final_idx = snap_to_density_peak(
                density_values,
                raw_idx,
                low_idx,
                high_idx,
                rules["snap_radius"],
                base_idx,
            )

        final_indices.append(int(final_idx))
        raw_candidates.append(int(raw_idx))
        snapped_candidates.append(int(final_idx))
        shift_values.append(abs(int(final_idx) - int(base_idx)))
        relocation_records.append(
            {
                "position_idx": int(pos),
                "base_idx": int(base_idx),
                "raw_candidate_idx": int(raw_idx),
                "final_idx": int(final_idx),
                "abs_shift": int(abs(int(final_idx) - int(base_idx))),
                "window_low": int(window_low),
                "window_high": int(window_high),
            }
        )

    final_indices.append(int(base_indices[-1]))
    raw_candidates.append(int(base_indices[-1]))
    snapped_candidates.append(int(base_indices[-1]))

    relocated_shifts = [shift for shift in shift_values if int(shift) > 0]
    return {
        "final_indices": list(final_indices),
        "raw_candidates": list(raw_candidates),
        "snapped_candidates": list(snapped_candidates),
        "relocations": relocation_records,
        "shift_values": list(shift_values),
        "relocated_shifts": list(relocated_shifts),
    }


def build_fixed_budget_item(build_item, position_idx, final_idx, base_idx, density_value, source_type, source_role):
    was_relocated = bool(int(final_idx) != int(base_idx))
    if not was_relocated:
        return build_item(
            final_idx,
            source_type="uniform",
            source_role="uniform",
            candidate_role="uniform",
            promotion_source="uniform",
        )
    return build_item(
        final_idx,
        source_type=str(source_type),
        source_role=str(source_role),
        candidate_role=str(source_role),
        rerank_score=float(density_value),
        semantic_relation="{}_density_peak".format(str(source_type)),
        is_inserted=False,
        is_relocated=True,
        replaced_uniform_index=int(base_idx),
        promotion_source="{}_action".format(str(source_type)),
        promotion_reason="fixed_budget_relocation",
        window_id=int(position_idx),
    )


def build_short_plan(context, policy_name, rules):
    build_item = context["build_item"]
    base_indices = list(context["uniform_base_indices"])
    items = []
    for frame_idx in base_indices:
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
        "uniform_base_indices": list(base_indices),
        "keyframe_indices": list(base_indices),
        "keyframe_items": items,
        "summary": {
            "policy_name": str(policy_name),
            "uniform_count": int(len(base_indices)),
            "num_uniform_base": int(len(base_indices)),
            "final_keyframe_count": int(len(base_indices)),
            "num_final_keyframes": int(len(base_indices)),
            "fixed_budget": True,
            "relocated_count": 0,
            "avg_abs_shift": 0.0,
            "max_abs_shift": 0,
            "extra_kf_ratio": 0.0,
            "num_boundary_selected": 0,
            "num_support_selected": 0,
            "num_boundary_relocated": 0,
            "num_boundary_inserted": 0,
            "num_support_inserted": 0,
            "num_promoted_support_inserted": 0,
            "num_burst_windows_triggered": 0,
        },
        "policy_meta": {
            "rules": dict(rules),
            "signal_enabled": False,
        },
    }


def build_signal_summary(policy_name, base_indices, final_indices, relocated_shifts, signal_prefix, signal_bundle):
    summary = {
        "policy_name": str(policy_name),
        "uniform_count": int(len(base_indices)),
        "num_uniform_base": int(len(base_indices)),
        "final_keyframe_count": int(len(final_indices)),
        "num_final_keyframes": int(len(final_indices)),
        "fixed_budget": True,
        "relocated_count": int(len(relocated_shifts)),
        "avg_abs_shift": float(mean_value(relocated_shifts)),
        "max_abs_shift": int(max(relocated_shifts) if relocated_shifts else 0),
        "extra_kf_ratio": 0.0,
        "num_boundary_selected": 0,
        "num_support_selected": 0,
        "num_boundary_relocated": 0,
        "num_boundary_inserted": 0,
        "num_support_inserted": 0,
        "num_promoted_support_inserted": 0,
        "num_burst_windows_triggered": 0,
    }
    summary["{}_displacement_mean".format(signal_prefix)] = float(mean_value(signal_bundle["displacement"]))
    summary["{}_displacement_max".format(signal_prefix)] = float(max_value(signal_bundle["displacement"]))
    summary["{}_velocity_mean".format(signal_prefix)] = float(mean_value(signal_bundle["velocity"]))
    summary["{}_velocity_max".format(signal_prefix)] = float(max_value(signal_bundle["velocity"]))
    summary["{}_acceleration_mean".format(signal_prefix)] = float(mean_value(signal_bundle["acceleration"]))
    summary["{}_acceleration_max".format(signal_prefix)] = float(max_value(signal_bundle["acceleration"]))
    summary["{}_density_mean".format(signal_prefix)] = float(mean_value(signal_bundle["density"]))
    summary["{}_density_max".format(signal_prefix)] = float(max_value(signal_bundle["density"]))
    summary["{}_action_total".format(signal_prefix)] = float(signal_bundle["action"][-1]) if signal_bundle["action"] else 0.0
    return summary
