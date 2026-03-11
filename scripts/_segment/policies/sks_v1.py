#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _common import log_info


def _scaled_gap(kf_gap, ratio, min_value=1):
    return max(int(min_value), int(round(float(kf_gap) * float(ratio))))


def _clip_int(value, low, high):
    value = int(value)
    low = int(low)
    high = int(high)
    if low > high:
        return int(low)
    return int(min(max(value, low), high))


def _mean(values):
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _max(values):
    if not values:
        return 0.0
    return float(max(float(v) for v in values))


def _load_semantic_rows(context):
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
        semantic_rows, semantic_meta = compute_semantic_rows(
            frame_paths,
            cache_dir,
            smooth_window=5,
            timestamps=timestamps,
        )
    except RuntimeError as e:
        if "open_clip" in str(e):
            raise RuntimeError(
                "sks_v1 failed to initialize OpenCLIP in the segment python environment. "
                "Configure environments.phases.segment.python with an interpreter that already has torch/open_clip, or keep --segment_policy uniform."
            )
        raise
    return semantic_rows, semantic_meta


def _row_map(rows):
    return {int(row.get("frame_idx", 0)): row for row in rows}


def _pick_action_candidate(action_values, target_value, low_idx, high_idx, base_idx):
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


def _snap_to_density_peak(density_values, anchor_idx, low_idx, high_idx, snap_radius, base_idx):
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


def _build_item(build_item, position_idx, final_idx, base_idx, density_value):
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
        source_type="semantic",
        source_role="sks_candidate",
        candidate_role="sks_candidate",
        rerank_score=float(density_value),
        semantic_relation="semantic_density_peak",
        is_inserted=False,
        is_relocated=True,
        replaced_uniform_index=int(base_idx),
        promotion_source="semantic_action",
        promotion_reason="fixed_budget_relocation",
        window_id=int(position_idx),
    )


def build_policy_plan(context):
    build_item = context["build_item"]
    base_indices = list(context["uniform_base_indices"])
    kf_gap = int(context["kf_gap"])
    used_last_idx = int(context["used_last_idx"])
    used_frame_count = int(context["frame_count_used"])

    rules = {
        "relocate_radius": int(_scaled_gap(kf_gap, 0.75)),
        "min_gap": int(max(2, _scaled_gap(kf_gap, 0.33))),
        "snap_radius": int(max(1, _scaled_gap(kf_gap, 0.20))),
        "density_eps": 0.03,
        "density_alpha": 0.7,
        "density_beta": 0.3,
        "smooth_window": 5,
    }

    if len(base_indices) <= 2:
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
                "policy_name": "sks_v1",
                "uniform_count": int(len(base_indices)),
                "num_uniform_base": int(len(base_indices)),
                "final_keyframe_count": int(len(base_indices)),
                "num_final_keyframes": int(len(base_indices)),
                "fixed_budget": True,
                "relocated_count": 0,
                "avg_abs_shift": 0.0,
                "max_abs_shift": 0,
                "semantic_velocity_mean": 0.0,
                "semantic_velocity_max": 0.0,
                "semantic_acceleration_mean": 0.0,
                "semantic_acceleration_max": 0.0,
                "num_boundary_selected": 0,
                "num_support_selected": 0,
                "num_boundary_relocated": 0,
                "num_boundary_inserted": 0,
                "num_support_inserted": 0,
                "num_promoted_support_inserted": 0,
                "num_burst_windows_triggered": 0,
                "extra_kf_ratio": 0.0,
            },
            "policy_meta": {
                "rules": rules,
                "semantic_enabled": False,
            },
        }

    semantic_rows, semantic_meta = _load_semantic_rows(context)
    used_rows = [row for row in semantic_rows if int(row.get("frame_idx", 0)) <= used_last_idx]
    if len(used_rows) != used_frame_count:
        used_rows = list(semantic_rows[:used_frame_count])
    semantic_map = _row_map(used_rows)

    displacement = [float(semantic_map[idx].get("semantic_displacement", 0.0)) for idx in range(used_frame_count)]
    velocity = [float(semantic_map[idx].get("semantic_velocity", 0.0)) for idx in range(used_frame_count)]
    acceleration = [float(semantic_map[idx].get("semantic_acceleration", 0.0)) for idx in range(used_frame_count)]
    density = [float(semantic_map[idx].get("semantic_density", 0.0)) for idx in range(used_frame_count)]
    action = [float(semantic_map[idx].get("semantic_action", 0.0)) for idx in range(used_frame_count)]
    action_end = float(action[-1]) if action else 0.0

    final_indices = [int(base_indices[0])]
    raw_candidates = [int(base_indices[0])]
    snapped_candidates = [int(base_indices[0])]
    relocation_records = []

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
            fallback_idx = _clip_int(base_idx, prev_idx + rules["min_gap"], min(order_high, used_last_idx))
            raw_idx = int(fallback_idx)
            final_idx = int(fallback_idx)
        else:
            target_value = float(pos) / float(len(base_indices) - 1) * float(action_end)
            raw_idx = _pick_action_candidate(action, target_value, low_idx, high_idx, base_idx)
            final_idx = _snap_to_density_peak(
                density,
                raw_idx,
                low_idx,
                high_idx,
                rules["snap_radius"],
                base_idx,
            )

        final_indices.append(int(final_idx))
        raw_candidates.append(int(raw_idx))
        snapped_candidates.append(int(final_idx))
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

    keyframe_items = []
    shift_values = []
    for pos, base_idx in enumerate(base_indices):
        final_idx = int(final_indices[pos])
        if pos > 0 and pos < len(base_indices) - 1:
            shift_values.append(abs(int(final_idx) - int(base_idx)))
        keyframe_items.append(
            _build_item(
                build_item,
                pos,
                final_idx,
                base_idx,
                density[final_idx] if final_idx < len(density) else 0.0,
            )
        )

    relocated_shifts = [shift for shift in shift_values if int(shift) > 0]
    log_info(
        "sks_v1 selected: uniform_base={} relocated={} final={}".format(
            len(base_indices),
            len(relocated_shifts),
            len(final_indices),
        )
    )

    return {
        "frame_count_used": int(context["frame_count_used"]),
        "tail_drop": int(context["tail_drop"]),
        "uniform_base_indices": list(base_indices),
        "keyframe_indices": list(final_indices),
        "keyframe_items": keyframe_items,
        "summary": {
            "policy_name": "sks_v1",
            "uniform_count": int(len(base_indices)),
            "num_uniform_base": int(len(base_indices)),
            "final_keyframe_count": int(len(final_indices)),
            "num_final_keyframes": int(len(final_indices)),
            "fixed_budget": True,
            "relocated_count": int(len(relocated_shifts)),
            "avg_abs_shift": float(_mean(relocated_shifts)),
            "max_abs_shift": int(max(relocated_shifts) if relocated_shifts else 0),
            "semantic_displacement_mean": float(_mean(displacement)),
            "semantic_displacement_max": float(_max(displacement)),
            "semantic_velocity_mean": float(_mean(velocity)),
            "semantic_velocity_max": float(_max(velocity)),
            "semantic_acceleration_mean": float(_mean(acceleration)),
            "semantic_acceleration_max": float(_max(acceleration)),
            "semantic_density_mean": float(_mean(density)),
            "semantic_density_max": float(_max(density)),
            "num_boundary_selected": 0,
            "num_support_selected": 0,
            "num_boundary_relocated": 0,
            "num_boundary_inserted": 0,
            "num_support_inserted": 0,
            "num_promoted_support_inserted": 0,
            "num_burst_windows_triggered": 0,
            "extra_kf_ratio": 0.0,
        },
        "policy_meta": {
            "rules": rules,
            "semantic_backend": str(semantic_meta.get("backend", "") or ""),
            "semantic_model_name": str(semantic_meta.get("model_name", "") or ""),
            "semantic_dt_sec": float(semantic_meta.get("dt_sec", 0.0) or 0.0),
            "semantic_dt_source": str(semantic_meta.get("dt_source", "") or ""),
            "signal_stats": dict(semantic_meta.get("signal_stats", {}) or {}),
            "uniform_base_indices": list(base_indices),
            "raw_candidates": list(raw_candidates),
            "snapped_candidates": list(snapped_candidates),
            "relocations": relocation_records,
        },
    }
