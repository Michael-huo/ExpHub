#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from _common import log_info


def _scaled_gap(kf_gap, ratio, min_value=1):
    return max(int(min_value), int(round(float(kf_gap) * float(ratio))))


def _mark_uniform_keyframes(rows, uniform_indices):
    keyframe_set = set(int(x) for x in uniform_indices)
    for row in rows:
        row["is_uniform_keyframe"] = bool(int(row.get("frame_idx", 0)) in keyframe_set)


def _compute_candidate_points(context):
    try:
        from _segment.research import (
            DEFAULT_PEAK_CONFIG,
            DEFAULT_SCORE_WEIGHTS,
            annotate_peaks,
            apply_scores,
            build_candidate_points,
            compute_frame_signal_rows,
            compute_semantic_rows,
        )
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        if missing_name in ("torch", "open_clip"):
            raise RuntimeError(
                "semantic_guarded_v1 requires torch/open_clip in the segment python environment. "
                "Use --sys_py with an interpreter that already has these dependencies, or keep --segment_policy uniform."
            )
        raise

    cache_dir = context["policy_cache_dir"]
    frame_paths = context["frame_paths"]
    timestamps = context["timestamps"]

    log_info("semantic_guarded_v1 semantic observe start: frames={}".format(len(frame_paths)))
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
                "semantic_guarded_v1 failed to initialize OpenCLIP in the segment python environment. "
                "Use --sys_py with an interpreter that already has torch/open_clip, or keep --segment_policy uniform."
            )
        raise
    rows, signal_meta = compute_frame_signal_rows(frame_paths, timestamps, semantic_rows=semantic_rows)
    _mark_uniform_keyframes(rows, context["uniform_base_indices"])
    rows, score_meta = apply_scores(
        rows,
        weights=dict(DEFAULT_SCORE_WEIGHTS),
        smooth_window=5,
        use_blur_in_score=False,
        use_semantic_in_score=False,
    )
    peak_cfg = dict(DEFAULT_PEAK_CONFIG)
    rows, peak_meta = annotate_peaks(
        rows,
        window_radius=int(peak_cfg["window_radius"]),
        threshold_std=float(peak_cfg["threshold_std"]),
        min_peak_distance=int(peak_cfg["min_peak_distance"]),
        min_peak_score_raw=float(peak_cfg["min_peak_score_raw"]),
        min_peak_prominence=float(peak_cfg["min_peak_prominence"]),
        edge_margin=int(peak_cfg["edge_margin"]),
    )
    candidate_points = build_candidate_points(rows, peak_meta)
    return {
        "rows": rows,
        "semantic_meta": semantic_meta,
        "signal_meta": signal_meta,
        "score_meta": score_meta,
        "peak_meta": peak_meta,
        "candidate_points": candidate_points,
    }


def _filter_candidates(candidate_points, role_name, used_last_idx):
    summary = candidate_points.get("candidate_roles_summary", {})
    items = list(summary.get(role_name, []))
    out = []
    for item in items:
        frame_idx = int(item.get("frame_idx", -1))
        if frame_idx < 0 or frame_idx > int(used_last_idx):
            continue
        out.append(dict(item))
    out.sort(key=lambda item: (-float(item.get("rerank_score", 0.0)), int(item.get("frame_idx", 0))))
    return out


def _can_place(frame_idx, current_indices, min_distance, skip_idx=None):
    frame_idx = int(frame_idx)
    skip_idx = int(skip_idx) if skip_idx is not None else None
    for existing_idx in current_indices:
        existing_idx = int(existing_idx)
        if skip_idx is not None and existing_idx == skip_idx:
            continue
        if abs(existing_idx - frame_idx) < int(min_distance):
            return False
    return True


def _nearest_gap(frame_idx, current_indices):
    if not current_indices:
        return None
    return min(abs(int(frame_idx) - int(existing_idx)) for existing_idx in current_indices)


def _select_merged_boundary_candidates(boundary_candidates, merge_radius):
    kept = []
    for item in boundary_candidates:
        frame_idx = int(item.get("frame_idx", 0))
        too_close = False
        for kept_item in kept:
            if abs(frame_idx - int(kept_item.get("frame_idx", 0))) <= int(merge_radius):
                too_close = True
                break
        if too_close:
            continue
        kept.append(dict(item))
    return kept


def _nearest_uniform_target(frame_idx, item_map, attach_radius):
    best = None
    best_dist = None
    for existing_idx in sorted(item_map.keys()):
        item = item_map[int(existing_idx)]
        if str(item.get("source_type")) != "uniform":
            continue
        dist = abs(int(frame_idx) - int(existing_idx))
        if dist > int(attach_radius):
            continue
        if best is None or dist < best_dist:
            best = int(existing_idx)
            best_dist = int(dist)
    return best


def build_policy_plan(context):
    build_item = context["build_item"]
    base_indices = list(context["uniform_base_indices"])
    kf_gap = int(context["kf_gap"])
    rules = {
        "boundary_attach_radius": int(_scaled_gap(kf_gap, 0.4)),
        "boundary_merge_radius": int(_scaled_gap(kf_gap, 0.25)),
        "support_trigger_gap": int(_scaled_gap(kf_gap, 0.75)),
        "support_min_distance": int(_scaled_gap(kf_gap, 0.5)),
        "support_window": int(max(kf_gap, int(round(2.0 * float(kf_gap))))),
        "max_support_per_window": 1,
        "min_kf_distance": int(_scaled_gap(kf_gap, 0.3)),
        "max_extra_kf_ratio": 0.35,
    }

    analysis = _compute_candidate_points(context)
    candidate_points = analysis["candidate_points"]
    counts = dict(candidate_points.get("counts", {}))
    boundary_candidates = _filter_candidates(
        candidate_points,
        "boundary_candidates",
        context["used_last_idx"],
    )
    support_candidates = _filter_candidates(
        candidate_points,
        "support_candidates",
        context["used_last_idx"],
    )
    semantic_only_candidates = _filter_candidates(
        candidate_points,
        "semantic_only_candidates",
        context["used_last_idx"],
    )
    suppressed_candidates = _filter_candidates(
        candidate_points,
        "suppressed_candidates",
        context["used_last_idx"],
    )
    boundary_candidates = _select_merged_boundary_candidates(
        boundary_candidates,
        rules["boundary_merge_radius"],
    )

    base_items = {}
    for frame_idx in base_indices:
        base_items[int(frame_idx)] = build_item(
            frame_idx,
            source_type="uniform",
            source_role="uniform",
            candidate_role="uniform",
            promotion_source="uniform",
        )

    item_map = dict(base_items)
    max_extra_count = int(math.floor(float(len(base_indices)) * float(rules["max_extra_kf_ratio"]) + 1e-9))
    extra_count = 0
    boundary_selected = 0
    boundary_relocated = 0
    boundary_inserted = 0
    support_selected = 0
    support_inserted = 0
    selected_support_indices = []

    for candidate in boundary_candidates:
        frame_idx = int(candidate.get("frame_idx", 0))
        attach_target = _nearest_uniform_target(
            frame_idx,
            item_map,
            rules["boundary_attach_radius"],
        )
        if attach_target is not None:
            if _can_place(
                frame_idx,
                item_map.keys(),
                rules["min_kf_distance"],
                skip_idx=attach_target,
            ):
                was_relocated = int(attach_target) != int(frame_idx)
                boundary_selected += 1
                if was_relocated:
                    boundary_relocated += 1
                    del item_map[int(attach_target)]
                item_map[int(frame_idx)] = build_item(
                    frame_idx,
                    source_type="boundary",
                    source_role="boundary_candidate",
                    candidate_role="boundary_candidate",
                    rerank_score=candidate.get("rerank_score"),
                    semantic_relation=candidate.get("semantic_relation", ""),
                    is_inserted=False,
                    is_relocated=bool(was_relocated),
                    replaced_uniform_index=int(attach_target) if was_relocated else None,
                    promotion_source="boundary",
                    promotion_reason="boundary_attach_replace" if was_relocated else "boundary_attach_keep",
                )
            continue

        if extra_count >= max_extra_count:
            continue
        if not _can_place(frame_idx, item_map.keys(), rules["min_kf_distance"]):
            continue
        item_map[int(frame_idx)] = build_item(
            frame_idx,
            source_type="boundary",
            source_role="boundary_candidate",
            candidate_role="boundary_candidate",
            rerank_score=candidate.get("rerank_score"),
            semantic_relation=candidate.get("semantic_relation", ""),
            is_inserted=True,
            is_relocated=False,
            replaced_uniform_index=None,
            promotion_source="boundary",
            promotion_reason="boundary_insert",
        )
        boundary_selected += 1
        boundary_inserted += 1
        extra_count += 1

    support_budget_left = max(0, int(max_extra_count - extra_count))
    for candidate in support_candidates:
        if support_inserted >= support_budget_left:
            break
        frame_idx = int(candidate.get("frame_idx", 0))
        if frame_idx in item_map:
            continue
        nearest_gap = _nearest_gap(frame_idx, item_map.keys())
        if nearest_gap is None or nearest_gap < int(rules["support_trigger_gap"]):
            continue
        if not _can_place(frame_idx, item_map.keys(), rules["support_min_distance"]):
            continue
        blocked = False
        for existing_idx in selected_support_indices:
            if abs(int(existing_idx) - frame_idx) < int(rules["support_window"]):
                blocked = True
                break
        if blocked:
            continue
        item_map[int(frame_idx)] = build_item(
            frame_idx,
            source_type="support",
            source_role="support_candidate",
            candidate_role="support_candidate",
            rerank_score=candidate.get("rerank_score"),
            semantic_relation=candidate.get("semantic_relation", ""),
            is_inserted=True,
            is_relocated=False,
            replaced_uniform_index=None,
            promotion_source="support",
            promotion_reason="support_gap_fill",
            window_id=int(frame_idx // max(1, rules["support_window"])),
        )
        selected_support_indices.append(int(frame_idx))
        support_selected += 1
        support_inserted += 1
        extra_count += 1

    final_indices = sorted(int(x) for x in item_map.keys())
    keyframe_items = [item_map[int(frame_idx)] for frame_idx in final_indices]
    log_info(
        "semantic_guarded_v1 selected: boundary={} support={} final={}".format(
            boundary_selected,
            support_selected,
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
            "policy_name": "semantic_guarded_v1",
            "num_uniform_base": int(len(base_indices)),
            "num_boundary_selected": int(boundary_selected),
            "num_support_selected": int(support_selected),
            "num_boundary_relocated": int(boundary_relocated),
            "num_boundary_inserted": int(boundary_inserted),
            "num_support_inserted": int(support_inserted),
            "num_promoted_support_inserted": 0,
            "num_burst_windows_triggered": 0,
            "num_final_keyframes": int(len(final_indices)),
            "extra_kf_ratio": float(max(0, len(final_indices) - len(base_indices)) / float(len(base_indices) or 1)),
            "num_semantic_only_observed": int(len(semantic_only_candidates)),
            "num_suppressed_observed": int(len(suppressed_candidates)),
        },
        "policy_meta": {
            "rules": rules,
            "candidate_counts": counts,
            "uniform_base_indices": list(base_indices),
            "boundary_candidates_considered": list(boundary_candidates),
            "support_candidates_considered": list(support_candidates),
            "semantic_only_candidates_observed": list(semantic_only_candidates),
            "suppressed_candidates_observed": list(suppressed_candidates),
            "semantic_meta": dict(analysis["semantic_meta"]),
            "signal_meta": dict(analysis["signal_meta"]),
            "score_meta": dict(analysis["score_meta"]),
            "peak_meta": {
                "peak_count": int(analysis["peak_meta"].get("peak_count", 0)),
                "suppressed_peak_count": int(analysis["peak_meta"].get("suppressed_peak_count", 0)),
                "threshold": float(analysis["peak_meta"].get("threshold", 0.0)),
                "min_peak_distance": int(analysis["peak_meta"].get("min_peak_distance", 0)),
                "min_peak_prominence": float(analysis["peak_meta"].get("min_peak_prominence", 0.0)),
            },
        },
    }
