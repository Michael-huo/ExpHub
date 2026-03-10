#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from _common import log_info
from _segment.policies import semantic_guarded_v1 as v1


def _window_id(frame_idx, window_size):
    window_size = max(1, int(window_size))
    return int(int(frame_idx) // int(window_size))


def _candidate_sort_key(item):
    return (
        -float(item.get("rerank_score", 0.0)),
        -float(item.get("nonsemantic_support", 0.0)),
        -float(item.get("local_prominence", 0.0)),
        int(item.get("frame_idx", 0)),
    )


def _can_place_against_sources(frame_idx, item_map, min_distance, source_types):
    frame_idx = int(frame_idx)
    source_type_set = set(str(x) for x in source_types)
    for existing_idx, item in item_map.items():
        if str(item.get("source_type", "")) not in source_type_set:
            continue
        if abs(int(existing_idx) - frame_idx) < int(min_distance):
            return False
    return True


def _select_support_candidates(support_candidates, item_map, rules, support_budget, extra_count, max_total_extra_count, build_item):
    selected_support_indices = []
    support_inserted = 0
    window_loads = {}

    for candidate in sorted(support_candidates, key=_candidate_sort_key):
        if support_inserted >= int(support_budget):
            break
        if int(extra_count) >= int(max_total_extra_count):
            break

        frame_idx = int(candidate.get("frame_idx", 0))
        if frame_idx in item_map:
            continue
        if float(candidate.get("rerank_score", 0.0)) < float(rules["min_support_rerank"]):
            continue

        nearest_gap = v1._nearest_gap(frame_idx, item_map.keys())
        if nearest_gap is None or nearest_gap < int(rules["support_trigger_gap"]):
            continue
        if not _can_place_against_sources(
            frame_idx,
            item_map,
            rules["support_min_distance"],
            ("support", "boundary"),
        ):
            continue

        window_id = _window_id(frame_idx, rules["support_window"])
        if int(window_loads.get(window_id, 0)) >= int(rules["max_support_per_window"]):
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
            window_id=window_id,
        )
        selected_support_indices.append(int(frame_idx))
        window_loads[window_id] = int(window_loads.get(window_id, 0)) + 1
        support_inserted += 1
        extra_count += 1

    return {
        "item_map": item_map,
        "window_loads": window_loads,
        "support_inserted": int(support_inserted),
        "extra_count": int(extra_count),
        "selected_support_indices": list(selected_support_indices),
    }


def _eligible_promoted_candidates(suppressed_candidates, item_map, rules):
    allowed_relations = set(
        [
            "semantic_and_nonsemantic_high",
            "nonsemantic_high_only",
        ]
    )
    eligible = []
    for candidate in suppressed_candidates:
        peak_reason = str(candidate.get("peak_suppressed_reason", "") or "")
        if "low_prominence" not in peak_reason:
            continue
        if "edge_margin" in peak_reason:
            continue
        if str(candidate.get("role_if_selected", "")) == "semantic_only_candidate":
            continue
        if str(candidate.get("semantic_relation", "")) not in allowed_relations:
            continue
        if int(candidate.get("rerank_rank", 10 ** 9)) > int(rules["promoted_max_rerank_rank"]):
            continue
        if float(candidate.get("rerank_score", 0.0)) < float(rules["promoted_min_rerank"]):
            continue

        frame_idx = int(candidate.get("frame_idx", 0))
        if frame_idx in item_map:
            continue
        nearest_gap = v1._nearest_gap(frame_idx, item_map.keys())
        if nearest_gap is None or nearest_gap < int(rules["promoted_min_distance"]):
            continue
        if not _can_place_against_sources(
            frame_idx,
            item_map,
            rules["support_min_distance"],
            ("support", "boundary"),
        ):
            continue
        eligible.append(dict(candidate))

    eligible.sort(key=_candidate_sort_key)
    return eligible


def _insert_promoted_candidates(
    promoted_candidates,
    item_map,
    window_loads,
    rules,
    promoted_budget,
    extra_count,
    max_total_extra_count,
    build_item,
):
    promoted_inserted = 0
    inserted_frame_indices = set()

    for candidate in promoted_candidates:
        if promoted_inserted >= int(promoted_budget):
            break
        if int(extra_count) >= int(max_total_extra_count):
            break

        frame_idx = int(candidate.get("frame_idx", 0))
        if frame_idx in item_map or frame_idx in inserted_frame_indices:
            continue

        nearest_gap = v1._nearest_gap(frame_idx, item_map.keys())
        if nearest_gap is None or nearest_gap < int(rules["promoted_min_distance"]):
            continue
        if not _can_place_against_sources(
            frame_idx,
            item_map,
            rules["support_min_distance"],
            ("support", "boundary"),
        ):
            continue

        window_id = _window_id(frame_idx, rules["support_window"])
        if int(window_loads.get(window_id, 0)) >= int(rules["max_support_per_window"]):
            continue

        item_map[int(frame_idx)] = build_item(
            frame_idx,
            source_type="support",
            source_role="promoted_support_candidate",
            candidate_role="promoted_support_candidate",
            rerank_score=candidate.get("rerank_score"),
            semantic_relation=candidate.get("semantic_relation", ""),
            is_inserted=True,
            is_relocated=False,
            replaced_uniform_index=None,
            promotion_source="suppressed_high",
            promotion_reason="suppressed_high_low_prominence",
            window_id=window_id,
        )
        inserted_frame_indices.add(int(frame_idx))
        window_loads[window_id] = int(window_loads.get(window_id, 0)) + 1
        promoted_inserted += 1
        extra_count += 1

    return {
        "item_map": item_map,
        "window_loads": window_loads,
        "promoted_inserted": int(promoted_inserted),
        "extra_count": int(extra_count),
        "inserted_frame_indices": inserted_frame_indices,
    }


def _build_burst_window_stats(rows, promoted_candidates, item_map, burst_window):
    groups = {}
    for row in rows:
        window_id = _window_id(row.get("frame_idx", 0), burst_window)
        group = groups.setdefault(
            window_id,
            {
                "score_sum": 0.0,
                "semantic_sum": 0.0,
                "row_count": 0,
                "promoted_candidates": [],
                "final_keyframe_count": 0,
            },
        )
        group["score_sum"] += float(row.get("score_smooth", 0.0))
        group["semantic_sum"] += float(row.get("semantic_smooth", 0.0))
        group["row_count"] += 1

    for candidate in promoted_candidates:
        window_id = _window_id(candidate.get("frame_idx", 0), burst_window)
        groups.setdefault(
            window_id,
            {
                "score_sum": 0.0,
                "semantic_sum": 0.0,
                "row_count": 0,
                "promoted_candidates": [],
                "final_keyframe_count": 0,
            },
        )["promoted_candidates"].append(dict(candidate))

    for frame_idx in item_map.keys():
        window_id = _window_id(frame_idx, burst_window)
        groups.setdefault(
            window_id,
            {
                "score_sum": 0.0,
                "semantic_sum": 0.0,
                "row_count": 0,
                "promoted_candidates": [],
                "final_keyframe_count": 0,
            },
        )["final_keyframe_count"] += 1

    return groups


def _insert_burst_promotions(
    rows,
    promoted_candidates,
    item_map,
    rules,
    promoted_budget_left,
    extra_count,
    max_total_extra_count,
    build_item,
):
    if int(promoted_budget_left) <= 0 or int(extra_count) >= int(max_total_extra_count):
        return {
            "item_map": item_map,
            "burst_inserted": 0,
            "extra_count": int(extra_count),
            "burst_window_ids": set(),
        }

    global_score_mean = 0.0
    global_semantic_mean = 0.0
    if rows:
        global_score_mean = float(sum(float(row.get("score_smooth", 0.0)) for row in rows) / float(len(rows)))
        global_semantic_mean = float(sum(float(row.get("semantic_smooth", 0.0)) for row in rows) / float(len(rows)))

    groups = _build_burst_window_stats(rows, promoted_candidates, item_map, rules["burst_window"])
    burst_inserted = 0
    burst_window_ids = set()

    for window_id in sorted(groups.keys()):
        if burst_inserted >= int(promoted_budget_left):
            break
        if int(extra_count) >= int(max_total_extra_count):
            break

        group = groups[window_id]
        candidates = sorted(group.get("promoted_candidates", []), key=_candidate_sort_key)
        if len(candidates) < 2:
            continue

        row_count = int(group.get("row_count", 0))
        score_mean = float(group.get("score_sum", 0.0) / float(row_count)) if row_count > 0 else 0.0
        semantic_mean = float(group.get("semantic_sum", 0.0) / float(row_count)) if row_count > 0 else 0.0
        activity_strong = bool(score_mean >= global_score_mean or semantic_mean >= global_semantic_mean)
        if not activity_strong:
            continue
        if int(group.get("final_keyframe_count", 0)) > int(rules["burst_max_final_keyframes"]):
            continue

        for candidate in candidates:
            frame_idx = int(candidate.get("frame_idx", 0))
            if frame_idx in item_map:
                continue
            nearest_gap = v1._nearest_gap(frame_idx, item_map.keys())
            if nearest_gap is None or nearest_gap < int(rules["promoted_min_distance"]):
                continue
            if not _can_place_against_sources(
                frame_idx,
                item_map,
                rules["support_min_distance"],
                ("support", "boundary"),
            ):
                continue

            item_map[int(frame_idx)] = build_item(
                frame_idx,
                source_type="support",
                source_role="promoted_support_candidate",
                candidate_role="promoted_support_candidate",
                rerank_score=candidate.get("rerank_score"),
                semantic_relation=candidate.get("semantic_relation", ""),
                is_inserted=True,
                is_relocated=False,
                replaced_uniform_index=None,
                promotion_source="suppressed_high",
                promotion_reason="suppressed_high_burst_window",
                window_id=window_id,
            )
            burst_inserted += 1
            extra_count += 1
            burst_window_ids.add(int(window_id))
            break

    return {
        "item_map": item_map,
        "burst_inserted": int(burst_inserted),
        "extra_count": int(extra_count),
        "burst_window_ids": burst_window_ids,
        "global_score_mean": float(global_score_mean),
        "global_semantic_mean": float(global_semantic_mean),
    }


def build_policy_plan(context):
    build_item = context["build_item"]
    base_indices = list(context["uniform_base_indices"])
    kf_gap = int(context["kf_gap"])
    rules = {
        "boundary_attach_radius": int(v1._scaled_gap(kf_gap, 0.4)),
        "boundary_merge_radius": int(v1._scaled_gap(kf_gap, 0.25)),
        "support_trigger_gap": int(v1._scaled_gap(kf_gap, 0.3)),
        "support_min_distance": int(v1._scaled_gap(kf_gap, 0.25)),
        "support_window": int(max(1, int(round(1.5 * float(kf_gap))))),
        "max_support_per_window": 1,
        "min_support_rerank": 0.30,
        "promoted_min_distance": 4,
        "promoted_min_rerank": 0.45,
        "promoted_max_rerank_rank": 12,
        "burst_window": int(max(1, int(round(1.5 * float(kf_gap))))),
        "burst_max_final_keyframes": 2,
        "min_kf_distance": int(v1._scaled_gap(kf_gap, 0.3)),
        "max_support_extra_ratio": 0.12,
        "max_promoted_extra_ratio": 0.08,
        "max_extra_kf_ratio": 0.40,
    }

    analysis = v1._compute_candidate_points(context)
    candidate_points = analysis["candidate_points"]
    counts = dict(candidate_points.get("counts", {}))
    boundary_candidates = v1._filter_candidates(
        candidate_points,
        "boundary_candidates",
        context["used_last_idx"],
    )
    support_candidates = v1._filter_candidates(
        candidate_points,
        "support_candidates",
        context["used_last_idx"],
    )
    semantic_only_candidates = v1._filter_candidates(
        candidate_points,
        "semantic_only_candidates",
        context["used_last_idx"],
    )
    suppressed_candidates = v1._filter_candidates(
        candidate_points,
        "suppressed_candidates",
        context["used_last_idx"],
    )
    boundary_candidates = v1._select_merged_boundary_candidates(
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
    max_total_extra_count = int(math.floor(float(len(base_indices)) * float(rules["max_extra_kf_ratio"]) + 1e-9))
    extra_count = 0
    boundary_selected = 0
    boundary_relocated = 0
    boundary_inserted = 0

    for candidate in boundary_candidates:
        frame_idx = int(candidate.get("frame_idx", 0))
        attach_target = v1._nearest_uniform_target(
            frame_idx,
            item_map,
            rules["boundary_attach_radius"],
        )
        if attach_target is not None:
            if v1._can_place(
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

        if extra_count >= max_total_extra_count:
            continue
        if not v1._can_place(frame_idx, item_map.keys(), rules["min_kf_distance"]):
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

    support_budget = int(math.floor(float(len(base_indices)) * float(rules["max_support_extra_ratio"]) + 1e-9))
    support_budget = min(int(support_budget), max(0, int(max_total_extra_count - extra_count)))
    support_result = _select_support_candidates(
        support_candidates=support_candidates,
        item_map=item_map,
        rules=rules,
        support_budget=support_budget,
        extra_count=extra_count,
        max_total_extra_count=max_total_extra_count,
        build_item=build_item,
    )
    item_map = support_result["item_map"]
    support_inserted = int(support_result["support_inserted"])
    extra_count = int(support_result["extra_count"])
    window_loads = dict(support_result["window_loads"])

    promoted_budget_total = int(math.floor(float(len(base_indices)) * float(rules["max_promoted_extra_ratio"]) + 1e-9))
    promoted_budget_total = min(int(promoted_budget_total), max(0, int(max_total_extra_count - extra_count)))
    eligible_promoted = _eligible_promoted_candidates(
        suppressed_candidates=suppressed_candidates,
        item_map=item_map,
        rules=rules,
    )
    promoted_result = _insert_promoted_candidates(
        promoted_candidates=eligible_promoted,
        item_map=item_map,
        window_loads=window_loads,
        rules=rules,
        promoted_budget=promoted_budget_total,
        extra_count=extra_count,
        max_total_extra_count=max_total_extra_count,
        build_item=build_item,
    )
    item_map = promoted_result["item_map"]
    window_loads = promoted_result["window_loads"]
    promoted_inserted = int(promoted_result["promoted_inserted"])
    extra_count = int(promoted_result["extra_count"])
    already_promoted = set(int(x) for x in promoted_result["inserted_frame_indices"])

    remaining_promoted = []
    for candidate in eligible_promoted:
        if int(candidate.get("frame_idx", 0)) in already_promoted:
            continue
        remaining_promoted.append(candidate)

    burst_result = _insert_burst_promotions(
        rows=analysis["rows"],
        promoted_candidates=remaining_promoted,
        item_map=item_map,
        rules=rules,
        promoted_budget_left=max(0, int(promoted_budget_total - promoted_inserted)),
        extra_count=extra_count,
        max_total_extra_count=max_total_extra_count,
        build_item=build_item,
    )
    item_map = burst_result["item_map"]
    burst_inserted = int(burst_result["burst_inserted"])
    extra_count = int(burst_result["extra_count"])
    promoted_inserted += int(burst_inserted)
    burst_window_ids = set(int(x) for x in burst_result.get("burst_window_ids", set()))

    final_indices = sorted(int(x) for x in item_map.keys())
    keyframe_items = [item_map[int(frame_idx)] for frame_idx in final_indices]
    support_selected = int(support_inserted)

    log_info(
        "semantic_guarded_v2 selected: boundary={} support={} promoted={} burst={} final={}".format(
            boundary_selected,
            support_inserted,
            promoted_inserted,
            len(burst_window_ids),
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
            "policy_name": "semantic_guarded_v2",
            "num_uniform_base": int(len(base_indices)),
            "num_boundary_selected": int(boundary_selected),
            "num_boundary_relocated": int(boundary_relocated),
            "num_boundary_inserted": int(boundary_inserted),
            "num_support_selected": int(support_selected),
            "num_support_inserted": int(support_inserted),
            "num_promoted_support_inserted": int(promoted_inserted),
            "num_burst_windows_triggered": int(len(burst_window_ids)),
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
            "support_candidates_considered": sorted(list(support_candidates), key=_candidate_sort_key),
            "eligible_promoted_candidates": list(eligible_promoted),
            "semantic_only_candidates_observed": list(semantic_only_candidates),
            "suppressed_candidates_observed": list(suppressed_candidates),
            "burst_window_ids_triggered": sorted(int(x) for x in burst_window_ids),
            "burst_activity_thresholds": {
                "global_score_mean": float(burst_result.get("global_score_mean", 0.0)),
                "global_semantic_mean": float(burst_result.get("global_semantic_mean", 0.0)),
                "max_final_keyframes_per_window": int(rules["burst_max_final_keyframes"]),
            },
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
