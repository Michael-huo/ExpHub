#!/usr/bin/env python3
# -*- coding: utf-8 -*-


DEFAULT_RERANK_WEIGHTS = {
    "score_smooth": 0.5,
    "semantic_smooth": 0.3,
    "local_prominence": 0.2,
}

DEFAULT_ROLE_RULES = {
    "boundary_candidate": "selected peak with high score_smooth, high semantic_smooth, and high local_prominence",
    "support_candidate": "selected peak with strong nonsemantic support or structure change, but not enough semantic evidence for boundary",
    "semantic_only_candidate": "selected peak with high semantic support while nonsemantic support stays below boundary/support level",
    "suppressed": "peak candidate rejected by peak rules or kept only for analysis",
}


def _percentile(values, q):
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    q = min(1.0, max(0.0, float(q)))
    pos = q * float(len(vals) - 1)
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - float(lo)
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _normalize_map(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    if not values:
        return {}
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return {int(row.get("frame_idx", idx)): 0.0 for idx, row in enumerate(rows)}
    out = {}
    scale = float(vmax - vmin)
    for idx, row in enumerate(rows):
        frame_idx = int(row.get("frame_idx", idx))
        out[frame_idx] = float((values[idx] - vmin) / scale)
    return out


def _max_signal(value_a, value_b, value_c):
    return float(max(float(value_a), float(value_b), float(value_c)))


def _semantic_relation(row, thresholds):
    semantic_high = float(row.get("semantic_smooth", 0.0)) >= float(thresholds["semantic_smooth"])
    nonsemantic_high = float(row.get("score_smooth", 0.0)) >= float(thresholds["score_smooth"])
    if semantic_high and nonsemantic_high:
        return "semantic_and_nonsemantic_high"
    if semantic_high and not nonsemantic_high:
        return "semantic_high_only"
    if (not semantic_high) and nonsemantic_high:
        return "nonsemantic_high_only"
    return "semantic_and_nonsemantic_low"


def _build_reason_thresholds(rows, peak_meta):
    return {
        "appearance_delta": _percentile([row.get("appearance_delta", 0.0) for row in rows], 0.75),
        "brightness_jump": _percentile([row.get("brightness_jump", 0.0) for row in rows], 0.75),
        "feature_motion": _percentile([row.get("feature_motion", 0.0) for row in rows], 0.75),
        "semantic_smooth": _percentile([row.get("semantic_smooth", 0.0) for row in rows], 0.75),
        "local_prominence": max(
            float(peak_meta.get("min_peak_prominence", 0.0)),
            _percentile([row.get("local_prominence", 0.0) for row in rows], 0.75),
        ),
    }


def _build_relation_thresholds(rows):
    return {
        "semantic_smooth": max(
            1e-6,
            _percentile([row.get("semantic_smooth", 0.0) for row in rows], 0.75),
        ),
        "score_smooth": max(
            1e-6,
            _percentile([row.get("score_smooth", 0.0) for row in rows], 0.75),
        ),
    }


def _build_support_maps(rows):
    score_norm = _normalize_map(rows, "score_smooth")
    semantic_norm = _normalize_map(rows, "semantic_smooth")
    prominence_norm = _normalize_map(rows, "local_prominence")
    appearance_norm = _normalize_map(rows, "appearance_delta")
    brightness_norm = _normalize_map(rows, "brightness_jump")
    motion_norm = _normalize_map(rows, "feature_motion")

    support_rows = []
    for row in rows:
        frame_idx = int(row["frame_idx"])
        structure_support = _max_signal(
            appearance_norm.get(frame_idx, 0.0),
            brightness_norm.get(frame_idx, 0.0),
            motion_norm.get(frame_idx, 0.0),
        )
        semantic_support = float(semantic_norm.get(frame_idx, 0.0))
        nonsemantic_support = float(
            0.5 * float(score_norm.get(frame_idx, 0.0))
            + 0.3 * float(prominence_norm.get(frame_idx, 0.0))
            + 0.2 * float(structure_support)
        )
        support_rows.append(
            {
                "frame_idx": frame_idx,
                "score_norm": float(score_norm.get(frame_idx, 0.0)),
                "semantic_norm": float(semantic_support),
                "prominence_norm": float(prominence_norm.get(frame_idx, 0.0)),
                "appearance_norm": float(appearance_norm.get(frame_idx, 0.0)),
                "brightness_norm": float(brightness_norm.get(frame_idx, 0.0)),
                "motion_norm": float(motion_norm.get(frame_idx, 0.0)),
                "structure_support": float(structure_support),
                "semantic_support": float(semantic_support),
                "nonsemantic_support": float(nonsemantic_support),
            }
        )
    return {int(item["frame_idx"]): item for item in support_rows}


def _build_role_thresholds(rows, support_map, reason_thresholds, relation_thresholds):
    support_values = [support_map[int(row["frame_idx"])] for row in rows]
    return {
        "score_smooth_high": float(relation_thresholds["score_smooth"]),
        "semantic_smooth_high": float(relation_thresholds["semantic_smooth"]),
        "local_prominence_high": float(reason_thresholds["local_prominence"]),
        "structure_support_high": float(_percentile([item["structure_support"] for item in support_values], 0.75)),
        "nonsemantic_support_high": float(_percentile([item["nonsemantic_support"] for item in support_values], 0.75)),
        "semantic_support_high": float(_percentile([item["semantic_support"] for item in support_values], 0.75)),
    }


def _rerank_score(support_values, weights):
    return float(
        float(weights["score_smooth"]) * float(support_values["score_norm"])
        + float(weights["semantic_smooth"]) * float(support_values["semantic_norm"])
        + float(weights["local_prominence"]) * float(support_values["prominence_norm"])
    )


def _support_signal_high(row, thresholds):
    if float(row.get("appearance_delta", 0.0)) >= float(thresholds["appearance_delta"]):
        return True
    if float(row.get("brightness_jump", 0.0)) >= float(thresholds["brightness_jump"]):
        return True
    if float(row.get("feature_motion", 0.0)) >= float(thresholds["feature_motion"]):
        return True
    return False


def _determine_role(row, selected, semantic_relation, reason_thresholds, role_thresholds, support_values):
    role_reasons = []
    score_high = float(row.get("score_smooth", 0.0)) >= float(role_thresholds["score_smooth_high"])
    semantic_high = float(row.get("semantic_smooth", 0.0)) >= float(role_thresholds["semantic_smooth_high"])
    prominence_high = float(row.get("local_prominence", 0.0)) >= float(role_thresholds["local_prominence_high"])
    structure_high = float(support_values["structure_support"]) >= float(role_thresholds["structure_support_high"])
    nonsemantic_high = float(support_values["nonsemantic_support"]) >= float(role_thresholds["nonsemantic_support_high"])
    semantic_support_high = float(support_values["semantic_support"]) >= float(role_thresholds["semantic_support_high"])

    if not bool(selected):
        role_reasons.append("suppressed_by_peak_rules")
        if row.get("peak_suppressed_reason"):
            role_reasons.append("peak_suppressed:{}".format(row.get("peak_suppressed_reason")))
        return "suppressed", role_reasons

    if semantic_relation == "semantic_and_nonsemantic_high" and score_high and semantic_high and prominence_high:
        role_reasons.extend(
            [
                "selected_peak",
                "semantic_and_nonsemantic_high",
                "local_prominence_high",
            ]
        )
        return "boundary_candidate", role_reasons

    if semantic_relation == "semantic_high_only" and semantic_support_high and not nonsemantic_high:
        role_reasons.extend(
            [
                "selected_peak",
                "semantic_high_only",
                "observe_only_watchpoint",
            ]
        )
        return "semantic_only_candidate", role_reasons

    if nonsemantic_high and (structure_high or _support_signal_high(row, reason_thresholds)):
        role_reasons.extend(
            [
                "selected_peak",
                "nonsemantic_support_high",
                "structure_or_change_signal_support",
            ]
        )
        if semantic_relation == "nonsemantic_high_only":
            role_reasons.append("nonsemantic_high_only")
        return "support_candidate", role_reasons

    if semantic_support_high and not prominence_high:
        role_reasons.extend(
            [
                "selected_peak",
                "semantic_support_high",
                "prominence_not_boundary_level",
            ]
        )
        return "semantic_only_candidate", role_reasons

    role_reasons.extend(
        [
            "selected_peak",
            "fallback_support_candidate",
        ]
    )
    return "support_candidate", role_reasons


def _base_reasons(row, reason_thresholds):
    reasons = []
    if float(row.get("appearance_delta", 0.0)) >= float(reason_thresholds["appearance_delta"]):
        reasons.append("high_appearance_delta")
    if float(row.get("brightness_jump", 0.0)) >= float(reason_thresholds["brightness_jump"]):
        reasons.append("high_brightness_jump")
    if float(row.get("feature_motion", 0.0)) >= float(reason_thresholds["feature_motion"]):
        reasons.append("high_feature_motion")
    if float(row.get("semantic_smooth", 0.0)) >= float(reason_thresholds["semantic_smooth"]):
        reasons.append("high_semantic_smooth")
    if float(row.get("local_prominence", 0.0)) >= float(reason_thresholds["local_prominence"]):
        reasons.append("strong_local_prominence")
    if bool(row.get("is_uniform_keyframe", False)):
        reasons.append("uniform_anchor_overlap")
    if not reasons:
        reasons.append("local_peak_candidate")
    return reasons


def _candidate_record(row, selected, reason_thresholds, relation_thresholds, role_thresholds, rerank_weights, support_values):
    semantic_relation = _semantic_relation(row, relation_thresholds)
    rerank_score = _rerank_score(support_values, rerank_weights)
    candidate_role, role_reasons = _determine_role(
        row,
        selected=selected,
        semantic_relation=semantic_relation,
        reason_thresholds=reason_thresholds,
        role_thresholds=role_thresholds,
        support_values=support_values,
    )

    reasons = _base_reasons(row, reason_thresholds)
    reasons.extend(role_reasons)
    if not bool(selected) and row.get("peak_suppressed_reason"):
        reasons.append("suppressed:{}".format(row.get("peak_suppressed_reason")))

    nonsemantic_support = float(support_values["nonsemantic_support"])
    semantic_support = float(support_values["semantic_support"])
    boundary_support_ratio = float(semantic_support / max(1e-6, nonsemantic_support))
    return {
        "frame_idx": int(row.get("frame_idx", 0)),
        "ts_sec": float(row.get("ts_sec", 0.0)),
        "file_name": row.get("file_name", ""),
        "score_raw": float(row.get("score_raw", 0.0)),
        "score_smooth": float(row.get("score_smooth", 0.0)),
        "semantic_delta": float(row.get("semantic_delta", 0.0)),
        "semantic_smooth": float(row.get("semantic_smooth", 0.0)),
        "semantic_relation": semantic_relation,
        "candidate_role": candidate_role,
        "candidate_confidence": float(rerank_score),
        "rerank_score": float(rerank_score),
        "semantic_support": float(semantic_support),
        "nonsemantic_support": float(nonsemantic_support),
        "boundary_support_ratio": float(boundary_support_ratio),
        "role_reasons": list(role_reasons),
        "peak_rank": int(row.get("peak_rank", 0)),
        "local_prominence": float(row.get("local_prominence", 0.0)),
        "is_uniform_keyframe": bool(row.get("is_uniform_keyframe", False)),
        "reasons": list(reasons),
        "selected_peak": bool(selected),
        "peak_suppressed_reason": row.get("peak_suppressed_reason", ""),
    }


def _counts_from_roles(records):
    return {
        "boundary_candidate_count": int(len([item for item in records if item["candidate_role"] == "boundary_candidate"])),
        "support_candidate_count": int(len([item for item in records if item["candidate_role"] == "support_candidate"])),
        "semantic_only_candidate_count": int(len([item for item in records if item["candidate_role"] == "semantic_only_candidate"])),
        "suppressed_candidate_count": int(len([item for item in records if item["candidate_role"] == "suppressed"])),
        "total_candidate_count": int(len(records)),
    }


def _records_by_role(records, role_name):
    items = [item for item in records if item["candidate_role"] == role_name]
    items.sort(key=lambda item: (-float(item["rerank_score"]), item["frame_idx"]))
    return items


def build_candidate_points(rows, peak_meta):
    reason_thresholds = _build_reason_thresholds(rows, peak_meta)
    relation_thresholds = _build_relation_thresholds(rows)
    support_map = _build_support_maps(rows)
    role_thresholds = _build_role_thresholds(rows, support_map, reason_thresholds, relation_thresholds)
    rerank_weights = dict(DEFAULT_RERANK_WEIGHTS)
    role_rules = dict(DEFAULT_ROLE_RULES)
    row_map = {int(row["frame_idx"]): row for row in rows}

    selected_candidates = []
    for item in peak_meta.get("selected_candidates", []):
        row = row_map.get(int(item["frame_idx"]))
        if row is None:
            continue
        support_values = support_map.get(int(row["frame_idx"]), {})
        selected_candidates.append(
            _candidate_record(
                row,
                selected=True,
                reason_thresholds=reason_thresholds,
                relation_thresholds=relation_thresholds,
                role_thresholds=role_thresholds,
                rerank_weights=rerank_weights,
                support_values=support_values,
            )
        )

    suppressed_candidates = []
    for item in peak_meta.get("suppressed_candidates", []):
        row = row_map.get(int(item["frame_idx"]))
        if row is None:
            continue
        support_values = support_map.get(int(row["frame_idx"]), {})
        suppressed_candidates.append(
            _candidate_record(
                row,
                selected=False,
                reason_thresholds=reason_thresholds,
                relation_thresholds=relation_thresholds,
                role_thresholds=role_thresholds,
                rerank_weights=rerank_weights,
                support_values=support_values,
            )
        )

    all_candidates = list(selected_candidates) + list(suppressed_candidates)
    all_candidates.sort(key=lambda item: (-float(item["rerank_score"]), item["frame_idx"]))
    for rank, item in enumerate(all_candidates, start=1):
        item["rerank_rank"] = int(rank)

    selected_candidates.sort(key=lambda item: (-float(item["rerank_score"]), item["frame_idx"]))
    suppressed_candidates.sort(key=lambda item: (-float(item["rerank_score"]), item["frame_idx"]))

    counts = _counts_from_roles(all_candidates)
    roles_summary = {
        "boundary_candidates": _records_by_role(all_candidates, "boundary_candidate"),
        "support_candidates": _records_by_role(all_candidates, "support_candidate"),
        "semantic_only_candidates": _records_by_role(all_candidates, "semantic_only_candidate"),
        "suppressed_candidates": _records_by_role(all_candidates, "suppressed"),
        "counts": counts,
        "top_reranked_candidates": list(all_candidates[:10]),
    }

    return {
        "selected_candidates": selected_candidates,
        "suppressed_candidates": suppressed_candidates,
        "reranked_candidates": all_candidates,
        "reason_thresholds": reason_thresholds,
        "relation_thresholds": relation_thresholds,
        "role_thresholds": role_thresholds,
        "rerank_weights": rerank_weights,
        "role_rules": role_rules,
        "counts": counts,
        "candidate_roles_summary": roles_summary,
    }
