#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


def _build_reasons(row, thresholds):
    reasons = []
    if float(row.get("appearance_delta", 0.0)) >= thresholds["appearance_delta"]:
        reasons.append("high_appearance_delta")
    if float(row.get("brightness_jump", 0.0)) >= thresholds["brightness_jump"]:
        reasons.append("high_brightness_jump")
    if float(row.get("feature_motion", 0.0)) >= thresholds["feature_motion"]:
        reasons.append("high_feature_motion")
    if float(row.get("local_prominence", 0.0)) >= thresholds["local_prominence"]:
        reasons.append("strong_local_prominence")
    if bool(row.get("is_uniform_keyframe", False)):
        reasons.append("uniform_anchor_overlap")
    if not reasons:
        reasons.append("local_peak_candidate")
    return reasons


def _role_hint(reasons):
    if "high_appearance_delta" in reasons or "high_brightness_jump" in reasons:
        return "boundary_candidate"
    if "high_feature_motion" in reasons:
        return "support_keyframe_candidate"
    return "candidate_point"


def _candidate_record(row, reasons, selected=True):
    return {
        "frame_idx": int(row.get("frame_idx", 0)),
        "ts_sec": float(row.get("ts_sec", 0.0)),
        "file_name": row.get("file_name", ""),
        "score_raw": float(row.get("score_raw", 0.0)),
        "score_smooth": float(row.get("score_smooth", 0.0)),
        "peak_rank": int(row.get("peak_rank", 0)),
        "local_prominence": float(row.get("local_prominence", 0.0)),
        "is_uniform_keyframe": bool(row.get("is_uniform_keyframe", False)),
        "reasons": list(reasons),
        "candidate_role": _role_hint(reasons),
        "selected_peak": bool(selected),
        "peak_suppressed_reason": row.get("peak_suppressed_reason", ""),
    }


def build_candidate_points(rows, peak_meta):
    thresholds = {
        "appearance_delta": _percentile([row.get("appearance_delta", 0.0) for row in rows], 0.75),
        "brightness_jump": _percentile([row.get("brightness_jump", 0.0) for row in rows], 0.75),
        "feature_motion": _percentile([row.get("feature_motion", 0.0) for row in rows], 0.75),
        "local_prominence": max(
            float(peak_meta.get("min_peak_prominence", 0.0)),
            _percentile([row.get("local_prominence", 0.0) for row in rows], 0.75),
        ),
    }
    row_map = {int(row["frame_idx"]): row for row in rows}

    selected = []
    for item in peak_meta.get("selected_candidates", []):
        row = row_map.get(int(item["frame_idx"]))
        if row is None:
            continue
        reasons = _build_reasons(row, thresholds)
        selected.append(_candidate_record(row, reasons, selected=True))

    suppressed = []
    for item in peak_meta.get("suppressed_candidates", []):
        row = row_map.get(int(item["frame_idx"]))
        if row is None:
            continue
        reasons = _build_reasons(row, thresholds)
        if row.get("peak_suppressed_reason"):
            reasons.append("suppressed:{}".format(row.get("peak_suppressed_reason")))
        suppressed.append(_candidate_record(row, reasons, selected=False))

    selected.sort(key=lambda item: (item["peak_rank"] if item["peak_rank"] > 0 else 10 ** 9, item["frame_idx"]))
    suppressed.sort(key=lambda item: item["frame_idx"])

    return {
        "selected_candidates": selected,
        "suppressed_candidates": suppressed,
        "reason_thresholds": thresholds,
    }
