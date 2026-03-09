#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math


DEFAULT_PEAK_CONFIG = {
    "window_radius": 2,
    "threshold_std": 0.5,
    "min_peak_distance": 4,
    "min_peak_score_raw": 0.04,
    "min_peak_prominence": 0.01,
    "edge_margin": 2,
}


def _mean_and_std(values):
    if not values:
        return 0.0, 0.0
    mean_val = float(sum(values) / float(len(values)))
    sq_mean = float(sum(v * v for v in values) / float(len(values)))
    std_val = float(math.sqrt(max(0.0, sq_mean - mean_val * mean_val)))
    return mean_val, std_val


def _local_prominence(values, idx, radius):
    left = max(0, idx - radius)
    right = min(len(values), idx + radius + 1)
    left_vals = values[left:idx]
    right_vals = values[idx + 1:right]
    if not left_vals or not right_vals:
        return 0.0
    baseline = max(min(left_vals), min(right_vals))
    return float(max(0.0, values[idx] - baseline))


def _is_local_peak(values, idx, radius):
    left = max(0, idx - radius)
    right = min(len(values), idx + radius + 1)
    window = values[left:right]
    if not window:
        return False
    value = values[idx]
    if value < max(window):
        return False
    if idx > left and value == values[idx - 1]:
        return False
    return True


def annotate_peaks(
    rows,
    window_radius=2,
    threshold_std=0.5,
    min_peak_distance=4,
    min_peak_score_raw=0.04,
    min_peak_prominence=0.01,
    edge_margin=2,
):
    values = [float(row.get("score_smooth", 0.0)) for row in rows]
    mean_val, std_val = _mean_and_std(values)
    threshold = float(mean_val + float(threshold_std) * std_val)
    radius = max(1, int(window_radius))
    edge_margin = max(0, int(edge_margin))
    min_peak_distance = max(1, int(min_peak_distance))

    for row in rows:
        row["is_peak"] = False
        row["peak_rank"] = 0
        row["local_prominence"] = 0.0
        row["peak_suppressed_reason"] = ""

    local_candidates = []
    suppressed_candidates = []
    for idx, row in enumerate(rows):
        prominence = _local_prominence(values, idx, radius)
        row["local_prominence"] = float(prominence)
        if not _is_local_peak(values, idx, radius):
            continue

        reasons = []
        if idx < edge_margin or idx >= max(0, len(rows) - edge_margin):
            reasons.append("edge_margin")
        if float(values[idx]) < threshold:
            reasons.append("below_smooth_threshold")
        if float(row.get("score_raw", 0.0)) < float(min_peak_score_raw):
            reasons.append("low_score_raw")
        if float(prominence) < float(min_peak_prominence):
            reasons.append("low_prominence")

        candidate = {
            "frame_idx": int(idx),
            "score_smooth": float(values[idx]),
            "score_raw": float(row.get("score_raw", 0.0)),
            "local_prominence": float(prominence),
            "reasons": list(reasons),
        }
        local_candidates.append(candidate)

    eligible = []
    for candidate in local_candidates:
        if candidate["reasons"]:
            rows[candidate["frame_idx"]]["peak_suppressed_reason"] = ";".join(candidate["reasons"])
            suppressed_candidates.append(dict(candidate))
            continue
        eligible.append(candidate)

    selected = []
    for candidate in sorted(eligible, key=lambda item: (-item["score_smooth"], item["frame_idx"])):
        too_close = None
        for keep in selected:
            if abs(candidate["frame_idx"] - keep["frame_idx"]) < min_peak_distance:
                too_close = keep
                break
        if too_close is not None:
            candidate = dict(candidate)
            candidate["reasons"] = ["min_peak_distance"]
            rows[candidate["frame_idx"]]["peak_suppressed_reason"] = "min_peak_distance"
            suppressed_candidates.append(candidate)
            continue
        selected.append(dict(candidate))

    selected.sort(key=lambda item: (-item["score_smooth"], item["frame_idx"]))
    for rank, candidate in enumerate(selected, start=1):
        idx = int(candidate["frame_idx"])
        rows[idx]["is_peak"] = True
        rows[idx]["peak_rank"] = int(rank)
        rows[idx]["peak_suppressed_reason"] = ""

    selected_sorted = sorted(selected, key=lambda item: item["frame_idx"])
    suppressed_sorted = sorted(suppressed_candidates, key=lambda item: item["frame_idx"])
    meta = {
        "window_radius": int(radius),
        "threshold_std": float(threshold_std),
        "threshold": float(threshold),
        "min_peak_distance": int(min_peak_distance),
        "min_peak_score_raw": float(min_peak_score_raw),
        "min_peak_prominence": float(min_peak_prominence),
        "edge_margin": int(edge_margin),
        "local_peak_count": int(len(local_candidates)),
        "eligible_peak_count": int(len(eligible)),
        "peak_count": int(len(selected_sorted)),
        "suppressed_peak_count": int(len(suppressed_sorted)),
        "selected_candidates": selected_sorted,
        "suppressed_candidates": suppressed_sorted,
    }
    return rows, meta
