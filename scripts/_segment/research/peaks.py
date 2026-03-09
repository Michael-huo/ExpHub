#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, List, Tuple


DEFAULT_PEAK_CONFIG = {
    "window_radius": 2,
    "threshold_std": 0.5,
}


def annotate_peaks(rows, window_radius=2, threshold_std=0.5):
    values = [float(row.get("score_smooth", 0.0)) for row in rows]
    if not values:
        return rows, {
            "window_radius": int(window_radius),
            "threshold_std": float(threshold_std),
            "threshold": 0.0,
            "peak_count": 0,
        }

    mean_val = float(sum(values) / float(len(values)))
    sq_mean = float(sum(v * v for v in values) / float(len(values)))
    std_val = float(math.sqrt(max(0.0, sq_mean - mean_val * mean_val)))
    threshold = float(mean_val + float(threshold_std) * std_val)
    radius = max(1, int(window_radius))

    candidates = []
    for idx, value in enumerate(values):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        window = values[left:right]
        if not window:
            continue
        if value < threshold:
            continue
        if value < max(window):
            continue
        if idx > left and value == values[idx - 1]:
            continue
        candidates.append((idx, value))

    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    rank_map = {}
    for rank, item in enumerate(ranked, start=1):
        rank_map[item[0]] = int(rank)

    for idx, row in enumerate(rows):
        is_peak = idx in rank_map
        row["is_peak"] = bool(is_peak)
        row["peak_rank"] = int(rank_map.get(idx, 0))

    meta = {
        "window_radius": int(radius),
        "threshold_std": float(threshold_std),
        "threshold": float(threshold),
        "peak_count": int(len(ranked)),
    }
    return rows, meta
