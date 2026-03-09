#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional


DEFAULT_SCORE_WEIGHTS = {
    "appearance_delta": 1.0,
    "brightness_jump": 0.35,
    "feature_motion": 0.75,
    "semantic_delta": 0.0,
}


def _moving_average(values, window_size):
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    out = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        window = values[left:right]
        if not window:
            out.append(0.0)
        else:
            out.append(float(sum(window) / float(len(window))))
    return out, window_size


def apply_scores(rows, weights=None, smooth_window=5):
    score_weights = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        score_weights.update(weights)

    raw_values = []
    for row in rows:
        semantic_delta = row.get("semantic_delta")
        if semantic_delta is None:
            semantic_delta = 0.0
        raw = (
            float(score_weights["appearance_delta"]) * float(row.get("appearance_delta", 0.0))
            + float(score_weights["brightness_jump"]) * float(row.get("brightness_jump", 0.0))
            + float(score_weights["feature_motion"]) * float(row.get("feature_motion", 0.0))
            + float(score_weights["semantic_delta"]) * float(semantic_delta)
        )
        row["score_raw"] = float(raw)
        raw_values.append(float(raw))

    smooth_values, actual_window = _moving_average(raw_values, smooth_window)
    for row, score_smooth in zip(rows, smooth_values):
        row["score_smooth"] = float(score_smooth)

    meta = {
        "score_weights": score_weights,
        "smoothing": {
            "method": "moving_average",
            "window_size": int(actual_window),
        },
    }
    return rows, meta
