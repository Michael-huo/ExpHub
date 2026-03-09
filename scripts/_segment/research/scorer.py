#!/usr/bin/env python3
# -*- coding: utf-8 -*-


DEFAULT_SCORE_WEIGHTS = {
    "appearance_delta": 1.0,
    "brightness_jump": 0.35,
    "feature_motion": 0.75,
    "semantic_delta": 0.0,
    "blur_score": 0.0,
}

DEFAULT_OBSERVED_SIGNALS = [
    "appearance_delta",
    "brightness_jump",
    "blur_score",
    "feature_motion",
    "semantic_delta",
    "semantic_smooth",
]

DEFAULT_SCORED_SIGNALS = [
    "appearance_delta",
    "brightness_jump",
    "feature_motion",
]


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


def apply_scores(rows, weights=None, smooth_window=5, use_blur_in_score=False, use_semantic_in_score=False):
    score_weights = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        score_weights.update(weights)

    scored_signals = list(DEFAULT_SCORED_SIGNALS)
    if bool(use_semantic_in_score) and "semantic_delta" not in scored_signals:
        scored_signals.append("semantic_delta")
    if bool(use_blur_in_score) and "blur_score" not in scored_signals:
        scored_signals.append("blur_score")

    raw_values = []
    for row in rows:
        semantic_delta = row.get("semantic_delta")
        if semantic_delta is None:
            semantic_delta = 0.0

        contribs = {
            "appearance_delta": float(score_weights["appearance_delta"]) * float(row.get("appearance_delta", 0.0)),
            "brightness_jump": float(score_weights["brightness_jump"]) * float(row.get("brightness_jump", 0.0)),
            "feature_motion": float(score_weights["feature_motion"]) * float(row.get("feature_motion", 0.0)),
        }
        if bool(use_semantic_in_score):
            contribs["semantic_delta"] = float(score_weights["semantic_delta"]) * float(semantic_delta)
        if bool(use_blur_in_score):
            contribs["blur_score"] = float(score_weights["blur_score"]) * float(row.get("blur_score", 0.0))

        score_raw = float(sum(contribs.values()))
        row["score_raw"] = float(score_raw)
        row["scored_signal_contribs"] = dict(contribs)
        raw_values.append(float(score_raw))

    smooth_values, actual_window = _moving_average(raw_values, smooth_window)
    for row, score_smooth in zip(rows, smooth_values):
        row["score_smooth"] = float(score_smooth)

    meta = {
        "score_weights": score_weights,
        "smoothing": {
            "method": "moving_average",
            "window_size": int(actual_window),
        },
        "use_blur_in_score": bool(use_blur_in_score),
        "use_semantic_in_score": bool(use_semantic_in_score),
        "observed_signals": list(DEFAULT_OBSERVED_SIGNALS),
        "scored_signals": list(scored_signals),
    }
    return rows, meta
