#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def moving_average(values, window_size):
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


def minmax_normalize(values):
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    arr = (arr - vmin) / float(vmax - vmin)
    return [float(x) for x in arr.tolist()]


def resolve_dt(timestamps=None, fps=None):
    if fps is not None:
        try:
            fps_value = float(fps)
        except Exception:
            fps_value = 0.0
        if fps_value > 0.0:
            return float(1.0 / fps_value), "fps"

    if timestamps:
        diffs = []
        prev_t = None
        for ts_sec in timestamps:
            cur_t = float(ts_sec)
            if prev_t is not None:
                dt = float(cur_t - prev_t)
                if dt > 1e-9:
                    diffs.append(dt)
            prev_t = cur_t
        if diffs:
            return float(np.median(np.asarray(diffs, dtype=np.float32))), "timestamps_median"

    return 1.0, "unit_dt_fallback"


def compute_velocity(displacement, dt):
    dt = max(float(dt), 1e-9)
    values = []
    for value in displacement:
        values.append(float(value) / dt)
    return values


def compute_acceleration(velocity, dt):
    dt = max(float(dt), 1e-9)
    values = [0.0 for _ in velocity]
    for idx in range(1, len(velocity)):
        values[idx] = float(abs(float(velocity[idx]) - float(velocity[idx - 1])) / dt)
    return values


def cumulative_sum(values):
    out = []
    total = 0.0
    for value in values:
        total += float(value)
        out.append(float(total))
    return out


def series_stats(values):
    if not values:
        return {
            "mean": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }
