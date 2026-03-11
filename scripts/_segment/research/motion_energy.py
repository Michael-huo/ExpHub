#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import numpy as np
from scripts._common import log_err, log_info

from .kinematics import (
    compute_acceleration,
    compute_velocity,
    cumulative_sum,
    minmax_normalize,
    moving_average,
    resolve_dt,
    series_stats,
)

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)


DEFAULT_MOTION_DENSITY_EPS = 0.03
DEFAULT_MOTION_DENSITY_ALPHA = 0.7
DEFAULT_MOTION_DENSITY_BETA = 0.3
DEFAULT_MOTION_SMOOTH_WINDOW = 5
DEFAULT_MOTION_BLUR_KERNEL = 5


def _read_gray_blur_image(path, blur_kernel_size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("failed to read frame: {}".format(path))
    kernel = max(1, int(blur_kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img


def _compute_motion_displacement(frame_paths, blur_kernel_size):
    displacement = []
    prev_gray = None
    for frame_path in frame_paths:
        cur_gray = _read_gray_blur_image(frame_path, blur_kernel_size)
        if prev_gray is None:
            displacement.append(0.0)
        else:
            diff = np.abs(cur_gray.astype(np.float32) - prev_gray.astype(np.float32))
            displacement.append(float(np.mean(diff) / 255.0))
        prev_gray = cur_gray
    return displacement


def compute_motion_rows(
    frame_paths,
    smooth_window=DEFAULT_MOTION_SMOOTH_WINDOW,
    timestamps=None,
    fps=None,
    density_eps=DEFAULT_MOTION_DENSITY_EPS,
    density_alpha=DEFAULT_MOTION_DENSITY_ALPHA,
    density_beta=DEFAULT_MOTION_DENSITY_BETA,
    blur_kernel_size=DEFAULT_MOTION_BLUR_KERNEL,
):
    frame_paths = [Path(p).resolve() for p in frame_paths]

    log_info("motion energy observe start: frames={}".format(len(frame_paths)))
    t0 = time.time()
    dt_sec, dt_source = resolve_dt(timestamps=timestamps, fps=fps)
    motion_displacement = _compute_motion_displacement(frame_paths, blur_kernel_size)
    motion_velocity = compute_velocity(motion_displacement, dt_sec)
    motion_velocity_smooth, velocity_window = moving_average(motion_velocity, smooth_window)
    motion_acceleration = compute_acceleration(motion_velocity, dt_sec)
    motion_acceleration_smooth, acceleration_window = moving_average(motion_acceleration, smooth_window)
    motion_density = []
    motion_velocity_norm = minmax_normalize(motion_velocity_smooth)
    motion_acceleration_norm = minmax_normalize(motion_acceleration_smooth)

    for idx in range(len(frame_paths)):
        motion_density.append(
            float(density_eps)
            + float(density_alpha) * float(motion_velocity_norm[idx])
            + float(density_beta) * float(motion_acceleration_norm[idx])
        )
    motion_action = cumulative_sum(motion_density)

    rows = []
    for idx, frame_path in enumerate(frame_paths):
        rows.append(
            {
                "frame_idx": int(idx),
                "file_name": Path(frame_path).name,
                "motion_displacement": float(motion_displacement[idx]),
                "motion_velocity": float(motion_velocity[idx]),
                "motion_velocity_smooth": float(motion_velocity_smooth[idx]),
                "motion_velocity_norm": float(motion_velocity_norm[idx]),
                "motion_acceleration": float(motion_acceleration[idx]),
                "motion_acceleration_smooth": float(motion_acceleration_smooth[idx]),
                "motion_acceleration_norm": float(motion_acceleration_norm[idx]),
                "motion_density": float(motion_density[idx]),
                "motion_action": float(motion_action[idx]),
            }
        )

    elapsed = float(time.time() - t0)
    log_info("motion energy observe done: frames={} elapsed={:.3f}s".format(len(frame_paths), elapsed))
    meta = {
        "enabled": True,
        "backend": "motion_energy",
        "cache_hit": False,
        "dt_sec": float(dt_sec),
        "dt_source": str(dt_source),
        "num_frames": int(len(frame_paths)),
        "observe_sec": float(elapsed),
        "preprocess": {
            "grayscale": True,
            "gaussian_blur": {
                "kernel_size": int(blur_kernel_size),
                "sigma": 0.0,
            },
        },
        "velocity_smoothing": {
            "method": "moving_average",
            "window_size": int(velocity_window),
        },
        "acceleration_smoothing": {
            "method": "moving_average",
            "window_size": int(acceleration_window),
        },
        "density": {
            "eps": float(density_eps),
            "alpha": float(density_alpha),
            "beta": float(density_beta),
        },
        "signal_stats": {
            "motion_displacement": series_stats(motion_displacement),
            "motion_velocity": series_stats(motion_velocity),
            "motion_acceleration": series_stats(motion_acceleration),
            "motion_density": series_stats(motion_density),
        },
    }
    return rows, meta
