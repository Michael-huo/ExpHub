#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import numpy as np
from exphub.common.logging import log_err, log_info

from .kinematics import (
    compute_velocity,
    resolve_dt,
    series_stats,
)

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)

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
    timestamps=None,
    fps=None,
    blur_kernel_size=DEFAULT_MOTION_BLUR_KERNEL,
):
    frame_paths = [Path(p).resolve() for p in frame_paths]

    log_info("motion energy observe start: frames={}".format(len(frame_paths)))
    t0 = time.time()
    dt_sec, dt_source = resolve_dt(timestamps=timestamps, fps=fps)
    motion_displacement = _compute_motion_displacement(frame_paths, blur_kernel_size)
    motion_velocity = compute_velocity(motion_displacement, dt_sec)

    rows = []
    for idx, frame_path in enumerate(frame_paths):
        rows.append(
            {
                "frame_idx": int(idx),
                "file_name": Path(frame_path).name,
                "motion_displacement": float(motion_displacement[idx]),
                "motion_velocity": float(motion_velocity[idx]),
            }
        )

    elapsed = float(time.time() - t0)
    log_info("motion energy observe done: frames={} elapsed={:.3f}s".format(len(frame_paths), elapsed))
    meta = {
        "enabled": True,
        "backend": "motion_energy",
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
        "signal_stats": {
            "motion_displacement": series_stats(motion_displacement),
            "motion_velocity": series_stats(motion_velocity),
        },
    }
    return rows, meta
