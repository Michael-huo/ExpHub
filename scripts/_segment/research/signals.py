#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scripts._common import log_err, log_warn

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)


_FEATURE_CFG = {
    "max_corners": 200,
    "quality_level": 0.01,
    "min_distance": 7,
    "block_size": 7,
}


def _read_gray_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("failed to read frame: {}".format(path))
    return img


def _appearance_delta(prev_gray, cur_gray):
    if prev_gray is None:
        return 0.0
    diff = np.abs(cur_gray.astype(np.float32) - prev_gray.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def _brightness_jump(prev_gray, cur_gray):
    if prev_gray is None:
        return 0.0
    prev_mean = float(np.mean(prev_gray)) / 255.0
    cur_mean = float(np.mean(cur_gray)) / 255.0
    return float(abs(cur_mean - prev_mean))


def _blur_score(cur_gray):
    lap = cv2.Laplacian(cur_gray, cv2.CV_32F)
    return float(np.var(lap))


def _feature_motion(prev_gray, cur_gray):
    if prev_gray is None:
        return 0.0

    try:
        points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=int(_FEATURE_CFG["max_corners"]),
            qualityLevel=float(_FEATURE_CFG["quality_level"]),
            minDistance=float(_FEATURE_CFG["min_distance"]),
            blockSize=int(_FEATURE_CFG["block_size"]),
        )
        if points is None or len(points) == 0:
            return 0.0

        points_next, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, points, None)
        if points_next is None or status is None:
            return 0.0

        status = status.reshape(-1)
        if not np.any(status == 1):
            return 0.0

        good_prev = points[status == 1].reshape(-1, 2)
        good_next = points_next[status == 1].reshape(-1, 2)
        if good_prev.shape[0] == 0:
            return 0.0

        disp = np.linalg.norm(good_next - good_prev, axis=1)
        diag = float(math.hypot(cur_gray.shape[0], cur_gray.shape[1]))
        if diag <= 0.0:
            return 0.0
        return float(np.median(disp) / diag)
    except Exception as e:
        log_warn("feature motion fallback to 0 for current pair: {}".format(e))
        return 0.0


def compute_frame_signal_rows(frame_paths, timestamps, semantic_enabled=False):
    rows = []
    prev_gray = None

    for idx, frame_path in enumerate(frame_paths):
        gray = _read_gray_image(frame_path)
        row = {
            "frame_idx": int(idx),
            "ts_sec": float(timestamps[idx]),
            "file_name": Path(frame_path).name,
            "appearance_delta": float(_appearance_delta(prev_gray, gray)),
            "brightness_jump": float(_brightness_jump(prev_gray, gray)),
            "blur_score": float(_blur_score(gray)),
            "feature_motion": float(_feature_motion(prev_gray, gray)),
            "semantic_delta": float(0.0) if not semantic_enabled else None,
        }
        rows.append(row)
        prev_gray = gray

    meta = {
        "enabled_signals": [
            "appearance_delta",
            "brightness_jump",
            "blur_score",
            "feature_motion",
            "semantic_delta",
        ],
        "semantic_enabled": bool(semantic_enabled),
        "feature_motion_method": "goodFeaturesToTrack + calcOpticalFlowPyrLK median displacement / image diagonal",
        "appearance_delta_method": "mean absolute grayscale difference / 255",
        "brightness_jump_method": "absolute mean grayscale brightness change / 255",
        "blur_score_method": "variance of grayscale Laplacian",
    }
    return rows, meta
