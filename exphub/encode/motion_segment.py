from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info


MOTION_LABELS = ("stop", "forward", "left_turn", "right_turn", "mixed")
TURN_LABELS = ("forward", "left_turn", "right_turn")
MAIN_MOTION_LABELS = ("forward", "left_turn", "right_turn")

TRACKING_COLORS = {
    "ok": "#E8F5E9",
    "weak": "#FFF59D",
    "lost": "#FFCDD2",
}
MOTION_COLORS = {
    "stop": "#ECEFF1",
    "forward": "#E8E3F4",
    "left_turn": "#D9EEF7",
    "right_turn": "#F6DEC9",
    "mixed": "#E5E7EA",
}
TRACKING_LINE_COLOR = "#7A3E9D"
PC_LINE_COLOR = "#1F77B4"
PC_GRID_ROWS = 4
PC_GRID_COLS = 6
PC_MIN_BLOCK_SIZE = 24
PC_EPS = 1e-6


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _time_at(prepare_result, idx):
    values = list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])
    if idx < 0 or idx >= len(values):
        raise RuntimeError("prepare_result frame_index_map missing abs time for frame {}".format(int(idx)))
    return float(values[int(idx)])


def _read_gray(path, max_width=416):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("failed to read prepared frame: {}".format(path))
    h, w = image.shape[:2]
    if w > int(max_width):
        scale = float(max_width) / float(w)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _read_rgb(path, max_width=416):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read prepared frame: {}".format(path))
    h, w = image.shape[:2]
    if w > int(max_width):
        scale = float(max_width) / float(w)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _moving_average(values, k):
    arr = np.asarray(values, dtype=np.float32)
    if k <= 1 or arr.size == 0:
        return arr
    pad = int(k) // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(int(k), dtype=np.float32) / float(k)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _median_filter(values, k):
    arr = np.asarray(values, dtype=np.float32)
    if k <= 1 or arr.size == 0:
        return arr
    if int(k) % 2 == 0:
        k += 1
    pad = int(k) // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)
    for idx in range(arr.size):
        out[idx] = float(np.median(padded[idx : idx + int(k)]))
    return out


def _smooth(values, median_k=5, mean_k=9):
    return _moving_average(_median_filter(values, median_k), mean_k)


def _normalize01(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _robust_scale(values, eps=1e-6):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    p90 = float(np.percentile(np.abs(finite), 90))
    if p90 > eps:
        return p90
    return max(float(np.max(np.abs(finite))), eps)


def _weighted_mean(values, weights, default=0.0):
    arr = np.asarray(values, dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    if arr.size == 0 or weight_arr.size == 0:
        return float(default)
    weight_sum = float(np.sum(weight_arr))
    if weight_sum <= PC_EPS:
        return float(default)
    return float(np.sum(arr * weight_arr) / weight_sum)


def _weighted_percentile(values, weights, percentile, default=1.0):
    arr = np.asarray(values, dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    mask = np.isfinite(arr) & np.isfinite(weight_arr) & (weight_arr > 0.0)
    if not np.any(mask):
        return float(default)
    arr = arr[mask]
    weight_arr = weight_arr[mask]
    order = np.argsort(arr)
    arr = arr[order]
    weight_arr = weight_arr[order]
    cumulative = np.cumsum(weight_arr)
    cutoff = float(percentile) / 100.0 * float(cumulative[-1])
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = int(np.clip(idx, 0, len(arr) - 1))
    return float(arr[idx])


def _regions(mask):
    items = np.asarray(mask, dtype=bool)
    out = []
    idx = 0
    while idx < len(items):
        if not bool(items[idx]):
            idx += 1
            continue
        start = idx
        while idx + 1 < len(items) and bool(items[idx + 1]):
            idx += 1
        out.append((int(start), int(idx)))
        idx += 1
    return out


def _fill_gaps(mask, max_gap):
    arr = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or arr.size == 0:
        return arr
    idx = 0
    while idx < len(arr):
        if bool(arr[idx]):
            idx += 1
            continue
        start = idx
        while idx + 1 < len(arr) and not bool(arr[idx + 1]):
            idx += 1
        end = idx
        if start > 0 and end + 1 < len(arr) and bool(arr[start - 1]) and bool(arr[end + 1]):
            if end - start + 1 <= int(max_gap):
                arr[start : end + 1] = True
        idx += 1
    return arr


def _orb_raw(prev_gray, curr_gray, orb, matcher):
    import cv2

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    num_kp_prev = 0 if kp1 is None else len(kp1)
    num_kp_curr = 0 if kp2 is None else len(kp2)
    raw_matches = []
    good_matches = []
    if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    num_good = len(good_matches)
    num_inliers = 0
    inlier_ratio = 0.0
    median_disp = 0.0
    affine_ok = 0
    tx = 0.0
    ty = 0.0
    rot_deg = 0.0
    if num_good > 0:
        pts1 = np.float32([kp1[item.queryIdx].pt for item in good_matches])
        pts2 = np.float32([kp2[item.trainIdx].pt for item in good_matches])
        disp = np.linalg.norm(pts2 - pts1, axis=1)
        median_disp = float(np.median(disp))
        if num_good >= 12:
            matrix, inliers = cv2.estimateAffinePartial2D(
                pts1,
                pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if matrix is not None and inliers is not None:
                affine_ok = 1
                inlier_mask = inliers.ravel().astype(bool)
                num_inliers = int(np.sum(inlier_mask))
                inlier_ratio = float(num_inliers) / float(max(1, num_good))
                tx = float(matrix[0, 2])
                ty = float(matrix[1, 2])
                rot_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))

    return {
        "num_kp_prev": int(num_kp_prev),
        "num_kp_curr": int(num_kp_curr),
        "num_raw_matches": int(len(raw_matches)),
        "num_good_matches": int(num_good),
        "num_inliers": int(num_inliers),
        "inlier_ratio": float(inlier_ratio),
        "median_disp": float(median_disp),
        "affine_ok": int(affine_ok),
        "tx": float(tx),
        "ty": float(ty),
        "rot_deg": float(rot_deg),
    }


def _tracking_quality(row):
    good_s = min(float(row.get("num_good_matches", 0)) / 80.0, 1.0)
    inlier_s = min(float(row.get("num_inliers", 0)) / 40.0, 1.0)
    ratio = float(row.get("inlier_ratio", 0.0) or 0.0)
    ratio_s = float(np.clip((ratio - 0.25) / 0.45, 0.0, 1.0))
    pose_s = 1.0 if int(row.get("affine_ok", 0) or 0) else 0.0
    return float(0.30 * good_s + 0.35 * inlier_s + 0.20 * ratio_s + 0.15 * pose_s)


def _tracking_states(qualities):
    states = []
    current = "ok"
    bad_count = 0
    good_count = 0
    for value in qualities:
        quality = float(value)
        if current != "lost":
            if quality <= 0.45:
                bad_count += 1
            else:
                bad_count = 0
            if bad_count >= 5:
                current = "lost"
                bad_count = 0
                good_count = 0
        else:
            if quality >= 0.55:
                good_count += 1
            else:
                good_count = 0
            if good_count >= 5:
                current = "ok"
                good_count = 0
                bad_count = 0
        states.append("lost" if current == "lost" else ("weak" if quality < 0.60 else "ok"))
    return states


def _orb_pairs(grays):
    import cv2

    pair_count = max(0, len(grays) - 1)
    if pair_count <= 0:
        return []
    orb = cv2.ORB_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_rows = [_orb_raw(grays[idx], grays[idx + 1], orb, matcher) for idx in range(pair_count)]
    quality = _smooth([_tracking_quality(row) for row in raw_rows], 5, 9)
    tx = _smooth([row.get("tx", 0.0) for row in raw_rows], 3, 5)
    ty = _smooth([row.get("ty", 0.0) for row in raw_rows], 3, 5)
    rot = _smooth([row.get("rot_deg", 0.0) for row in raw_rows], 3, 5)
    states = _tracking_states(quality)

    out = []
    for idx, row in enumerate(raw_rows):
        item = dict(row)
        item.update(
            {
                "pair_index": int(idx),
                "frame_start_idx": int(idx),
                "frame_end_idx": int(idx + 1),
                "tracking_quality": float(quality[idx]),
                "tracking_state": str(states[idx]),
                "tx": float(tx[idx]),
                "ty": float(ty[idx]),
                "rot_deg": float(rot[idx]),
            }
        )
        out.append(item)
    return out


def _loss_intervals(orb_rows):
    if not orb_rows:
        return []
    health = np.asarray([float(row.get("tracking_quality", 0.0)) for row in orb_rows], dtype=np.float32)
    tracking = [str(row.get("tracking_state", "ok") or "ok") for row in orb_rows]
    core_mask = np.asarray([state == "lost" for state in tracking], dtype=bool)
    unreliable = np.asarray(
        [(state != "ok") or (quality < 0.55) for state, quality in zip(tracking, health)],
        dtype=bool,
    )
    if np.any(unreliable):
        unreliable = _fill_gaps(unreliable, 4)
        kernel = np.ones(25, dtype=np.int32)
        unreliable = np.convolve(unreliable.astype(np.int32), kernel, mode="same") > 0

    regions = _regions(unreliable)
    core_regions = _regions(core_mask)
    intervals = []
    for core_start, core_end in core_regions:
        chosen = None
        for start, end in regions:
            if not (end < core_start or start > core_end):
                chosen = (start, end)
                break
        start, end = chosen if chosen is not None else (core_start, core_end)
        start = max(0, int(start) - 8)
        end = min(len(orb_rows) - 1, int(end) + 12)
        if end - start + 1 < 20:
            pad = max(0, (20 - (end - start + 1) + 1) // 2)
            start = max(0, start - pad)
            end = min(len(orb_rows) - 1, end + pad)
        intervals.append(
            {
                "loss_interval_id": int(len(intervals)),
                "pair_start": int(start),
                "pair_end": int(end),
                "start_idx": int(start),
                "end_idx": int(end + 1),
                "core_pair_start": int(core_start),
                "core_pair_end": int(core_end),
                "core_start_idx": int(core_start),
                "core_end_idx": int(core_end + 1),
                "min_tracking_quality": float(np.min(health[start : end + 1])),
            }
        )

    if not intervals:
        return []
    merged = [intervals[0]]
    for item in intervals[1:]:
        prev = merged[-1]
        if int(item["pair_start"]) <= int(prev["pair_end"]) + 1:
            prev["pair_end"] = max(int(prev["pair_end"]), int(item["pair_end"]))
            prev["end_idx"] = int(prev["pair_end"]) + 1
            prev["core_pair_start"] = min(int(prev["core_pair_start"]), int(item["core_pair_start"]))
            prev["core_pair_end"] = max(int(prev["core_pair_end"]), int(item["core_pair_end"]))
            prev["core_start_idx"] = int(prev["core_pair_start"])
            prev["core_end_idx"] = int(prev["core_pair_end"]) + 1
            prev["min_tracking_quality"] = min(float(prev["min_tracking_quality"]), float(item["min_tracking_quality"]))
        else:
            merged.append(item)
    for idx, item in enumerate(merged):
        item["loss_interval_id"] = int(idx)
    return merged


def _pc_empty_result(block_count=0):
    return {
        "yaw_coeff": 0.0,
        "forward_x_coeff": 0.0,
        "vertical_coeff": 0.0,
        "forward_y_coeff": 0.0,
        "expansion_coeff": 0.0,
        "same_sign_consistency": 0.0,
        "expansion_consistency": 0.0,
        "motion_intensity": 0.0,
        "state_confidence": 0.0,
        "block_count": int(block_count),
        "reliable_block_count": 0,
        "_magnitude_samples": [],
        "_magnitude_weights": [],
    }


def _pc_row_weight(row, grid_rows):
    row = int(row)
    grid_rows = max(int(grid_rows), 1)
    if row >= grid_rows - 1:
        return 0.70
    return 1.0


def _pc_block_records(prev_gray, curr_gray, grid_rows=PC_GRID_ROWS, grid_cols=PC_GRID_COLS, min_block_size=PC_MIN_BLOCK_SIZE):
    import cv2

    h, w = prev_gray.shape
    grid_rows = int(grid_rows)
    grid_cols = int(grid_cols)
    block_h = h // grid_rows
    block_w = w // int(grid_cols)
    if block_h < min_block_size or block_w < min_block_size:
        return []

    records = []
    hann = cv2.createHanningWindow((block_w, block_h), cv2.CV_32F)
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * block_h
            x0 = col * block_w
            p = prev_gray[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            q = curr_gray[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            texture = float(np.std(p))
            (dx, dy), response = cv2.phaseCorrelate(p, q, hann)
            if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(response)):
                dx, dy, response = 0.0, 0.0, 0.0
            response = max(0.0, float(response))
            texture_factor = float(np.clip((texture - 1.0) / 14.0, 0.0, 1.0))
            response_factor = float(np.clip(response, 0.0, 1.0))
            reliability = float(response_factor * texture_factor)
            row_weight = float(_pc_row_weight(row, grid_rows))
            weight = float(reliability * row_weight)
            center_x = float(x0) + 0.5 * float(block_w)
            center_y = float(y0) + 0.5 * float(block_h)
            x_pos = (center_x - 0.5 * float(w)) / max(0.5 * float(w), 1.0)
            y_pos = (center_y - 0.5 * float(h)) / max(0.5 * float(h), 1.0)
            records.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "x0": int(x0),
                    "y0": int(y0),
                    "bw": int(block_w),
                    "bh": int(block_h),
                    "x_pos": float(x_pos),
                    "y_pos": float(y_pos),
                    "dx": float(dx),
                    "dy": float(dy),
                    "response": float(response),
                    "texture": float(texture),
                    "reliability": float(reliability),
                    "weight": float(weight),
                }
            )
    return records


def _pc_block(prev_gray, curr_gray, grid_rows=PC_GRID_ROWS, grid_cols=PC_GRID_COLS, min_block_size=PC_MIN_BLOCK_SIZE):
    records = _pc_block_records(prev_gray, curr_gray, grid_rows, grid_cols, min_block_size)

    if not records:
        return _pc_empty_result(0)

    dxs = np.asarray([item["dx"] for item in records], dtype=np.float32)
    dys = np.asarray([item["dy"] for item in records], dtype=np.float32)
    responses = np.asarray([item["response"] for item in records], dtype=np.float32)
    weights = np.asarray([item["weight"] for item in records], dtype=np.float32)
    x_pos = np.asarray([item["x_pos"] for item in records], dtype=np.float32)
    y_pos = np.asarray([item["y_pos"] for item in records], dtype=np.float32)
    magnitudes = np.sqrt(dxs * dxs + dys * dys).astype(np.float32)
    weight_sum = float(np.sum(weights))
    reliable_count = int(np.sum(weights > PC_EPS))
    if weight_sum <= PC_EPS:
        result = _pc_empty_result(len(records))
        result["_magnitude_samples"] = [float(item) for item in magnitudes]
        result["_magnitude_weights"] = [1.0 for _ in records]
        return result

    x_center = _weighted_mean(x_pos, weights)
    y_center = _weighted_mean(y_pos, weights)
    x_c = x_pos - float(x_center)
    y_c = y_pos - float(y_center)
    forward_x = float(np.sum(weights * x_c * dxs) / (np.sum(weights * x_c * x_c) + PC_EPS))
    forward_y = float(np.sum(weights * y_c * dys) / (np.sum(weights * y_c * y_c) + PC_EPS))
    residual_dx = dxs - float(forward_x) * x_c
    yaw = _weighted_mean(residual_dx, weights)
    vertical = _weighted_mean(dys - float(forward_y) * y_c, weights)
    radial_projection = x_c * dxs + y_c * dys
    radial_norm = x_c * x_c + y_c * y_c
    expansion = float(np.sum(weights * radial_projection) / (np.sum(weights * radial_norm) + PC_EPS))
    expansion_consistency = float(np.sum(weights[radial_projection > 0.0]) / (weight_sum + PC_EPS))
    same_pos = float(np.sum(weights[residual_dx >= 0.0]))
    same_neg = float(np.sum(weights[residual_dx < 0.0]))
    same_sign = float(max(same_pos, same_neg) / (weight_sum + PC_EPS))
    return {
        "yaw_coeff": float(yaw),
        "forward_x_coeff": float(forward_x),
        "vertical_coeff": float(vertical),
        "forward_y_coeff": float(forward_y),
        "expansion_coeff": float(expansion),
        "same_sign_consistency": float(np.clip(same_sign, 0.0, 1.0)),
        "expansion_consistency": float(np.clip(expansion_consistency, 0.0, 1.0)),
        "motion_intensity": float(np.sum(magnitudes * weights) / (weight_sum + PC_EPS)),
        "state_confidence": float(np.clip(np.mean(np.maximum(responses, 0.0)), 0.0, 1.0)),
        "block_count": int(len(records)),
        "reliable_block_count": int(reliable_count),
        "_magnitude_samples": [float(item) for item in magnitudes],
        "_magnitude_weights": [float(max(item["weight"], PC_EPS)) for item in records],
    }


def _pc_evidence(grays):
    pair_count = max(0, len(grays) - 1)
    if pair_count <= 0:
        return []
    raw_rows = [_pc_block(grays[idx], grays[idx + 1]) for idx in range(pair_count)]
    magnitude_samples = []
    magnitude_weights = []
    for row in raw_rows:
        magnitude_samples.extend(list(row.get("_magnitude_samples") or []))
        magnitude_weights.extend(list(row.get("_magnitude_weights") or []))
    robust_scale = max(_weighted_percentile(magnitude_samples, magnitude_weights, 75, default=1.0), PC_EPS)
    motion = _smooth([row["motion_intensity"] for row in raw_rows], 5, 9)
    yaw = _smooth([row["yaw_coeff"] for row in raw_rows], 5, 9)
    forward_x = _smooth([row["forward_x_coeff"] for row in raw_rows], 5, 9)
    vertical = _smooth([row["vertical_coeff"] for row in raw_rows], 5, 9)
    forward_y = _smooth([row["forward_y_coeff"] for row in raw_rows], 5, 9)
    expansion = _smooth([row["expansion_coeff"] for row in raw_rows], 5, 9)
    same_sign = np.clip(_smooth([row["same_sign_consistency"] for row in raw_rows], 3, 5), 0.0, 1.0)
    expansion_consistency = np.clip(_smooth([row["expansion_consistency"] for row in raw_rows], 3, 5), 0.0, 1.0)
    confidence = _smooth([row["state_confidence"] for row in raw_rows], 5, 9)
    out = []
    for idx, row in enumerate(raw_rows):
        yaw_norm = abs(float(yaw[idx])) / robust_scale
        expansion_norm = max(float(expansion[idx]), 0.0) / robust_scale
        turn_score = float(yaw_norm * float(same_sign[idx]))
        forward_score = float(expansion_norm * float(expansion_consistency[idx]))
        left_score = float(turn_score if float(yaw[idx]) > 0.0 else 0.0)
        right_score = float(turn_score if float(yaw[idx]) < 0.0 else 0.0)
        signed_yaw_score = float(math.copysign(yaw_norm, float(yaw[idx]))) if abs(float(yaw[idx])) > PC_EPS else 0.0
        out.append(
            {
                "pair_index": int(idx),
                "frame_start_idx": int(idx),
                "frame_end_idx": int(idx + 1),
                "yaw_coeff": float(yaw[idx]),
                "forward_x_coeff": float(forward_x[idx]),
                "vertical_coeff": float(vertical[idx]),
                "forward_y_coeff": float(forward_y[idx]),
                "expansion_coeff": float(expansion[idx]),
                "same_sign_consistency": float(same_sign[idx]),
                "expansion_consistency": float(expansion_consistency[idx]),
                "yaw_norm": float(yaw_norm),
                "expansion_norm": float(expansion_norm),
                "turn_score": float(turn_score),
                "forward_score": float(forward_score),
                "left_score": float(left_score),
                "right_score": float(right_score),
                "steering_response": float(signed_yaw_score),
                "forward_response": float(forward_score),
                "motion_intensity": float(motion[idx]),
                "state_confidence": float(confidence[idx]),
                "robust_scale": float(robust_scale),
                "block_count": int(row.get("block_count", 0)),
                "reliable_block_count": int(row.get("reliable_block_count", 0)),
                "motion_state": "forward",
            }
        )
    return out


def _in_loss_mask(pair_count, loss_intervals):
    mask = np.zeros(int(pair_count), dtype=bool)
    for interval in list(loss_intervals or []):
        start = max(0, int(interval.get("pair_start", 0) or 0))
        end = min(int(pair_count) - 1, int(interval.get("pair_end", -1) or -1))
        if end >= start:
            mask[start : end + 1] = True
    return mask


def _run_count(labels):
    values = list(labels or [])
    if not values:
        return 0
    count = 1
    for prev, cur in zip(values[:-1], values[1:]):
        if str(prev) != str(cur):
            count += 1
    return int(count)


def _turn_label(steering):
    # PC convention validated against the reference script: positive horizontal steering means left_turn.
    return "left_turn" if float(steering) > 0.0 else "right_turn"


def _estimate_states(pc_rows, loss_intervals):
    del loss_intervals
    if not pc_rows:
        return []
    motion = np.asarray([float(row["motion_intensity"]) for row in pc_rows], dtype=np.float32)
    m_scale = _robust_scale(motion)

    out = []
    for idx, row in enumerate(pc_rows):
        item = dict(row)
        left_score = float(item.get("left_score", 0.0) or 0.0)
        right_score = float(item.get("right_score", 0.0) or 0.0)
        forward_score = float(item.get("forward_score", 0.0) or 0.0)
        turn_score = max(left_score, right_score)
        candidate = "forward"
        if turn_score > max(forward_score * 1.4, 0.10) and float(item.get("same_sign_consistency", 0.0) or 0.0) >= 0.62:
            candidate = "left_turn" if left_score >= right_score else "right_turn"
        item["raw_motion_state"] = str(candidate)
        item["steering_strength"] = float(turn_score)
        item["forward_strength"] = float(forward_score)
        item["motion_strength"] = float(motion[idx]) / max(float(m_scale), PC_EPS)
        item["motion_state"] = str(candidate)
        out.append(item)
    return out


def _dominant_label(scores):
    scores = _as_dict(scores)
    if not scores:
        return "forward"
    left_score = float(scores.get("left_score", 0.0) or 0.0)
    right_score = float(scores.get("right_score", 0.0) or 0.0)
    forward_score = float(scores.get("forward_score", 0.0) or 0.0)
    same_sign = float(scores.get("same_sign_consistency", 0.0) or 0.0)
    ok_fraction = float(scores.get("ok_pair_fraction", 0.0) or 0.0)
    lost_fraction = float(scores.get("lost_pair_fraction", 0.0) or 0.0)
    turn_score = max(left_score, right_score)
    turn_label = "left_turn" if left_score >= right_score else "right_turn"
    if ok_fraction >= 0.67 and lost_fraction <= 0.05:
        same_min = 0.70
        score_min = 0.12
        dominance = 1.60
    else:
        same_min = 0.60
        score_min = 0.10
        dominance = 1.25
    if same_sign >= same_min and turn_score >= score_min and turn_score >= dominance * max(forward_score, PC_EPS):
        return turn_label
    return "forward"


def _mean_score(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _window_scores(pc_rows, orb_rows, scales=None):
    del scales
    tracking_values = [float(row.get("tracking_quality", 0.0) or 0.0) for row in orb_rows]
    ok_count = len([row for row in orb_rows if str(row.get("tracking_state", "") or "") == "ok"])
    weak_count = len([row for row in orb_rows if str(row.get("tracking_state", "") or "") == "weak"])
    lost_count = len([row for row in orb_rows if str(row.get("tracking_state", "") or "") == "lost"])
    total = max(1, len(orb_rows))
    has_tracking = bool(orb_rows)
    motion = _mean_score(pc_rows, "motion_intensity")
    steering = _mean_score(pc_rows, "steering_response")
    forward = _mean_score(pc_rows, "forward_response")
    confidence = _mean_score(pc_rows, "state_confidence")
    yaw = _mean_score(pc_rows, "yaw_coeff")
    expansion = _mean_score(pc_rows, "expansion_coeff")
    yaw_norm = _mean_score(pc_rows, "yaw_norm")
    expansion_norm = _mean_score(pc_rows, "expansion_norm")
    same_sign = _mean_score(pc_rows, "same_sign_consistency")
    expansion_consistency = _mean_score(pc_rows, "expansion_consistency")
    left_score = _mean_score(pc_rows, "left_score")
    right_score = _mean_score(pc_rows, "right_score")
    turn_score = max(float(left_score), float(right_score))
    forward_score = _mean_score(pc_rows, "forward_score")
    return {
        "translational": float(motion),
        "directional": float(steering),
        "confidence": float(confidence),
        "motion_intensity": float(motion),
        "steering_response": float(steering),
        "forward_response": float(forward),
        "yaw_coeff": float(yaw),
        "expansion_coeff": float(expansion),
        "yaw_norm": float(yaw_norm),
        "expansion_norm": float(expansion_norm),
        "same_sign_consistency": float(same_sign),
        "expansion_consistency": float(expansion_consistency),
        "left_score": float(left_score),
        "right_score": float(right_score),
        "turn_score": float(turn_score),
        "forward_score": float(forward_score),
        "motion_strength": float(_mean_score(pc_rows, "motion_strength")),
        "steering_strength": float(turn_score),
        "forward_strength": float(forward_score),
        "steering_sign_fraction": float(same_sign),
        "tracking_quality": float(np.mean(tracking_values)) if tracking_values else 0.0,
        "ok_pair_fraction": (float(ok_count) / float(total)) if has_tracking else 1.0,
        "weak_pair_fraction": (float(weak_count) / float(total)) if has_tracking else 0.0,
        "lost_pair_fraction": (float(lost_count) / float(total)) if has_tracking else 0.0,
    }


def _project_legal(pc_rows, orb_rows, loss_intervals, legal_positions):
    loss_mask = _in_loss_mask(len(pc_rows), loss_intervals)
    windows = []
    for win_idx in range(len(legal_positions) - 1):
        start_idx = int(legal_positions[win_idx])
        end_idx = int(legal_positions[win_idx + 1])
        pair_start = max(0, start_idx)
        pair_end = min(len(pc_rows) - 1, end_idx - 1)
        pc_slice = pc_rows[pair_start : pair_end + 1] if pair_end >= pair_start else []
        orb_slice = orb_rows[pair_start : pair_end + 1] if pair_end >= pair_start else []
        loss_overlap = bool(np.any(loss_mask[pair_start : pair_end + 1])) if pair_end >= pair_start else False
        scores = _window_scores(pc_slice, orb_slice)
        scores["loss_overlap"] = bool(loss_overlap)
        label = _dominant_label(scores)
        windows.append(
            {
                "window_index": int(win_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "pair_start": int(pair_start),
                "pair_end": int(pair_end),
                "motion_label": str(label),
                "scores": scores,
            }
        )

    if len(windows) >= 3:
        labels = [str(item["motion_label"]) for item in windows]
        for idx in range(1, len(windows) - 1):
            if labels[idx - 1] == labels[idx + 1] and labels[idx] != labels[idx - 1]:
                confidence = float(_as_dict(windows[idx].get("scores")).get("confidence", 0.0) or 0.0)
                if confidence < 0.78:
                    windows[idx]["motion_label"] = labels[idx - 1]
                    scores = dict(_as_dict(windows[idx].get("scores")))
                    scores["stabilized_from"] = str(labels[idx])
                    scores["confidence"] = float(max(confidence, 0.68))
                    windows[idx]["scores"] = scores
    return windows


def _window_runs(windows):
    runs = []
    idx = 0
    items = list(windows or [])
    while idx < len(items):
        label = str(items[idx].get("motion_label", "forward") or "forward")
        start = idx
        while idx + 1 < len(items) and str(items[idx + 1].get("motion_label", "forward") or "forward") == label:
            idx += 1
        runs.append({"start": int(start), "end": int(idx), "motion_label": str(label)})
        idx += 1
    return runs


def _run_windows(windows, run):
    return list(windows[int(run["start"]) : int(run["end"]) + 1])


def _run_frame_len(windows, run):
    rows = _run_windows(windows, run)
    if not rows:
        return 0
    return int(rows[-1]["end_idx"]) - int(rows[0]["start_idx"])


def _run_score(rows, key):
    values = [float(_as_dict(row.get("scores")).get(key, 0.0) or 0.0) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _label_response(label, rows):
    label = str(label or "forward")
    if label == "forward":
        return _run_score(rows, "forward_strength") + 0.25 * _run_score(rows, "motion_strength")
    if label in ("left_turn", "right_turn"):
        return _run_score(rows, "steering_strength") + 0.15 * _run_score(rows, "motion_strength")
    return 0.0


def _strong_turn_run(windows, run):
    label = str(run.get("motion_label", "") or "")
    if label not in ("left_turn", "right_turn"):
        return False
    rows = _run_windows(windows, run)
    steering_strength = _run_score(rows, "steering_strength")
    forward_strength = _run_score(rows, "forward_strength")
    sign_fraction = _run_score(rows, "steering_sign_fraction")
    return bool(steering_strength >= 0.12 and sign_fraction >= 0.65 and steering_strength >= 1.25 * max(forward_strength, PC_EPS))


def _neighbor_choice(windows, run, left_run, right_run):
    rows = _run_windows(windows, run)
    left_rows = _run_windows(windows, left_run) if left_run is not None else []
    right_rows = _run_windows(windows, right_run) if right_run is not None else []
    left_label = str(left_run.get("motion_label", "forward") if left_run is not None else "forward")
    right_label = str(right_run.get("motion_label", "forward") if right_run is not None else "forward")
    left_len = float(_run_frame_len(windows, left_run)) if left_run is not None else 0.0
    right_len = float(_run_frame_len(windows, right_run)) if right_run is not None else 0.0
    left_score = _label_response(left_label, left_rows + rows) + 0.01 * left_len
    right_score = _label_response(right_label, right_rows + rows) + 0.01 * right_len
    return left_label if left_score >= right_score else right_label


def _set_window_label(windows, start, end, label, reason):
    for idx in range(int(start), int(end) + 1):
        windows[idx]["motion_label"] = str(label)
        scores = dict(_as_dict(windows[idx].get("scores")))
        scores["consolidated_from"] = str(scores.get("consolidated_from", "") or scores.get("stabilized_from", "") or "")
        if not scores["consolidated_from"]:
            scores["consolidated_from"] = str(reason)
        else:
            scores["consolidated_from"] = "{},{}".format(scores["consolidated_from"], str(reason))
        windows[idx]["scores"] = scores


def _consolidate_windows(windows, fps, num_frames):
    items = [dict(item, scores=dict(_as_dict(item.get("scores")))) for item in list(windows or [])]
    if len(items) <= 1:
        return items, {"min_segment_sec": 0.0, "strong_turn_short_segment_count": 0}

    fps_i = max(int(fps or 0), 1)
    clip_duration_sec = float(max(0, int(num_frames) - 1)) / float(fps_i)
    min_segment_sec = min(1.5, max(0.0, clip_duration_sec * 0.35))
    min_segment_frames = int(round(min_segment_sec * float(fps_i)))

    for _ in range(4):
        changed = False
        runs = _window_runs(items)
        for run_idx, run in enumerate(runs):
            label = str(run.get("motion_label", "forward") or "forward")
            if label not in MAIN_MOTION_LABELS:
                label = "forward"
                _set_window_label(items, run["start"], run["end"], label, "unsupported_final_label")
                changed = True

            is_boundary = run_idx == 0 or run_idx == len(runs) - 1
            frame_len = _run_frame_len(items, run)
            is_short = frame_len < int(min_segment_frames)
            strong_turn = _strong_turn_run(items, run)
            if is_boundary or not is_short or strong_turn:
                continue

            left_run = runs[run_idx - 1] if run_idx > 0 else None
            right_run = runs[run_idx + 1] if run_idx + 1 < len(runs) else None
            if left_run is None and right_run is None:
                continue
            left_label = str(left_run.get("motion_label", "forward") if left_run is not None else "forward")
            right_label = str(right_run.get("motion_label", "forward") if right_run is not None else "forward")
            if left_label == right_label:
                new_label = left_label
            elif label in ("left_turn", "right_turn") and not strong_turn:
                new_label = "forward" if "forward" in (left_label, right_label) else _neighbor_choice(items, run, left_run, right_run)
            else:
                new_label = _neighbor_choice(items, run, left_run, right_run)
            if new_label not in MAIN_MOTION_LABELS:
                new_label = "forward"
            _set_window_label(items, run["start"], run["end"], new_label, "short_segment")
            changed = True
            break
        if not changed:
            break

    strong_turn_short_count = len(
        [
            run
            for run in _window_runs(items)
            if _run_frame_len(items, run) < int(min_segment_frames) and _strong_turn_run(items, run)
        ]
    )
    return items, {
        "min_segment_sec": float(min_segment_sec),
        "strong_turn_short_segment_count": int(strong_turn_short_count),
    }


def _finish_motion_state(current, score_rows, motion_states, prepare_result):
    keys = [
        "translational",
        "directional",
        "confidence",
        "motion_intensity",
        "steering_response",
        "forward_response",
        "yaw_coeff",
        "expansion_coeff",
        "yaw_norm",
        "expansion_norm",
        "same_sign_consistency",
        "expansion_consistency",
        "left_score",
        "right_score",
        "turn_score",
        "forward_score",
        "motion_strength",
        "steering_strength",
        "forward_strength",
        "steering_sign_fraction",
        "tracking_quality",
        "ok_pair_fraction",
        "weak_pair_fraction",
        "lost_pair_fraction",
    ]
    scores = {}
    for key in keys:
        scores[key] = float(np.mean([float(row.get(key, 0.0) or 0.0) for row in score_rows])) if score_rows else 0.0
    stabilized = [str(row.get("stabilized_from", "") or "") for row in score_rows if str(row.get("stabilized_from", "") or "")]
    if stabilized:
        scores["stabilized_from"] = ",".join(sorted(set(stabilized)))
    motion_states.append(
        {
            "motion_state_id": "",
            "start_idx": int(current["start_idx"]),
            "end_idx": int(current["end_idx"]),
            "motion_state_start": int(current["start_idx"]),
            "motion_state_end": int(current["end_idx"]),
            "start_abs_time_sec": _time_at(prepare_result, int(current["start_idx"])),
            "end_abs_time_sec": _time_at(prepare_result, int(current["end_idx"])),
            "motion_label": str(current["motion_label"]),
            "scores": scores,
        }
    )


def _merge_windows(windows, prepare_result):
    motion_states = []
    current = None
    current_scores = []
    for window in windows:
        label = str(window.get("motion_label", "mixed") or "mixed")
        if current is None or label != current["motion_label"]:
            if current is not None:
                _finish_motion_state(current, current_scores, motion_states, prepare_result)
            current = {
                "start_idx": int(window["start_idx"]),
                "end_idx": int(window["end_idx"]),
                "motion_label": str(label),
            }
            current_scores = [dict(_as_dict(window.get("scores")))]
        else:
            current["end_idx"] = int(window["end_idx"])
            current_scores.append(dict(_as_dict(window.get("scores"))))
    if current is not None:
        _finish_motion_state(current, current_scores, motion_states, prepare_result)
    for idx, item in enumerate(motion_states):
        item["motion_state_id"] = "motion_{:04d}".format(int(idx))
    return motion_states


def _sample_indices(start, end, max_frames=8):
    start = int(max(0, start))
    end = int(max(start, end))
    count = min(int(max_frames), end - start + 1)
    if count <= 0:
        return [start]
    return np.linspace(start, end, num=count, dtype=int).tolist()


def _resize_pad(image, target_w, target_h, pad_value=255):
    import cv2

    if image is None or image.size == 0:
        return np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(float(target_w) / max(1, w), float(target_h) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _resize_direct(image, target_w, target_h, pad_value=255):
    import cv2

    if image is None or image.size == 0:
        return np.full((int(target_h), int(target_w), 3), pad_value, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.resize(arr, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)


def _pc_details(prev_gray, curr_gray):
    return [item for item in _pc_block_records(prev_gray, curr_gray) if float(item.get("weight", 0.0) or 0.0) > PC_EPS]


def _pc_thumb(grays, pair_index, thumb_w=160, thumb_h=100):
    import cv2

    pair_index = int(np.clip(pair_index, 0, max(0, len(grays) - 2)))
    prev_gray = grays[pair_index]
    curr_gray = grays[pair_index + 1]
    h, w = prev_gray.shape
    canvas = np.full((thumb_h, thumb_w, 3), 246, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (thumb_w - 1, thumb_h - 1), (190, 196, 203), 1, cv2.LINE_AA)
    for row in range(PC_GRID_ROWS + 1):
        y = int(round(row * thumb_h / float(PC_GRID_ROWS)))
        cv2.line(canvas, (0, min(thumb_h - 1, y)), (thumb_w - 1, min(thumb_h - 1, y)), (205, 211, 218), 1, cv2.LINE_AA)
    for col in range(PC_GRID_COLS + 1):
        x = int(round(col * thumb_w / float(PC_GRID_COLS)))
        cv2.line(canvas, (min(thumb_w - 1, x), 0), (min(thumb_w - 1, x), thumb_h - 1), (205, 211, 218), 1, cv2.LINE_AA)
    for item in _pc_details(prev_gray, curr_gray):
        x1 = int(round(float(item["x0"]) / max(1, w) * thumb_w))
        x2 = int(round(float(item["x0"] + item["bw"]) / max(1, w) * thumb_w))
        y1 = int(round(float(item["y0"]) / max(1, h) * thumb_h))
        y2 = int(round(float(item["y0"] + item["bh"]) / max(1, h) * thumb_h))
        x1, x2 = int(np.clip(x1, 0, thumb_w - 1)), int(np.clip(x2, x1 + 1, thumb_w))
        y1, y2 = int(np.clip(y1, 0, thumb_h - 1)), int(np.clip(y2, y1 + 1, thumb_h))
        dx = float(item["dx"])
        dy = float(item["dy"])
        base = np.asarray((226, 128, 92), dtype=np.float32) if dx >= 0 else np.asarray((84, 156, 203), dtype=np.float32)
        strength = float(np.clip(0.24 + 0.16 * float(item.get("reliability", 0.0) or 0.0) + 0.10 * abs(dx), 0.20, 0.82))
        fill = np.clip((1.0 - strength) * 246.0 + strength * base, 0, 255).astype(np.uint8)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), tuple(int(v) for v in fill), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (228, 232, 236), 1, cv2.LINE_AA)
        cx = int(round((x1 + x2 - 1) / 2.0))
        cy = int(round((y1 + y2 - 1) / 2.0))
        scale = min(11.0, max(2.8, 1.85 * math.sqrt(dx * dx + dy * dy) + 2.0))
        ex = int(np.clip(cx + dx * scale, x1 + 5, x2 - 5))
        ey = int(np.clip(cy + dy * scale, y1 + 5, y2 - 5))
        color = (156, 58, 36) if dx >= 0 else (20, 91, 142)
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.34)
    return canvas


def _sample_strip(frames, grays, start, end):
    import cv2

    indices = _sample_indices(start, end, 8)
    return _sample_strip_indices(frames, grays, indices)


def _sample_strip_indices(frames, grays, indices):
    import cv2

    indices = [int(idx) for idx in list(indices or [])]
    if not indices:
        indices = [0]
    thumb_w = 160
    thumb_h = 100
    gap_x = 12
    gap_y = 12
    total_w = len(indices) * thumb_w + max(0, len(indices) - 1) * gap_x
    total_h = 2 * thumb_h + gap_y
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    for col, frame_idx in enumerate(indices):
        x0 = col * (thumb_w + gap_x)
        rgb = _resize_pad(_read_rgb(frames[min(frame_idx, len(frames) - 1)]), thumb_w, thumb_h)
        cv2.putText(rgb, str(int(frame_idx)), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(rgb, (0, 0), (thumb_w - 1, thumb_h - 1), (226, 229, 232), 1, cv2.LINE_AA)
        canvas[0:thumb_h, x0 : x0 + thumb_w] = rgb
        canvas[thumb_h + gap_y : 2 * thumb_h + gap_y, x0 : x0 + thumb_w] = _pc_thumb(grays, frame_idx, thumb_w, thumb_h)
    return canvas


def _overview_sample_strip(frames, grays, indices):
    import cv2

    indices = [int(idx) for idx in list(indices or [])]
    if not indices:
        indices = [0]
    thumb_w = 180
    thumb_h = 95
    gap_x = 8
    gap_y = 6
    total_w = len(indices) * thumb_w + max(0, len(indices) - 1) * gap_x
    total_h = 2 * thumb_h + gap_y
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    for col, frame_idx in enumerate(indices):
        x0 = col * (thumb_w + gap_x)
        safe_idx = int(np.clip(frame_idx, 0, max(0, len(frames) - 1)))
        pair_idx = int(np.clip(frame_idx, 0, max(0, len(grays) - 2)))
        rgb = _resize_direct(_read_rgb(frames[safe_idx]), thumb_w, thumb_h)
        cv2.putText(rgb, str(int(pair_idx)), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(rgb, (0, 0), (thumb_w - 1, thumb_h - 1), (226, 229, 232), 1, cv2.LINE_AA)
        canvas[0:thumb_h, x0 : x0 + thumb_w] = rgb
        canvas[thumb_h + gap_y : 2 * thumb_h + gap_y, x0 : x0 + thumb_w] = _pc_thumb(grays, pair_idx, thumb_w, thumb_h)
    return canvas


def _overview_sample_indices(pair_count, max_frames=8):
    pair_count = max(0, int(pair_count))
    if pair_count <= 0:
        return [0]
    count = min(int(max_frames), max(6, pair_count))
    count = max(1, min(count, pair_count))
    indices = np.linspace(0, pair_count - 1, num=count, dtype=int).tolist()
    return sorted({int(np.clip(item, 0, pair_count - 1)) for item in indices})


def _plot_background(ax, labels, x_start, colors, alpha):
    if not labels:
        return
    current = labels[0]
    start = 0
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            color = colors.get(current)
            if color:
                ax.axvspan(x_start + start, x_start + idx, color=color, alpha=alpha, lw=0)
            current = labels[idx]
            start = idx
    color = colors.get(current)
    if color:
        ax.axvspan(x_start + start, x_start + len(labels) - 1, color=color, alpha=alpha, lw=0)


def _style_axis(ax, ylabel=None):
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(True, axis="y", color="#E2E6EA", linewidth=0.6, alpha=0.8, linestyle="--")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA1A9")
    ax.spines["bottom"].set_color("#9AA1A9")
    ax.tick_params(axis="both", colors="#4A4F55", length=3.0, width=0.8)


def _pc_summary(pair_states, interval):
    start = int(interval.get("pair_start", 0) or 0)
    end = int(interval.get("pair_end", -1) or -1)
    if end < start:
        return "none"
    parts = []
    idx = start
    while idx <= end and idx < len(pair_states):
        label = str(pair_states[idx].get("motion_state", "mixed") or "mixed")
        run_end = idx
        while run_end + 1 <= end and run_end + 1 < len(pair_states):
            if str(pair_states[run_end + 1].get("motion_state", "mixed") or "mixed") != label:
                break
            run_end += 1
        if label != "mixed":
            parts.append("{}[{}-{}]".format(label, int(idx), int(run_end)))
        idx = run_end + 1
    return "; ".join(parts) if parts else "none"


def _plot_loss_interval(path, frames, grays, interval, orb_rows, pair_states, local_context=40):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not pair_states:
        return
    pair_start = max(0, int(interval.get("pair_start", 0) or 0) - int(local_context))
    pair_end = min(len(pair_states) - 1, int(interval.get("pair_end", len(pair_states) - 1) or 0) + int(local_context))
    x = np.arange(pair_start, pair_end + 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(16, 7.7), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=[3.4, 1.05, 1.70])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax1)

    strip = _sample_strip(frames, grays, pair_start, pair_end)
    ax0.imshow(strip, extent=[pair_start, pair_end, 0, 1], aspect="auto")
    ax0.set_xlim(pair_start, pair_end)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.text(-0.012, 0.745, "RGB", transform=ax0.transAxes, ha="right", va="center", fontsize=11, fontweight="bold", color="#4A4F55")
    ax0.text(-0.012, 0.255, "PC", transform=ax0.transAxes, ha="right", va="center", fontsize=11, fontweight="bold", color="#4A4F55")
    ax0.set_title(
        "Loss Interval #{} | PC: {}".format(int(interval.get("loss_interval_id", 0) or 0), _pc_summary(pair_states, interval)),
        pad=8,
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    local_orb = orb_rows[pair_start : pair_end + 1]
    quality = np.asarray([float(row.get("tracking_quality", 0.0)) for row in local_orb], dtype=np.float32)
    _plot_background(ax1, [str(row.get("tracking_state", "ok") or "ok") for row in local_orb], pair_start, TRACKING_COLORS, 0.8)
    ax1.plot(x, quality, color=TRACKING_LINE_COLOR, linewidth=2.0, label="tracking_quality")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("ORB Tracking Quality", pad=6, loc="left")
    _style_axis(ax1, "Quality")
    ax1.tick_params(axis="x", length=0, labelbottom=False)

    local_pc = pair_states[pair_start : pair_end + 1]
    steering = _normalize01(np.abs([float(row.get("steering_response", 0.0)) for row in local_pc]))
    forward = _normalize01([float(row.get("forward_score", row.get("forward_response", 0.0)) or 0.0) for row in local_pc])
    _plot_background(ax2, [str(row.get("motion_state", "mixed") or "mixed") for row in local_pc], pair_start, MOTION_COLORS, 0.42)
    ax2.plot(x, steering, color=PC_LINE_COLOR, linewidth=2.0, alpha=0.95, label="PC normalized yaw")
    ax2.plot(x, forward, color=PC_LINE_COLOR, linewidth=1.6, alpha=0.72, linestyle="--", label="PC normalized expansion")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Motion Responses with Estimated PC State Background", pad=6, loc="left")
    _style_axis(ax2, "Response")
    ax2.set_xlabel("Frame Pair Index", labelpad=8)
    for ax in (ax1, ax2):
        ax.set_xlim(pair_start, pair_end)
        ax.margins(x=0)

    handles = [Patch(color=color, label="ORB: {}".format(label.upper())) for label, color in TRACKING_COLORS.items()]
    handles += [Patch(color=MOTION_COLORS[label], label=label) for label in ("forward", "left_turn", "right_turn", "stop", "mixed")]
    handles += [
        Line2D([0], [0], color=TRACKING_LINE_COLOR, lw=2.0, label="tracking_quality"),
        Line2D([0], [0], color=PC_LINE_COLOR, lw=2.0, label="PC normalized yaw"),
        Line2D([0], [0], color=PC_LINE_COLOR, lw=1.6, linestyle="--", label="PC normalized expansion"),
    ]
    legend = fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.992), frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.96)
    frame.set_edgecolor("#D8DDE3")
    frame.set_linewidth(0.8)
    fig.align_ylabels([ax1, ax2])
    fig.subplots_adjust(left=0.09, right=0.965, top=0.80, bottom=0.075, hspace=0.16)
    plt.savefig(str(path), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _write_diagnostics(out_path, frames, grays, loss_intervals, orb_rows, pair_states):
    if out_path is None or not loss_intervals:
        return []
    out_dir = Path(out_path).resolve().parent
    paths = []
    for idx, interval in enumerate(loss_intervals):
        fig_path = out_dir / "loss_interval_{:02d}.png".format(int(idx))
        _plot_loss_interval(fig_path, frames, grays, interval, orb_rows, pair_states)
        paths.append(fig_path.name)
    return paths


def _plot_motion_overview(path, frames, grays, pair_states, motion_states, num_frames):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not pair_states or not motion_states:
        return False

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.5, 4.0), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.55, 0.60])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])

    pair_count = len(pair_states)
    sample_indices = _overview_sample_indices(pair_count, max_frames=8)
    strip = _overview_sample_strip(frames, grays, sample_indices)
    x_max = max(0, int(num_frames) - 1)
    ax0.imshow(strip)
    ax0.axis("off")
    ax0.text(-0.010, 0.745, "RGB", transform=ax0.transAxes, ha="right", va="center", fontsize=10.5, fontweight="bold", color="#4A4F55")
    ax0.text(-0.010, 0.255, "PC", transform=ax0.transAxes, ha="right", va="center", fontsize=10.5, fontweight="bold", color="#4A4F55")

    x = np.arange(len(pair_states))
    steering = _normalize01(np.abs([float(row.get("steering_response", 0.0) or 0.0) for row in pair_states]))
    state_labels = ["mixed" for _ in range(len(pair_states))]
    for state in motion_states:
        label = str(state.get("motion_label", "forward") or "forward")
        start = max(0, int(state.get("start_idx", 0) or 0))
        end = min(len(pair_states) - 1, int(state.get("end_idx", start) or start) - 1)
        if end >= start:
            state_labels[start : end + 1] = [label for _ in range(end - start + 1)]
    current = state_labels[0]
    start = 0
    for idx in range(1, len(state_labels)):
        if state_labels[idx] != current:
            color = MOTION_COLORS.get(current)
            if color:
                ax1.axvspan(start, idx, color=color, alpha=0.28, lw=0, zorder=0)
            current = state_labels[idx]
            start = idx
    color = MOTION_COLORS.get(current)
    if color:
        ax1.axvspan(start, len(state_labels) - 1, color=color, alpha=0.28, lw=0, zorder=0)
    for state in motion_states:
        start = int(state.get("start_idx", 0) or 0)
        ax1.axvline(start, color="#333333", linewidth=0.6, alpha=0.25, zorder=2)
    ax1.axvline(int(motion_states[-1].get("end_idx", x_max)), color="#333333", linewidth=0.6, alpha=0.25, zorder=2)
    ax1.plot(x, steering, color=PC_LINE_COLOR, linewidth=2.0, alpha=0.95, label="PC motion response", zorder=3)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks([0, 1])
    _style_axis(ax1, "Response")
    ax1.set_xlabel("Frame / Pair Index", labelpad=8)
    ax1.set_xlim(0, x_max)
    ax1.margins(x=0)

    display_labels = {"forward": "Forward", "left_turn": "Left turn", "right_turn": "Right turn"}
    handles = [Patch(color=MOTION_COLORS[label], label=display_labels[label]) for label in MAIN_MOTION_LABELS]
    handles += [
        Line2D([0], [0], color=PC_LINE_COLOR, lw=2.0, label="PC motion response"),
    ]
    legend = fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.975), frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.96)
    frame.set_edgecolor("#D8DDE3")
    frame.set_linewidth(0.8)
    fig.align_ylabels([ax1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.12, hspace=0.08)
    plt.savefig(str(path), dpi=230, facecolor="white", transparent=False)
    plt.close(fig)
    return True


def _write_motion_overview(out_path, frames, grays, pair_states, motion_states, num_frames):
    if out_path is None:
        return ""
    fig_path = Path(out_path).resolve().parent / "motion_overview.png"
    if _plot_motion_overview(fig_path, frames, grays, pair_states, motion_states, num_frames):
        return fig_path.name
    return ""


def _legal_continuity_ok(motion_states, legal_positions, num_frames):
    try:
        _validate_motion_states(motion_states, legal_positions, num_frames)
    except Exception:
        return False
    return True


def _validate_motion_states(motion_states, legal_positions, num_frames):
    legal_set = set(int(item) for item in legal_positions)
    if not motion_states:
        raise RuntimeError("motion state builder produced zero states")
    if int(motion_states[0]["start_idx"]) != 0:
        raise RuntimeError("motion_states must start at frame 0")
    if int(motion_states[-1]["end_idx"]) != int(num_frames) - 1:
        raise RuntimeError("motion_states must end at final prepared frame")
    previous_end = None
    for item in motion_states:
        start = int(item.get("start_idx"))
        end = int(item.get("end_idx"))
        if start not in legal_set or end not in legal_set:
            raise RuntimeError("motion_state boundary is not legal: start={} end={}".format(start, end))
        if previous_end is not None and start != previous_end:
            raise RuntimeError("motion_states must join with shared endpoints: prev_end={} start={}".format(previous_end, start))
        if str(item.get("motion_label", "") or "") not in MOTION_LABELS:
            raise RuntimeError("unsupported motion_label: {}".format(item.get("motion_label")))
        previous_end = end


def build_motion_segments(prepare_result, frames_dir, out_path=None):
    started = time.time()
    profile = {
        "version": 1,
        "read_gray_sec": 0.0,
        "motion_estimation_sec": 0.0,
        "phase_correlation_sec": 0.0,
        "orb_tracking_sec": None,
        "optical_flow_sec": None,
        "motion_benchmark_sec": None,
        "write_json_sec": 0.0,
        "total_sec": 0.0,
    }
    prepare = _as_dict(prepare_result)
    frame_dir = ensure_dir(frames_dir, "prepare frames dir")
    frames = list_frames_sorted(frame_dir)
    num_frames = int(prepare.get("num_frames", len(frames)) or len(frames))
    if len(frames) != num_frames:
        raise RuntimeError("prepare frame count mismatch: files={} prepare_result={}".format(len(frames), num_frames))

    legal_grid = _as_dict(prepare.get("legal_grid"))
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    if len(legal_positions) < 2:
        raise RuntimeError("prepare_result legal_grid requires at least two legal positions")
    if legal_positions[0] != 0:
        raise RuntimeError("legal positions must start at 0")
    if legal_positions[-1] != num_frames - 1:
        raise RuntimeError(
            "last legal position must match final prepared frame: last_legal={} final_frame={}".format(
                legal_positions[-1],
                num_frames - 1,
            )
        )

    log_info("motion estimation backend=phase_correlation")
    phase_started = time.perf_counter()
    grays = [_read_gray(frame) for frame in frames]
    profile["read_gray_sec"] = float(time.perf_counter() - phase_started)
    orb_rows = []
    loss_intervals = []
    phase_started = time.perf_counter()
    pc_rows = _pc_evidence(grays)
    motion_estimation_sec = float(time.perf_counter() - phase_started)
    profile["motion_estimation_sec"] = float(motion_estimation_sec)
    # PC timing is the full phase-correlation evidence stage, including block statistics
    # and response smoothing; it is not the pure cv2.phaseCorrelate kernel time.
    profile["phase_correlation_sec"] = float(motion_estimation_sec)
    pair_states = _estimate_states(pc_rows, loss_intervals)
    windows = _project_legal(pair_states, orb_rows, loss_intervals, legal_positions)
    projected_window_count = int(len(windows))
    projected_run_count = _run_count([str(item.get("motion_label", "forward") or "forward") for item in windows])
    fps_value = int(prepare.get("target_fps", legal_grid.get("fps", 0)) or 0)
    windows, consolidation_meta = _consolidate_windows(windows, fps_value, num_frames)
    motion_states = _merge_windows(windows, prepare)
    _validate_motion_states(motion_states, legal_positions, num_frames)
    legal_continuity_ok = _legal_continuity_ok(motion_states, legal_positions, num_frames)
    final_mixed_count = len([item for item in motion_states if str(item.get("motion_label", "") or "") == "mixed"])
    final_stop_count = len([item for item in motion_states if str(item.get("motion_label", "") or "") == "stop"])
    diagnostic_figures = []
    motion_overview_figure = _write_motion_overview(out_path, frames, grays, pair_states, motion_states, num_frames)
    motion_diagnostics = {
        "raw_candidate_run_count": int(_run_count([str(row.get("raw_motion_state", "forward") or "forward") for row in pair_states])),
        "stabilized_run_count": int(_run_count([str(row.get("motion_state", "forward") or "forward") for row in pair_states])),
        "projected_window_count": int(projected_window_count),
        "projected_run_count": int(projected_run_count),
        "consolidated_window_run_count": int(_run_count([str(item.get("motion_label", "forward") or "forward") for item in windows])),
        "final_motion_state_count": int(len(motion_states)),
        "final_mixed_count": int(final_mixed_count),
        "final_stop_count": int(final_stop_count),
        "legal_continuity_ok": bool(legal_continuity_ok),
        "min_segment_sec": float(consolidation_meta.get("min_segment_sec", 0.0)),
        "strong_turn_short_segment_count": int(consolidation_meta.get("strong_turn_short_segment_count", 0)),
    }

    motion_boundaries = [
        {
            "frame_idx": int(item["start_idx"]),
            "motion_state_id": str(item["motion_state_id"]),
            "boundary_type": "motion_state_start",
        }
        for item in motion_states
    ] + (
        [
            {
                "frame_idx": int(motion_states[-1]["end_idx"]),
                "motion_state_id": str(motion_states[-1]["motion_state_id"]),
                "boundary_type": "motion_state_end",
            }
        ]
        if motion_states
        else []
    )
    summary_elapsed_sec = float(time.time() - started)
    profile["total_sec"] = float(summary_elapsed_sec)
    payload = {
        "source": "encode.motion_state",
        "fps": int(fps_value),
        "motion_labels": list(MOTION_LABELS),
        "motion_states": motion_states,
        "motion_boundaries": motion_boundaries,
        "motion_backend": "phase_correlation",
        "tracking_backend": None,
        "tracking_quality": [],
        "tracking_state": [],
        "loss_intervals": [],
        "pc_motion_evidence": [
            {
                "pair_index": int(row["pair_index"]),
                "frame_start_idx": int(row["frame_start_idx"]),
                "frame_end_idx": int(row["frame_end_idx"]),
                "yaw_coeff": float(row.get("yaw_coeff", 0.0)),
                "expansion_coeff": float(row.get("expansion_coeff", 0.0)),
                "same_sign_consistency": float(row.get("same_sign_consistency", 0.0)),
                "expansion_consistency": float(row.get("expansion_consistency", 0.0)),
                "forward_score": float(row.get("forward_score", 0.0)),
                "left_score": float(row.get("left_score", 0.0)),
                "right_score": float(row.get("right_score", 0.0)),
                "steering_response": float(row["steering_response"]),
                "forward_response": float(row["forward_response"]),
                "motion_intensity": float(row["motion_intensity"]),
                "state_confidence": float(row["state_confidence"]),
                "motion_state": str(row["motion_state"]),
            }
            for row in pair_states
        ],
        "diagnostic_figures": list(diagnostic_figures),
        "motion_overview_figure": str(motion_overview_figure),
        "motion_diagnostics": motion_diagnostics,
        "summary": {
            "motion_state_count": int(len(motion_states)),
            "motion_boundary_count": int(len(motion_states) + (1 if motion_states else 0)),
            "legal_position_count": int(len(legal_positions)),
            "loss_interval_count": int(len(loss_intervals)),
            "diagnostic_figure_count": int(len(diagnostic_figures)),
            "motion_overview_figure": str(motion_overview_figure),
            "elapsed_sec": float(summary_elapsed_sec),
            "profile": dict(profile),
        },
    }
    if out_path is not None:
        write_started = time.perf_counter()
        write_json_atomic(out_path, payload, indent=2)
        write_json_sec = float(time.perf_counter() - write_started)
        payload["summary"]["profile"]["write_json_sec"] = float(write_json_sec)
        payload["summary"]["profile"]["total_sec"] = float(time.time() - started)
        payload["summary"]["elapsed_sec"] = float(payload["summary"]["profile"]["total_sec"])
        write_json_atomic(out_path, payload, indent=2)
        log_info(
            "motion states generated: backend=phase_correlation count={} path={}".format(
                len(motion_states),
                Path(out_path).resolve(),
            )
        )
        log_info(
            "motion diagnostics: raw_runs={} stabilized_runs={} projected_runs={} final_states={} mixed={} stop={} legal_continuity_ok={}".format(
                motion_diagnostics["raw_candidate_run_count"],
                motion_diagnostics["stabilized_run_count"],
                motion_diagnostics["projected_run_count"],
                motion_diagnostics["final_motion_state_count"],
                motion_diagnostics["final_mixed_count"],
                motion_diagnostics["final_stop_count"],
                motion_diagnostics["legal_continuity_ok"],
            )
        )
        if motion_overview_figure:
            log_info("motion overview figure: {}".format(str(Path(out_path).resolve().parent / motion_overview_figure)))
    return payload


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mainline", action="store_true")
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--out_path", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_mainline:
        raise SystemExit("--run-mainline is required")
    prepare_result = read_json_dict(ensure_file(args.prepare_result, "prepare result"))
    build_motion_segments(prepare_result, args.frames_dir, args.out_path)


if __name__ == "__main__":
    main()
