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


def _pc_block(prev_gray, curr_gray, top_ratio=0.6, grid_rows=2, grid_cols=4, min_block_size=24):
    import cv2

    h, w = prev_gray.shape
    top_h = max(int(h * top_ratio), int(min_block_size) * int(grid_rows))
    top_h = min(top_h, h)
    prev = prev_gray[:top_h, :]
    curr = curr_gray[:top_h, :]
    block_h = top_h // int(grid_rows)
    block_w = w // int(grid_cols)
    if block_h < min_block_size or block_w < min_block_size:
        return {
            "motion_intensity": 0.0,
            "steering_response": 0.0,
            "forward_response": 0.0,
            "state_confidence": 0.0,
            "block_count": 0,
        }

    records = []
    hann = cv2.createHanningWindow((block_w, block_h), cv2.CV_32F)
    for row in range(int(grid_rows)):
        for col in range(int(grid_cols)):
            y0 = row * block_h
            x0 = col * block_w
            p = prev[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            q = curr[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            texture = float(np.std(p))
            if texture < 2.0:
                continue
            (dx, dy), response = cv2.phaseCorrelate(p, q, hann)
            weight = max(0.0, float(response)) * max(texture, 1e-3)
            if weight <= 0.0:
                continue
            center_x = float(x0) + 0.5 * float(block_w)
            horizontal_pos = (center_x - 0.5 * float(w)) / max(0.5 * float(w), 1.0)
            records.append((float(dx), float(dy), float(response), float(weight), float(horizontal_pos)))

    if not records:
        return {
            "motion_intensity": 0.0,
            "steering_response": 0.0,
            "forward_response": 0.0,
            "state_confidence": 0.0,
            "block_count": 0,
        }

    dxs = np.asarray([item[0] for item in records], dtype=np.float32)
    dys = np.asarray([item[1] for item in records], dtype=np.float32)
    responses = np.asarray([item[2] for item in records], dtype=np.float32)
    weights = np.asarray([item[3] for item in records], dtype=np.float32)
    positions = np.asarray([item[4] for item in records], dtype=np.float32)
    weight_sum = float(np.sum(weights)) + 1e-6
    return {
        "motion_intensity": float(np.sum(np.sqrt(dxs * dxs + dys * dys) * weights) / weight_sum),
        "steering_response": float(np.sum(dxs * weights) / weight_sum),
        "forward_response": float(np.sum(np.maximum(0.0, positions * dxs) * weights) / weight_sum),
        "state_confidence": float(np.clip(np.mean(np.maximum(responses, 0.0)), 0.0, 1.0)),
        "block_count": int(len(records)),
    }


def _pc_evidence(grays):
    pair_count = max(0, len(grays) - 1)
    if pair_count <= 0:
        return []
    raw_rows = [_pc_block(grays[idx], grays[idx + 1]) for idx in range(pair_count)]
    motion = _smooth([row["motion_intensity"] for row in raw_rows], 5, 9)
    steering = _smooth([row["steering_response"] for row in raw_rows], 5, 9)
    forward = _smooth([row["forward_response"] for row in raw_rows], 5, 9)
    confidence = _smooth([row["state_confidence"] for row in raw_rows], 5, 9)
    out = []
    for idx, row in enumerate(raw_rows):
        out.append(
            {
                "pair_index": int(idx),
                "frame_start_idx": int(idx),
                "frame_end_idx": int(idx + 1),
                "steering_response": float(steering[idx]),
                "forward_response": float(forward[idx]),
                "motion_intensity": float(motion[idx]),
                "state_confidence": float(confidence[idx]),
                "block_count": int(row.get("block_count", 0)),
                "motion_state": "mixed",
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


def _state_from_response(steering, forward, s_scale, f_scale, in_loss=False):
    s_norm = abs(float(steering)) / max(float(s_scale), 1e-6)
    f_norm = max(float(forward), 0.0) / max(float(f_scale), 1e-6)
    turn_label = _turn_label(steering)
    if s_norm >= 0.45 and s_norm >= 0.80 * max(f_norm, 1e-6):
        return turn_label
    if f_norm >= 0.18 and f_norm >= 0.75 * max(s_norm, 1e-6):
        return "forward"
    if s_norm >= 0.32 and s_norm >= 0.80 * max(f_norm, 1e-6):
        return turn_label if s_norm >= f_norm else "forward"
    return "forward"


def _switch_allowed(current, candidate, steering_strength, forward_strength, confidence, block_count):
    if candidate == current:
        return True
    if int(block_count) <= 0 or float(confidence) < 0.04:
        return False
    steering_strength = float(steering_strength)
    forward_strength = float(forward_strength)
    if candidate == "forward":
        if current in ("left_turn", "right_turn"):
            return forward_strength >= 0.18 and (
                forward_strength >= 1.05 * max(steering_strength, 1e-6) or steering_strength < 0.22
            )
        return True
    if candidate in ("left_turn", "right_turn"):
        if current == "forward":
            return steering_strength >= 0.45 and steering_strength >= 0.80 * max(forward_strength, 1e-6)
        return steering_strength >= 0.45 and steering_strength >= 0.90 * max(forward_strength, 1e-6)
    return False


def _stabilize_candidates(candidates, strengths, confirm_pairs=8):
    if not candidates:
        return []
    current = str(candidates[0] or "forward")
    if current not in MAIN_MOTION_LABELS:
        current = "forward"
    pending = None
    pending_count = 0
    out = []
    for idx, candidate in enumerate(candidates):
        candidate = str(candidate or "forward")
        if candidate not in MAIN_MOTION_LABELS:
            candidate = "forward"
        meta = strengths[idx] if idx < len(strengths) else {}
        if candidate == current:
            pending = None
            pending_count = 0
            out.append(current)
            continue
        if _switch_allowed(
            current,
            candidate,
            meta.get("steering_strength", 0.0),
            meta.get("forward_strength", 0.0),
            meta.get("state_confidence", 0.0),
            meta.get("block_count", 0),
        ):
            if pending == candidate:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1
            if pending_count >= int(confirm_pairs):
                current = candidate
                pending = None
                pending_count = 0
        else:
            pending = None
            pending_count = 0
        out.append(current)
    return out


def _replace_range(states, start, end, value):
    for idx in range(max(0, int(start)), min(len(states) - 1, int(end)) + 1):
        states[idx] = str(value)


def _absorb_short(states, loss_intervals, min_len=8):
    states = list(states)
    ranges = []
    if loss_intervals:
        ranges = [(int(item["pair_start"]), int(item["pair_end"])) for item in loss_intervals]
    else:
        ranges = [(0, len(states) - 1)] if states else []
    for _ in range(2):
        changed = False
        for start, end in ranges:
            idx = max(0, start)
            end = min(len(states) - 1, end)
            while idx <= end:
                current = states[idx]
                run_end = idx
                while run_end + 1 <= end and states[run_end + 1] == current:
                    run_end += 1
                if current != "mixed" and run_end - idx + 1 < int(min_len):
                    left = states[idx - 1] if idx - 1 >= start else None
                    right = states[run_end + 1] if run_end + 1 <= end else None
                    if left == right and left not in (None, "mixed"):
                        new_state = left
                    elif right not in (None, "mixed"):
                        new_state = right
                    elif left not in (None, "mixed"):
                        new_state = left
                    else:
                        new_state = "forward"
                    _replace_range(states, idx, run_end, new_state)
                    changed = True
                idx = run_end + 1
        if not changed:
            break
    return states


def _estimate_states(pc_rows, loss_intervals):
    if not pc_rows:
        return []
    steering = np.asarray([float(row["steering_response"]) for row in pc_rows], dtype=np.float32)
    forward = np.asarray([float(row["forward_response"]) for row in pc_rows], dtype=np.float32)
    motion = np.asarray([float(row["motion_intensity"]) for row in pc_rows], dtype=np.float32)
    confidence = np.asarray([float(row["state_confidence"]) for row in pc_rows], dtype=np.float32)
    loss_mask = _in_loss_mask(len(pc_rows), loss_intervals)
    s_scale = _robust_scale(np.abs(steering))
    f_scale = _robust_scale(forward)
    m_scale = _robust_scale(motion)

    raw_states = []
    strengths = []
    for idx, row in enumerate(pc_rows):
        m_norm = float(motion[idx]) / max(float(m_scale), 1e-6)
        conf = float(confidence[idx])
        steering_strength = abs(float(steering[idx])) / max(float(s_scale), 1e-6)
        forward_strength = max(float(forward[idx]), 0.0) / max(float(f_scale), 1e-6)
        block_count = int(row.get("block_count", 0) or 0)
        if conf < 0.04 or block_count <= 0:
            state = raw_states[-1] if raw_states else "forward"
        else:
            state = _state_from_response(steering[idx], forward[idx], s_scale, f_scale, bool(loss_mask[idx]))
            if m_norm < 0.08 and state != (raw_states[-1] if raw_states else "forward"):
                state = raw_states[-1] if raw_states else "forward"
        if state not in MAIN_MOTION_LABELS:
            state = "forward"
        raw_states.append(state)
        strengths.append(
            {
                "steering_strength": float(steering_strength),
                "forward_strength": float(forward_strength),
                "motion_strength": float(m_norm),
                "state_confidence": float(conf),
                "block_count": int(block_count),
            }
        )

    states = _stabilize_candidates(raw_states, strengths, confirm_pairs=8)
    states = _absorb_short(states, loss_intervals, min_len=8)
    out = []
    for idx, row in enumerate(pc_rows):
        item = dict(row)
        item["raw_motion_state"] = str(raw_states[idx])
        item["steering_strength"] = float(strengths[idx]["steering_strength"])
        item["forward_strength"] = float(strengths[idx]["forward_strength"])
        item["motion_strength"] = float(strengths[idx]["motion_strength"])
        item["motion_state"] = str(states[idx])
        out.append(item)
    return out


def _dominant_label(rows, loss_overlap, scales):
    if not rows:
        return "forward"
    scales = _as_dict(scales)
    motion_scale = max(float(scales.get("motion", 1.0) or 1.0), 1e-6)
    steering_scale = max(float(scales.get("steering", 1.0) or 1.0), 1e-6)
    forward_scale = max(float(scales.get("forward", 1.0) or 1.0), 1e-6)
    mean_steering = _mean_score(rows, "steering_response")
    mean_forward = max(_mean_score(rows, "forward_response"), 0.0)
    mean_steering_strength = abs(float(mean_steering)) / steering_scale
    mean_forward_strength = float(mean_forward) / forward_scale
    turn_label = _turn_label(mean_steering)
    turn_fraction = float(
        len(
            [
                row
                for row in rows
                if str(row.get("raw_motion_state", row.get("motion_state", "")) or "") == turn_label
                or str(row.get("motion_state", "") or "") == turn_label
            ]
        )
    ) / float(max(1, len(rows)))
    if mean_steering_strength >= 0.55 and mean_steering_strength >= 0.70 * max(mean_forward_strength, 1e-6) and turn_fraction >= 0.40:
        return turn_label
    if mean_forward_strength >= 0.20 and mean_forward_strength >= 1.25 * max(mean_steering_strength, 1e-6):
        return "forward"
    if mean_steering_strength >= 0.45 and mean_steering_strength >= 0.85 * max(mean_forward_strength, 1e-6):
        return turn_label

    weights = {label: 0.0 for label in MAIN_MOTION_LABELS}
    for row in rows:
        label = str(row.get("motion_state", "forward") or "forward")
        if label not in MAIN_MOTION_LABELS:
            label = "forward"
        weight = max(float(row.get("state_confidence", 0.0) or 0.0), 0.05)
        weight += min(float(row.get("motion_intensity", 0.0) or 0.0) / motion_scale, 1.0) * 0.20
        weights[label] = float(weights.get(label, 0.0) + weight)

        steering = float(row.get("steering_response", 0.0) or 0.0)
        forward = max(float(row.get("forward_response", 0.0) or 0.0), 0.0)
        steering_score = abs(steering) / steering_scale
        forward_score = forward / forward_scale
        if steering > 0.0:
            weights["left_turn"] += steering_score * 0.75
        elif steering < 0.0:
            weights["right_turn"] += steering_score * 0.75
        weights["forward"] += forward_score * 0.55

    ranked = sorted(weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
    best_label = str(ranked[0][0])
    if bool(loss_overlap) and best_label not in MAIN_MOTION_LABELS:
        return "forward"
    return best_label if best_label in MAIN_MOTION_LABELS else "forward"


def _mean_score(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _window_scores(pc_rows, orb_rows, scales=None):
    scales = _as_dict(scales)
    steering_scale = max(float(scales.get("steering", 1.0) or 1.0), 1e-6)
    forward_scale = max(float(scales.get("forward", 1.0) or 1.0), 1e-6)
    motion_scale = max(float(scales.get("motion", 1.0) or 1.0), 1e-6)
    tracking_values = [float(row.get("tracking_quality", 0.0) or 0.0) for row in orb_rows]
    lost_count = len([row for row in orb_rows if str(row.get("tracking_state", "") or "") == "lost"])
    total = max(1, len(orb_rows))
    motion = _mean_score(pc_rows, "motion_intensity")
    steering = _mean_score(pc_rows, "steering_response")
    forward = _mean_score(pc_rows, "forward_response")
    confidence = _mean_score(pc_rows, "state_confidence")
    if steering >= 0.0:
        sign_count = len([row for row in pc_rows if float(row.get("steering_response", 0.0) or 0.0) >= 0.0])
    else:
        sign_count = len([row for row in pc_rows if float(row.get("steering_response", 0.0) or 0.0) < 0.0])
    sign_fraction = float(sign_count) / float(max(1, len(pc_rows)))
    return {
        "translational": float(motion),
        "directional": float(steering),
        "confidence": float(confidence),
        "motion_intensity": float(motion),
        "steering_response": float(steering),
        "forward_response": float(forward),
        "motion_strength": float(motion) / motion_scale,
        "steering_strength": abs(float(steering)) / steering_scale,
        "forward_strength": max(float(forward), 0.0) / forward_scale,
        "steering_sign_fraction": float(sign_fraction),
        "tracking_quality": float(np.mean(tracking_values)) if tracking_values else 0.0,
        "lost_pair_fraction": float(lost_count) / float(total),
    }


def _project_legal(pc_rows, orb_rows, loss_intervals, legal_positions):
    scales = {
        "motion": _robust_scale([float(row.get("motion_intensity", 0.0)) for row in pc_rows]),
        "steering": _robust_scale([abs(float(row.get("steering_response", 0.0))) for row in pc_rows]),
        "forward": _robust_scale([float(row.get("forward_response", 0.0)) for row in pc_rows]),
    }
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
        label = _dominant_label(pc_slice, loss_overlap, scales)
        windows.append(
            {
                "window_index": int(win_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "pair_start": int(pair_start),
                "pair_end": int(pair_end),
                "motion_label": str(label),
                "scores": _window_scores(pc_slice, orb_slice, scales),
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
    return bool(steering_strength >= 0.35 and sign_fraction >= 0.62 and steering_strength >= 0.75 * max(forward_strength, 1e-6))


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
        "motion_strength",
        "steering_strength",
        "forward_strength",
        "steering_sign_fraction",
        "tracking_quality",
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


def _pc_details(prev_gray, curr_gray):
    import cv2

    h, w = prev_gray.shape
    top_h = min(max(int(h * 0.6), 48), h)
    block_h = top_h // 2
    block_w = w // 4
    if block_h < 24 or block_w < 24:
        return []
    details = []
    hann = cv2.createHanningWindow((block_w, block_h), cv2.CV_32F)
    for row in range(2):
        for col in range(4):
            y0 = row * block_h
            x0 = col * block_w
            p = prev_gray[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            q = curr_gray[y0 : y0 + block_h, x0 : x0 + block_w].astype(np.float32)
            texture = float(np.std(p))
            if texture < 2.0:
                continue
            (dx, dy), response = cv2.phaseCorrelate(p, q, hann)
            if max(0.0, float(response)) * max(texture, 1e-3) <= 0.0:
                continue
            details.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "bw": int(block_w),
                    "bh": int(block_h),
                    "dx": float(dx),
                    "dy": float(dy),
                    "response": float(response),
                }
            )
    return details


def _pc_thumb(grays, pair_index, thumb_w=160, thumb_h=100):
    import cv2

    pair_index = int(np.clip(pair_index, 0, max(0, len(grays) - 2)))
    prev_gray = grays[pair_index]
    curr_gray = grays[pair_index + 1]
    h, w = prev_gray.shape
    top_h = min(max(int(h * 0.6), 48), h)
    canvas = np.full((thumb_h, thumb_w, 3), 244, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (thumb_w - 1, thumb_h - 1), (223, 226, 229), 1, cv2.LINE_AA)
    for row in range(3):
        y = int(round(row * thumb_h / 2.0))
        cv2.line(canvas, (0, min(thumb_h - 1, y)), (thumb_w - 1, min(thumb_h - 1, y)), (232, 235, 238), 1, cv2.LINE_AA)
    for col in range(5):
        x = int(round(col * thumb_w / 4.0))
        cv2.line(canvas, (min(thumb_w - 1, x), 0), (min(thumb_w - 1, x), thumb_h - 1), (232, 235, 238), 1, cv2.LINE_AA)
    for item in _pc_details(prev_gray, curr_gray):
        x1 = int(round(float(item["x0"]) / max(1, w) * thumb_w))
        x2 = int(round(float(item["x0"] + item["bw"]) / max(1, w) * thumb_w))
        y1 = int(round(float(item["y0"]) / max(1, top_h) * thumb_h))
        y2 = int(round(float(item["y0"] + item["bh"]) / max(1, top_h) * thumb_h))
        x1, x2 = int(np.clip(x1, 0, thumb_w - 1)), int(np.clip(x2, x1 + 1, thumb_w))
        y1, y2 = int(np.clip(y1, 0, thumb_h - 1)), int(np.clip(y2, y1 + 1, thumb_h))
        dx = float(item["dx"])
        dy = float(item["dy"])
        base = np.asarray((215, 188, 179), dtype=np.float32) if dx >= 0 else np.asarray((173, 202, 217), dtype=np.float32)
        strength = float(np.clip(0.20 + 0.08 * abs(dx) + 0.30 * max(float(item["response"]), 0.0), 0.16, 0.72))
        fill = np.clip((1.0 - strength) * 244.0 + strength * base, 0, 255).astype(np.uint8)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), tuple(int(v) for v in fill), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (240, 242, 244), 1, cv2.LINE_AA)
        cx = int(round((x1 + x2 - 1) / 2.0))
        cy = int(round((y1 + y2 - 1) / 2.0))
        scale = min(7.5, max(1.6, 1.25 * math.sqrt(dx * dx + dy * dy) + 1.2))
        ex = int(np.clip(cx + dx * scale, x1 + 5, x2 - 5))
        ey = int(np.clip(cy + dy * scale, y1 + 5, y2 - 5))
        color = (155, 103, 86) if dx >= 0 else (83, 129, 150)
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 1, cv2.LINE_AA, tipLength=0.30)
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
    thumb_w = 260
    thumb_h = int(round(thumb_w * 0.58))
    gap_x = max(12, int(round(thumb_w * 0.08)))
    gap_y = max(10, int(round(thumb_h * 0.10)))
    total_w = len(indices) * thumb_w + max(0, len(indices) - 1) * gap_x
    total_h = 2 * thumb_h + gap_y
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    for col, frame_idx in enumerate(indices):
        x0 = col * (thumb_w + gap_x)
        safe_idx = int(np.clip(frame_idx, 0, max(0, len(frames) - 1)))
        pair_idx = int(np.clip(frame_idx, 0, max(0, len(grays) - 2)))
        rgb = _resize_pad(_read_rgb(frames[safe_idx]), thumb_w, thumb_h)
        cv2.putText(rgb, str(int(pair_idx)), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
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
    forward = _normalize01([float(row.get("forward_response", 0.0)) for row in local_pc])
    _plot_background(ax2, [str(row.get("motion_state", "mixed") or "mixed") for row in local_pc], pair_start, MOTION_COLORS, 0.42)
    ax2.plot(x, steering, color=PC_LINE_COLOR, linewidth=2.0, alpha=0.95, label="PC steering_response")
    ax2.plot(x, forward, color=PC_LINE_COLOR, linewidth=1.6, alpha=0.72, linestyle="--", label="PC forward_response")
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
        Line2D([0], [0], color=PC_LINE_COLOR, lw=2.0, label="PC steering_response"),
        Line2D([0], [0], color=PC_LINE_COLOR, lw=1.6, linestyle="--", label="PC forward_response"),
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
    fig = plt.figure(figsize=(16, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 2.15])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])

    pair_count = len(pair_states)
    sample_indices = _overview_sample_indices(pair_count, max_frames=8)
    strip = _overview_sample_strip(frames, grays, sample_indices)
    x_max = max(0, int(num_frames) - 1)
    ax0.imshow(strip)
    ax0.axis("off")
    ax0.text(-0.012, 0.745, "RGB", transform=ax0.transAxes, ha="right", va="center", fontsize=11, fontweight="bold", color="#4A4F55")
    ax0.text(-0.012, 0.255, "PC", transform=ax0.transAxes, ha="right", va="center", fontsize=11, fontweight="bold", color="#4A4F55")
    ax0.set_title("Motion Overview Samples", pad=8, loc="left", fontsize=12, fontweight="bold")

    x = np.arange(len(pair_states))
    steering = _normalize01(np.abs([float(row.get("steering_response", 0.0) or 0.0) for row in pair_states]))
    forward = _normalize01([float(row.get("forward_response", 0.0) or 0.0) for row in pair_states])
    state_labels = ["mixed" for _ in range(len(pair_states))]
    for state in motion_states:
        label = str(state.get("motion_label", "forward") or "forward")
        start = max(0, int(state.get("start_idx", 0) or 0))
        end = min(len(pair_states) - 1, int(state.get("end_idx", start) or start) - 1)
        if end >= start:
            state_labels[start : end + 1] = [label for _ in range(end - start + 1)]
    _plot_background(ax1, state_labels, 0, MOTION_COLORS, 0.42)
    for state in motion_states:
        start = int(state.get("start_idx", 0) or 0)
        ax1.axvline(start, color="#333333", linewidth=0.75, alpha=0.35)
    ax1.axvline(int(motion_states[-1].get("end_idx", x_max)), color="#333333", linewidth=0.75, alpha=0.35)
    ax1.plot(x, steering, color=PC_LINE_COLOR, linewidth=2.0, alpha=0.95, label="PC abs(steering_response)")
    ax1.plot(x, forward, color=PC_LINE_COLOR, linewidth=1.7, alpha=0.72, linestyle="--", label="PC forward_response")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Phase Correlation Motion Responses", pad=7, loc="left")
    _style_axis(ax1, "Response")
    ax1.set_xlabel("Frame / Pair Index", labelpad=8)
    ax1.set_xlim(0, x_max)
    ax1.margins(x=0)

    handles = [Patch(color=MOTION_COLORS[label], label=label) for label in MAIN_MOTION_LABELS]
    handles += [
        Line2D([0], [0], color=PC_LINE_COLOR, lw=2.0, label="PC abs(steering_response)"),
        Line2D([0], [0], color=PC_LINE_COLOR, lw=1.7, linestyle="--", label="PC forward_response"),
    ]
    legend = fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.975), frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.96)
    frame.set_edgecolor("#D8DDE3")
    frame.set_linewidth(0.8)
    fig.align_ylabels([ax1])
    fig.subplots_adjust(left=0.09, right=0.965, top=0.875, bottom=0.085, hspace=0.18)
    plt.savefig(str(path), dpi=230, bbox_inches="tight")
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

    log_info("motion estimation backend=phase_correlation tracking_backend=orb_quality")
    grays = [_read_gray(frame) for frame in frames]
    orb_rows = _orb_pairs(grays)
    loss_intervals = _loss_intervals(orb_rows)
    pc_rows = _pc_evidence(grays)
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
    diagnostic_figures = _write_diagnostics(out_path, frames, grays, loss_intervals, orb_rows, pair_states)
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

    payload = {
        "source": "encode.motion_state",
        "fps": int(fps_value),
        "motion_labels": list(MOTION_LABELS),
        "motion_states": motion_states,
        "motion_boundaries": [
            {
                "frame_idx": int(item["start_idx"]),
                "motion_state_id": str(item["motion_state_id"]),
                "boundary_type": "motion_state_start",
            }
            for item in motion_states
        ]
        + (
            [
                {
                    "frame_idx": int(motion_states[-1]["end_idx"]),
                    "motion_state_id": str(motion_states[-1]["motion_state_id"]),
                    "boundary_type": "motion_state_end",
                }
            ]
            if motion_states
            else []
        ),
        "motion_backend": "phase_correlation",
        "tracking_backend": "orb_quality",
        "tracking_quality": [
            {
                "pair_index": int(row["pair_index"]),
                "frame_start_idx": int(row["frame_start_idx"]),
                "frame_end_idx": int(row["frame_end_idx"]),
                "tracking_quality": float(row["tracking_quality"]),
            }
            for row in orb_rows
        ],
        "tracking_state": [
            {
                "pair_index": int(row["pair_index"]),
                "tracking_state": str(row["tracking_state"]),
            }
            for row in orb_rows
        ],
        "loss_intervals": loss_intervals,
        "pc_motion_evidence": [
            {
                "pair_index": int(row["pair_index"]),
                "frame_start_idx": int(row["frame_start_idx"]),
                "frame_end_idx": int(row["frame_end_idx"]),
                "steering_response": float(row["steering_response"]),
                "forward_response": float(row["forward_response"]),
                "motion_intensity": float(row["motion_intensity"]),
                "state_confidence": float(row["state_confidence"]),
                "raw_motion_state": str(row.get("raw_motion_state", row["motion_state"])),
                "steering_strength": float(row.get("steering_strength", 0.0)),
                "forward_strength": float(row.get("forward_strength", 0.0)),
                "motion_strength": float(row.get("motion_strength", 0.0)),
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
            "elapsed_sec": float(time.time() - started),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
        log_info(
            "motion states generated: backend=phase_correlation count={} loss_intervals={} path={}".format(
                len(motion_states),
                len(loss_intervals),
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
        if diagnostic_figures:
            log_info("loss interval diagnostic figures: {}".format(", ".join(diagnostic_figures)))
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
