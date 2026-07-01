from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from exphub.common.io import (
    ensure_dir,
    list_frames_sorted,
    replace_nonempty_file,
    unique_sibling_temp_path,
    write_csv_atomic,
    write_json_atomic,
)
from exphub.common.logging import log_info, log_warn

from .motion_segment import (
    MOTION_COLORS,
    PC_EPS,
    PC_GRID_COLS,
    PC_GRID_ROWS,
    _as_dict,
    _estimate_states,
    _loss_intervals,
    _normalize01,
    _orb_pairs,
    _pc_evidence,
    _pc_thumb,
    _read_gray,
    _read_rgb,
    _robust_scale,
    _run_count,
)


BENCHMARK_LABELS = ("forward", "left_turn", "right_turn", "mixed", "failed")
BENCHMARK_COLORS = dict(MOTION_COLORS)
BENCHMARK_COLORS.update({"failed": "#FFCDD2"})
THRESHOLDS = {
    "orb_failed_quality": 0.45,
    "orb_weak_quality": 0.55,
    "of_turn_score": 0.08,
    "of_turn_dominance": 1.25,
    "of_mixed_ratio": 0.80,
}
ORB_LABEL_CONFIG = {
    "turn_abs_px_threshold": 1.25,
    "turn_norm_threshold": 0.35,
    "smoothing_window": 9,
    "min_segment_length": 8,
    "orb_turn_sign": 1,
    "turn_sign_convention": "label = left_turn when orb_turn_sign * smoothed_tx > 0; flip orb_turn_sign if dataset convention is reversed",
}
LABEL_STYLE = {
    "fontsize": 10.0,
    "fontweight": "bold",
    "color": "#4A4F55",
}


def _resize_direct(image, target_w, target_h):
    import cv2

    if image is None or image.size == 0:
        return np.full((int(target_h), int(target_w), 3), 255, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.resize(arr, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)


def _centered_mean(values, window):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    if window <= 1:
        return arr
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _safe_float(value, default=0.0):
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _relative(exp_dir, path):
    target = Path(path).resolve()
    try:
        return target.relative_to(Path(exp_dir).resolve()).as_posix()
    except Exception:
        return target.as_posix()


def _labels(values):
    return [str(item if item in BENCHMARK_LABELS else "mixed") for item in list(values or [])]


def _segment_count(labels):
    return _run_count(_labels(labels))


def _method_summary(runtime_sec, labels, pair_count, extra=None):
    labels = _labels(labels)
    pair_count = int(pair_count)
    valid_count = len([item for item in labels if item != "failed"])
    summary = {
        "runtime_sec": float(runtime_sec),
        "time_per_pair_ms": float(runtime_sec) * 1000.0 / float(max(1, pair_count)),
        "valid_rate": float(valid_count) / float(max(1, pair_count)),
        "valid_output_rate_pct": 100.0 * float(valid_count) / float(max(1, pair_count)),
        "segment_count": int(_segment_count(labels)),
        "labels": sorted(set(labels)),
    }
    if extra:
        summary.update(dict(extra))
    return summary


def _pc_method(grays):
    started = time.perf_counter()
    rows = _estimate_states(_pc_evidence(grays), [])
    runtime_sec = float(time.perf_counter() - started)
    labels = [str(row.get("motion_state", "forward") or "forward") for row in rows]
    for row, label in zip(rows, labels):
        row["benchmark_label"] = label if label in BENCHMARK_LABELS else "mixed"
    return rows, labels, runtime_sec


def _orb_failed(row):
    quality = _safe_float(row.get("tracking_quality"))
    tracking_state = str(row.get("tracking_state", "") or "")
    return bool(tracking_state == "lost" or quality < float(THRESHOLDS["orb_failed_quality"]))


def _stabilize_orb_labels(labels, failed_mask, min_segment_length):
    out = list(labels or [])
    failed_mask = [bool(item) for item in list(failed_mask or [])]
    min_segment_length = max(1, int(min_segment_length))
    idx = 0
    while idx < len(out):
        label = str(out[idx])
        start = idx
        while idx + 1 < len(out) and str(out[idx + 1]) == label:
            idx += 1
        end = idx
        run_len = int(end - start + 1)
        if label in ("left_turn", "right_turn", "mixed") and run_len < min_segment_length:
            left_label = out[start - 1] if start > 0 and not failed_mask[start - 1] else ""
            right_label = out[end + 1] if end + 1 < len(out) and not failed_mask[end + 1] else ""
            if left_label and left_label == right_label:
                replacement = left_label
            elif left_label in ("left_turn", "right_turn", "forward"):
                replacement = left_label
            elif right_label in ("left_turn", "right_turn", "forward"):
                replacement = right_label
            else:
                replacement = "forward"
            for pos in range(start, end + 1):
                if not failed_mask[pos]:
                    out[pos] = replacement
        idx += 1
    for pos, failed in enumerate(failed_mask):
        if failed:
            out[pos] = "failed"
    return out


def _orb_labels(rows):
    rows = list(rows or [])
    if not rows:
        return []
    failed_mask = [_orb_failed(row) for row in rows]
    tx_values = [_safe_float(row.get("tx")) for row in rows]
    smoothed_tx = _centered_mean(tx_values, ORB_LABEL_CONFIG["smoothing_window"])
    valid_tx = np.asarray([abs(float(value)) for value, failed in zip(smoothed_tx, failed_mask) if not failed], dtype=np.float32)
    robust_tx = max(_robust_scale(valid_tx), PC_EPS)
    labels = []
    for idx, row in enumerate(rows):
        if failed_mask[idx]:
            labels.append("failed")
            continue
        tx = float(smoothed_tx[idx])
        norm = abs(tx) / robust_tx
        abs_ok = abs(tx) >= float(ORB_LABEL_CONFIG["turn_abs_px_threshold"])
        norm_ok = norm >= float(ORB_LABEL_CONFIG["turn_norm_threshold"])
        if abs_ok and norm_ok:
            signed = float(ORB_LABEL_CONFIG["orb_turn_sign"]) * tx
            label = "left_turn" if signed > 0.0 else "right_turn"
        else:
            label = "forward"
        row["orb_turn_response"] = float(tx)
        row["orb_turn_response_norm"] = float(norm)
        labels.append(label)
    labels = _stabilize_orb_labels(labels, failed_mask, ORB_LABEL_CONFIG["min_segment_length"])
    return labels


def _orb_method(grays):
    started = time.perf_counter()
    rows = _orb_pairs(grays)
    runtime_sec = float(time.perf_counter() - started)
    labels = _orb_labels(rows)
    for row, label in zip(rows, labels):
        row["benchmark_label"] = label
        row["orb_valid"] = bool(label != "failed")
    failed_intervals = _loss_intervals(rows)
    lost_or_weak = len([row for row in rows if str(row.get("tracking_state", "") or "") != "ok"])
    extra = {
        "failed_interval_count": int(len(failed_intervals)),
        "lost_or_weak_ratio": float(lost_or_weak) / float(max(1, len(rows))),
    }
    return rows, labels, runtime_sec, failed_intervals, extra


def _of_pair(prev_gray, curr_gray):
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if flow is None or flow.ndim != 3 or flow.shape[2] != 2 or not np.all(np.isfinite(flow)):
        return {"valid": False, "label": "failed", "steering_response": 0.0, "forward_response": 0.0}

    h, w = prev_gray.shape[:2]
    block_h = h // PC_GRID_ROWS
    block_w = w // PC_GRID_COLS
    records = []
    for row in range(PC_GRID_ROWS):
        for col in range(PC_GRID_COLS):
            y0 = row * block_h
            x0 = col * block_w
            block = flow[y0 : y0 + block_h, x0 : x0 + block_w]
            if block.size == 0:
                continue
            dx = float(np.mean(block[..., 0]))
            dy = float(np.mean(block[..., 1]))
            mag = float(np.sqrt(dx * dx + dy * dy))
            cx = float(x0) + 0.5 * float(block_w)
            cy = float(y0) + 0.5 * float(block_h)
            records.append(
                {
                    "x_pos": (cx - 0.5 * float(w)) / max(0.5 * float(w), 1.0),
                    "y_pos": (cy - 0.5 * float(h)) / max(0.5 * float(h), 1.0),
                    "dx": dx,
                    "dy": dy,
                    "weight": max(mag, PC_EPS),
                }
            )
    if not records:
        return {"valid": False, "label": "failed", "steering_response": 0.0, "forward_response": 0.0}

    dxs = np.asarray([item["dx"] for item in records], dtype=np.float32)
    dys = np.asarray([item["dy"] for item in records], dtype=np.float32)
    x_pos = np.asarray([item["x_pos"] for item in records], dtype=np.float32)
    y_pos = np.asarray([item["y_pos"] for item in records], dtype=np.float32)
    weights = np.asarray([item["weight"] for item in records], dtype=np.float32)
    weight_sum = float(np.sum(weights))
    if weight_sum <= PC_EPS:
        return {"valid": False, "label": "failed", "steering_response": 0.0, "forward_response": 0.0}

    x_center = float(np.sum(x_pos * weights) / (weight_sum + PC_EPS))
    y_center = float(np.sum(y_pos * weights) / (weight_sum + PC_EPS))
    x_c = x_pos - x_center
    y_c = y_pos - y_center
    forward_x = float(np.sum(weights * x_c * dxs) / (np.sum(weights * x_c * x_c) + PC_EPS))
    residual_dx = dxs - forward_x * x_c
    yaw = float(np.sum(residual_dx * weights) / (weight_sum + PC_EPS))
    radial_projection = x_c * dxs + y_c * dys
    radial_norm = x_c * x_c + y_c * y_c
    expansion = float(np.sum(weights * radial_projection) / (np.sum(weights * radial_norm) + PC_EPS))
    same_pos = float(np.sum(weights[residual_dx >= 0.0]))
    same_neg = float(np.sum(weights[residual_dx < 0.0]))
    same_sign = float(max(same_pos, same_neg) / (weight_sum + PC_EPS))
    expansion_consistency = float(np.sum(weights[radial_projection > 0.0]) / (weight_sum + PC_EPS))
    scale = max(_robust_scale(np.sqrt(dxs * dxs + dys * dys)), PC_EPS)
    steering = math.copysign(abs(yaw) / scale, yaw) if abs(yaw) > PC_EPS else 0.0
    turn_score = abs(steering) * same_sign
    forward_response = max(expansion, 0.0) / scale * expansion_consistency
    label = "forward"
    if turn_score >= float(THRESHOLDS["of_turn_score"]):
        if forward_response >= float(THRESHOLDS["of_turn_score"]):
            ratio = min(turn_score, forward_response) / max(turn_score, forward_response, PC_EPS)
            label = "mixed" if ratio >= float(THRESHOLDS["of_mixed_ratio"]) else label
        if label == "forward" and turn_score >= float(THRESHOLDS["of_turn_dominance"]) * max(forward_response, PC_EPS):
            label = "left_turn" if steering > 0.0 else "right_turn"
    return {
        "valid": True,
        "label": label,
        "steering_response": float(steering),
        "forward_response": float(forward_response),
        "yaw_coeff": float(yaw),
        "expansion_coeff": float(expansion),
        "same_sign_consistency": float(same_sign),
        "expansion_consistency": float(expansion_consistency),
    }


def _of_method(grays):
    started = time.perf_counter()
    rows = []
    for idx in range(max(0, len(grays) - 1)):
        row = _of_pair(grays[idx], grays[idx + 1])
        row.update({"pair_index": int(idx), "frame_start_idx": int(idx), "frame_end_idx": int(idx + 1)})
        rows.append(row)
    runtime_sec = float(time.perf_counter() - started)
    labels = [str(row.get("label", "failed") or "failed") for row in rows]
    return rows, labels, runtime_sec


def _write_csv(path, pc_rows, orb_rows, of_rows, pair_count):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_idx",
                "pc_label",
                "orb_label",
                "of_label",
                "pc_steering_response",
                "pc_forward_response",
                "orb_tracking_quality",
                "orb_valid",
                "of_steering_response",
                "of_forward_response",
            ],
        )
        writer.writeheader()
        for idx in range(int(pair_count)):
            pc = pc_rows[idx] if idx < len(pc_rows) else {}
            orb = orb_rows[idx] if idx < len(orb_rows) else {}
            of = of_rows[idx] if idx < len(of_rows) else {}
            writer.writerow(
                {
                    "frame_idx": int(idx),
                    "pc_label": str(pc.get("benchmark_label", pc.get("motion_state", "failed")) or "failed"),
                    "orb_label": str(orb.get("benchmark_label", "failed") or "failed"),
                    "of_label": str(of.get("label", "failed") or "failed"),
                    "pc_steering_response": _safe_float(pc.get("steering_response")),
                    "pc_forward_response": _safe_float(pc.get("forward_response")),
                    "orb_tracking_quality": _safe_float(orb.get("tracking_quality")),
                    "orb_valid": bool(orb.get("orb_valid", False)),
                    "of_steering_response": _safe_float(of.get("steering_response")),
                    "of_forward_response": _safe_float(of.get("forward_response")),
                }
            )
    tmp_path.replace(path)


def _write_summary_files(json_path, csv_path, report, identity=None):
    identity = dict(identity or {})
    rows = []
    for method, payload in dict(report.get("methods") or {}).items():
        item = dict(payload or {})
        pair_count = int(report.get("frame_pair_count", 0) or 0)
        valid_rate = item.get("valid_rate")
        valid_pair_count = None
        try:
            valid_pair_count = int(round(float(valid_rate) * float(pair_count)))
        except Exception:
            valid_pair_count = None
        row = {
            "method": method,
            "dataset": identity.get("dataset", ""),
            "sequence": identity.get("sequence", ""),
            "tag": identity.get("tag", ""),
            "frame_count": int(report.get("frame_count", 0) or 0),
            "pair_count": pair_count,
            "valid_pair_count": "" if valid_pair_count is None else valid_pair_count,
            "valid_rate": item.get("valid_rate"),
            "total_time_s": item.get("runtime_sec"),
            "avg_time_ms_per_pair": item.get("time_per_pair_ms"),
            "segment_count": item.get("segment_count"),
            "labels": "|".join(str(label) for label in list(item.get("labels") or [])),
        }
        rows.append(row)
    summary = {
        "schema_version": 1,
        "source": "exphub.encode.motion_benchmark",
        "identity": identity,
        "frame_count": int(report.get("frame_count", 0) or 0),
        "pair_count": int(report.get("frame_pair_count", 0) or 0),
        "methods": {row["method"]: dict(row) for row in rows},
        "warnings": list(report.get("warnings") or []),
    }
    write_json_atomic(json_path, summary, indent=2)
    fieldnames = [
        "method",
        "dataset",
        "sequence",
        "tag",
        "frame_count",
        "pair_count",
        "valid_pair_count",
        "valid_rate",
        "total_time_s",
        "avg_time_ms_per_pair",
        "segment_count",
        "labels",
    ]
    write_csv_atomic(csv_path, fieldnames, rows)
    return summary


def _text_tile(text, thumb_w, thumb_h, bg=(248, 249, 250)):
    import cv2

    tile = np.full((thumb_h, thumb_w, 3), bg, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (thumb_w - 1, thumb_h - 1), (210, 214, 219), 1, cv2.LINE_AA)
    cv2.putText(tile, str(text), (10, thumb_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (78, 84, 91), 1, cv2.LINE_AA)
    return tile


def _status_overlay(image, text, color=(220, 38, 38), alpha=0.30):
    import cv2

    if not text:
        return image
    out = image.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, -1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.64
    thickness = 2
    (tw, th), _ = cv2.getTextSize(str(text), font, scale, thickness)
    x = max(4, (out.shape[1] - tw) // 2)
    y = max(th + 4, (out.shape[0] + th) // 2)
    cv2.putText(out, str(text), (x, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, str(text), (x, y), font, scale, (70, 0, 0), thickness, cv2.LINE_AA)
    return out


def _orb_thumb(frames, grays, pair_index, thumb_w, thumb_h, orb_row=None):
    import cv2

    pair_index = int(np.clip(pair_index, 0, max(0, len(grays) - 2)))
    prev_gray = grays[pair_index]
    curr_gray = grays[pair_index + 1]
    gray_bg = _resize_direct(cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2RGB), thumb_w, thumb_h)
    canvas = cv2.addWeighted(gray_bg, 0.76, np.full_like(gray_bg, 18), 0.24, 0.0)
    orb = cv2.ORB_create(nfeatures=900)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return _status_overlay(canvas, "LOW FEATURE", alpha=0.42)
    good = []
    for pair in matcher.knnMatch(des1, des2, k=2):
        if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if not good:
        return _status_overlay(canvas, "LOW FEATURE", alpha=0.42)
    good = sorted(good, key=lambda item: (float(item.distance), -float(kp1[item.queryIdx].response)))
    h, w = prev_gray.shape[:2]
    sx = float(thumb_w) / max(1.0, float(w))
    sy = float(thumb_h) / max(1.0, float(h))
    quality = _safe_float(_as_dict(orb_row).get("tracking_quality"), 1.0)
    label = str(_as_dict(orb_row).get("benchmark_label", "") or "")
    state = str(_as_dict(orb_row).get("tracking_state", "") or "")
    max_points = 40 if label == "failed" or state == "lost" else (70 if quality < float(THRESHOLDS["orb_weak_quality"]) else 95)
    for match in good[:max_points]:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        p1 = (int(np.clip(x1 * sx, 0, thumb_w - 1)), int(np.clip(y1 * sy, 0, thumb_h - 1)))
        p2 = (int(np.clip(x2 * sx, 0, thumb_w - 1)), int(np.clip(y2 * sy, 0, thumb_h - 1)))
        cv2.arrowedLine(canvas, p1, p2, (0, 210, 0), 1, cv2.LINE_AA, tipLength=0.22)
        cv2.rectangle(canvas, (p1[0] - 3, p1[1] - 3), (p1[0] + 3, p1[1] + 3), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (80, 255, 80), -1, cv2.LINE_AA)
    if label == "failed" or state == "lost":
        canvas = _status_overlay(canvas, "FAILED", alpha=0.34)
    elif quality < float(THRESHOLDS["orb_weak_quality"]) or state == "weak":
        canvas = _status_overlay(canvas, "WEAK", color=(214, 117, 20), alpha=0.22)
    cv2.rectangle(canvas, (0, 0), (thumb_w - 1, thumb_h - 1), (200, 220, 200), 1, cv2.LINE_AA)
    return canvas


def _flow_to_rgb(flow):
    import cv2

    dx = flow[..., 0]
    dy = flow[..., 1]
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.asarray(ang / 2.0, dtype=np.uint8)
    hsv[..., 1] = 220
    vmax = float(np.percentile(mag[np.isfinite(mag)], 95)) if np.any(np.isfinite(mag)) else 1.0
    vmax = max(vmax, 1e-6)
    hsv[..., 2] = np.asarray(np.clip(mag / vmax * 255.0, 35.0, 255.0), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _of_thumb(frames, grays, pair_index, thumb_w, thumb_h):
    import cv2

    pair_index = int(np.clip(pair_index, 0, max(0, len(grays) - 2)))
    flow = cv2.calcOpticalFlowFarneback(
        grays[pair_index],
        grays[pair_index + 1],
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0,
    )
    if flow is None or not np.all(np.isfinite(flow)):
        return _text_tile("OF unavailable", thumb_w, thumb_h)
    h, w = flow.shape[:2]
    flow_rgb = _flow_to_rgb(flow)
    canvas = cv2.resize(flow_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    for y in np.linspace(0.18 * h, 0.82 * h, 4):
        for x in np.linspace(0.12 * w, 0.88 * w, 6):
            x_i = int(np.clip(round(x), 0, w - 1))
            y_i = int(np.clip(round(y), 0, h - 1))
            dx, dy = flow[y_i, x_i]
            sx = float(thumb_w) / max(1.0, float(w))
            sy = float(thumb_h) / max(1.0, float(h))
            p1 = (int(x_i * sx), int(y_i * sy))
            p2 = (
                int(np.clip((x_i + 4.0 * float(dx)) * sx, 0, thumb_w - 1)),
                int(np.clip((y_i + 4.0 * float(dy)) * sy, 0, thumb_h - 1)),
            )
            cv2.arrowedLine(canvas, p1, p2, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.circle(canvas, p1, 2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (thumb_w - 1, thumb_h - 1), (226, 229, 232), 1, cv2.LINE_AA)
    return canvas


def _sample_strip(frames, grays, indices, warnings, orb_rows=None):
    import cv2

    thumb_w = 180
    thumb_h = 95
    gap_x = 8
    gap_y = 5
    indices = [int(item) for item in list(indices or [])]
    total_w = len(indices) * thumb_w + max(0, len(indices) - 1) * gap_x
    total_h = 4 * thumb_h + 3 * gap_y
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    for col, pair_idx in enumerate(indices):
        x0 = col * (thumb_w + gap_x)
        safe_idx = int(np.clip(pair_idx, 0, max(0, len(frames) - 1)))
        rows = []
        rgb = _resize_direct(_read_rgb(frames[safe_idx]), thumb_w, thumb_h)
        cv2.putText(rgb, str(int(pair_idx)), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(rgb)
        rows.append(_pc_thumb(grays, pair_idx, thumb_w, thumb_h))
        try:
            orb_row = orb_rows[pair_idx] if orb_rows is not None and pair_idx < len(orb_rows) else None
            rows.append(_orb_thumb(frames, grays, pair_idx, thumb_w, thumb_h, orb_row=orb_row))
        except Exception as exc:
            warnings.append("ORB visualization unavailable for pair {}: {}".format(int(pair_idx), exc))
            rows.append(_text_tile("ORB unavailable", thumb_w, thumb_h))
        try:
            rows.append(_of_thumb(frames, grays, pair_idx, thumb_w, thumb_h))
        except Exception as exc:
            warnings.append("OF visualization unavailable for pair {}: {}".format(int(pair_idx), exc))
            rows.append(_text_tile("OF unavailable", thumb_w, thumb_h))
        for row_idx, tile in enumerate(rows):
            y0 = row_idx * (thumb_h + gap_y)
            canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = tile
    return canvas


def _plot_band(ax, labels, title, sample_indices=None):
    labels = _labels(labels)
    if not labels:
        labels = ["failed"]
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] == current:
            continue
        ax.axvspan(start, idx, color=BENCHMARK_COLORS.get(current, "#E5E7EA"), alpha=0.88, lw=0)
        current = labels[idx]
        start = idx
    ax.axvspan(start, len(labels), color=BENCHMARK_COLORS.get(current, "#E5E7EA"), alpha=0.88, lw=0)
    for sample_idx in list(sample_indices or []):
        if 0 <= int(sample_idx) < len(labels):
            ax.axvline(int(sample_idx) + 0.5, color="#2B2F33", linewidth=0.55, alpha=0.55)
    ax.set_xlim(0, max(1, len(labels)))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(
        title,
        rotation=0,
        ha="right",
        va="center",
        labelpad=34,
        fontsize=LABEL_STYLE["fontsize"],
        fontweight=LABEL_STYLE["fontweight"],
        color=LABEL_STYLE["color"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#9AA1A9")
    ax.tick_params(axis="x", colors="#4A4F55", length=2.0, width=0.7, labelsize=8.5)


def _write_overview(path, frames, grays, pc_rows, pc_labels, orb_rows, orb_labels, of_rows, of_labels, warnings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = Path(path).resolve()
    pair_count = max(len(pc_labels), len(orb_labels), len(of_labels))
    sample_indices = np.linspace(0, max(0, pair_count - 1), num=min(8, max(1, pair_count)), dtype=int).tolist()
    has_strip = True
    try:
        strip = _sample_strip(frames, grays, sample_indices, warnings, orb_rows=orb_rows)
    except Exception as exc:
        has_strip = False
        strip = None
        warnings.append("sample-row visualization unavailable: {}".format(exc))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    height_ratios = [3.9, 0.18, 0.18, 0.18] if has_strip else [0.26, 0.26, 0.26]
    fig = plt.figure(figsize=(13.5, 5.5 if has_strip else 2.4), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)
    row = 0
    if has_strip:
        ax_img = fig.add_subplot(grid[row, 0])
        row += 1
        ax_img.imshow(strip)
        ax_img.axis("off")
        for y_pos, label in [(0.875, "RGB"), (0.625, "PC"), (0.375, "ORB"), (0.125, "OF")]:
            ax_img.text(-0.010, y_pos, label, transform=ax_img.transAxes, ha="right", va="center", **LABEL_STYLE)
        ax_img.set_title("Encode Motion Benchmark Samples", pad=8, loc="left")

    ax_pc = fig.add_subplot(grid[row, 0])
    row += 1
    ax_orb = fig.add_subplot(grid[row, 0], sharex=ax_pc)
    row += 1
    ax_of = fig.add_subplot(grid[row, 0], sharex=ax_pc)
    row += 1
    _plot_band(ax_pc, pc_labels, "PC", sample_indices)
    _plot_band(ax_orb, orb_labels, "ORB", sample_indices)
    _plot_band(ax_of, of_labels, "OF", sample_indices)
    ax_pc.tick_params(axis="x", length=0, labelbottom=False)
    ax_orb.tick_params(axis="x", length=0, labelbottom=False)
    ax_of.set_xlabel("Frame Pair Index", labelpad=8)

    handles = [Patch(color=BENCHMARK_COLORS[label], label=label) for label in BENCHMARK_LABELS]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.982), frameon=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.875, bottom=0.085, hspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=220, facecolor="white", transparent=False)
    plt.close(fig)


def _write_bands_only_overview(path, pc_labels, orb_labels, of_labels, warnings, sample_indices=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = Path(path).resolve()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.5, 2.6), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.26, 0.26, 0.26])
    axes = [fig.add_subplot(grid[idx, 0]) for idx in range(3)]
    _plot_band(axes[0], pc_labels, "PC", sample_indices)
    _plot_band(axes[1], orb_labels, "ORB", sample_indices)
    _plot_band(axes[2], of_labels, "OF", sample_indices)
    axes[0].tick_params(axis="x", length=0, labelbottom=False)
    axes[1].tick_params(axis="x", length=0, labelbottom=False)
    axes[2].set_xlabel("Frame Pair Index", labelpad=8)
    axes[0].set_title("Encode Motion Benchmark Segmentation Bands", pad=8, loc="left")
    if warnings:
        axes[2].text(
            0.0,
            -0.85,
            "Visualization fallback: {}".format("; ".join(warnings[-2:])),
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#6B7280",
        )
    fig.legend(
        handles=[Patch(color=BENCHMARK_COLORS[label], label=label) for label in BENCHMARK_LABELS],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.22, hspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=220, facecolor="white", transparent=False)
    plt.close(fig)


def run_motion_benchmark(
    prepare_result,
    frames_dir,
    encode_dir,
    exp_dir=None,
    formal_encode_sec_without_benchmark=None,
    identity=None,
):
    started = time.perf_counter()
    encode_dir = Path(encode_dir).resolve()
    exp_dir = Path(exp_dir).resolve() if exp_dir is not None else encode_dir.parent
    frame_dir = ensure_dir(frames_dir, "prepare frames dir")
    frames = list_frames_sorted(frame_dir)
    prepare = _as_dict(prepare_result)
    frame_count = int(prepare.get("num_frames", len(frames)) or len(frames))
    if len(frames) != frame_count:
        raise RuntimeError("prepare frame count mismatch: files={} prepare_result={}".format(len(frames), frame_count))
    pair_count = max(0, frame_count - 1)
    warnings = []

    log_info("encode motion benchmark start: methods=pc,orb,optical_flow")
    read_started = time.perf_counter()
    grays = [_read_gray(frame) for frame in frames]
    read_gray_sec = float(time.perf_counter() - read_started)

    pc_rows, pc_labels, pc_runtime = _pc_method(grays)
    orb_rows, orb_labels, orb_runtime, failed_intervals, orb_extra = _orb_method(grays)
    of_rows, of_labels, of_runtime = _of_method(grays)

    csv_path = encode_dir / "pairs.csv"
    overview_path = encode_dir / "overview.png"
    report_path = encode_dir / "report.json"
    summary_path = encode_dir / "summary.json"
    summary_csv_path = encode_dir / "summary.csv"
    _write_csv(csv_path, pc_rows, orb_rows, of_rows, pair_count)
    overview_temp_path = unique_sibling_temp_path(overview_path)
    try:
        _write_overview(overview_temp_path, frames, grays, pc_rows, pc_labels, orb_rows, orb_labels, of_rows, of_labels, warnings)
    except Exception as exc:
        warnings.append("benchmark overview fallback used: {}".format(exc))
        fallback_indices = np.linspace(0, max(0, pair_count - 1), num=min(8, max(1, pair_count)), dtype=int).tolist()
        _write_bands_only_overview(overview_temp_path, pc_labels, orb_labels, of_labels, warnings, sample_indices=fallback_indices)
    replace_nonempty_file(overview_temp_path, overview_path, "motion benchmark overview")

    motion_benchmark_sec = float(time.perf_counter() - started)
    pc_cost = max(float(pc_runtime), PC_EPS)
    report = {
        "enabled": True,
        "benchmark_enabled": True,
        "benchmark_scope": "experimental_overhead_not_default_inference_cost",
        "frame_count": int(frame_count),
        "frame_pair_count": int(pair_count),
        "read_gray_sec": float(read_gray_sec),
        "motion_benchmark_sec": float(motion_benchmark_sec),
        "formal_encode_sec_without_benchmark": (
            None if formal_encode_sec_without_benchmark is None else float(formal_encode_sec_without_benchmark)
        ),
        "thresholds": dict(THRESHOLDS),
        "orb_label_config": dict(ORB_LABEL_CONFIG),
        "methods": {
            "phase_correlation": _method_summary(pc_runtime, pc_labels, pair_count),
            "orb": _method_summary(orb_runtime, orb_labels, pair_count, orb_extra),
            "optical_flow": _method_summary(of_runtime, of_labels, pair_count),
        },
        "relative_cost": {
            "pc_as_1x": {
                "orb": float(orb_runtime) / pc_cost,
                "optical_flow": float(of_runtime) / pc_cost,
            }
        },
        "outputs": {
            "csv": _relative(exp_dir, csv_path),
            "overview": _relative(exp_dir, overview_path),
            "summary": _relative(exp_dir, summary_path),
            "summary_csv": _relative(exp_dir, summary_csv_path),
        },
        "warnings": list(dict.fromkeys(warnings)),
        "visualization_warnings": list(dict.fromkeys(warnings)),
        "orb_failed_intervals": failed_intervals,
    }
    write_json_atomic(report_path, report, indent=2)
    _write_summary_files(
        summary_path,
        summary_csv_path,
        report,
        identity=identity,
    )
    log_info(
        "encode motion benchmark done: report={} overview={}".format(
            _relative(exp_dir, report_path),
            _relative(exp_dir, overview_path),
        )
    )
    if warnings:
        log_warn("encode motion benchmark warnings: {}".format("; ".join(warnings[:4])))
    return report
