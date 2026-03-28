#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts._common import ensure_dir, ensure_file, list_frames_sorted, write_json_atomic
from scripts._segment.research.kinematics import moving_average
from scripts._segment.signal_extraction import (
    FORMAL_STATE_INPUT_COLUMNS,
    FORMAL_STATE_INPUT_SIGNALS,
    build_formal_state_input_rows,
    extract_signal_timeseries_from_frames,
)


STATE_LOW = "low_state"
STATE_HIGH = "high_state"

DEFAULT_INPUT_SIGNALS = list(FORMAL_STATE_INPUT_SIGNALS)
DEFAULT_NORMALIZATION_METHOD = "processed_formal_state_inputs_only"
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_WEIGHTS = {
    "motion_velocity": 0.75,
    "semantic_velocity": 0.25,
}
DEFAULT_ENTER_TH = 0.65
DEFAULT_EXIT_TH = 0.45
DEFAULT_MIN_HIGH_LEN = 24
DEFAULT_MIN_LOW_LEN = 24
DEFAULT_GLITCH_MERGE_LEN = 12

REPORT_SCHEMA_VERSION = "state_report.v3"
_INPUT_CSV_NAME = "signal_timeseries.csv"
_OUTPUT_JSON_NAME = "state_segments.json"
_OUTPUT_REPORT_NAME = "state_report.json"

_LEGACY_STATE_OUTPUT_NAMES = [
    "density_schedule.csv",
    "density_schedule_overview.png",
    "state_segments.csv",
    "state_timeline.csv",
    "state_candidate_compare.json",
    "state_segmentation_meta.json",
    "state_segmentation_overview.png",
    "state_signal_overlay.png",
    "state_signal_candidate_compare.png",
]

_DETECTOR_TYPE = "high_risk_interval_detector"


def _optional_float(value):
    text = str(value or "").strip()
    if text == "":
        return None
    return float(text)


def _read_signal_rows(csv_path):
    csv_path = ensure_file(csv_path, name="state segmentation input csv")
    rows = []
    required_columns = set(["frame_idx", "timestamp"] + list(DEFAULT_INPUT_SIGNALS))
    with open(str(csv_path), "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(required_columns.difference(fieldnames))
        if missing:
            raise SystemExit(
                "[ERR] state segmentation input missing columns: {} ({})".format(
                    ", ".join(missing),
                    csv_path,
                )
            )
        for row in reader:
            rows.append(
                {
                    "frame_idx": int(row.get("frame_idx", 0) or 0),
                    "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                    "appearance_delta": float(row.get("appearance_delta", 0.0) or 0.0),
                    "motion_velocity": float(row.get("motion_velocity", 0.0) or 0.0),
                    "blur_score": float(row.get("blur_score", 0.0) or 0.0),
                    "semantic_velocity": float(row.get("semantic_velocity", 0.0) or 0.0),
                    "motion_velocity_state_input": _optional_float(row.get("motion_velocity_state_input")),
                    "semantic_velocity_state_input": _optional_float(row.get("semantic_velocity_state_input")),
                }
            )
    if not rows:
        raise SystemExit("[ERR] state segmentation input is empty: {}".format(csv_path))
    return rows


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _read_timestamps(path):
    values = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            values.append(float(text))
    return values


def _coerce_signal_rows(rows):
    coerced = []
    for row in list(rows or []):
        coerced.append(
            {
                "frame_idx": int(row.get("frame_idx", 0) or 0),
                "timestamp": float(row.get("timestamp", row.get("ts_sec", 0.0)) or 0.0),
                "appearance_delta": float(row.get("appearance_delta", 0.0) or 0.0),
                "motion_velocity": float(row.get("motion_velocity", 0.0) or 0.0),
                "blur_score": float(row.get("blur_score", 0.0) or 0.0),
                "semantic_velocity": float(row.get("semantic_velocity", 0.0) or 0.0),
                "motion_velocity_state_input": _optional_float(row.get("motion_velocity_state_input")),
                "semantic_velocity_state_input": _optional_float(row.get("semantic_velocity_state_input")),
            }
        )
    if not coerced:
        raise ValueError("state segmentation requires non-empty signal rows")
    return coerced


def _formal_input_series(rows, signal_name):
    processed_column = FORMAL_STATE_INPUT_COLUMNS[signal_name]
    values = []
    has_all_values = True
    for row in list(rows or []):
        value = row.get(processed_column)
        if value is None:
            has_all_values = False
            break
        values.append(float(value))
    return has_all_values, values


def _resolve_formal_state_inputs(rows):
    rows = list(rows or [])
    if not rows:
        return [], {
            "source_mode": "missing",
            "signal_names": list(DEFAULT_INPUT_SIGNALS),
            "processed_columns": dict(FORMAL_STATE_INPUT_COLUMNS),
            "signals": {},
            "analysis_only_note": (
                "Signals outside formal_state_inputs remain analysis sidecar observations and do not "
                "enter the official state mainline score."
            ),
        }

    persisted_meta = {}
    has_all_persisted = True
    for signal_name in DEFAULT_INPUT_SIGNALS:
        available, values = _formal_input_series(rows, signal_name)
        if not available:
            has_all_persisted = False
            break
        persisted_meta[signal_name] = {
            "raw_column": str(signal_name),
            "processed_column": str(FORMAL_STATE_INPUT_COLUMNS[signal_name]),
            "processed_stats": _series_stats(values),
        }

    if has_all_persisted:
        _rebuilt_rows, rebuilt_meta = build_formal_state_input_rows(rows)
        rebuilt_meta["source_mode"] = "persisted_formal_state_inputs"
        for signal_name in DEFAULT_INPUT_SIGNALS:
            rebuilt_meta.setdefault("signals", {}).setdefault(signal_name, {})
            rebuilt_meta["signals"][signal_name]["processed_stats"] = dict(persisted_meta[signal_name]["processed_stats"])
        return rows, {
            "source_mode": str(rebuilt_meta.get("source_mode", "persisted_formal_state_inputs")),
            "signal_names": list(rebuilt_meta.get("signal_names", []) or []),
            "processed_columns": dict(rebuilt_meta.get("processed_columns", {}) or {}),
            "pipeline": list(rebuilt_meta.get("pipeline", []) or []),
            "signals": dict(rebuilt_meta.get("signals", {}) or {}),
            "analysis_only_note": str(rebuilt_meta.get("analysis_only_note", "") or ""),
        }

    rebuilt_rows, rebuild_meta = build_formal_state_input_rows(rows)
    rebuild_meta["source_mode"] = "rebuilt_from_observed_raw_signals"
    return rebuilt_rows, rebuild_meta


def _series_stats(values):
    arr = np.asarray(list(values or []), dtype=np.float32)
    if arr.size <= 0:
        return {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "std": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
        }
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def _mean(values):
    values = list(values or [])
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _clamp01(value):
    return float(min(1.0, max(0.0, float(value))))


def _series_quantile(values, q, default_value=0.0):
    arr = np.asarray(list(values or []), dtype=np.float32)
    if arr.size <= 0:
        return float(default_value)
    return float(np.quantile(arr, float(q)))


def _trailing_mean(values, window):
    values = [float(value) for value in list(values or [])]
    window = max(1, int(window))
    out = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += float(value)
        if idx >= int(window):
            running_sum -= float(values[idx - int(window)])
        denom = min(idx + 1, int(window))
        out.append(float(running_sum / float(max(1, denom))))
    return out


def _median_dt(timestamps):
    diffs = []
    prev_value = None
    for value in list(timestamps or []):
        current = float(value)
        if prev_value is not None:
            diff = float(current - prev_value)
            if diff > 1e-9:
                diffs.append(diff)
        prev_value = current
    if not diffs:
        return 0.0
    return float(np.median(np.asarray(diffs, dtype=np.float32)))


def _duration_sec(timestamps, start_idx, end_idx, dt_sec):
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(timestamps):
        return 0.0
    if end_idx == start_idx:
        return float(dt_sec) if float(dt_sec) > 0.0 else 0.0
    duration = float(timestamps[end_idx]) - float(timestamps[start_idx])
    if float(dt_sec) > 0.0:
        duration += float(dt_sec)
    return max(0.0, float(duration))


def _resolve_weights(weights):
    resolved = dict(DEFAULT_WEIGHTS)
    for key, value in dict(weights or {}).items():
        if key in resolved:
            resolved[key] = float(value)
    total = float(sum(max(0.0, float(value)) for value in resolved.values()))
    if total <= 1e-9:
        resolved = dict(DEFAULT_WEIGHTS)
        total = float(sum(DEFAULT_WEIGHTS.values()))
    normalized = {}
    for key in sorted(resolved.keys()):
        normalized[key] = float(max(0.0, float(resolved[key])) / float(total))
    return normalized


def _build_state_score(motion_values, semantic_values, weights, smoothing_window):
    score_raw = []
    for idx in range(len(motion_values)):
        score_raw.append(
            float(
                weights["motion_velocity"] * float(motion_values[idx])
                + weights["semantic_velocity"] * float(semantic_values[idx])
            )
        )
    score_values, actual_window = moving_average(score_raw, int(smoothing_window))
    score_values = [float(min(1.0, max(0.0, value))) for value in list(score_values or [])]
    return list(score_raw), list(score_values), {
        "method": "weighted_sum_plus_moving_average",
        "description": (
            "Official state score uses only processed motion_velocity and semantic_velocity, "
            "then applies one light moving-average smoothing pass."
        ),
        "weights": dict(weights),
        "smoothing_window": int(actual_window),
        "input_range": [0.0, 1.0],
        "score_raw_stats": _series_stats(score_raw),
        "score_stats": _series_stats(score_values),
    }


def _fill_short_false_gaps(mask, max_gap):
    filled = [bool(item) for item in list(mask or [])]
    max_gap = max(0, int(max_gap))
    if max_gap <= 0 or len(filled) <= 2:
        return filled, {
            "maximum_gap_frames": int(max_gap),
            "merge_count": 0,
            "applied_gaps": [],
        }

    applied = []
    idx = 0
    length = len(filled)
    while idx < length:
        if bool(filled[idx]):
            idx += 1
            continue
        gap_start = int(idx)
        while idx < length and not bool(filled[idx]):
            idx += 1
        gap_end = int(idx - 1)
        gap_len = int(gap_end - gap_start + 1)
        if gap_start > 0 and idx < length and gap_len <= int(max_gap):
            for fill_idx in range(gap_start, gap_end + 1):
                filled[fill_idx] = True
            applied.append(
                {
                    "gap_start": int(gap_start),
                    "gap_end": int(gap_end),
                    "gap_frames": int(gap_len),
                }
            )
    return filled, {
        "maximum_gap_frames": int(max_gap),
        "merge_count": int(len(applied)),
        "applied_gaps": applied,
    }


def _active_intervals(mask):
    intervals = []
    start_idx = None
    for idx, flag in enumerate(list(mask or [])):
        if bool(flag):
            if start_idx is None:
                start_idx = int(idx)
            continue
        if start_idx is not None:
            intervals.append((int(start_idx), int(idx - 1)))
            start_idx = None
    if start_idx is not None:
        intervals.append((int(start_idx), int(len(mask) - 1)))
    return intervals


def _merge_close_intervals(intervals, max_gap):
    intervals = list(intervals or [])
    max_gap = max(0, int(max_gap))
    if not intervals:
        return [], {
            "merge_count": 0,
            "maximum_gap_frames": int(max_gap),
            "applied_merges": [],
        }

    merged = [list(intervals[0])]
    applied = []
    for start_idx, end_idx in list(intervals[1:]):
        prev_start, prev_end = merged[-1]
        gap = int(start_idx - prev_end - 1)
        if gap <= int(max_gap):
            merged[-1][1] = int(end_idx)
            applied.append(
                {
                    "left_end": int(prev_end),
                    "right_start": int(start_idx),
                    "gap_frames": int(max(0, gap)),
                }
            )
        else:
            merged.append([int(start_idx), int(end_idx)])
    return [(int(item[0]), int(item[1])) for item in merged], {
        "merge_count": int(len(applied)),
        "maximum_gap_frames": int(max_gap),
        "applied_merges": applied,
    }


def _merge_intervals_on_elevated_gap(intervals, state_scores, gap_floor, max_gap):
    intervals = list(intervals or [])
    state_scores = list(state_scores or [])
    gap_floor = float(gap_floor)
    max_gap = max(0, int(max_gap))
    if not intervals:
        return [], {
            "merge_count": 0,
            "maximum_gap_frames": int(max_gap),
            "gap_floor": float(gap_floor),
            "applied_merges": [],
        }

    merged = [list(intervals[0])]
    applied = []
    for start_idx, end_idx in list(intervals[1:]):
        prev_start, prev_end = merged[-1]
        gap_len = int(start_idx - prev_end - 1)
        gap_slice = list(state_scores[int(prev_end + 1) : int(start_idx)] or [])
        gap_mean = _mean(gap_slice)
        if gap_len > 0 and gap_len <= int(max_gap) and float(gap_mean) >= float(gap_floor):
            merged[-1][1] = int(end_idx)
            applied.append(
                {
                    "left_end": int(prev_end),
                    "right_start": int(start_idx),
                    "gap_frames": int(gap_len),
                    "gap_mean": float(gap_mean),
                }
            )
        else:
            merged.append([int(start_idx), int(end_idx)])
    return [(int(item[0]), int(item[1])) for item in merged], {
        "merge_count": int(len(applied)),
        "maximum_gap_frames": int(max_gap),
        "gap_floor": float(gap_floor),
        "applied_merges": applied,
    }


def _filter_short_intervals(intervals, min_len):
    kept = []
    dropped = []
    min_len = max(1, int(min_len))
    for start_idx, end_idx in list(intervals or []):
        duration = int(end_idx - start_idx + 1)
        if duration < int(min_len):
            dropped.append(
                {
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "duration_frames": int(duration),
                }
            )
            continue
        kept.append((int(start_idx), int(end_idx)))
    return kept, {
        "minimum_interval_frames": int(min_len),
        "dropped_count": int(len(dropped)),
        "dropped_intervals": dropped,
    }


def _expand_interval_to_regime(start_idx, end_idx, detector_rows, max_extension):
    detector_rows = list(detector_rows or [])
    frame_count = len(detector_rows)
    start_idx = max(0, int(start_idx))
    end_idx = min(int(frame_count - 1), int(end_idx))
    max_extension = max(0, int(max_extension))
    start_extension = 0
    end_extension = 0

    while start_idx > 0 and start_extension < int(max_extension):
        prev_idx = int(start_idx - 1)
        prev_row = dict(detector_rows[prev_idx] or {})
        prev_regime_mean = float(prev_row.get("regime_mean", 0.0) or 0.0)
        prev_hold_floor = float(prev_row.get("hold_floor", 0.0) or 0.0)
        prev_detector_score = float(prev_row.get("detector_score", 0.0) or 0.0)
        prev_evidence = float(prev_row.get("positive_evidence", 0.0) or 0.0)
        if (
            prev_regime_mean >= float(prev_hold_floor)
            or prev_detector_score >= 0.30
            or prev_evidence >= 0.18
        ):
            start_idx -= 1
            start_extension += 1
            continue
        break

    while end_idx + 1 < frame_count and end_extension < int(max_extension):
        next_idx = int(end_idx + 1)
        next_row = dict(detector_rows[next_idx] or {})
        next_regime_mean = float(next_row.get("regime_mean", 0.0) or 0.0)
        next_hold_floor = float(next_row.get("hold_floor", 0.0) or 0.0)
        next_detector_score = float(next_row.get("detector_score", 0.0) or 0.0)
        next_evidence = float(next_row.get("positive_evidence", 0.0) or 0.0)
        if (
            next_regime_mean >= float(next_hold_floor)
            or next_detector_score >= 0.25
            or next_evidence >= 0.15
        ):
            end_idx += 1
            end_extension += 1
            continue
        break

    return int(start_idx), int(end_idx), {
        "start_extension_frames": int(start_extension),
        "end_extension_frames": int(end_extension),
    }


def _expand_intervals_to_regimes(intervals, detector_rows, shoulder_extension_frames):
    detector_rows = list(detector_rows or [])
    hold_floors = [float(row.get("hold_floor", 0.0) or 0.0) for row in detector_rows]
    global_hold_floor = _series_quantile(hold_floors, 0.50, default_value=0.0)
    expanded = []
    for start_idx, end_idx in list(intervals or []):
        expanded_start, expanded_end, extension_meta = _expand_interval_to_regime(
            start_idx=start_idx,
            end_idx=end_idx,
            detector_rows=detector_rows,
            max_extension=shoulder_extension_frames,
        )
        expanded.append(
            {
                "source_interval": [int(start_idx), int(end_idx)],
                "expanded_interval": [int(expanded_start), int(expanded_end)],
                "shoulder_extension": extension_meta,
            }
        )
    return [(int(item["expanded_interval"][0]), int(item["expanded_interval"][1])) for item in expanded], {
        "global_hold_floor": float(global_hold_floor),
        "expanded_count": int(len(expanded)),
        "shoulder_extension_frames": int(max(0, int(shoulder_extension_frames))),
        "expanded_intervals": expanded,
    }


def _filter_low_score_intervals(intervals, state_scores, score_stats):
    score_stats = dict(score_stats or {})
    score_mean_floor = max(
        _series_quantile(state_scores, 0.60, default_value=float(score_stats.get("q50", 0.0) or 0.0)),
        float(score_stats.get("mean", 0.0) or 0.0) + 0.02,
    )
    score_peak_floor = max(
        float(score_stats.get("q90", 0.0) or 0.0),
        _series_quantile(state_scores, 0.90, default_value=float(score_stats.get("q90", 0.0) or 0.0)),
    )
    kept = []
    dropped = []
    for start_idx, end_idx in list(intervals or []):
        score_slice = list(state_scores[int(start_idx) : int(end_idx) + 1] or [])
        interval_mean = _mean(score_slice)
        interval_peak = float(max(score_slice)) if score_slice else 0.0
        if interval_mean >= float(score_mean_floor) or interval_peak >= float(score_peak_floor):
            kept.append((int(start_idx), int(end_idx)))
            continue
        dropped.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "score_mean": float(interval_mean),
                "score_peak": float(interval_peak),
            }
        )
    return kept, {
        "score_mean_floor": float(score_mean_floor),
        "score_peak_floor": float(score_peak_floor),
        "dropped_count": int(len(dropped)),
        "dropped_intervals": dropped,
    }


def _resolve_detector_windows(frame_count, smoothing_window, min_low_len):
    recent_window = max(6, min(12, int(max(4, smoothing_window))))
    regime_window = max(int(recent_window * 3), int(recent_window + 24))
    baseline_window = max(int(min_low_len * 3), int(regime_window * 2))
    regime_window = min(int(regime_window), int(max(recent_window + 2, frame_count)))
    baseline_window = min(int(max(regime_window + 2, baseline_window)), int(max(regime_window + 2, frame_count)))
    shoulder_extension = min(int(regime_window), int(max(recent_window * 2, regime_window)))
    return {
        "recent_window_frames": int(recent_window),
        "regime_window_frames": int(regime_window),
        "baseline_window_frames": int(baseline_window),
        "shoulder_extension_frames": int(max(0, shoulder_extension)),
    }


def _build_regime_change_rows(scores, smoothing_window, min_low_len):
    scores = [float(value) for value in list(scores or [])]
    frame_count = int(len(scores))
    window_meta = _resolve_detector_windows(frame_count, smoothing_window, min_low_len)
    recent_window = int(window_meta["recent_window_frames"])
    regime_window = int(window_meta["regime_window_frames"])
    baseline_window = int(window_meta["baseline_window_frames"])
    global_stats = _series_stats(scores)
    global_q50 = _series_quantile(scores, 0.50, default_value=float(global_stats.get("q50", 0.0) or 0.0))
    global_q65 = _series_quantile(scores, 0.65, default_value=float(global_stats.get("q50", 0.0) or 0.0))
    global_q90 = _series_quantile(scores, 0.90, default_value=float(global_stats.get("q90", 0.0) or 0.0))
    floor_std = max(0.025, float(global_stats.get("std", 0.0) or 0.0) * 0.20)
    recent_means = _trailing_mean(scores, recent_window)
    regime_means = _trailing_mean(scores, regime_window)
    baseline_alpha_up = 1.0 / float(max(1, baseline_window * 4))
    baseline_alpha_down = 1.0 / float(max(1, recent_window * 2))
    baseline_mean = float(regime_means[0]) if regime_means else 0.0
    baseline_var = 0.0
    posterior = 0.0
    rows = []

    for idx in range(frame_count):
        recent_mean = float(recent_means[idx])
        regime_mean = float(regime_means[idx])
        alpha = float(baseline_alpha_up) if regime_mean > baseline_mean else float(baseline_alpha_down)
        baseline_mean = float((1.0 - alpha) * baseline_mean + alpha * regime_mean)
        baseline_diff = float(regime_mean - baseline_mean)
        baseline_var = float((1.0 - alpha) * baseline_var + alpha * (baseline_diff * baseline_diff))
        baseline_std = max(float(np.sqrt(max(0.0, baseline_var))), float(floor_std))

        activation_floor = max(
            float(global_q65),
            float(baseline_mean) + 0.20 * float(baseline_std),
        )
        hold_floor = max(
            float(global_q50) + 0.01,
            float(baseline_mean) + 0.02 * float(baseline_std),
        )
        level_excess = max(0.0, float(regime_mean) - float(activation_floor))
        hold_excess = max(0.0, float(regime_mean) - float(hold_floor))
        shift_excess = max(0.0, float(regime_mean) - float(baseline_mean))
        recovery_excess = max(
            0.0,
            max(float(global_q50) + 0.01, float(baseline_mean) - 0.02) - float(regime_mean),
        )

        level_proxy = _clamp01(level_excess / max(0.08, float(global_q90) - float(global_q65)))
        hold_proxy = _clamp01(hold_excess / max(0.10, float(global_q90) - float(global_q50)))
        shift_proxy = _clamp01(shift_excess / max(0.10, 2.8 * float(baseline_std)))
        recovery_proxy = _clamp01(recovery_excess / max(0.07, float(global_q65) - float(global_q50) + 0.03))
        positive_evidence = (
            0.45 * float(level_proxy)
            + 0.35 * float(hold_proxy)
            + 0.20 * float(shift_proxy)
        )
        detector_target = float(
            0.50 * float(hold_proxy)
            + 0.30 * float(level_proxy)
            + 0.20 * float(shift_proxy)
        )
        posterior = _clamp01(
            float(posterior) * 0.97
            + 0.11 * float(positive_evidence)
            - 0.06 * float(recovery_proxy)
        )
        if detector_target > posterior:
            posterior = _clamp01(0.90 * float(posterior) + 0.10 * float(detector_target))
        else:
            posterior = _clamp01(0.96 * float(posterior) + 0.04 * float(detector_target))

        rows.append(
            {
                "frame_idx": int(idx),
                "baseline_mean": float(baseline_mean),
                "baseline_std": float(baseline_std),
                "recent_mean": float(recent_mean),
                "regime_mean": float(regime_mean),
                "activation_floor": float(activation_floor),
                "hold_floor": float(hold_floor),
                "regime_shift": float(shift_excess),
                "level_proxy": float(level_proxy),
                "hold_proxy": float(hold_proxy),
                "shift_proxy": float(shift_proxy),
                "positive_evidence": float(positive_evidence),
                "recovery_evidence": float(recovery_proxy),
                "detector_target": float(detector_target),
                "detector_score": float(posterior),
            }
        )

    return rows, {
        "detector_type": _DETECTOR_TYPE,
        "online": True,
        "trailing_style": True,
        "uses_local_mean": True,
        "uses_local_spread": True,
        "baseline_window_frames": int(baseline_window),
        "recent_window_frames": int(recent_window),
        "regime_window_frames": int(regime_window),
        "shoulder_extension_frames": int(window_meta["shoulder_extension_frames"]),
        "global_score_stats": global_stats,
        "global_q50": float(global_q50),
        "global_q65": float(global_q65),
        "global_q90": float(global_q90),
        "floor_std": float(floor_std),
        "baseline_update_mode": "asymmetric_ewma",
        "posterior_update_mode": "slow_rise_slow_decay",
        "baseline_alpha_up": float(baseline_alpha_up),
        "baseline_alpha_down": float(baseline_alpha_down),
    }


def _apply_detector_hysteresis(detector_scores, enter_th, exit_th):
    mask = []
    active = False
    for score in list(detector_scores or []):
        score = float(score)
        if active:
            active = bool(score >= float(exit_th))
        else:
            active = bool(score >= float(enter_th))
        mask.append(bool(active))
    return mask


def _build_high_risk_intervals(
    detector_rows,
    detector_window_meta,
    state_scores,
    score_stats,
    enter_th,
    exit_th,
    min_high_len,
    glitch_merge_len,
):
    detector_scores = [float(row.get("detector_score", 0.0) or 0.0) for row in list(detector_rows or [])]
    hysteresis_mask = _apply_detector_hysteresis(detector_scores, enter_th, exit_th)
    gap_filled_mask, gap_merge_meta = _fill_short_false_gaps(hysteresis_mask, glitch_merge_len)
    raw_intervals = _active_intervals(gap_filled_mask)
    expanded_intervals, expand_meta = _expand_intervals_to_regimes(
        intervals=raw_intervals,
        detector_rows=detector_rows,
        shoulder_extension_frames=int((detector_window_meta or {}).get("shoulder_extension_frames", 0) or 0),
    )
    regime_merge_gap = max(
        int(glitch_merge_len),
        int((detector_window_meta or {}).get("recent_window_frames", 0) or 0) * 2,
        int((detector_window_meta or {}).get("shoulder_extension_frames", 0) or 0),
    )
    regime_merged_intervals, regime_merge_meta = _merge_intervals_on_elevated_gap(
        intervals=expanded_intervals,
        state_scores=state_scores,
        gap_floor=max(0.0, float((expand_meta.get("global_hold_floor", 0.0) or 0.0)) - 0.01),
        max_gap=regime_merge_gap,
    )
    merged_intervals, interval_merge_meta = _merge_close_intervals(regime_merged_intervals, glitch_merge_len)
    high_intervals, filter_meta = _filter_short_intervals(merged_intervals, min_high_len)
    high_intervals, score_filter_meta = _filter_low_score_intervals(high_intervals, state_scores, score_stats)
    return list(high_intervals), {
        "detector_type": _DETECTOR_TYPE,
        "online": True,
        "trailing_style": True,
        "enter_th": float(enter_th),
        "exit_th": float(exit_th),
        "minimum_interval_frames": int(min_high_len),
        "gap_merge_frames": int(glitch_merge_len),
        "shoulder_extension_frames": int((detector_window_meta or {}).get("shoulder_extension_frames", 0) or 0),
        "gap_merge": gap_merge_meta,
        "regime_expand": expand_meta,
        "regime_merge": regime_merge_meta,
        "interval_merge": interval_merge_meta,
        "short_interval_filter": filter_meta,
        "score_level_filter": score_filter_meta,
        "raw_interval_count": int(len(raw_intervals)),
        "final_interval_count": int(len(high_intervals)),
        "raw_active_frame_ratio": (
            float(sum(1 for flag in hysteresis_mask if bool(flag))) / float(len(hysteresis_mask))
            if hysteresis_mask
            else 0.0
        ),
    }


def _state_labels_from_intervals(frame_count, high_intervals):
    labels = [STATE_LOW for _ in range(max(0, int(frame_count)))]
    for start_idx, end_idx in list(high_intervals or []):
        for idx in range(int(start_idx), int(end_idx) + 1):
            if 0 <= int(idx) < len(labels):
                labels[int(idx)] = STATE_HIGH
    return labels


def _build_runs(state_labels):
    if not state_labels:
        return []

    runs = []
    start_idx = 0
    current = state_labels[0]
    for idx in range(1, len(state_labels)):
        if state_labels[idx] != current:
            runs.append(
                {
                    "label": current,
                    "start_idx": int(start_idx),
                    "end_idx": int(idx - 1),
                    "length": int(idx - start_idx),
                }
            )
            start_idx = idx
            current = state_labels[idx]
    runs.append(
        {
            "label": current,
            "start_idx": int(start_idx),
            "end_idx": int(len(state_labels) - 1),
            "length": int(len(state_labels) - start_idx),
        }
    )
    return runs


def _build_segments(rows, state_labels, state_scores, detector_scores, dt_sec):
    segments = []
    runs = _build_runs(state_labels)
    timestamps = [float(row.get("timestamp", 0.0) or 0.0) for row in list(rows or [])]
    for segment_id, run in enumerate(runs):
        start_idx = int(run["start_idx"])
        end_idx = int(run["end_idx"])
        start_row = rows[start_idx]
        end_row = rows[end_idx]
        score_slice = list(state_scores[start_idx : end_idx + 1] or [])
        detector_slice = list(detector_scores[start_idx : end_idx + 1] or [])
        state_label = str(run["label"])
        segments.append(
            {
                "segment_id": int(segment_id),
                "state_label": str(state_label),
                "risk_level": "high_risk" if state_label == STATE_HIGH else "background",
                "start_frame": int(start_row["frame_idx"]),
                "end_frame": int(end_row["frame_idx"]),
                "start_time": float(start_row["timestamp"]),
                "end_time": float(end_row["timestamp"]),
                "duration_frames": int(end_idx - start_idx + 1),
                "duration_sec": float(_duration_sec(timestamps, start_idx, end_idx, dt_sec)),
                "state_score_mean": _mean(score_slice),
                "state_score_peak": float(max(score_slice)) if score_slice else 0.0,
                "detector_score_mean": _mean(detector_slice),
                "detector_score_peak": float(max(detector_slice)) if detector_slice else 0.0,
            }
        )
    return segments


def _high_risk_interval_digest(segments):
    rows = []
    for segment in list(segments or []):
        if str(segment.get("state_label", STATE_LOW)) != STATE_HIGH:
            continue
        rows.append(
            {
                "segment_id": int(segment.get("segment_id", 0) or 0),
                "start_frame": int(segment.get("start_frame", 0) or 0),
                "end_frame": int(segment.get("end_frame", 0) or 0),
                "start_time": float(segment.get("start_time", 0.0) or 0.0),
                "end_time": float(segment.get("end_time", 0.0) or 0.0),
                "length_frames": int(segment.get("duration_frames", 0) or 0),
                "length_sec": float(segment.get("duration_sec", 0.0) or 0.0),
                "score_mean": float(segment.get("state_score_mean", 0.0) or 0.0),
                "score_peak": float(segment.get("state_score_peak", 0.0) or 0.0),
                "detector_score_mean": float(segment.get("detector_score_mean", 0.0) or 0.0),
                "detector_score_peak": float(segment.get("detector_score_peak", 0.0) or 0.0),
            }
        )
    return rows


def _state_frame_statistics(segments, frame_count):
    high_frames = 0
    low_frames = 0
    for segment in list(segments or []):
        frames = int(segment.get("duration_frames", 0) or 0)
        if str(segment.get("state_label", STATE_LOW)) == STATE_HIGH:
            high_frames += frames
        else:
            low_frames += frames
    total_frames = max(0, int(frame_count))
    return {
        "frame_count": int(total_frames),
        "high_state_frames": int(high_frames),
        "low_state_frames": int(low_frames),
        "high_state_ratio": float(float(high_frames) / float(total_frames)) if total_frames > 0 else 0.0,
        "low_state_ratio": float(float(low_frames) / float(total_frames)) if total_frames > 0 else 0.0,
    }


def _build_interval_summary(segments, frame_count):
    high_segments = [item for item in list(segments or []) if str(item.get("state_label", STATE_LOW)) == STATE_HIGH]
    low_segments = [item for item in list(segments or []) if str(item.get("state_label", STATE_LOW)) == STATE_LOW]
    high_frames = 0
    for item in list(high_segments):
        high_frames += int(item.get("duration_frames", 0) or 0)
    return {
        "segment_count": int(len(list(segments or []))),
        "high_state_segment_count": int(len(high_segments)),
        "low_state_segment_count": int(len(low_segments)),
        "high_state_frame_count": int(high_frames),
        "high_state_frame_ratio": float(float(high_frames) / float(frame_count)) if int(frame_count) > 0 else 0.0,
        "high_risk_interval_count": int(len(high_segments)),
    }


def _build_interpretation(high_intervals, frame_count):
    high_intervals = list(high_intervals or [])
    if not high_intervals:
        return "No sustained high-risk interval detected in this sample."
    if len(high_intervals) == 1:
        item = dict(high_intervals[0])
        center = 0.5 * float(int(item.get("start_frame", 0) or 0) + int(item.get("end_frame", 0) or 0))
        middle = 0.5 * float(max(0, int(frame_count) - 1))
        if middle > 0.0 and abs(float(center) - float(middle)) / float(middle) <= 0.35:
            return "Single middle high-risk interval covers the main elevated-risk regime in this sample."
        return "Single sustained high-risk interval detected in this sample."
    return "Multiple high-risk intervals detected in this sample."


def build_state_report(result):
    meta = dict(result.get("meta", {}) or {})
    summary = dict(meta.get("summary", {}) or {})
    segments = list(result.get("segments", []) or [])
    frame_rows = list(result.get("frame_rows", []) or [])
    score_meta = dict(meta.get("score", {}) or {})
    detector_meta = dict(meta.get("detector", {}) or {})
    high_intervals = list(meta.get("high_risk_intervals", []) or [])
    coverage_ratio = float(meta.get("high_risk_coverage_ratio", 0.0) or 0.0)

    report = {
        "report_schema_version": str(REPORT_SCHEMA_VERSION),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp_dir": str(meta.get("source_exp_dir", "") or ""),
        "artifact_contract": {
            "default_outputs": [
                _OUTPUT_REPORT_NAME,
                "state_overview.png",
                _OUTPUT_JSON_NAME,
            ],
            "fact_source": _OUTPUT_JSON_NAME,
        },
        "formal_inputs": {
            "signal_names": list(meta.get("input_signals", []) or []),
            "source_mode": str((meta.get("formal_state_inputs", {}) or {}).get("source_mode", "") or ""),
            "processed_columns": dict((meta.get("formal_state_inputs", {}) or {}).get("processed_columns", {}) or {}),
            "pipeline": list((meta.get("formal_state_inputs", {}) or {}).get("pipeline", []) or []),
            "signals": dict((meta.get("formal_state_inputs", {}) or {}).get("signals", {}) or {}),
        },
        "state_score_summary": {
            "method": str(score_meta.get("method", "") or ""),
            "description": str(score_meta.get("description", "") or ""),
            "weights": dict(score_meta.get("weights", {}) or {}),
            "smoothing_window": int(score_meta.get("smoothing_window", 0) or 0),
            "stats": dict(score_meta.get("score_stats", {}) or {}),
        },
        "detector_summary": {
            "detector_type": str(detector_meta.get("detector_type", _DETECTOR_TYPE) or _DETECTOR_TYPE),
            "online": bool(detector_meta.get("online", True)),
            "trailing_style": bool(detector_meta.get("trailing_style", True)),
            "uses_local_mean": bool(detector_meta.get("uses_local_mean", True)),
            "uses_local_spread": bool(detector_meta.get("uses_local_spread", True)),
            "baseline_window_frames": int(detector_meta.get("baseline_window_frames", 0) or 0),
            "recent_window_frames": int(detector_meta.get("recent_window_frames", 0) or 0),
            "regime_window_frames": int(detector_meta.get("regime_window_frames", 0) or 0),
            "enter_th": float(detector_meta.get("enter_th", 0.0) or 0.0),
            "exit_th": float(detector_meta.get("exit_th", 0.0) or 0.0),
            "minimum_interval_frames": int(detector_meta.get("minimum_interval_frames", 0) or 0),
            "gap_merge_frames": int(detector_meta.get("gap_merge_frames", 0) or 0),
            "shoulder_extension_frames": int(detector_meta.get("shoulder_extension_frames", 0) or 0),
            "gap_merge_count": int(detector_meta.get("gap_merge_count", 0) or 0),
            "merged_interval_count": int(detector_meta.get("merged_interval_count", 0) or 0),
            "dropped_short_interval_count": int(detector_meta.get("dropped_short_interval_count", 0) or 0),
            "dropped_low_score_interval_count": int(detector_meta.get("dropped_low_score_interval_count", 0) or 0),
        },
        "high_risk_intervals": list(high_intervals),
        "coverage_ratio": float(coverage_ratio),
        "interval_sequence_summary": {
            "summary": summary,
            "frame_statistics": _state_frame_statistics(segments, len(frame_rows)),
        },
        "key_parameters": {
            "enter_th": float(meta.get("enter_th", 0.0) or 0.0),
            "exit_th": float(meta.get("exit_th", 0.0) or 0.0),
            "min_high_len": int(meta.get("min_high_len", 0) or 0),
            "min_low_len": int(meta.get("min_low_len", 0) or 0),
            "glitch_merge_len": int(((meta.get("merge_rule", {}) or {}).get("maximum_gap_frames", 0) or 0)),
        },
        "interpretation": str(meta.get("sample_note", "") or ""),
    }
    return report


def write_state_report(path, report):
    write_json_atomic(path, report, indent=2)


def _remove_legacy_state_outputs(output_dir):
    output_dir = Path(output_dir).resolve()
    for name in _LEGACY_STATE_OUTPUT_NAMES:
        path = output_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception:
            continue


def _build_detector_report_meta(detector_meta, detector_window_meta, enter_th, exit_th, min_high_len, glitch_merge_len):
    return {
        "detector_type": str(detector_meta.get("detector_type", _DETECTOR_TYPE) or _DETECTOR_TYPE),
        "online": bool(detector_meta.get("online", True)),
        "trailing_style": bool(detector_meta.get("trailing_style", True)),
        "uses_local_mean": bool(detector_window_meta.get("uses_local_mean", True)),
        "uses_local_spread": bool(detector_window_meta.get("uses_local_spread", True)),
        "baseline_window_frames": int(detector_window_meta.get("baseline_window_frames", 0) or 0),
        "recent_window_frames": int(detector_window_meta.get("recent_window_frames", 0) or 0),
        "regime_window_frames": int(detector_window_meta.get("regime_window_frames", 0) or 0),
        "enter_th": float(enter_th),
        "exit_th": float(exit_th),
        "minimum_interval_frames": int(min_high_len),
        "gap_merge_frames": int(glitch_merge_len),
        "shoulder_extension_frames": int(
            detector_meta.get("shoulder_extension_frames", 0)
            or detector_window_meta.get("shoulder_extension_frames", 0)
            or 0
        ),
        "gap_merge_count": int(((detector_meta.get("gap_merge", {}) or {}).get("merge_count", 0) or 0)),
        "merged_interval_count": int(((detector_meta.get("interval_merge", {}) or {}).get("merge_count", 0) or 0)),
        "dropped_short_interval_count": int(((detector_meta.get("short_interval_filter", {}) or {}).get("dropped_count", 0) or 0)),
        "dropped_low_score_interval_count": int(((detector_meta.get("score_level_filter", {}) or {}).get("dropped_count", 0) or 0)),
    }


def _load_state_input_rows(exp_dir, segment_dir):
    input_csv = segment_dir / "signal_extraction" / _INPUT_CSV_NAME
    if input_csv.is_file():
        return _read_signal_rows(input_csv), input_csv

    frames_dir = ensure_dir(segment_dir / "frames", name="segment frames dir")
    timestamps = _read_timestamps(ensure_file(segment_dir / "timestamps.txt", name="segment timestamps"))
    keyframes_meta_path = segment_dir / "keyframes" / "keyframes_meta.json"
    keyframes_meta = _read_json(keyframes_meta_path) if keyframes_meta_path.is_file() else {}
    signal_payload = extract_signal_timeseries_from_frames(
        frame_paths=list_frames_sorted(frames_dir),
        timestamps=timestamps,
        exp_dir=exp_dir,
        segment_dir=segment_dir,
        keyframes_meta=keyframes_meta,
        output_dir=segment_dir / ".segment_cache" / "state_signal_prepare",
        cache_dir=segment_dir / ".segment_cache" / "signal_extraction",
    )
    return list(signal_payload.get("rows", []) or []), None


def compute_state_segments(
    rows,
    exp_dir=None,
    input_csv=None,
    output_dir=None,
    normalization_method=DEFAULT_NORMALIZATION_METHOD,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    enter_th=DEFAULT_ENTER_TH,
    exit_th=DEFAULT_EXIT_TH,
    min_high_len=DEFAULT_MIN_HIGH_LEN,
    min_low_len=DEFAULT_MIN_LOW_LEN,
    glitch_merge_len=DEFAULT_GLITCH_MERGE_LEN,
    weights=None,
):
    del normalization_method

    rows = _coerce_signal_rows(rows)
    exp_dir_path = Path(exp_dir).resolve() if exp_dir is not None else None
    output_dir_path = Path(output_dir).resolve() if output_dir is not None else None
    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    smoothing_window = max(1, int(smoothing_window))
    min_high_len = max(1, int(min_high_len))
    min_low_len = max(1, int(min_low_len))
    glitch_merge_len = max(0, int(glitch_merge_len))
    weights = _resolve_weights(weights)

    rows, formal_state_input_meta = _resolve_formal_state_inputs(rows)
    motion_values = [float(row["motion_velocity_state_input"]) for row in rows]
    semantic_values = [float(row["semantic_velocity_state_input"]) for row in rows]
    timestamps = [float(row["timestamp"]) for row in rows]
    dt_sec = _median_dt(timestamps)

    state_scores_raw, state_scores, score_meta = _build_state_score(
        motion_values=motion_values,
        semantic_values=semantic_values,
        weights=weights,
        smoothing_window=smoothing_window,
    )

    detector_rows, detector_window_meta = _build_regime_change_rows(
        scores=state_scores,
        smoothing_window=smoothing_window,
        min_low_len=min_low_len,
    )
    high_intervals_idx, detector_meta = _build_high_risk_intervals(
        detector_rows=detector_rows,
        detector_window_meta=detector_window_meta,
        state_scores=state_scores,
        score_stats=dict(score_meta.get("score_stats", {}) or {}),
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        glitch_merge_len=glitch_merge_len,
    )

    detector_scores = [float(row.get("detector_score", 0.0) or 0.0) for row in list(detector_rows or [])]
    state_labels = _state_labels_from_intervals(len(rows), high_intervals_idx)
    segments = _build_segments(
        rows=rows,
        state_labels=state_labels,
        state_scores=state_scores,
        detector_scores=detector_scores,
        dt_sec=dt_sec,
    )
    summary = _build_interval_summary(segments, len(rows))
    high_risk_intervals = _high_risk_interval_digest(segments)
    high_risk_coverage_ratio = float(summary.get("high_state_frame_ratio", 0.0) or 0.0)

    frame_rows = []
    detector_map = dict((int(row.get("frame_idx", 0) or 0), row) for row in list(detector_rows or []))
    for idx, row in enumerate(list(rows or [])):
        detector_row = detector_map.get(int(idx), {})
        frame_rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "motion_velocity_raw": float(row["motion_velocity"]),
                "semantic_velocity_raw": float(row["semantic_velocity"]),
                "motion_velocity_state_signal": float(motion_values[idx]),
                "semantic_velocity_state_signal": float(semantic_values[idx]),
                "state_score_raw": float(state_scores_raw[idx]),
                "state_score": float(state_scores[idx]),
                "detector_score": float(detector_row.get("detector_score", 0.0) or 0.0),
                "trailing_baseline_mean": float(detector_row.get("baseline_mean", 0.0) or 0.0),
                "trailing_recent_mean": float(detector_row.get("recent_mean", 0.0) or 0.0),
                "trailing_regime_mean": float(detector_row.get("regime_mean", 0.0) or 0.0),
                "detector_hold_floor": float(detector_row.get("hold_floor", 0.0) or 0.0),
                "state_label": str(state_labels[idx]),
                "is_high_risk": bool(str(state_labels[idx]) == STATE_HIGH),
            }
        )

    sample_note = _build_interpretation(high_risk_intervals, len(rows))
    detector_report_meta = _build_detector_report_meta(
        detector_meta=detector_meta,
        detector_window_meta=detector_window_meta,
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        glitch_merge_len=glitch_merge_len,
    )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp_dir": str(exp_dir_path) if exp_dir_path is not None else "",
        "input_csv": str(Path(input_csv).resolve()) if input_csv else "",
        "input_signals": list(DEFAULT_INPUT_SIGNALS),
        "formal_state_inputs": dict(formal_state_input_meta),
        "weights": dict(weights),
        "score": score_meta,
        "enter_th": float(enter_th),
        "exit_th": float(exit_th),
        "min_high_len": int(min_high_len),
        "min_low_len": int(min_low_len),
        "merge_rule": dict((detector_meta.get("gap_merge", {}) or {})),
        "detector": detector_report_meta,
        "summary": summary,
        "high_risk_intervals": high_risk_intervals,
        "high_risk_coverage_ratio": float(high_risk_coverage_ratio),
        "sample_note": str(sample_note),
    }

    json_payload = {
        "source_exp_dir": str(exp_dir_path) if exp_dir_path is not None else "",
        "input_csv": str(Path(input_csv).resolve()) if input_csv else "",
        "summary": summary,
        "detector": {
            "detector_type": str(detector_report_meta["detector_type"]),
            "online": bool(detector_report_meta["online"]),
            "trailing_style": bool(detector_report_meta["trailing_style"]),
        },
        "high_risk_intervals": high_risk_intervals,
        "segments": segments,
    }

    return {
        "exp_dir": exp_dir_path,
        "segment_dir": output_dir_path.parent if output_dir_path is not None else None,
        "input_csv": Path(input_csv).resolve() if input_csv else None,
        "output_dir": output_dir_path,
        "frame_rows": frame_rows,
        "segments": segments,
        "meta": meta,
        "json_payload": json_payload,
    }


def run_state_segmentation(
    exp_dir,
    normalization_method=DEFAULT_NORMALIZATION_METHOD,
    smoothing_window=DEFAULT_SMOOTHING_WINDOW,
    enter_th=DEFAULT_ENTER_TH,
    exit_th=DEFAULT_EXIT_TH,
    min_high_len=DEFAULT_MIN_HIGH_LEN,
    min_low_len=DEFAULT_MIN_LOW_LEN,
    glitch_merge_len=DEFAULT_GLITCH_MERGE_LEN,
    weights=None,
):
    exp_dir = ensure_dir(exp_dir, name="experiment dir")
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    rows, input_csv = _load_state_input_rows(exp_dir, segment_dir)
    output_dir = (segment_dir / "state_segmentation").resolve()
    return compute_state_segments(
        rows=rows,
        exp_dir=exp_dir,
        input_csv=input_csv,
        output_dir=output_dir,
        normalization_method=normalization_method,
        smoothing_window=smoothing_window,
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        min_low_len=min_low_len,
        glitch_merge_len=glitch_merge_len,
        weights=weights,
    )


def write_state_segmentation_outputs(result):
    output_dir = ensure_dir(result["output_dir"], name="state segmentation output dir")
    json_path = output_dir / _OUTPUT_JSON_NAME
    report_path = output_dir / _OUTPUT_REPORT_NAME
    report = build_state_report(result)
    write_json_atomic(json_path, result["json_payload"], indent=2)
    write_state_report(report_path, report)
    _remove_legacy_state_outputs(output_dir)
    return {
        "json_path": json_path,
        "report_path": report_path,
        "report": report,
    }
