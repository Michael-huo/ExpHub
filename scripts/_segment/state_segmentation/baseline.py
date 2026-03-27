#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts._common import ensure_dir, ensure_file, write_json_atomic
from scripts._segment.research.kinematics import minmax_normalize, moving_average
from scripts._segment.signal_extraction import (
    FORMAL_STATE_INPUT_COLUMNS,
    FORMAL_STATE_INPUT_SIGNALS,
    build_formal_state_input_rows,
)


STATE_LOW = "low_state"
STATE_HIGH = "high_state"

DEFAULT_INPUT_SIGNALS = list(FORMAL_STATE_INPUT_SIGNALS)
DEFAULT_NORMALIZATION_METHOD = "robust_zscore_per_signal"
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
DEFAULT_SCORE_CALIBRATION_MARGIN = 0.10
DEFAULT_VALIDATION_IMAGE_WEIGHT = 0.20

REPORT_SCHEMA_VERSION = "state_report.v1"
_INPUT_CSV_NAME = "signal_timeseries.csv"
_OUTPUT_JSON_NAME = "state_segments.json"
_OUTPUT_REPORT_NAME = "state_report.json"

_VALIDATION_CANDIDATE_ORDER = [
    "official_current",
    "candidate_blur",
    "candidate_appearance",
]

_VALIDATION_CANDIDATE_META = {
    "official_current": {
        "display_name": "official_current",
        "description": "Current official state mainline candidate using motion_velocity + semantic_velocity.",
        "is_formal_mainline": True,
        "input_display_names": [
            "motion_velocity",
            "semantic_velocity",
        ],
    },
    "candidate_blur": {
        "display_name": "candidate_blur",
        "description": "Validation sidecar candidate using motion_velocity + blur_risk(processed blur_score) + semantic_velocity.",
        "is_formal_mainline": False,
        "input_display_names": [
            "motion_velocity",
            "blur_risk (processed blur_score)",
            "semantic_velocity",
        ],
    },
    "candidate_appearance": {
        "display_name": "candidate_appearance",
        "description": "Validation sidecar candidate using motion_velocity + processed appearance_delta + semantic_velocity.",
        "is_formal_mainline": False,
        "input_display_names": [
            "motion_velocity",
            "appearance_delta (processed validation sidecar)",
            "semantic_velocity",
        ],
    },
}

_VALIDATION_BLUR_PREPROCESSING = {
    "clip_quantiles": [0.05, 0.95],
    "smoothing_window": 5,
    "invert_after_normalize": True,
}

_VALIDATION_APPEARANCE_PREPROCESSING = {
    "clip_quantiles": [0.02, 0.98],
    "smoothing_window": 5,
    "invert_after_normalize": False,
}

_LEGACY_STATE_OUTPUT_NAMES = [
    "density_schedule.csv",
    "density_schedule_overview.png",
    "state_segments.csv",
    "state_timeline.csv",
    "state_candidate_compare.json",
    "state_segmentation_meta.json",
    "state_segmentation_overview.png",
    "state_signal_overlay.png",
]


def _float_list(values):
    return [float(value) for value in values]


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


def _optional_float(value):
    text = str(value or "").strip()
    if text == "":
        return None
    return float(text)


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

    has_all_persisted = True
    persisted_meta = {}
    for signal_name in DEFAULT_INPUT_SIGNALS:
        available, values = _formal_input_series(rows, signal_name)
        if not available:
            has_all_persisted = False
            break
        persisted_meta[signal_name] = {
            "raw_column": str(signal_name),
            "processed_column": str(FORMAL_STATE_INPUT_COLUMNS[signal_name]),
            "processed_stats": {
                "min": float(min(values)) if values else 0.0,
                "mean": float(sum(values) / float(len(values))) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
            },
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


def _median_dt(timestamps):
    diffs = []
    prev_value = None
    for value in timestamps:
        current = float(value)
        if prev_value is not None:
            diff = float(current - prev_value)
            if diff > 1e-9:
                diffs.append(diff)
        prev_value = current
    if not diffs:
        return 0.0
    return float(np.median(np.asarray(diffs, dtype=np.float32)))


def _quantile(values, q):
    arr = np.asarray(list(values or []), dtype=np.float32)
    if arr.size <= 0:
        return 0.0
    return float(np.quantile(arr, float(q)))


def _robust_clip(values, low_quantile, high_quantile):
    arr = np.asarray(list(values or []), dtype=np.float32)
    if arr.size <= 0:
        return [], {
            "low_quantile": float(low_quantile),
            "high_quantile": float(high_quantile),
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "degenerated": True,
        }
    lower_bound = float(np.quantile(arr, float(low_quantile)))
    upper_bound = float(np.quantile(arr, float(high_quantile)))
    if upper_bound < lower_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    clipped = np.clip(arr, lower_bound, upper_bound)
    return [float(value) for value in clipped.tolist()], {
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "degenerated": bool(abs(float(upper_bound) - float(lower_bound)) < 1e-12),
    }


def _preprocess_validation_signal(values, clip_quantiles, smoothing_window, invert_after_normalize):
    clipped_values, clip_meta = _robust_clip(values, clip_quantiles[0], clip_quantiles[1])
    normalized_values = minmax_normalize(clipped_values)
    if bool(invert_after_normalize):
        normalized_values = [float(1.0 - float(value)) for value in normalized_values]
    smoothed_values, actual_window = moving_average(normalized_values, int(smoothing_window))
    return [float(value) for value in smoothed_values], {
        "clip": clip_meta,
        "normalization": {
            "method": "minmax_after_robust_clip",
            "range": [0.0, 1.0],
            "invert_after_normalize": bool(invert_after_normalize),
        },
        "smoothing": {
            "method": "moving_average",
            "window_size": int(actual_window),
        },
    }


def _compatibility_calibrate_scores(raw_scores, enter_th, exit_th, margin=DEFAULT_SCORE_CALIBRATION_MARGIN):
    raw_scores = [float(value) for value in list(raw_scores or [])]
    if not raw_scores:
        return [], {
            "method": "affine_iqr_compatibility",
            "applied": False,
            "reason": "empty_scores",
        }

    q25 = _quantile(raw_scores, 0.25)
    q75 = _quantile(raw_scores, 0.75)
    q90 = _quantile(raw_scores, 0.90)
    raw_min = float(min(raw_scores))
    raw_max = float(max(raw_scores))
    raw_span = float(q75 - q25)
    should_calibrate = bool(q75 < float(enter_th) or q90 < float(enter_th) + 0.05)

    if not should_calibrate:
        return list(raw_scores), {
            "method": "affine_iqr_compatibility",
            "applied": False,
            "reason": "distribution_already_compatible",
            "source_quantiles": {
                "q25": float(q25),
                "q75": float(q75),
                "q90": float(q90),
                "min": float(raw_min),
                "max": float(raw_max),
            },
        }

    target_q25 = float(exit_th) - float(margin)
    target_q75 = float(enter_th) + float(margin)
    if raw_span < 1e-6:
        return list(raw_scores), {
            "method": "affine_iqr_compatibility",
            "applied": False,
            "reason": "degenerate_iqr",
            "source_quantiles": {
                "q25": float(q25),
                "q75": float(q75),
                "q90": float(q90),
                "min": float(raw_min),
                "max": float(raw_max),
            },
        }

    scale = float((target_q75 - target_q25) / raw_span)
    offset = float(target_q25 - float(scale) * float(q25))
    calibrated_scores = [float(scale * float(value) + offset) for value in raw_scores]
    return calibrated_scores, {
        "method": "affine_iqr_compatibility",
        "applied": True,
        "reason": "restore_fixed_threshold_compatibility",
        "source_quantiles": {
            "q25": float(q25),
            "q75": float(q75),
            "q90": float(q90),
            "min": float(raw_min),
            "max": float(raw_max),
        },
        "target_quantiles": {
            "q25": float(target_q25),
            "q75": float(target_q75),
        },
        "scale": float(scale),
        "offset": float(offset),
    }


def _resolve_glitch_merge_len(smoothing_window, configured_len):
    base_len = max(int(configured_len), int(smoothing_window))
    return max(3, int(base_len))


def _apply_state_machine(scores, enter_th, exit_th, min_high_len, min_low_len):
    if not scores:
        return []

    states = [STATE_LOW for _ in scores]
    current_state = STATE_LOW
    high_run = 0
    low_run = 0

    for idx, score in enumerate(scores):
        if current_state == STATE_LOW:
            if float(score) >= float(enter_th):
                high_run += 1
            else:
                high_run = 0
            if high_run >= int(min_high_len):
                start_idx = idx - int(min_high_len) + 1
                for write_idx in range(max(0, start_idx), idx + 1):
                    states[write_idx] = STATE_HIGH
                current_state = STATE_HIGH
                high_run = 0
                low_run = 0
        else:
            states[idx] = STATE_HIGH
            if float(score) <= float(exit_th):
                low_run += 1
            else:
                low_run = 0
            if low_run >= int(min_low_len):
                start_idx = idx - int(min_low_len) + 1
                for write_idx in range(max(0, start_idx), idx + 1):
                    states[write_idx] = STATE_LOW
                current_state = STATE_LOW
                high_run = 0
                low_run = 0

    return states


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


def _merge_short_runs(state_labels, min_segment_len):
    merged = list(state_labels)
    applied_rules = []
    merge_count = 0

    while True:
        runs = _build_runs(merged)
        target_run = None
        for idx, run in enumerate(runs):
            if int(run["length"]) < int(min_segment_len):
                target_run = (idx, run, runs)
                break
        if target_run is None:
            break

        run_idx, run, runs = target_run
        prev_run = runs[run_idx - 1] if run_idx > 0 else None
        next_run = runs[run_idx + 1] if run_idx + 1 < len(runs) else None
        target_label = str(run["label"])
        reason = "kept"

        if prev_run is not None and next_run is not None and prev_run["label"] == next_run["label"]:
            target_label = str(prev_run["label"])
            reason = "merge_short_island_between_same_labels"
        elif prev_run is not None and next_run is not None:
            if int(prev_run["length"]) >= int(next_run["length"]):
                target_label = str(prev_run["label"])
                reason = "merge_short_run_into_longer_previous_neighbor"
            else:
                target_label = str(next_run["label"])
                reason = "merge_short_run_into_longer_next_neighbor"
        elif prev_run is not None:
            target_label = str(prev_run["label"])
            reason = "merge_short_edge_run_into_previous_neighbor"
        elif next_run is not None:
            target_label = str(next_run["label"])
            reason = "merge_short_edge_run_into_next_neighbor"

        if target_label == run["label"]:
            break

        for idx in range(int(run["start_idx"]), int(run["end_idx"]) + 1):
            merged[idx] = target_label

        merge_count += 1
        applied_rules.append(
            {
                "source_label": str(run["label"]),
                "target_label": str(target_label),
                "start_idx": int(run["start_idx"]),
                "end_idx": int(run["end_idx"]),
                "length": int(run["length"]),
                "reason": reason,
            }
        )

    return merged, {
        "minimum_segment_len": int(min_segment_len),
        "merge_count": int(merge_count),
        "applied_rules": applied_rules,
        "description": (
            "Iteratively merge any segment shorter than {0} frames into adjacent context: "
            "prefer same-label neighbors on both sides, otherwise merge into the longer neighbor."
        ).format(int(min_segment_len)),
    }


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


def _build_segments(rows, state_labels, state_scores, dt_sec):
    segments = []
    runs = _build_runs(state_labels)
    timestamps = [row["timestamp"] for row in rows]
    for segment_id, run in enumerate(runs):
        start_idx = int(run["start_idx"])
        end_idx = int(run["end_idx"])
        frame_slice = rows[start_idx : end_idx + 1]
        score_slice = state_scores[start_idx : end_idx + 1]
        start_row = frame_slice[0]
        end_row = frame_slice[-1]
        duration_frames = int(end_idx - start_idx + 1)
        segments.append(
            {
                "segment_id": int(segment_id),
                "start_frame": int(start_row["frame_idx"]),
                "end_frame": int(end_row["frame_idx"]),
                "start_time": float(start_row["timestamp"]),
                "end_time": float(end_row["timestamp"]),
                "duration_frames": int(duration_frames),
                "duration_sec": float(_duration_sec(timestamps, start_idx, end_idx, dt_sec)),
                "state_label": str(run["label"]),
                "state_score_mean": float(np.mean(np.asarray(score_slice, dtype=np.float32))) if score_slice else 0.0,
                "state_score_peak": float(np.max(np.asarray(score_slice, dtype=np.float32))) if score_slice else 0.0,
            }
        )
    return segments


def _build_summary(segments, frame_count):
    high_segments = [item for item in segments if item.get("state_label") == STATE_HIGH]
    low_segments = [item for item in segments if item.get("state_label") == STATE_LOW]
    high_frames = 0
    for item in high_segments:
        high_frames += int(item.get("duration_frames", 0) or 0)
    return {
        "segment_count": int(len(segments)),
        "high_state_segment_count": int(len(high_segments)),
        "low_state_segment_count": int(len(low_segments)),
        "high_state_frame_count": int(high_frames),
        "high_state_frame_ratio": float(float(high_frames) / float(frame_count)) if int(frame_count) > 0 else 0.0,
    }


def _compute_candidate_track(
    rows,
    components,
    enter_th,
    exit_th,
    min_high_len,
    min_low_len,
    glitch_merge_len,
    dt_sec,
):
    score_raw = []
    for idx in range(len(rows)):
        value = 0.0
        for component in list(components or []):
            value += float(component.get("weight", 0.0) or 0.0) * float(component.get("values", [0.0])[idx])
        score_raw.append(float(value))

    score_values, calibration_meta = _compatibility_calibrate_scores(
        score_raw,
        enter_th=enter_th,
        exit_th=exit_th,
    )
    raw_states = _apply_state_machine(
        scores=score_values,
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        min_low_len=min_low_len,
    )
    merged_states, merge_meta = _merge_short_runs(raw_states, glitch_merge_len)
    segments = _build_segments(rows, merged_states, score_values, dt_sec)
    return {
        "score_raw": list(score_raw),
        "score": list(score_values),
        "states": list(merged_states),
        "segments": list(segments),
        "summary": _build_summary(segments, len(rows)),
        "score_calibration": dict(calibration_meta),
        "merge_rule": dict(merge_meta),
    }


def _candidate_high_segment_digest(segments):
    rows = []
    for segment in list(segments or []):
        if str(segment.get("state_label", STATE_LOW)) != STATE_HIGH:
            continue
        rows.append(
            {
                "start_frame": int(segment.get("start_frame", 0) or 0),
                "end_frame": int(segment.get("end_frame", 0) or 0),
                "duration_frames": int(segment.get("duration_frames", 0) or 0),
                "duration_sec": float(segment.get("duration_sec", 0.0) or 0.0),
            }
        )
    return rows


def _build_validation_sidecar(
    rows,
    motion_values,
    semantic_values,
    official_track,
    enter_th,
    exit_th,
    min_high_len,
    min_low_len,
    glitch_merge_len,
    dt_sec,
):
    blur_raw = [float(row.get("blur_score", 0.0) or 0.0) for row in list(rows or [])]
    blur_values, blur_preprocess = _preprocess_validation_signal(
        blur_raw,
        clip_quantiles=list(_VALIDATION_BLUR_PREPROCESSING["clip_quantiles"]),
        smoothing_window=int(_VALIDATION_BLUR_PREPROCESSING["smoothing_window"]),
        invert_after_normalize=bool(_VALIDATION_BLUR_PREPROCESSING["invert_after_normalize"]),
    )
    appearance_raw = [float(row.get("appearance_delta", 0.0) or 0.0) for row in list(rows or [])]
    appearance_values, appearance_preprocess = _preprocess_validation_signal(
        appearance_raw,
        clip_quantiles=list(_VALIDATION_APPEARANCE_PREPROCESSING["clip_quantiles"]),
        smoothing_window=int(_VALIDATION_APPEARANCE_PREPROCESSING["smoothing_window"]),
        invert_after_normalize=bool(_VALIDATION_APPEARANCE_PREPROCESSING["invert_after_normalize"]),
    )

    candidate_tracks = {
        "official_current": {
            "display_name": str(_VALIDATION_CANDIDATE_META["official_current"]["display_name"]),
            "description": str(_VALIDATION_CANDIDATE_META["official_current"]["description"]),
            "is_formal_mainline": True,
            "input_display_names": list(_VALIDATION_CANDIDATE_META["official_current"]["input_display_names"]),
            "score_raw": list(official_track.get("score_raw", []) or []),
            "score": list(official_track.get("score", []) or []),
            "states": list(official_track.get("states", []) or []),
            "segments": list(official_track.get("segments", []) or []),
            "summary": dict(official_track.get("summary", {}) or {}),
            "score_calibration": dict(official_track.get("score_calibration", {}) or {}),
            "merge_rule": dict(official_track.get("merge_rule", {}) or {}),
        },
        "candidate_blur": dict(
            {
                "display_name": str(_VALIDATION_CANDIDATE_META["candidate_blur"]["display_name"]),
                "description": str(_VALIDATION_CANDIDATE_META["candidate_blur"]["description"]),
                "is_formal_mainline": False,
                "input_display_names": list(_VALIDATION_CANDIDATE_META["candidate_blur"]["input_display_names"]),
                "validation_preprocessing": {
                    "blur_score": dict(blur_preprocess),
                },
            },
            **_compute_candidate_track(
                rows=rows,
                components=[
                    {"weight": 0.60, "values": motion_values},
                    {"weight": float(DEFAULT_VALIDATION_IMAGE_WEIGHT), "values": blur_values},
                    {"weight": 0.20, "values": semantic_values},
                ],
                enter_th=enter_th,
                exit_th=exit_th,
                min_high_len=min_high_len,
                min_low_len=min_low_len,
                glitch_merge_len=glitch_merge_len,
                dt_sec=dt_sec,
            )
        ),
        "candidate_appearance": dict(
            {
                "display_name": str(_VALIDATION_CANDIDATE_META["candidate_appearance"]["display_name"]),
                "description": str(_VALIDATION_CANDIDATE_META["candidate_appearance"]["description"]),
                "is_formal_mainline": False,
                "input_display_names": list(_VALIDATION_CANDIDATE_META["candidate_appearance"]["input_display_names"]),
                "validation_preprocessing": {
                    "appearance_delta": dict(appearance_preprocess),
                },
            },
            **_compute_candidate_track(
                rows=rows,
                components=[
                    {"weight": 0.60, "values": motion_values},
                    {"weight": float(DEFAULT_VALIDATION_IMAGE_WEIGHT), "values": appearance_values},
                    {"weight": 0.20, "values": semantic_values},
                ],
                enter_th=enter_th,
                exit_th=exit_th,
                min_high_len=min_high_len,
                min_low_len=min_low_len,
                glitch_merge_len=glitch_merge_len,
                dt_sec=dt_sec,
            )
        ),
    }

    report_candidates = {}
    for candidate_name in _VALIDATION_CANDIDATE_ORDER:
        track = dict(candidate_tracks.get(candidate_name, {}) or {})
        summary = dict(track.get("summary", {}) or {})
        report_candidates[candidate_name] = {
            "display_name": str(track.get("display_name", candidate_name) or candidate_name),
            "description": str(track.get("description", "") or ""),
            "is_formal_mainline": bool(track.get("is_formal_mainline", False)),
            "input_display_names": list(track.get("input_display_names", []) or []),
            "high_frame_ratio": float(summary.get("high_state_frame_ratio", 0.0) or 0.0),
            "high_segment_count": int(summary.get("high_state_segment_count", 0) or 0),
            "high_segments": _candidate_high_segment_digest(track.get("segments", [])),
            "score_calibration": dict(track.get("score_calibration", {}) or {}),
        }
        if "validation_preprocessing" in track:
            report_candidates[candidate_name]["validation_preprocessing"] = dict(track.get("validation_preprocessing", {}) or {})

    return {
        "frame_indices": [int(row.get("frame_idx", 0) or 0) for row in list(rows or [])],
        "official_current_name": "official_current",
        "report": {
            "official_current_remains_formal_mainline": True,
            "artifacts": {
                "comparison_plot": "segment/state_segmentation/state_signal_candidate_compare.png",
            },
            "description": (
                "Validation sidecar comparison reuses the current state score pipeline, including processed inputs, "
                "compatibility calibration, thresholding, and state-machine decoding. Only official_current drives "
                "formal state segmentation outputs; blur_score and appearance_delta remain validation-only sidecar signals."
            ),
            "candidates": report_candidates,
        },
        "candidates": candidate_tracks,
    }


def _frame_time_lookup(frame_rows):
    out = {}
    for row in list(frame_rows or []):
        out[int(row.get("frame_idx", 0) or 0)] = float(row.get("timestamp", 0.0) or 0.0)
    return out


def _segment_digest(segments):
    rows = []
    for segment in list(segments or []):
        rows.append(
            {
                "segment_id": int(segment.get("segment_id", 0) or 0),
                "state_label": str(segment.get("state_label", STATE_LOW)),
                "start_frame": int(segment.get("start_frame", 0) or 0),
                "end_frame": int(segment.get("end_frame", 0) or 0),
                "duration_frames": int(segment.get("duration_frames", 0) or 0),
                "duration_sec": float(segment.get("duration_sec", 0.0) or 0.0),
                "state_score_mean": float(segment.get("state_score_mean", 0.0) or 0.0),
                "state_score_peak": float(segment.get("state_score_peak", 0.0) or 0.0),
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


def _density_window_statistics(schedule_runs):
    zone_frames = {
        "low": 0,
        "transition": 0,
        "high": 0,
    }
    for run in list(schedule_runs or []):
        zone_name = str(run.get("schedule_zone", "low") or "low")
        zone_frames[zone_name] = int(zone_frames.get(zone_name, 0)) + int(run.get("duration_frames", 0) or 0)
    total_frames = int(sum(zone_frames.values()))
    return {
        "window_count": int(len(list(schedule_runs or []))),
        "low_window_count": int(len([row for row in list(schedule_runs or []) if str(row.get("schedule_zone", "low") or "low") == "low"])),
        "transition_window_count": int(len([row for row in list(schedule_runs or []) if str(row.get("schedule_zone", "low") or "low") == "transition"])),
        "high_window_count": int(len([row for row in list(schedule_runs or []) if str(row.get("schedule_zone", "low") or "low") == "high"])),
        "zone_frames": zone_frames,
        "zone_ratios": {
            "low": float(float(zone_frames["low"]) / float(total_frames)) if total_frames > 0 else 0.0,
            "transition": float(float(zone_frames["transition"]) / float(total_frames)) if total_frames > 0 else 0.0,
            "high": float(float(zone_frames["high"]) / float(total_frames)) if total_frames > 0 else 0.0,
        },
    }


def build_state_timeline_rows(result, schedule_runs=None):
    frame_lookup = _frame_time_lookup(result.get("frame_rows", []))
    rows = []
    for segment in list(result.get("segments", []) or []):
        rows.append(
            {
                "row_type": "state_segment",
                "row_id": int(segment.get("segment_id", 0) or 0),
                "start_frame": int(segment.get("start_frame", 0) or 0),
                "end_frame": int(segment.get("end_frame", 0) or 0),
                "start_time": float(segment.get("start_time", 0.0) or 0.0),
                "end_time": float(segment.get("end_time", 0.0) or 0.0),
                "duration_frames": int(segment.get("duration_frames", 0) or 0),
                "duration_sec": float(segment.get("duration_sec", 0.0) or 0.0),
                "state_label": str(segment.get("state_label", STATE_LOW)),
                "schedule_zone": "",
                "target_gap": "",
                "anchor_count": "",
                "state_score_mean": float(segment.get("state_score_mean", 0.0) or 0.0),
                "state_score_peak": float(segment.get("state_score_peak", 0.0) or 0.0),
            }
        )

    for run in list(schedule_runs or []):
        start_frame = int(run.get("start_frame", 0) or 0)
        end_frame = int(run.get("end_frame", 0) or 0)
        rows.append(
            {
                "row_type": "density_window",
                "row_id": int(run.get("run_id", 0) or 0),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": float(frame_lookup.get(int(start_frame), 0.0)),
                "end_time": float(frame_lookup.get(int(end_frame), 0.0)),
                "duration_frames": int(run.get("duration_frames", 0) or 0),
                "duration_sec": float(max(0.0, float(frame_lookup.get(int(end_frame), 0.0)) - float(frame_lookup.get(int(start_frame), 0.0)))),
                "state_label": "",
                "schedule_zone": str(run.get("schedule_zone", "low") or "low"),
                "target_gap": int(run.get("target_gap", 0) or 0),
                "anchor_count": int(run.get("anchor_count", 0) or 0),
                "state_score_mean": "",
                "state_score_peak": "",
            }
        )
    return rows


def _timeline_fieldnames():
    return [
        "row_type",
        "row_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "duration_frames",
        "duration_sec",
        "state_label",
        "schedule_zone",
        "target_gap",
        "anchor_count",
        "state_score_mean",
        "state_score_peak",
    ]


def write_state_timeline_csv(path, rows):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _timeline_fieldnames()
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_state_report(result, schedule_runs=None, final_indices=None, policy_meta=None, analysis_summary=None, signal_report=None, candidate_sidecar=None):
    meta = dict(result.get("meta", {}) or {})
    summary = dict(meta.get("summary", {}) or {})
    segments = list(result.get("segments", []) or [])
    frame_rows = list(result.get("frame_rows", []) or [])
    schedule_runs = list(schedule_runs or [])
    policy_meta = dict(policy_meta or {})
    final_indices = [int(idx) for idx in list(final_indices or [])]

    report = {
        "report_schema_version": str(REPORT_SCHEMA_VERSION),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp_dir": str(meta.get("source_exp_dir", "") or ""),
        "input_artifacts": {
            "signal_timeseries_csv": "segment/signal_extraction/signal_timeseries.csv",
            "state_segments_json": "segment/state_segmentation/{}".format(_OUTPUT_JSON_NAME),
        },
        "artifact_contract": {
            "default_files": [
                _OUTPUT_REPORT_NAME,
                "state_overview.png",
            ],
            "factsource_files": [
                _OUTPUT_JSON_NAME,
            ],
            "sidecar_files": [
                "state_signal_candidate_compare.png",
            ],
            "legacy_default_outputs_replaced": list(_LEGACY_STATE_OUTPUT_NAMES),
        },
        "state_segmentation": {
            "summary": summary,
            "frame_statistics": _state_frame_statistics(segments, len(frame_rows)),
            "params": {
                "input_signals": list(meta.get("input_signals", []) or []),
                "formal_state_inputs": dict(meta.get("formal_state_inputs", {}) or {}),
                "normalization_method": dict(meta.get("normalization_method", {}) or {}),
                "smoothing": dict(meta.get("smoothing", {}) or {}),
                "weights": dict(meta.get("weights", {}) or {}),
                "score_calibration": dict(meta.get("score_calibration", {}) or {}),
                "enter_th": float(meta.get("enter_th", 0.0) or 0.0),
                "exit_th": float(meta.get("exit_th", 0.0) or 0.0),
                "min_high_len": int(meta.get("min_high_len", 0) or 0),
                "min_low_len": int(meta.get("min_low_len", 0) or 0),
                "merge_rule": dict(meta.get("merge_rule", {}) or {}),
            },
            "state_labels": list(meta.get("state_labels", []) or []),
            "median_dt_sec": float(meta.get("median_dt_sec", 0.0) or 0.0),
            "segments": _segment_digest(segments),
        },
    }

    if schedule_runs or policy_meta:
        report["density_schedule"] = {
            "window_statistics": _density_window_statistics(schedule_runs),
            "summary": list(policy_meta.get("density_schedule_summary", []) or []),
            "rules": {
                "safe_gap": int(policy_meta.get("safe_gap", 0) or 0),
                "transition_gap": int(policy_meta.get("transition_gap", 0) or 0),
                "high_gap": int(policy_meta.get("high_gap", 0) or 0),
                "pre_transition_frames": int(policy_meta.get("pre_transition_frames", 0) or 0),
                "post_transition_frames": int(policy_meta.get("post_transition_frames", 0) or 0),
                "min_anchor_spacing": int(policy_meta.get("min_anchor_spacing", 0) or 0),
                "min_segment_frames": int(policy_meta.get("min_segment_frames", 0) or 0),
                "scheduling_rules": dict(policy_meta.get("scheduling_rules", {}) or {}),
            },
            "final_keyframes": {
                "count": int(len(final_indices)),
                "indices": list(final_indices),
                "min_final_gap": int(policy_meta.get("min_final_gap", 0) or 0),
                "min_final_segment_frames": int(policy_meta.get("min_final_segment_frames", 0) or 0),
                "short_segment_merge_count": int(policy_meta.get("short_segment_merge_count", 0) or 0),
            },
            "zone_statistics": {
                "high_state_count": int(policy_meta.get("high_state_count", 0) or 0),
                "low_state_count": int(policy_meta.get("low_state_count", 0) or 0),
                "transition_band_count": int(policy_meta.get("transition_band_count", 0) or 0),
                "state_segment_count": int(policy_meta.get("state_segment_count", 0) or 0),
            },
        }

    if isinstance(signal_report, dict) and signal_report:
        report["signal_context"] = {
            "signal_report_path": "segment/signal_extraction/signal_report.json",
            "formal_state_inputs": dict(signal_report.get("formal_state_inputs", {}) or {}),
            "representative_signals": dict(signal_report.get("representative_signals", {}) or {}),
            "family_groups": list(signal_report.get("family_groups", []) or []),
        }

    if isinstance(analysis_summary, dict) and analysis_summary:
        report["segment_analysis_summary"] = dict(analysis_summary)

    if isinstance(candidate_sidecar, dict) and candidate_sidecar:
        report["validation_sidecar"] = dict(candidate_sidecar)

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
    rows = _coerce_signal_rows(rows)
    exp_dir_path = Path(exp_dir).resolve() if exp_dir is not None else None
    output_dir_path = Path(output_dir).resolve() if output_dir is not None else None
    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    weights = dict(weights or DEFAULT_WEIGHTS)
    smoothing_window = max(1, int(smoothing_window))
    min_high_len = max(1, int(min_high_len))
    min_low_len = max(1, int(min_low_len))
    glitch_merge_len = _resolve_glitch_merge_len(smoothing_window, glitch_merge_len)

    rows, formal_state_input_meta = _resolve_formal_state_inputs(rows)
    motion_values = [float(row["motion_velocity_state_input"]) for row in rows]
    semantic_values = [float(row["semantic_velocity_state_input"]) for row in rows]

    timestamps = [float(row["timestamp"]) for row in rows]
    dt_sec = _median_dt(timestamps)
    official_track = _compute_candidate_track(
        rows=rows,
        components=[
            {"weight": float(weights.get("motion_velocity", 0.0) or 0.0), "values": motion_values},
            {"weight": float(weights.get("semantic_velocity", 0.0) or 0.0), "values": semantic_values},
        ],
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        min_low_len=min_low_len,
        glitch_merge_len=glitch_merge_len,
        dt_sec=dt_sec,
    )
    state_scores_raw = list(official_track.get("score_raw", []) or [])
    state_scores = list(official_track.get("score", []) or [])
    merged_states = list(official_track.get("states", []) or [])
    segments = list(official_track.get("segments", []) or [])
    summary = dict(official_track.get("summary", {}) or {})
    score_calibration_meta = dict(official_track.get("score_calibration", {}) or {})
    merge_meta = dict(official_track.get("merge_rule", {}) or {})
    validation_sidecar = _build_validation_sidecar(
        rows=rows,
        motion_values=motion_values,
        semantic_values=semantic_values,
        official_track=official_track,
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        min_low_len=min_low_len,
        glitch_merge_len=glitch_merge_len,
        dt_sec=dt_sec,
    )

    frame_rows = []
    for idx, row in enumerate(rows):
        frame_rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "motion_velocity_raw": float(row["motion_velocity"]),
                "blur_score_raw": float(row["blur_score"]),
                "semantic_velocity_raw": float(row["semantic_velocity"]),
                "motion_velocity_state_signal": float(motion_values[idx]),
                "semantic_velocity_state_signal": float(semantic_values[idx]),
                "state_score_raw": float(state_scores_raw[idx]),
                "state_score": float(state_scores[idx]),
                "state_label": str(merged_states[idx]),
            }
        )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp_dir": str(exp_dir_path) if exp_dir_path is not None else "",
        "input_csv": str(Path(input_csv).resolve()) if input_csv else "",
        "input_signals": list(DEFAULT_INPUT_SIGNALS),
        "formal_state_inputs": dict(formal_state_input_meta),
        "normalization_method": {
            "name": "minmax_after_robust_clip",
            "scope": "each official state input independently",
            "source_mode": str(formal_state_input_meta.get("source_mode", "")),
            "legacy_cli_fallback_only": True,
        },
        "smoothing": {
            "source_mode": str(formal_state_input_meta.get("source_mode", "")),
            "signals": dict((formal_state_input_meta.get("signals", {}) or {})),
        },
        "weights": {
            "motion_velocity": float(weights.get("motion_velocity", 0.0) or 0.0),
            "semantic_velocity": float(weights.get("semantic_velocity", 0.0) or 0.0),
        },
        "score_calibration": dict(score_calibration_meta),
        "enter_th": float(enter_th),
        "exit_th": float(exit_th),
        "min_high_len": int(min_high_len),
        "min_low_len": int(min_low_len),
        "merge_rule": merge_meta,
        "frame_count": int(len(rows)),
        "median_dt_sec": float(dt_sec),
        "state_labels": [STATE_LOW, STATE_HIGH],
        "summary": summary,
    }

    json_payload = {
        "source_exp_dir": str(exp_dir_path) if exp_dir_path is not None else "",
        "input_csv": str(Path(input_csv).resolve()) if input_csv else "",
        "summary": summary,
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
        "validation_sidecar": validation_sidecar,
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
    input_csv = segment_dir / "signal_extraction" / _INPUT_CSV_NAME
    rows = _read_signal_rows(input_csv)
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


def write_state_segmentation_outputs(result, schedule_runs=None, final_indices=None, policy_meta=None, analysis_summary=None, signal_report=None):
    output_dir = ensure_dir(result["output_dir"], name="state segmentation output dir")
    json_path = output_dir / _OUTPUT_JSON_NAME
    report_path = output_dir / _OUTPUT_REPORT_NAME
    candidate_sidecar = dict(result.get("validation_sidecar", {}) or {})
    report = build_state_report(
        result,
        schedule_runs=schedule_runs,
        final_indices=final_indices,
        policy_meta=policy_meta,
        analysis_summary=analysis_summary,
        signal_report=signal_report,
        candidate_sidecar=dict(candidate_sidecar.get("report", {}) or {}),
    )
    write_json_atomic(json_path, result["json_payload"])
    write_state_report(report_path, report)
    _remove_legacy_state_outputs(output_dir)
    return {
        "json_path": json_path,
        "report_path": report_path,
        "report": report,
    }
