#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts._common import ensure_dir, ensure_file, write_json_atomic
from scripts._segment.research.kinematics import moving_average


STATE_LOW = "low_state"
STATE_HIGH = "high_state"

DEFAULT_INPUT_SIGNALS = [
    "motion_velocity",
    "feature_motion",
]
DEFAULT_NORMALIZATION_METHOD = "robust_zscore_per_signal"
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_WEIGHTS = {
    "motion_velocity": 0.6,
    "feature_motion": 0.4,
}
DEFAULT_ENTER_TH = 0.65
DEFAULT_EXIT_TH = 0.45
DEFAULT_MIN_HIGH_LEN = 24
DEFAULT_MIN_LOW_LEN = 24
DEFAULT_GLITCH_MERGE_LEN = 12

REPORT_SCHEMA_VERSION = "state_report.v1"
_INPUT_CSV_NAME = "signal_timeseries.csv"
_OUTPUT_JSON_NAME = "state_segments.json"
_OUTPUT_TIMELINE_NAME = "state_timeline.csv"
_OUTPUT_REPORT_NAME = "state_report.json"

_LEGACY_STATE_OUTPUT_NAMES = [
    "density_schedule.csv",
    "density_schedule_overview.png",
    "state_segments.csv",
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
                    "motion_velocity": float(row.get("motion_velocity", 0.0) or 0.0),
                    "feature_motion": float(row.get("feature_motion", 0.0) or 0.0),
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
                "motion_velocity": float(row.get("motion_velocity", 0.0) or 0.0),
                "feature_motion": float(row.get("feature_motion", 0.0) or 0.0),
            }
        )
    if not coerced:
        raise ValueError("state segmentation requires non-empty signal rows")
    return coerced


def _zscore(values):
    arr = np.asarray(values, dtype=np.float32)
    mean_value = float(arr.mean()) if arr.size > 0 else 0.0
    std_value = float(arr.std()) if arr.size > 0 else 0.0
    if abs(std_value) < 1e-12:
        return [0.0 for _ in values], {
            "center": mean_value,
            "scale": 0.0,
            "scale_source": "std",
            "degenerated": True,
        }
    out = (arr - mean_value) / float(std_value)
    return _float_list(out.tolist()), {
        "center": mean_value,
        "scale": std_value,
        "scale_source": "std",
        "degenerated": False,
    }


def _robust_zscore(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return [], {
            "center": 0.0,
            "scale": 0.0,
            "scale_source": "mad",
            "degenerated": True,
        }
    median_value = float(np.median(arr))
    mad_value = float(np.median(np.abs(arr - median_value)))
    robust_scale = float(1.4826 * mad_value)
    if abs(robust_scale) < 1e-12:
        fallback_values, fallback_meta = _zscore(values)
        fallback_meta["center_source"] = "mean"
        fallback_meta["fallback_from"] = "mad"
        return fallback_values, fallback_meta
    out = (arr - median_value) / float(robust_scale)
    return _float_list(out.tolist()), {
        "center": median_value,
        "scale": robust_scale,
        "scale_source": "mad",
        "center_source": "median",
        "degenerated": False,
    }


def _standardize_signal(values, method):
    if str(method) == "zscore_per_signal":
        standardized, meta = _zscore(values)
        meta["method"] = "zscore_per_signal"
        return standardized, meta
    standardized, meta = _robust_zscore(values)
    meta["method"] = "robust_zscore_per_signal"
    return standardized, meta


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


def build_state_report(result, schedule_runs=None, final_indices=None, policy_meta=None, analysis_summary=None, signal_report=None):
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
                _OUTPUT_TIMELINE_NAME,
                "state_overview.png",
            ],
            "factsource_files": [
                _OUTPUT_JSON_NAME,
            ],
            "legacy_default_outputs_replaced": list(_LEGACY_STATE_OUTPUT_NAMES),
        },
        "state_segmentation": {
            "summary": summary,
            "frame_statistics": _state_frame_statistics(segments, len(frame_rows)),
            "params": {
                "input_signals": list(meta.get("input_signals", []) or []),
                "normalization_method": dict(meta.get("normalization_method", {}) or {}),
                "smoothing_window": int(meta.get("smoothing_window", 0) or 0),
                "weights": dict(meta.get("weights", {}) or {}),
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
            "representative_signals": dict(signal_report.get("representative_signals", {}) or {}),
            "family_groups": list(signal_report.get("family_groups", []) or []),
        }

    if isinstance(analysis_summary, dict) and analysis_summary:
        report["segment_analysis_summary"] = dict(analysis_summary)

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

    motion_values = [float(row["motion_velocity"]) for row in rows]
    feature_values = [float(row["feature_motion"]) for row in rows]
    motion_std, motion_norm_meta = _standardize_signal(motion_values, normalization_method)
    feature_std, feature_norm_meta = _standardize_signal(feature_values, normalization_method)
    motion_smooth, actual_window = moving_average(motion_std, smoothing_window)
    feature_smooth, actual_window = moving_average(feature_std, smoothing_window)

    state_scores = []
    for idx in range(len(rows)):
        score = (
            float(weights.get("motion_velocity", 0.0) or 0.0) * float(motion_smooth[idx])
            + float(weights.get("feature_motion", 0.0) or 0.0) * float(feature_smooth[idx])
        )
        state_scores.append(float(score))

    raw_states = _apply_state_machine(
        scores=state_scores,
        enter_th=enter_th,
        exit_th=exit_th,
        min_high_len=min_high_len,
        min_low_len=min_low_len,
    )
    merged_states, merge_meta = _merge_short_runs(raw_states, glitch_merge_len)

    timestamps = [float(row["timestamp"]) for row in rows]
    dt_sec = _median_dt(timestamps)
    segments = _build_segments(rows, merged_states, state_scores, dt_sec)
    summary = _build_summary(segments, len(rows))

    frame_rows = []
    for idx, row in enumerate(rows):
        frame_rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "motion_velocity_raw": float(row["motion_velocity"]),
                "feature_motion_raw": float(row["feature_motion"]),
                "motion_velocity_state_signal": float(motion_smooth[idx]),
                "feature_motion_state_signal": float(feature_smooth[idx]),
                "state_score": float(state_scores[idx]),
                "state_label": str(merged_states[idx]),
            }
        )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_exp_dir": str(exp_dir_path) if exp_dir_path is not None else "",
        "input_csv": str(Path(input_csv).resolve()) if input_csv else "",
        "input_signals": list(DEFAULT_INPUT_SIGNALS),
        "normalization_method": {
            "name": str(normalization_method),
            "scope": "each signal independently",
            "signals": {
                "motion_velocity": motion_norm_meta,
                "feature_motion": feature_norm_meta,
            },
        },
        "smoothing_window": int(actual_window),
        "weights": {
            "motion_velocity": float(weights.get("motion_velocity", 0.0) or 0.0),
            "feature_motion": float(weights.get("feature_motion", 0.0) or 0.0),
        },
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
    timeline_rows = build_state_timeline_rows(result, schedule_runs=schedule_runs)
    timeline_path = output_dir / _OUTPUT_TIMELINE_NAME
    json_path = output_dir / _OUTPUT_JSON_NAME
    report_path = output_dir / _OUTPUT_REPORT_NAME
    report = build_state_report(
        result,
        schedule_runs=schedule_runs,
        final_indices=final_indices,
        policy_meta=policy_meta,
        analysis_summary=analysis_summary,
        signal_report=signal_report,
    )
    write_state_timeline_csv(timeline_path, timeline_rows)
    write_json_atomic(json_path, result["json_payload"])
    write_state_report(report_path, report)
    _remove_legacy_state_outputs(output_dir)
    return {
        "timeline_path": timeline_path,
        "json_path": json_path,
        "report_path": report_path,
        "report": report,
    }
