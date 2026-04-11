#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from exphub.common.logging import log_info
from ._state_signals import compute_frame_signal_rows
from ._state_motion_energy import compute_motion_rows
from ._state_semantic_openclip import compute_semantic_rows
from ._state_kinematics import minmax_normalize, moving_average

FORMAL_STATE_INPUT_SIGNALS = [
    "motion_velocity",
    "semantic_velocity",
]

FORMAL_STATE_INPUT_COLUMNS = {
    "motion_velocity": "motion_velocity_state_input",
    "semantic_velocity": "semantic_velocity_state_input",
}

FORMAL_STATE_INPUT_DISPLAY_NAMES = {
    "motion_velocity": "motion_velocity (processed)",
    "semantic_velocity": "semantic_velocity (processed)",
}

FORMAL_STATE_INPUT_PREPROCESSING = {
    "motion_velocity": {
        "clip_quantiles": [0.02, 0.98],
        "smoothing_window": 9,
        "invert_after_normalize": False,
    },
    "semantic_velocity": {
        "clip_quantiles": [0.02, 0.98],
        "smoothing_window": 7,
        "invert_after_normalize": False,
    },
}

_SIGNAL_DESCRIPTIONS = {
    "motion_velocity": "Velocity-like motion signal used by current state segmentation.",
    "semantic_velocity": "Velocity-like semantic change signal.",
}


def _raw_signal_display_name(signal_name):
    return str(signal_name)


def _formal_state_input_display_name(signal_name):
    return str(FORMAL_STATE_INPUT_DISPLAY_NAMES.get(signal_name, "{} (processed)".format(signal_name)))


def _formal_state_input_description(signal_name):
    return (
        "Processed official state mainline input derived from {} after robust clipping, "
        "min-max normalization, and light smoothing."
    ).format(str(signal_name))


@contextmanager
def _quiet_semantic_library_noise():
    logger_names = [
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ]
    logger_states = []
    for name in logger_names:
        logger = logging.getLogger(name)
        logger_states.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for logger, level in logger_states:
            logger.setLevel(level)


def _series_stats(values):
    if not values:
        return {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


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
    return [float(x) for x in clipped.tolist()], {
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "degenerated": bool(abs(float(upper_bound) - float(lower_bound)) < 1e-12),
    }


def build_formal_state_input_rows(rows):
    rows = list(rows or [])
    if not rows:
        return [], {
            "signal_names": list(FORMAL_STATE_INPUT_SIGNALS),
            "processed_columns": dict(FORMAL_STATE_INPUT_COLUMNS),
            "display_names": dict(
                (signal_name, _formal_state_input_display_name(signal_name))
                for signal_name in FORMAL_STATE_INPUT_SIGNALS
            ),
            "raw_display_names": dict(
                (signal_name, _raw_signal_display_name(signal_name))
                for signal_name in FORMAL_STATE_INPUT_SIGNALS
            ),
            "pipeline": [
                "robust_clip",
                "minmax_normalize",
                "moving_average",
            ],
            "signals": {},
        }

    processed_values = {}
    signal_meta = {}
    for signal_name in FORMAL_STATE_INPUT_SIGNALS:
        config = dict(FORMAL_STATE_INPUT_PREPROCESSING.get(signal_name, {}) or {})
        clip_quantiles = list(config.get("clip_quantiles", [0.02, 0.98]) or [0.02, 0.98])
        raw_values = [float(row.get(signal_name, 0.0) or 0.0) for row in rows]
        clipped_values, clip_meta = _robust_clip(raw_values, clip_quantiles[0], clip_quantiles[1])
        normalized_values = minmax_normalize(clipped_values)
        invert_after_normalize = bool(config.get("invert_after_normalize", False))
        if invert_after_normalize:
            normalized_values = [float(1.0 - float(value)) for value in normalized_values]
        smoothed_values, actual_window = moving_average(
            normalized_values,
            int(config.get("smoothing_window", 1) or 1),
        )
        processed_column = FORMAL_STATE_INPUT_COLUMNS[signal_name]
        processed_values[processed_column] = [float(value) for value in smoothed_values]
        signal_meta[signal_name] = {
            "raw_column": str(signal_name),
            "processed_column": str(processed_column),
            "display_name": _formal_state_input_display_name(signal_name),
            "raw_display_name": _raw_signal_display_name(signal_name),
            "clip": clip_meta,
            "normalization": {
                "method": "minmax_after_robust_clip",
                "range": [0.0, 1.0],
                "invert_after_normalize": invert_after_normalize,
            },
            "smoothing": {
                "method": "moving_average",
                "window_size": int(actual_window),
            },
            "raw_semantics": str(_SIGNAL_DESCRIPTIONS.get(signal_name, "")),
            "processed_semantics": _formal_state_input_description(signal_name),
            "raw_stats": _series_stats(raw_values),
            "processed_stats": _series_stats(processed_values[processed_column]),
        }

    processed_rows = []
    for idx, row in enumerate(rows):
        item = dict(row)
        for signal_name in FORMAL_STATE_INPUT_SIGNALS:
            processed_column = FORMAL_STATE_INPUT_COLUMNS[signal_name]
            item[processed_column] = float(processed_values[processed_column][idx])
        processed_rows.append(item)

    return processed_rows, {
        "signal_names": list(FORMAL_STATE_INPUT_SIGNALS),
        "processed_columns": dict(FORMAL_STATE_INPUT_COLUMNS),
        "display_names": dict(
            (signal_name, _formal_state_input_display_name(signal_name))
            for signal_name in FORMAL_STATE_INPUT_SIGNALS
        ),
        "raw_display_names": dict(
            (signal_name, _raw_signal_display_name(signal_name))
            for signal_name in FORMAL_STATE_INPUT_SIGNALS
        ),
        "pipeline": [
            "robust_clip",
            "minmax_normalize",
            "moving_average",
        ],
        "signals": signal_meta,
    }


def extract_signal_timeseries_from_frames(
    frame_paths,
    timestamps,
    exp_dir=None,
    segment_dir=None,
    keyframes_meta=None,
    output_dir=None,
    cache_dir=None,
):
    del keyframes_meta
    del output_dir
    del cache_dir

    frame_paths = [Path(path).resolve() for path in list(frame_paths or [])]
    timestamps = [float(value) for value in list(timestamps or [])]
    if not frame_paths:
        raise ValueError("signal extraction requires non-empty frame_paths")
    if len(frame_paths) != len(timestamps):
        raise ValueError(
            "signal extraction frame/timestamp mismatch: frames={} timestamps={}".format(
                len(frame_paths), len(timestamps)
            )
        )

    segment_dir = Path(segment_dir).resolve() if segment_dir is not None else Path(frame_paths[0]).resolve().parent.parent
    exp_dir = Path(exp_dir).resolve() if exp_dir is not None else segment_dir.parent

    log_info("signal extraction start: exp_dir={} frames={}".format(exp_dir, len(frame_paths)))

    try:
        with _quiet_semantic_library_noise():
            semantic_rows, semantic_meta = compute_semantic_rows(
                frame_paths,
                timestamps=timestamps,
            )
    except ModuleNotFoundError as e:
        missing_name = str(getattr(e, "name", "") or "")
        if missing_name in ("torch", "open_clip"):
            raise SystemExit(
                "[ERR] semantic signal extraction requires torch/open_clip in the current python environment. "
                "Please rerun with the segmentclip interpreter or another environment that already has these dependencies."
            )
        raise
    except RuntimeError as e:
        text = str(e)
        if "open_clip" in text:
            raise SystemExit(
                "[ERR] semantic signal extraction failed to initialize OpenCLIP: {}. "
                "Please rerun with the configured segmentclip interpreter.".format(text)
            )
        raise

    log_info("signal extraction motion observe start: frames={}".format(len(frame_paths)))
    motion_rows, motion_meta = compute_motion_rows(
        frame_paths,
        timestamps=timestamps,
    )
    log_info(
        "signal extraction compose official state inputs: {}".format(
            ", ".join(FORMAL_STATE_INPUT_SIGNALS)
        )
    )
    frame_rows, frame_signal_meta = compute_frame_signal_rows(
        frame_paths,
        timestamps,
        semantic_rows=semantic_rows,
        motion_rows=motion_rows,
    )
    frame_rows, formal_state_input_meta = build_formal_state_input_rows(frame_rows)
    return {
        "exp_dir": exp_dir,
        "segment_dir": segment_dir,
        "output_dir": None,
        "rows": frame_rows,
        "timestamps": timestamps,
        "semantic_meta": semantic_meta,
        "motion_meta": motion_meta,
        "frame_signal_meta": frame_signal_meta,
        "formal_state_input_meta": formal_state_input_meta,
    }
