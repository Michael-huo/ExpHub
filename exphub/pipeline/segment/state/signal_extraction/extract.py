#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from exphub.common.logging import log_info, log_warn
from exphub.pipeline.segment.state.observed_signals import (
    compute_frame_signal_rows,
    compute_motion_rows,
    compute_semantic_rows,
)
from exphub.pipeline.segment.state.observed_signals.kinematics import minmax_normalize, moving_average

FORMAL_STATE_INPUT_SIGNALS = [
    "motion_velocity",
    "semantic_velocity",
]

FORMAL_STATE_INPUT_COLUMNS = {
    "motion_velocity": "motion_velocity_state_input",
    "semantic_velocity": "semantic_velocity_state_input",
}

RAW_SIGNAL_DISPLAY_NAMES = {
    "blur_score": "blur_score_raw (sharpness proxy)",
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

SELECTED_SIGNALS = [
    "feature_motion",
    "appearance_delta",
    "brightness_jump",
    "blur_score",
    "motion_displacement",
    "motion_velocity",
    "motion_acceleration",
    "semantic_delta",
    "semantic_velocity",
    "semantic_acceleration",
]

_SIGNAL_DESCRIPTIONS = {
    "feature_motion": "Visual feature drift estimated from adjacent frames.",
    "appearance_delta": "Appearance change magnitude between adjacent frames.",
    "brightness_jump": "Frame-to-frame brightness variation magnitude.",
    "blur_score": "Raw sharpness proxy from grayscale Laplacian variance; higher means sharper, not higher blur risk.",
    "motion_displacement": "Motion displacement magnitude from the motion estimator.",
    "motion_velocity": "Velocity-like motion signal used by current state segmentation.",
    "motion_acceleration": "Acceleration-like motion signal derived from velocity changes.",
    "semantic_delta": "Semantic embedding distance between adjacent frames.",
    "semantic_velocity": "Velocity-like semantic change signal.",
    "semantic_acceleration": "Acceleration-like semantic change signal.",
}


def _raw_signal_display_name(signal_name):
    return str(RAW_SIGNAL_DISPLAY_NAMES.get(signal_name, signal_name))


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


def _resolve_inline_outputs(segment_dir=None, output_dir=None, cache_dir=None):
    segment_dir = Path(segment_dir).resolve() if segment_dir is not None else None
    if output_dir is None:
        if segment_dir is None:
            raise ValueError("output_dir or segment_dir is required for inline signal extraction")
        output_dir = segment_dir / "signal_extraction"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        if segment_dir is None:
            raise ValueError("cache_dir or segment_dir is required for inline signal extraction")
        cache_dir = segment_dir / ".segment_cache" / "signal_extraction"
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "segment_dir": segment_dir,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
    }


def _normalize_cache_policy_name(policy_name):
    name = str(policy_name or "").strip().lower()
    if not name:
        name = "state"
    return name


def _existing_semantic_cache_dirs(segment_dir, keyframes_meta):
    cache_root = (segment_dir / ".segment_cache").resolve()
    candidates = []
    seen = set()

    policy_name = _normalize_cache_policy_name(
        keyframes_meta.get("policy_name", "") or keyframes_meta.get("policy", "") or ""
    )
    if policy_name:
        cache_dir = cache_root / policy_name
        cache_path = cache_dir / "semantic_embeddings.npz"
        if cache_path.is_file():
            resolved_dir = cache_dir.resolve()
            cache_key = str(resolved_dir)
            if cache_key not in seen:
                seen.add(cache_key)
                candidates.append(resolved_dir)

    if cache_root.is_dir():
        for cache_path in sorted(cache_root.rglob("semantic_embeddings.npz")):
            resolved_dir = cache_path.parent.resolve()
            cache_key = str(resolved_dir)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            candidates.append(resolved_dir)

    return candidates


def _resolve_semantic_cache_dir(segment_dir, keyframes_meta, extraction_cache_dir):
    existing_dirs = _existing_semantic_cache_dirs(segment_dir, keyframes_meta)
    if existing_dirs:
        cache_dir = existing_dirs[0]
        cache_path = cache_dir / "semantic_embeddings.npz"
        log_info("signal extraction reuse semantic cache: {}".format(cache_path))
        return {
            "cache_dir": cache_dir,
            "cache_path": cache_path,
            "cache_reused": True,
            "cache_source": "existing",
        }

    cache_dir = Path(extraction_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "semantic_embeddings.npz"
    log_warn(
        "signal extraction semantic cache missing under {} ; will rebuild embeddings into {}".format(
            segment_dir / ".segment_cache",
            cache_path,
        )
    )
    return {
        "cache_dir": cache_dir,
        "cache_path": cache_path,
        "cache_reused": False,
        "cache_source": "signal_extraction_rebuild",
    }


def _build_selected_rows(frame_rows):
    selected_rows = []
    for row in frame_rows:
        item = {
            "frame_idx": int(row.get("frame_idx", 0)),
            "timestamp": float(row.get("ts_sec", 0.0) or 0.0),
        }
        for signal_name in SELECTED_SIGNALS:
            item[signal_name] = float(row.get(signal_name, 0.0) or 0.0)
        selected_rows.append(item)
    return selected_rows


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
            "display_names": dict((signal_name, _formal_state_input_display_name(signal_name)) for signal_name in FORMAL_STATE_INPUT_SIGNALS),
            "raw_display_names": dict((signal_name, _raw_signal_display_name(signal_name)) for signal_name in FORMAL_STATE_INPUT_SIGNALS),
            "pipeline": [
                "robust_clip",
                "minmax_normalize",
                "moving_average",
            ],
            "signals": {},
            "analysis_only_note": (
                "Signals outside formal_state_inputs may still be extracted and plotted for analysis, "
                "but they do not enter the current official state mainline score."
            ),
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
        "display_names": dict((signal_name, _formal_state_input_display_name(signal_name)) for signal_name in FORMAL_STATE_INPUT_SIGNALS),
        "raw_display_names": dict((signal_name, _raw_signal_display_name(signal_name)) for signal_name in FORMAL_STATE_INPUT_SIGNALS),
        "pipeline": [
            "robust_clip",
            "minmax_normalize",
            "moving_average",
        ],
        "signals": signal_meta,
        "analysis_only_note": (
            "Signals outside formal_state_inputs may still be extracted and plotted for analysis, "
            "but they do not enter the current official state mainline score."
        ),
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

    inline_outputs = _resolve_inline_outputs(
        segment_dir=segment_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    segment_dir = inline_outputs["segment_dir"] or Path(frame_paths[0]).resolve().parent.parent
    exp_dir = Path(exp_dir).resolve() if exp_dir is not None else segment_dir.parent
    keyframes_meta = dict(keyframes_meta or {})

    log_info("signal extraction start: exp_dir={} frames={}".format(exp_dir, len(frame_paths)))
    semantic_cache = _resolve_semantic_cache_dir(
        segment_dir,
        keyframes_meta,
        inline_outputs["cache_dir"],
    )

    try:
        with _quiet_semantic_library_noise():
            semantic_rows, semantic_meta = compute_semantic_rows(
                frame_paths,
                semantic_cache["cache_dir"],
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
    log_info("signal extraction frame signal compose start: frames={}".format(len(frame_paths)))
    frame_rows, frame_signal_meta = compute_frame_signal_rows(
        frame_paths,
        timestamps,
        semantic_rows=semantic_rows,
        motion_rows=motion_rows,
    )
    selected_rows = _build_selected_rows(frame_rows)
    log_info(
        "signal extraction preprocess official state inputs: {}".format(
            ", ".join(FORMAL_STATE_INPUT_SIGNALS)
        )
    )
    selected_rows, formal_state_input_meta = build_formal_state_input_rows(selected_rows)
    return {
        "exp_dir": exp_dir,
        "segment_dir": segment_dir,
        "output_dir": inline_outputs["output_dir"],
        "rows": selected_rows,
        "timestamps": timestamps,
        "semantic_cache": semantic_cache,
        "semantic_meta": semantic_meta,
        "motion_meta": motion_meta,
        "frame_signal_meta": frame_signal_meta,
        "formal_state_input_meta": formal_state_input_meta,
    }
