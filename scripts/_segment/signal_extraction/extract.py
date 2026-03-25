#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import logging
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from scripts._common import ensure_dir, ensure_file, list_frames_sorted, log_info, log_warn, write_json_atomic
from scripts._segment.policies.naming import normalize_policy_name
from scripts._segment.research import compute_frame_signal_rows, compute_motion_rows, compute_semantic_rows


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

SIGNAL_FAMILIES = {
    "image": [
        "feature_motion",
        "appearance_delta",
        "brightness_jump",
        "blur_score",
    ],
    "motion": [
        "motion_displacement",
        "motion_velocity",
        "motion_acceleration",
    ],
    "semantic": [
        "semantic_delta",
        "semantic_velocity",
        "semantic_acceleration",
    ],
}

REPRESENTATIVE_SIGNALS = [
    "feature_motion",
    "motion_velocity",
    "semantic_velocity",
]

DEFAULT_PLOT_SMOOTH_WINDOW = 5

_TIMESERIES_COLUMNS = [
    "frame_idx",
    "timestamp",
] + list(SELECTED_SIGNALS)


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


def _resolve_segment_inputs(exp_dir):
    exp_dir = Path(exp_dir).resolve()
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    frames_dir = ensure_dir(segment_dir / "frames", name="segment frames dir")
    timestamps_path = ensure_file(segment_dir / "timestamps.txt", name="segment timestamps")
    keyframes_meta_path = segment_dir / "keyframes" / "keyframes_meta.json"
    keyframes_meta = _read_json(keyframes_meta_path) if keyframes_meta_path.is_file() else {}

    frame_paths = list_frames_sorted(frames_dir)
    timestamps = _read_timestamps(timestamps_path)
    if not frame_paths:
        raise SystemExit("[ERR] no frames found under {}".format(frames_dir))
    if len(frame_paths) != len(timestamps):
        raise SystemExit(
            "[ERR] frame count and timestamps count mismatch: frames={} timestamps={}".format(
                len(frame_paths), len(timestamps)
            )
        )

    output_dir = (segment_dir / "signal_extraction").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (segment_dir / ".segment_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = (cache_root / "signal_extraction").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "exp_dir": exp_dir,
        "segment_dir": segment_dir,
        "frames_dir": frames_dir,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "keyframes_meta": keyframes_meta,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "cache_root": cache_root,
    }


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


def _existing_semantic_cache_dirs(segment_dir, keyframes_meta):
    cache_root = (segment_dir / ".segment_cache").resolve()
    candidates = []
    seen = set()

    policy_name = normalize_policy_name(keyframes_meta.get("policy_name", "") or keyframes_meta.get("policy", "") or "")
    preferred_names = []
    if policy_name:
        preferred_names.append(policy_name)
    preferred_names.extend(["semantic", "risk", "segment_analyze"])

    for name in preferred_names:
        cache_dir = cache_root / name
        cache_path = cache_dir / "semantic_embeddings.npz"
        if cache_path.is_file():
            resolved_dir = cache_dir.resolve()
            cache_key = str(resolved_dir)
            if cache_key in seen:
                continue
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


def write_signal_timeseries_csv(path, rows):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_TIMESERIES_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _TIMESERIES_COLUMNS})


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
    }


def build_signal_extraction_meta(payload, plot_meta):
    timestamps = list(payload["timestamps"])
    timestamp_range = {
        "start": float(timestamps[0]) if timestamps else 0.0,
        "end": float(timestamps[-1]) if timestamps else 0.0,
    }
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_signals": list(SELECTED_SIGNALS),
        "signal_families": dict(SIGNAL_FAMILIES),
        "representative_signals": list(REPRESENTATIVE_SIGNALS),
        "smoothing_used_for_plot": dict(plot_meta.get("smoothing_used_for_plot", {})),
        "normalization_used_for_plot": dict(plot_meta.get("normalization_used_for_plot", {})),
        "source_exp_dir": str(payload["exp_dir"]),
        "frame_count": int(len(payload["rows"])),
        "timestamp_range": timestamp_range,
        "semantic_source": {
            "cache_dir": str(payload["semantic_cache"]["cache_dir"]),
            "cache_path": str(payload["semantic_cache"]["cache_path"]),
            "cache_reused": bool(payload["semantic_cache"]["cache_reused"]),
            "cache_source": str(payload["semantic_cache"]["cache_source"]),
            "backend": str(payload["semantic_meta"].get("backend", "")),
            "cache_hit": bool(payload["semantic_meta"].get("cache_hit", False)),
        },
        "motion_source": {
            "backend": str(payload["motion_meta"].get("backend", "")),
            "cache_reused": False,
            "note": "motion signals are computed directly from frames; no standalone reusable cache is defined today.",
        },
        "frame_signal_methods": {
            "feature_motion": str(payload["frame_signal_meta"].get("feature_motion_method", "")),
            "appearance_delta": str(payload["frame_signal_meta"].get("appearance_delta_method", "")),
            "brightness_jump": str(payload["frame_signal_meta"].get("brightness_jump_method", "")),
            "blur_score": str(payload["frame_signal_meta"].get("blur_score_method", "")),
        },
    }
    return meta


def write_signal_extraction_meta(path, meta):
    write_json_atomic(Path(path).resolve(), meta, indent=2)


def materialize_signal_extraction_outputs(payload, plot_smooth_window=DEFAULT_PLOT_SMOOTH_WINDOW):
    output_dir = Path(payload["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "signal_timeseries.csv"
    write_signal_timeseries_csv(csv_path, payload["rows"])
    log_info("signal extraction csv write: {}".format(csv_path))

    from .visualize import save_signal_plots

    plot_meta = save_signal_plots(
        output_dir=output_dir,
        rows=payload["rows"],
        smooth_window=plot_smooth_window,
    )
    meta = build_signal_extraction_meta(payload, plot_meta)
    meta_path = output_dir / "signal_extraction_meta.json"
    write_signal_extraction_meta(meta_path, meta)
    return {
        "csv_path": csv_path,
        "meta_path": meta_path,
        "meta": meta,
        "plot_meta": plot_meta,
        "plot_paths": {
            "image_family": output_dir / "signal_image_family.png",
            "motion_family": output_dir / "signal_motion_family.png",
            "semantic_family": output_dir / "signal_semantic_family.png",
            "representatives": output_dir / "signal_representatives.png",
        },
    }


def extract_signal_timeseries(exp_dir, plot_smooth_window=DEFAULT_PLOT_SMOOTH_WINDOW):
    payload = _resolve_segment_inputs(exp_dir)
    result = extract_signal_timeseries_from_frames(
        frame_paths=payload["frame_paths"],
        timestamps=payload["timestamps"],
        exp_dir=payload["exp_dir"],
        segment_dir=payload["segment_dir"],
        keyframes_meta=payload["keyframes_meta"],
        output_dir=payload["output_dir"],
        cache_dir=payload["cache_dir"],
    )
    io_paths = materialize_signal_extraction_outputs(result, plot_smooth_window=plot_smooth_window)
    result["csv_path"] = io_paths["csv_path"]
    result["meta_path"] = io_paths["meta_path"]
    return result
