#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze an existing segment directory without touching the main pipeline."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
else:
    _REPO_ROOT = Path(__file__).resolve().parents[1]

from exphub.context import ExperimentContext
from scripts._common import ensure_dir, ensure_file, list_frames_sorted, log_err, log_info, log_prog, write_json_atomic
from scripts._segment.research import (
    DEFAULT_PEAK_CONFIG,
    DEFAULT_SCORE_WEIGHTS,
    annotate_peaks,
    apply_scores,
    compute_frame_signal_rows,
    save_peaks_preview,
    save_score_curve,
    save_score_curve_with_keyframes,
)


FIELDNAMES = [
    "frame_idx",
    "ts_sec",
    "file_name",
    "appearance_delta",
    "brightness_jump",
    "blur_score",
    "feature_motion",
    "semantic_delta",
    "score_raw",
    "score_smooth",
    "is_peak",
    "peak_rank",
    "is_uniform_keyframe",
]


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Analyze an existing segment directory and emit research-side artifacts.")
    ap.add_argument("--exp_dir", default="", help="existing experiment directory; if set, overrides dataset/sequence/tag-based resolution")
    ap.add_argument("--exphub", default=str(_REPO_ROOT), help="ExpHub root used when resolving exp_dir from experiment parameters")
    ap.add_argument("--exp_root", default="", help="override experiments root when resolving from experiment parameters")

    ap.add_argument("--dataset", default="")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--h", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--dur", default="")
    ap.add_argument("--start_sec", default="")
    ap.add_argument("--kf_gap", type=int, default=0)

    ap.add_argument("--score_w_appearance", type=float, default=DEFAULT_SCORE_WEIGHTS["appearance_delta"])
    ap.add_argument("--score_w_brightness", type=float, default=DEFAULT_SCORE_WEIGHTS["brightness_jump"])
    ap.add_argument("--score_w_motion", type=float, default=DEFAULT_SCORE_WEIGHTS["feature_motion"])
    ap.add_argument("--score_w_semantic", type=float, default=DEFAULT_SCORE_WEIGHTS["semantic_delta"])
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--peak_window", type=int, default=DEFAULT_PEAK_CONFIG["window_radius"])
    ap.add_argument("--peak_threshold_std", type=float, default=DEFAULT_PEAK_CONFIG["threshold_std"])
    return ap



def _resolve_exp_dir(args):
    if args.exp_dir:
        return Path(args.exp_dir).resolve()

    required = [
        ("dataset", args.dataset),
        ("sequence", args.sequence),
        ("tag", args.tag),
        ("w", args.w),
        ("h", args.h),
        ("fps", args.fps),
        ("dur", args.dur),
        ("start_sec", args.start_sec),
    ]
    missing = [name for name, value in required if value in (None, "", 0, 0.0)]
    if missing:
        raise SystemExit("[ERR] missing args for exp_dir resolution: {}".format(", ".join(missing)))

    exphub_root = Path(args.exphub).resolve()
    exp_root_override = Path(args.exp_root).resolve() if args.exp_root else None
    ctx = ExperimentContext(
        exphub_root=exphub_root,
        dataset=str(args.dataset),
        sequence=str(args.sequence),
        tag=str(args.tag),
        w=int(args.w),
        h=int(args.h),
        start_sec=str(args.start_sec),
        dur=str(args.dur),
        fps=float(args.fps),
        kf_gap_input=int(args.kf_gap),
        exp_root_override=exp_root_override,
    )
    return ctx.exp_dir.resolve()



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



def _load_segment_inputs(segment_dir):
    frames_dir = ensure_dir(segment_dir / "frames", name="segment frames dir")
    timestamps_path = ensure_file(segment_dir / "timestamps.txt", name="segment timestamps")
    keyframes_meta_path = ensure_file(segment_dir / "keyframes" / "keyframes_meta.json", name="segment keyframes meta")
    step_meta_path = ensure_file(segment_dir / "step_meta.json", name="segment step meta")
    preprocess_meta_path = segment_dir / "preprocess_meta.json"

    frame_paths = list_frames_sorted(frames_dir)
    timestamps = _read_timestamps(timestamps_path)
    if len(frame_paths) != len(timestamps):
        raise SystemExit(
            "[ERR] frame count and timestamps count mismatch: frames={} timestamps={}".format(
                len(frame_paths), len(timestamps)
            )
        )

    keyframes_meta = _read_json(keyframes_meta_path)
    step_meta = _read_json(step_meta_path)
    preprocess_meta = _read_json(preprocess_meta_path) if preprocess_meta_path.is_file() else None
    return {
        "frames_dir": frames_dir,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "keyframes_meta": keyframes_meta,
        "step_meta": step_meta,
        "preprocess_meta": preprocess_meta,
    }



def _mark_uniform_keyframes(rows, keyframes_meta):
    indices = keyframes_meta.get("keyframe_indices") or []
    keyframe_set = set(int(x) for x in indices)
    for row in rows:
        row["is_uniform_keyframe"] = bool(int(row["frame_idx"]) in keyframe_set)
    return keyframe_set



def _write_csv(path, rows):
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})



def run_segment_analyze(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exp_dir = _resolve_exp_dir(args)
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    analysis_dir = exp_dir / "segment" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    log_prog("segment analyze start: exp_dir={}".format(exp_dir))
    data = _load_segment_inputs(segment_dir)
    rows, signal_meta = compute_frame_signal_rows(data["frame_paths"], data["timestamps"], semantic_enabled=False)
    keyframe_set = _mark_uniform_keyframes(rows, data["keyframes_meta"])

    score_weights = {
        "appearance_delta": float(args.score_w_appearance),
        "brightness_jump": float(args.score_w_brightness),
        "feature_motion": float(args.score_w_motion),
        "semantic_delta": float(args.score_w_semantic),
    }
    rows, score_meta = apply_scores(rows, weights=score_weights, smooth_window=int(args.smooth_window))
    rows, peak_meta = annotate_peaks(rows, window_radius=int(args.peak_window), threshold_std=float(args.peak_threshold_std))

    csv_path = analysis_dir / "frame_scores.csv"
    json_path = analysis_dir / "frame_scores.json"
    curve_path = analysis_dir / "score_curve.png"
    curve_kf_path = analysis_dir / "score_curve_with_keyframes.png"
    peaks_path = analysis_dir / "peaks_preview.png"
    meta_path = analysis_dir / "analysis_meta.json"

    _write_csv(csv_path, rows)
    write_json_atomic(str(json_path), rows, indent=2)
    save_score_curve(rows, curve_path)
    save_score_curve_with_keyframes(rows, curve_kf_path, sorted(keyframe_set))
    save_peaks_preview(rows, peaks_path)

    analysis_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(exp_dir),
        "source_segment_dir": str(segment_dir),
        "num_frames": int(len(rows)),
        "num_keyframes": int(len(keyframe_set)),
        "enabled_signals": list(signal_meta["enabled_signals"]),
        "signal_methods": {
            "appearance_delta": signal_meta["appearance_delta_method"],
            "brightness_jump": signal_meta["brightness_jump_method"],
            "blur_score": signal_meta["blur_score_method"],
            "feature_motion": signal_meta["feature_motion_method"],
            "semantic_delta": "disabled in commit 2A; outputs constant 0.0",
        },
        "score_weights": score_meta["score_weights"],
        "smoothing": score_meta["smoothing"],
        "peak_detection": peak_meta,
        "semantic_enabled": False,
        "uniform_keyframe_indices": sorted(int(x) for x in keyframe_set),
        "source_files": {
            "timestamps": str(segment_dir / "timestamps.txt"),
            "keyframes_meta": str(segment_dir / "keyframes" / "keyframes_meta.json"),
            "step_meta": str(segment_dir / "step_meta.json"),
            "preprocess_meta": str(segment_dir / "preprocess_meta.json") if data["preprocess_meta"] is not None else "",
        },
        "outputs": {
            "frame_scores_csv": str(csv_path),
            "frame_scores_json": str(json_path),
            "score_curve_png": str(curve_path),
            "score_curve_with_keyframes_png": str(curve_kf_path),
            "peaks_preview_png": str(peaks_path),
        },
    }
    write_json_atomic(str(meta_path), analysis_meta, indent=2)

    log_info("analysis_dir: {}".format(analysis_dir))
    log_info("frame_scores.csv rows: {}".format(len(rows)))
    log_info("uniform keyframes: {}".format(len(keyframe_set)))
    log_prog("segment analyze done: {}".format(analysis_dir))


if __name__ == "__main__":
    run_segment_analyze()
