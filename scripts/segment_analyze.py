#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze an existing segment directory without touching the main pipeline."""

import argparse
import csv
import json
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
from scripts._common import ensure_dir, ensure_file, list_frames_sorted, log_info, log_prog, write_json_atomic
from scripts._segment.research import (
    DEFAULT_PEAK_CONFIG,
    DEFAULT_SCORE_WEIGHTS,
    annotate_peaks,
    apply_scores,
    build_candidate_points,
    compute_frame_signal_rows,
    compute_semantic_rows,
    save_candidate_points_overview,
    save_candidate_roles_overview,
    save_peaks_preview,
    save_score_curve,
    save_score_curve_with_keyframes,
    save_semantic_curve,
    save_semantic_vs_nonsemantic,
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
    "semantic_smooth",
    "score_raw",
    "score_smooth",
    "local_prominence",
    "is_peak",
    "peak_rank",
    "peak_suppressed_reason",
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
    ap.add_argument("--score_w_blur", type=float, default=DEFAULT_SCORE_WEIGHTS["blur_score"])
    ap.add_argument("--score_use_blur", action="store_true", help="include blur_score in score_raw (default: disabled)")
    ap.add_argument("--score_use_semantic", action="store_true", help="include semantic_delta in score_raw (default: disabled)")
    ap.add_argument("--smooth_window", type=int, default=5)

    ap.add_argument("--peak_window", type=int, default=DEFAULT_PEAK_CONFIG["window_radius"])
    ap.add_argument("--peak_threshold_std", type=float, default=DEFAULT_PEAK_CONFIG["threshold_std"])
    ap.add_argument("--min_peak_distance", type=int, default=DEFAULT_PEAK_CONFIG["min_peak_distance"])
    ap.add_argument("--min_peak_score_raw", type=float, default=DEFAULT_PEAK_CONFIG["min_peak_score_raw"])
    ap.add_argument("--min_peak_prominence", type=float, default=DEFAULT_PEAK_CONFIG["min_peak_prominence"])
    ap.add_argument("--edge_margin", type=int, default=DEFAULT_PEAK_CONFIG["edge_margin"])
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



def _resolve_keyframe_sets(keyframes_meta):
    uniform_indices = keyframes_meta.get("uniform_base_indices") or keyframes_meta.get("keyframe_indices") or []
    final_indices = keyframes_meta.get("keyframe_indices") or uniform_indices
    uniform_set = set(int(x) for x in uniform_indices)
    final_set = set(int(x) for x in final_indices)
    return uniform_set, final_set


def _mark_uniform_keyframes(rows, uniform_keyframe_indices):
    keyframe_set = set(int(x) for x in uniform_keyframe_indices)
    for row in rows:
        row["is_uniform_keyframe"] = bool(int(row["frame_idx"]) in keyframe_set)
    return keyframe_set



def _write_csv(path, rows):
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})



def _peak_meta_public(peak_meta):
    keep_keys = [
        "window_radius",
        "threshold_std",
        "threshold",
        "min_peak_distance",
        "min_peak_score_raw",
        "min_peak_prominence",
        "edge_margin",
        "local_peak_count",
        "eligible_peak_count",
        "peak_count",
        "suppressed_peak_count",
    ]
    return {key: peak_meta.get(key) for key in keep_keys}



def run_segment_analyze(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exp_dir = _resolve_exp_dir(args)
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    analysis_dir = exp_dir / "segment" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    log_prog("segment analyze start: exp_dir={}".format(exp_dir))
    data = _load_segment_inputs(segment_dir)
    semantic_rows, semantic_meta = compute_semantic_rows(
        data["frame_paths"],
        analysis_dir,
        smooth_window=int(args.smooth_window),
    )
    rows, signal_meta = compute_frame_signal_rows(data["frame_paths"], data["timestamps"], semantic_rows=semantic_rows)
    uniform_keyframe_set, final_keyframe_set = _resolve_keyframe_sets(data["keyframes_meta"])
    keyframe_set = _mark_uniform_keyframes(rows, uniform_keyframe_set)

    score_weights = {
        "appearance_delta": float(args.score_w_appearance),
        "brightness_jump": float(args.score_w_brightness),
        "feature_motion": float(args.score_w_motion),
        "semantic_delta": float(args.score_w_semantic),
        "blur_score": float(args.score_w_blur),
    }
    rows, score_meta = apply_scores(
        rows,
        weights=score_weights,
        smooth_window=int(args.smooth_window),
        use_blur_in_score=bool(args.score_use_blur),
        use_semantic_in_score=bool(args.score_use_semantic),
    )
    rows, peak_meta = annotate_peaks(
        rows,
        window_radius=int(args.peak_window),
        threshold_std=float(args.peak_threshold_std),
        min_peak_distance=int(args.min_peak_distance),
        min_peak_score_raw=float(args.min_peak_score_raw),
        min_peak_prominence=float(args.min_peak_prominence),
        edge_margin=int(args.edge_margin),
    )
    candidate_points = build_candidate_points(rows, peak_meta)

    csv_path = analysis_dir / "frame_scores.csv"
    json_path = analysis_dir / "frame_scores.json"
    curve_path = analysis_dir / "score_curve.png"
    curve_kf_path = analysis_dir / "score_curve_with_keyframes.png"
    peaks_path = analysis_dir / "peaks_preview.png"
    candidate_points_path = analysis_dir / "candidate_points.json"
    candidate_roles_summary_path = analysis_dir / "candidate_roles_summary.json"
    candidate_overview_path = analysis_dir / "candidate_points_overview.png"
    candidate_roles_overview_path = analysis_dir / "candidate_roles_overview.png"
    semantic_curve_path = analysis_dir / "semantic_curve.png"
    semantic_vs_nonsemantic_path = analysis_dir / "semantic_vs_nonsemantic.png"
    meta_path = analysis_dir / "analysis_meta.json"

    _write_csv(csv_path, rows)
    write_json_atomic(str(json_path), rows, indent=2)
    write_json_atomic(str(candidate_points_path), candidate_points, indent=2)
    write_json_atomic(str(candidate_roles_summary_path), candidate_points.get("candidate_roles_summary", {}), indent=2)
    save_score_curve(rows, curve_path)
    save_score_curve_with_keyframes(rows, curve_kf_path, sorted(final_keyframe_set))
    save_peaks_preview(rows, peaks_path)
    save_candidate_points_overview(
        rows,
        candidate_overview_path,
        sorted(final_keyframe_set),
        candidate_points.get("selected_candidates", []),
    )
    save_candidate_roles_overview(
        rows,
        candidate_roles_overview_path,
        sorted(final_keyframe_set),
        candidate_points.get("candidate_roles_summary", {}),
    )
    save_semantic_curve(rows, semantic_curve_path)
    save_semantic_vs_nonsemantic(
        rows,
        semantic_vs_nonsemantic_path,
        sorted(final_keyframe_set),
        candidate_points.get("selected_candidates", []),
    )

    analysis_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(exp_dir),
        "source_segment_dir": str(segment_dir),
        "num_frames": int(len(rows)),
        "num_keyframes": int(len(final_keyframe_set)),
        "policy_name": data["keyframes_meta"].get("policy_name", "uniform"),
        "observed_signals": list(score_meta["observed_signals"]),
        "scored_signals": list(score_meta["scored_signals"]),
        "enabled_signals": list(signal_meta["enabled_signals"]),
        "signal_methods": {
            "appearance_delta": signal_meta["appearance_delta_method"],
            "brightness_jump": signal_meta["brightness_jump_method"],
            "blur_score": signal_meta["blur_score_method"],
            "feature_motion": signal_meta["feature_motion_method"],
            "semantic_delta": signal_meta["semantic_delta_method"],
            "semantic_smooth": signal_meta["semantic_smooth_method"],
        },
        "score_weights": score_meta["score_weights"],
        "use_blur_in_score": bool(score_meta["use_blur_in_score"]),
        "use_semantic_in_score": bool(score_meta["use_semantic_in_score"]),
        "smoothing": score_meta["smoothing"],
        "peak_detection": _peak_meta_public(peak_meta),
        "candidate_role_enabled": True,
        "role_rules": candidate_points.get("role_rules", {}),
        "rerank_weights": candidate_points.get("rerank_weights", {}),
        "role_thresholds": candidate_points.get("role_thresholds", {}),
        "semantic_enabled": bool(semantic_meta["enabled"]),
        "semantic_backend": semantic_meta["backend"],
        "semantic_model_name": semantic_meta["model_name"],
        "semantic_pretrained": semantic_meta["pretrained"],
        "semantic_device": semantic_meta["device"],
        "semantic_cache_path": semantic_meta["cache_path"],
        "semantic_cache_hit": bool(semantic_meta["cache_hit"]),
        "semantic_cache_lookup_sec": float(semantic_meta["cache_lookup_sec"]),
        "semantic_encode_sec": float(semantic_meta["encode_sec"]),
        "semantic_threshold": float(candidate_points.get("relation_thresholds", {}).get("semantic_smooth", 0.0)),
        "semantic_peak_enabled": bool(semantic_meta.get("semantic_peak_enabled", False)),
        "uniform_keyframe_indices": sorted(int(x) for x in uniform_keyframe_set),
        "final_keyframe_indices": sorted(int(x) for x in final_keyframe_set),
        "candidate_points": {
            "selected_count": int(len(candidate_points.get("selected_candidates", []))),
            "suppressed_count": int(len(candidate_points.get("suppressed_candidates", []))),
            "reason_thresholds": candidate_points.get("reason_thresholds", {}),
            "relation_thresholds": candidate_points.get("relation_thresholds", {}),
            "role_thresholds": candidate_points.get("role_thresholds", {}),
            "counts": candidate_points.get("counts", {}),
        },
        "boundary_candidate_count": int(candidate_points.get("counts", {}).get("boundary_candidate_count", 0)),
        "support_candidate_count": int(candidate_points.get("counts", {}).get("support_candidate_count", 0)),
        "semantic_only_candidate_count": int(candidate_points.get("counts", {}).get("semantic_only_candidate_count", 0)),
        "suppressed_candidate_count": int(candidate_points.get("counts", {}).get("suppressed_candidate_count", 0)),
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
            "candidate_points_json": str(candidate_points_path),
            "candidate_roles_summary_json": str(candidate_roles_summary_path),
            "candidate_points_overview_png": str(candidate_overview_path),
            "candidate_roles_overview_png": str(candidate_roles_overview_path),
            "semantic_embeddings_npz": semantic_meta["cache_path"],
            "semantic_curve_png": str(semantic_curve_path),
            "semantic_vs_nonsemantic_png": str(semantic_vs_nonsemantic_path),
        },
    }
    write_json_atomic(str(meta_path), analysis_meta, indent=2)

    log_info("analysis_dir: {}".format(analysis_dir))
    log_info("frame_scores.csv rows: {}".format(len(rows)))
    log_info("uniform base keyframes: {}".format(len(uniform_keyframe_set)))
    log_info("final keyframes: {}".format(len(final_keyframe_set)))
    log_info("candidate peaks: {}".format(len(candidate_points.get("selected_candidates", []))))
    log_info(
        "candidate roles: boundary={} support={} semantic_only={} suppressed={}".format(
            int(candidate_points.get("counts", {}).get("boundary_candidate_count", 0)),
            int(candidate_points.get("counts", {}).get("support_candidate_count", 0)),
            int(candidate_points.get("counts", {}).get("semantic_only_candidate_count", 0)),
            int(candidate_points.get("counts", {}).get("suppressed_candidate_count", 0)),
        )
    )
    log_info("semantic cache hit: {}".format(bool(semantic_meta["cache_hit"])))
    log_prog("segment analyze done: {}".format(analysis_dir))


if __name__ == "__main__":
    run_segment_analyze()
