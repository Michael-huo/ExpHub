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
    save_roles_overview,
    save_score_overview,
    save_semantic_overview,
)


FIELDNAMES = [
    "frame_idx",
    "ts_sec",
    "score_raw",
    "score_smooth",
    "semantic_delta",
    "semantic_smooth",
    "candidate_role",
    "is_candidate",
    "is_selected_candidate",
    "is_final_keyframe",
    "source_type",
    "source_role",
]

LEGACY_OUTPUT_NAMES = [
    "analysis_meta.json",
    "candidate_points.json",
    "candidate_roles_summary.json",
    "frame_scores.json",
    "peaks_preview.png",
    "score_curve.png",
    "score_curve_with_keyframes.png",
    "candidate_points_overview.png",
    "candidate_roles_overview.png",
    "semantic_curve.png",
    "semantic_vs_nonsemantic.png",
    "semantic_embeddings.npz",
]
DEFAULT_SEMANTIC_CACHE_NAME = "semantic_embeddings.npz"


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


def _load_segment_inputs(exp_dir, segment_dir):
    frames_dir = ensure_dir(segment_dir / "frames", name="segment frames dir")
    timestamps_path = ensure_file(segment_dir / "timestamps.txt", name="segment timestamps")
    keyframes_meta_path = ensure_file(segment_dir / "keyframes" / "keyframes_meta.json", name="segment keyframes meta")
    step_meta_path = ensure_file(segment_dir / "step_meta.json", name="segment step meta")
    preprocess_meta_path = segment_dir / "preprocess_meta.json"
    exp_meta_path = exp_dir / "exp_meta.json"

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
    exp_meta = _read_json(exp_meta_path) if exp_meta_path.is_file() else {}
    return {
        "frames_dir": frames_dir,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "keyframes_meta": keyframes_meta,
        "step_meta": step_meta,
        "preprocess_meta": preprocess_meta,
        "exp_meta": exp_meta,
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


def _summarize_final_keyframes(keyframes_meta):
    items = list(keyframes_meta.get("keyframes") or [])
    source_type_counts = {}
    source_role_counts = {}
    promotion_source_counts = {}
    promoted_items = []

    for item in items:
        source_type = str(item.get("source_type", "") or "unknown")
        source_role = str(item.get("source_role", "") or "unknown")
        promotion_source = str(item.get("promotion_source", "") or "")
        source_type_counts[source_type] = int(source_type_counts.get(source_type, 0)) + 1
        source_role_counts[source_role] = int(source_role_counts.get(source_role, 0)) + 1
        if promotion_source:
            promotion_source_counts[promotion_source] = int(promotion_source_counts.get(promotion_source, 0)) + 1
        if source_role == "promoted_support_candidate":
            promoted_items.append(dict(item))

    promoted_items.sort(key=lambda item: (-float(item.get("rerank_score", 0.0) or 0.0), int(item.get("frame_idx", 0))))
    return {
        "source_type_counts": source_type_counts,
        "source_role_counts": source_role_counts,
        "promotion_source_counts": promotion_source_counts,
        "promoted_items": promoted_items,
    }


def _write_csv(path, rows):
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDNAMES})


def _remove_legacy_outputs(analysis_dir):
    for name in LEGACY_OUTPUT_NAMES:
        path = analysis_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception:
            continue


def _candidate_maps(candidate_points):
    selected_map = {}
    all_map = {}

    for item in candidate_points.get("selected_candidates", []):
        frame_idx = int(item.get("frame_idx", 0))
        selected_map[frame_idx] = item
        all_map[frame_idx] = item

    for item in candidate_points.get("suppressed_candidates", []):
        frame_idx = int(item.get("frame_idx", 0))
        if frame_idx not in all_map:
            all_map[frame_idx] = item

    return selected_map, all_map


def _final_keyframe_map(keyframes_meta):
    item_map = {}
    for item in keyframes_meta.get("keyframes", []):
        item_map[int(item.get("frame_idx", 0))] = dict(item)
    return item_map


def _slim_rows(rows, candidate_points, keyframes_meta):
    selected_map, all_candidate_map = _candidate_maps(candidate_points)
    final_map = _final_keyframe_map(keyframes_meta)
    slim_rows = []

    for row in rows:
        frame_idx = int(row.get("frame_idx", 0))
        candidate = all_candidate_map.get(frame_idx, {})
        final_item = final_map.get(frame_idx, {})
        slim_rows.append(
            {
                "frame_idx": frame_idx,
                "ts_sec": float(row.get("ts_sec", 0.0)),
                "score_raw": float(row.get("score_raw", 0.0)),
                "score_smooth": float(row.get("score_smooth", 0.0)),
                "semantic_delta": float(row.get("semantic_delta", 0.0)),
                "semantic_smooth": float(row.get("semantic_smooth", 0.0)),
                "candidate_role": str(candidate.get("candidate_role", "") or ""),
                "is_candidate": bool(frame_idx in all_candidate_map),
                "is_selected_candidate": bool(frame_idx in selected_map),
                "is_final_keyframe": bool(frame_idx in final_map),
                "source_type": str(final_item.get("source_type", "") or ""),
                "source_role": str(final_item.get("source_role", "") or ""),
            }
        )
    return slim_rows


def _join_reason(parts):
    cleaned = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        cleaned.append(text)
    if not cleaned:
        return ""
    return ", ".join(cleaned[:4])


def _top_candidate_digest(items, limit):
    out = []
    for item in items[:limit]:
        out.append(
            {
                "frame_idx": int(item.get("frame_idx", 0)),
                "reason": _join_reason(item.get("role_reasons", []) or item.get("reasons", [])),
                "rerank_score": float(item.get("rerank_score", 0.0) or 0.0),
                "semantic_relation": str(item.get("semantic_relation", "") or ""),
            }
        )
    return out


def _top_promoted_digest(items, limit):
    out = []
    for item in items[:limit]:
        out.append(
            {
                "frame_idx": int(item.get("frame_idx", 0)),
                "reason": _join_reason([item.get("promotion_reason", ""), item.get("promotion_source", "")]),
                "rerank_score": float(item.get("rerank_score", 0.0) or 0.0),
                "semantic_relation": str(item.get("semantic_relation", "") or ""),
            }
        )
    return out


def _experiment_info(exp_dir, exp_meta, step_meta):
    params = dict(exp_meta.get("params", {}) or {})
    step_params = dict(step_meta.get("params", {}) or {})
    dataset = str(exp_meta.get("dataset", "") or "")
    sequence = str(exp_meta.get("sequence", "") or "")
    tag = str(exp_meta.get("tag", "") or "")

    if (not dataset or not sequence) and len(exp_dir.parts) >= 3:
        sequence = sequence or str(exp_dir.parent.name)
        dataset = dataset or str(exp_dir.parent.parent.name)

    return {
        "dataset": dataset,
        "sequence": sequence,
        "tag": tag,
        "start_sec": step_params.get("start_sec", params.get("start_sec", "")),
        "dur": step_params.get("dur", params.get("dur", "")),
        "fps": step_params.get("fps", params.get("fps", "")),
        "kf_gap": step_params.get("kf_gap", params.get("kf_gap", "")),
    }


def _analysis_summary(exp_dir, data, rows, candidate_points, keyframes_meta, final_keyframe_summary, score_meta, semantic_meta):
    exp_info = _experiment_info(exp_dir, data["exp_meta"], data["step_meta"])
    keyframe_summary = dict(keyframes_meta.get("summary", {}) or {})
    counts = dict(candidate_points.get("counts", {}) or {})
    roles_summary = dict(candidate_points.get("candidate_roles_summary", {}) or {})

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy_name": str(keyframes_meta.get("policy_name", "") or ""),
        "dataset": exp_info["dataset"],
        "sequence": exp_info["sequence"],
        "tag": exp_info["tag"],
        "start_sec": exp_info["start_sec"],
        "dur": exp_info["dur"],
        "fps": exp_info["fps"],
        "kf_gap": exp_info["kf_gap"],
        "frame_count_total": int(keyframes_meta.get("frame_count_total", len(rows))),
        "uniform_base_count": int(len(keyframes_meta.get("uniform_base_indices", []) or [])),
        "final_keyframe_count": int(len(keyframes_meta.get("keyframe_indices", []) or [])),
        "extra_kf_ratio": float(keyframe_summary.get("extra_kf_ratio", 0.0) or 0.0),
        "keyframe_bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
        "final_keyframe_source_counts": final_keyframe_summary["source_type_counts"],
        "final_keyframe_source_roles": final_keyframe_summary["source_role_counts"],
        "num_boundary_relocated": int(keyframe_summary.get("num_boundary_relocated", 0) or 0),
        "num_support_inserted": int(keyframe_summary.get("num_support_inserted", 0) or 0),
        "num_promoted_support_inserted": int(keyframe_summary.get("num_promoted_support_inserted", 0) or 0),
        "num_burst_windows_triggered": int(keyframe_summary.get("num_burst_windows_triggered", 0) or 0),
        "candidate_role_counts": counts,
        "selected_candidate_count": int(len(candidate_points.get("selected_candidates", []) or [])),
        "suppressed_candidate_count": int(len(candidate_points.get("suppressed_candidates", []) or [])),
        "uniform_base_indices": list(keyframes_meta.get("uniform_base_indices", []) or []),
        "final_keyframe_indices": list(keyframes_meta.get("keyframe_indices", []) or []),
        "semantic_backend": str(semantic_meta.get("backend", "") or ""),
        "semantic_model_name": str(semantic_meta.get("model_name", "") or ""),
        "use_semantic_in_score": bool(score_meta.get("use_semantic_in_score", False)),
        "semantic_cache_hit": bool(semantic_meta.get("cache_hit", False)),
        "fixed_budget": bool(keyframe_summary.get("fixed_budget", False)),
        "relocated_count": int(keyframe_summary.get("relocated_count", 0) or 0),
        "avg_abs_shift": float(keyframe_summary.get("avg_abs_shift", 0.0) or 0.0),
        "max_abs_shift": int(keyframe_summary.get("max_abs_shift", 0) or 0),
        "semantic_displacement_mean": float(keyframe_summary.get("semantic_displacement_mean", 0.0) or 0.0),
        "semantic_displacement_max": float(keyframe_summary.get("semantic_displacement_max", 0.0) or 0.0),
        "semantic_velocity_mean": float(keyframe_summary.get("semantic_velocity_mean", 0.0) or 0.0),
        "semantic_velocity_max": float(keyframe_summary.get("semantic_velocity_max", 0.0) or 0.0),
        "semantic_acceleration_mean": float(keyframe_summary.get("semantic_acceleration_mean", 0.0) or 0.0),
        "semantic_acceleration_max": float(keyframe_summary.get("semantic_acceleration_max", 0.0) or 0.0),
        "semantic_density_mean": float(keyframe_summary.get("semantic_density_mean", 0.0) or 0.0),
        "semantic_density_max": float(keyframe_summary.get("semantic_density_max", 0.0) or 0.0),
        "semantic_signal_stats": dict(semantic_meta.get("signal_stats", {}) or {}),
        "top_candidates": {
            "boundary": _top_candidate_digest(roles_summary.get("boundary_candidates", []) or [], 5),
            "support": _top_candidate_digest(roles_summary.get("support_candidates", []) or [], 5),
            "promoted": _top_promoted_digest(final_keyframe_summary.get("promoted_items", []) or [], 5),
        },
    }


def _roles_for_plot(candidate_points, final_keyframe_summary):
    roles_summary = dict(candidate_points.get("candidate_roles_summary", {}) or {})
    roles_summary["promoted_candidates"] = [
        {"frame_idx": int(item.get("frame_idx", 0))}
        for item in final_keyframe_summary.get("promoted_items", [])
    ]
    return roles_summary


def _resolve_semantic_cache_dir(segment_dir, keyframes_meta):
    policy_name = str(keyframes_meta.get("policy_name", "") or "").strip()
    if policy_name:
        policy_cache_dir = segment_dir / ".segment_cache" / policy_name
        if (policy_cache_dir / DEFAULT_SEMANTIC_CACHE_NAME).is_file():
            return policy_cache_dir
    return segment_dir / ".segment_cache" / "segment_analyze"


def run_segment_analyze(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exp_dir = _resolve_exp_dir(args)
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    analysis_dir = exp_dir / "segment" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_outputs(analysis_dir)

    log_prog("segment analyze start: exp_dir={}".format(exp_dir))
    data = _load_segment_inputs(exp_dir, segment_dir)
    cache_dir = _resolve_semantic_cache_dir(segment_dir, data["keyframes_meta"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    semantic_rows, semantic_meta = compute_semantic_rows(
        data["frame_paths"],
        cache_dir,
        smooth_window=int(args.smooth_window),
        timestamps=data["timestamps"],
    )
    rows, _signal_meta = compute_frame_signal_rows(data["frame_paths"], data["timestamps"], semantic_rows=semantic_rows)
    uniform_keyframe_set, final_keyframe_set = _resolve_keyframe_sets(data["keyframes_meta"])
    _mark_uniform_keyframes(rows, uniform_keyframe_set)
    final_keyframe_summary = _summarize_final_keyframes(data["keyframes_meta"])

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

    csv_rows = _slim_rows(rows, candidate_points, data["keyframes_meta"])
    roles_for_plot = _roles_for_plot(candidate_points, final_keyframe_summary)
    analysis_summary = _analysis_summary(
        exp_dir,
        data,
        rows,
        candidate_points,
        data["keyframes_meta"],
        final_keyframe_summary,
        score_meta,
        semantic_meta,
    )

    csv_path = analysis_dir / "frame_scores.csv"
    summary_path = analysis_dir / "analysis_summary.json"
    score_overview_path = analysis_dir / "score_overview.png"
    roles_overview_path = analysis_dir / "roles_overview.png"
    semantic_overview_path = analysis_dir / "semantic_overview.png"

    _write_csv(csv_path, csv_rows)
    write_json_atomic(str(summary_path), analysis_summary, indent=2)
    save_score_overview(rows, score_overview_path, sorted(final_keyframe_set), candidate_points.get("selected_candidates", []))
    save_roles_overview(rows, roles_overview_path, sorted(final_keyframe_set), roles_for_plot)
    save_semantic_overview(
        rows,
        semantic_overview_path,
        sorted(final_keyframe_set),
        candidate_points.get("selected_candidates", []),
        keyframe_items=data["keyframes_meta"].get("keyframes", []),
    )

    log_info("analysis_dir: {}".format(analysis_dir))
    log_info("analysis_summary.json written")
    log_info("frame_scores.csv rows: {}".format(len(csv_rows)))
    log_info("uniform base keyframes: {}".format(len(uniform_keyframe_set)))
    log_info("final keyframes: {}".format(len(final_keyframe_set)))
    log_info("final keyframe sources: {}".format(final_keyframe_summary["source_role_counts"]))
    log_info("candidate peaks: {}".format(len(candidate_points.get("selected_candidates", []))))
    log_info("semantic cache hit: {}".format(bool(semantic_meta.get("cache_hit", False))))
    log_info("semantic cache path: {}".format(semantic_meta.get("cache_path", "")))
    log_prog("segment analyze done: {}".format(analysis_dir))


if __name__ == "__main__":
    run_segment_analyze()
