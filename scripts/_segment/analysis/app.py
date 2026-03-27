#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze an existing segment directory for the current uniform/state mainline."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from exphub.context import ExperimentContext
from scripts._common import ensure_dir, ensure_file, list_frames_sorted, log_info, log_prog, log_warn, write_json_atomic
from scripts._segment.policies.naming import OFFICIAL_POLICY_NAMES, normalize_policy_name, policy_display_name
from scripts._segment.signal_extraction import extract_signal_timeseries
from scripts._segment.state_segmentation import run_state_segmentation, write_state_segmentation_outputs


_REPO_ROOT = Path(__file__).resolve().parents[3]

STALE_ANALYSIS_OUTPUT_NAMES = [
    "analysis_summary.json",
    "frame_scores.csv",
    "score_overview.png",
    "roles_overview.png",
    "semantic_overview.png",
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
    "risk_bundle.json",
    "risk_summary.json",
    "risk_timeseries.csv",
    "risk_windows.csv",
    "risk_curve.png",
    "risk_anchor_overview.png",
    "projection_overview.png",
    "proposed_schedule.json",
    "proposed_schedule.csv",
    "proposed_schedule_overview.png",
    "proposed_window_comparison.csv",
]


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Analyze an existing segment directory and refresh current uniform/state sidecar artifacts only."
    )
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
    ap.add_argument("--plot_smooth_window", type=int, default=5)
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


def _read_signal_rows(csv_path):
    rows = []
    with open(str(csv_path), "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {}
            for key, value in row.items():
                text = str(value or "").strip()
                if key == "frame_idx":
                    item[key] = int(text or 0)
                elif key == "timestamp":
                    item[key] = float(text or 0.0)
                elif text == "":
                    item[key] = ""
                else:
                    item[key] = float(text)
            rows.append(item)
    return rows


def _load_segment_inputs(exp_dir, segment_dir):
    frames_dir = ensure_dir(segment_dir / "frames", name="segment frames dir")
    timestamps_path = ensure_file(segment_dir / "timestamps.txt", name="segment timestamps")
    keyframes_meta_path = ensure_file(segment_dir / "keyframes" / "keyframes_meta.json", name="segment keyframes meta")
    step_meta_path = ensure_file(segment_dir / "step_meta.json", name="segment step meta")
    preprocess_meta_path = segment_dir / "preprocess_meta.json"
    deploy_schedule_path = segment_dir / "deploy_schedule.json"
    exp_meta_path = exp_dir / "exp_meta.json"

    frame_paths = list_frames_sorted(frames_dir)
    timestamps = _read_timestamps(timestamps_path)
    if len(frame_paths) != len(timestamps):
        raise SystemExit("[ERR] frame count and timestamps count mismatch: frames={} timestamps={}".format(len(frame_paths), len(timestamps)))

    return {
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "keyframes_meta": _read_json(keyframes_meta_path),
        "step_meta": _read_json(step_meta_path),
        "preprocess_meta": _read_json(preprocess_meta_path) if preprocess_meta_path.is_file() else None,
        "deploy_schedule": _read_json(deploy_schedule_path) if deploy_schedule_path.is_file() else None,
        "exp_meta": _read_json(exp_meta_path) if exp_meta_path.is_file() else {},
    }


def _policy_name(keyframes_meta):
    return normalize_policy_name(keyframes_meta.get("policy_name", "") or keyframes_meta.get("policy", "") or "uniform")


def _resolve_keyframe_sets(keyframes_meta):
    uniform_indices = list(keyframes_meta.get("uniform_base_indices") or keyframes_meta.get("keyframe_indices") or [])
    final_indices = list(keyframes_meta.get("keyframe_indices") or uniform_indices)
    return [int(x) for x in uniform_indices], [int(x) for x in final_indices]


def _remove_stale_outputs(analysis_dir):
    for name in STALE_ANALYSIS_OUTPUT_NAMES:
        path = analysis_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception:
            continue
    try:
        if analysis_dir.is_dir() and not list(analysis_dir.iterdir()):
            analysis_dir.rmdir()
    except Exception:
        pass


def _projection_maps(deploy_schedule):
    if not isinstance(deploy_schedule, dict):
        return {"raw_boundary_order": {}, "deploy_boundary_order": {}, "raw_segments": {}, "deploy_segments": {}}

    raw_boundary_order = {}
    for pos, frame_idx in enumerate(list(deploy_schedule.get("raw_keyframe_indices") or [])):
        raw_boundary_order[int(frame_idx)] = int(pos)
    deploy_boundary_order = {}
    for pos, frame_idx in enumerate(list(deploy_schedule.get("deploy_keyframe_indices") or [])):
        deploy_boundary_order[int(frame_idx)] = int(pos)

    raw_segments = {}
    deploy_segments = {}
    for item in list(deploy_schedule.get("segments") or []):
        raw_end_idx = int(item.get("raw_end_idx", 0) or 0)
        deploy_end_idx = int(item.get("deploy_end_idx", 0) or 0)
        payload = {
            "projection_segment_id": int(item.get("segment_id", 0) or 0),
            "projection_raw_gap": int(item.get("raw_gap", 0) or 0),
            "projection_deploy_gap": int(item.get("deploy_gap", 0) or 0),
            "projection_gap_error": int(item.get("gap_error", 0) or 0),
            "projection_boundary_shift": int(item.get("boundary_shift", 0) or 0),
        }
        raw_segments[raw_end_idx] = dict(payload)
        deploy_segments[deploy_end_idx] = dict(payload)
    return {
        "raw_boundary_order": raw_boundary_order,
        "deploy_boundary_order": deploy_boundary_order,
        "raw_segments": raw_segments,
        "deploy_segments": deploy_segments,
    }


def _projection_summary(deploy_schedule):
    if not isinstance(deploy_schedule, dict):
        return {}
    projection_stats = dict(deploy_schedule.get("projection_stats", {}) or {})
    return {
        "mean_abs_boundary_shift": float(projection_stats.get("mean_abs_boundary_shift", 0.0) or 0.0),
        "max_abs_boundary_shift": int(projection_stats.get("max_abs_boundary_shift", 0) or 0),
        "mean_abs_gap_error": float(projection_stats.get("mean_abs_gap_error", 0.0) or 0.0),
        "max_abs_gap_error": int(projection_stats.get("max_abs_gap_error", 0) or 0),
        "segment_count": int(projection_stats.get("segment_count", 0) or 0),
    }


def _signal_stats(rows):
    def _values(key):
        return [float(row.get(key, 0.0) or 0.0) for row in rows]

    def _mean(values):
        if not values:
            return 0.0
        return float(sum(values) / float(len(values)))

    def _max(values):
        if not values:
            return 0.0
        return float(max(values))

    summary = {"frame_count": int(len(rows))}
    for key in ("appearance_delta", "brightness_jump", "blur_score", "feature_motion", "motion_velocity", "semantic_velocity"):
        values = _values(key)
        summary["{}_mean".format(key)] = _mean(values)
        summary["{}_max".format(key)] = _max(values)
    return summary


def _state_row_map(frame_rows):
    row_map = {}
    for row in list(frame_rows or []):
        row_map[int(row.get("frame_idx", 0) or 0)] = row
    return row_map


def _build_timeseries_rows(signal_rows, keyframes_meta, deploy_schedule, state_result=None):
    uniform_indices, final_indices = _resolve_keyframe_sets(keyframes_meta)
    uniform_set = set(uniform_indices)
    final_set = set(final_indices)
    state_map = _state_row_map((state_result or {}).get("frame_rows", []))
    projection = _projection_maps(deploy_schedule)
    out_rows = []

    for row in list(signal_rows or []):
        frame_idx = int(row.get("frame_idx", 0) or 0)
        item = dict(row)
        item["is_uniform_anchor"] = bool(frame_idx in uniform_set)
        item["is_selected_keyframe"] = bool(frame_idx in final_set)
        item["is_active_keyframe"] = bool(frame_idx in final_set)
        item["is_raw_boundary"] = bool(frame_idx in projection["raw_boundary_order"])
        item["raw_boundary_order"] = projection["raw_boundary_order"].get(frame_idx)
        item["is_deploy_boundary"] = bool(frame_idx in projection["deploy_boundary_order"])
        item["deploy_boundary_order"] = projection["deploy_boundary_order"].get(frame_idx)
        projection_row = projection["raw_segments"].get(frame_idx) or projection["deploy_segments"].get(frame_idx) or {}
        for key in ("projection_segment_id", "projection_raw_gap", "projection_deploy_gap", "projection_gap_error", "projection_boundary_shift"):
            item[key] = projection_row.get(key)

        state_row = state_map.get(frame_idx)
        if state_row is not None:
            item["state_label"] = str(state_row.get("state_label", ""))
            item["state_score"] = float(state_row.get("state_score", 0.0) or 0.0)
            item["schedule_zone"] = str(state_row.get("schedule_zone", ""))
            item["target_gap"] = int(state_row.get("target_gap", 0) or 0)
        out_rows.append(item)
    return out_rows


def _write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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


def _build_summary(exp_dir, data, signal_rows, state_result=None):
    keyframes_meta = dict(data["keyframes_meta"])
    policy_name = _policy_name(keyframes_meta)
    exp_info = _experiment_info(exp_dir, data["exp_meta"], data["step_meta"])
    keyframe_summary = dict(keyframes_meta.get("summary", {}) or {})
    policy_meta = dict(keyframes_meta.get("policy_meta", {}) or {})
    uniform_indices, final_indices = _resolve_keyframe_sets(keyframes_meta)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy_name": str(policy_name),
        "policy_display_name": policy_display_name(policy_name),
        "dataset": exp_info["dataset"],
        "sequence": exp_info["sequence"],
        "tag": exp_info["tag"],
        "start_sec": exp_info["start_sec"],
        "dur": exp_info["dur"],
        "fps": exp_info["fps"],
        "kf_gap": exp_info["kf_gap"],
        "allocation": {
            "uniform_base_count": int(len(uniform_indices)),
            "final_keyframe_count": int(len(final_indices)),
            "keyframe_bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
            "extra_kf_ratio": float(keyframe_summary.get("extra_kf_ratio", 0.0) or 0.0),
        },
        "signals": _signal_stats(signal_rows),
    }
    projection = _projection_summary(data.get("deploy_schedule"))
    if projection:
        summary["projection"] = projection
    if policy_name == "state":
        state_block = {
            "state_segment_count": int(policy_meta.get("state_segment_count", 0) or 0),
            "high_state_count": int(policy_meta.get("high_state_count", 0) or 0),
            "low_state_count": int(policy_meta.get("low_state_count", 0) or 0),
            "transition_band_count": int(policy_meta.get("transition_band_count", 0) or 0),
            "safe_gap": int(policy_meta.get("safe_gap", 0) or 0),
            "transition_gap": int(policy_meta.get("transition_gap", 0) or 0),
            "high_gap": int(policy_meta.get("high_gap", 0) or 0),
            "min_final_gap": int(policy_meta.get("min_final_gap", 0) or 0),
            "min_final_segment_frames": int(policy_meta.get("min_final_segment_frames", 0) or 0),
            "short_segment_merge_count": int(policy_meta.get("short_segment_merge_count", 0) or 0),
        }
        if state_result is not None:
            state_block["state_segmentation_summary"] = dict((state_result.get("meta", {}) or {}).get("summary", {}) or {})
        summary["state_schedule"] = state_block
    return summary


def _official_log(keyframes_meta, signal_dir, state_dir=None, state_result=None):
    policy_name = _policy_name(keyframes_meta)
    uniform_indices, final_indices = _resolve_keyframe_sets(keyframes_meta)
    log_info("signal research dir: {}".format(signal_dir))
    log_info("uniform base keyframes: {}".format(len(uniform_indices)))
    log_info("final keyframes: {}".format(len(final_indices)))
    log_info("policy analyzed: {}".format(policy_name))
    if state_dir is not None:
        log_info("state research dir: {}".format(state_dir))
    if state_result is not None:
        summary = dict((state_result.get("meta", {}) or {}).get("summary", {}) or {})
        log_info("state segments: {}".format(int(summary.get("segment_count", 0) or 0)))
        log_info("high state frames: {}".format(int(summary.get("high_state_frame_count", 0) or 0)))


def run_segment_analyze(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exp_dir = _resolve_exp_dir(args)
    segment_dir = ensure_dir(exp_dir / "segment", name="segment dir")
    analysis_dir = exp_dir / "segment" / "analysis"
    _remove_stale_outputs(analysis_dir)

    log_prog("segment analyze start: exp_dir={}".format(exp_dir))
    data = _load_segment_inputs(exp_dir, segment_dir)
    policy_name = _policy_name(data["keyframes_meta"])
    if policy_name not in OFFICIAL_POLICY_NAMES:
        log_warn("segment analyze skipped for unsupported historical policy: {}".format(policy_name))
        return

    signal_payload = extract_signal_timeseries(exp_dir, plot_smooth_window=int(args.plot_smooth_window))
    signal_rows = list(signal_payload.get("rows", []) or _read_signal_rows(signal_payload["csv_path"]))

    state_result = None
    state_output_dir = None
    if policy_name == "state":
        state_output_dir = segment_dir / "state_segmentation"
        report_path = state_output_dir / "state_report.json"
        summary = _build_summary(exp_dir, data, signal_rows, state_result=state_result)
        if report_path.is_file():
            report = _read_json(report_path)
            if not isinstance(report, dict):
                report = {}
            report["segment_analysis_summary"] = summary
            report["signal_context"] = {
                "signal_report_path": "segment/signal_extraction/signal_report.json",
                "formal_state_inputs": dict((signal_payload.get("report", {}) or {}).get("formal_state_inputs", {}) or {}),
                "representative_signals": dict((signal_payload.get("report", {}) or {}).get("representative_signals", {}) or {}),
                "family_groups": list((signal_payload.get("report", {}) or {}).get("family_groups", []) or []),
            }
            write_json_atomic(str(report_path), report, indent=2)
        else:
            state_result = run_state_segmentation(exp_dir)
            write_state_segmentation_outputs(
                state_result,
                analysis_summary=summary,
                signal_report=signal_payload.get("report", {}),
            )
    else:
        summary = _build_summary(exp_dir, data, signal_rows, state_result=state_result)

    _official_log(
        data["keyframes_meta"],
        signal_payload["output_dir"],
        state_output_dir,
        state_result=state_result,
    )
    log_prog("segment analyze done: signal_dir={} state_dir={}".format(signal_payload["output_dir"], state_output_dir or ""))


if __name__ == "__main__":
    run_segment_analyze()
