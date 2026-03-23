#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze an existing segment directory without touching the main pipeline."""

import argparse
import csv
import json
import math
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
else:
    _REPO_ROOT = Path(__file__).resolve().parents[1]

from exphub.context import ExperimentContext
from scripts._common import ensure_dir, ensure_file, list_frames_sorted, log_info, log_prog, log_warn, write_json_atomic
from scripts._segment.policies.fixed_budget import allocate_fixed_budget, build_fixed_budget_rules
from scripts._segment.policies.naming import (
    OFFICIAL_POLICY_NAMES,
    normalize_policy_name,
    policy_display_name,
)
from scripts._segment.research import (
    build_risk_summary,
    compute_risk_bundle,
    save_allocation_overview,
    save_comparison_overview,
    save_kinematics_overview,
    save_projection_overview,
    save_risk_anchor_overview,
    save_risk_curve,
    compute_frame_signal_rows,
    compute_motion_rows,
    compute_semantic_rows,
    risk_bundle_to_dict,
    risk_frame_rows_to_dicts,
    risk_windows_to_dicts,
)


LEGACY_OUTPUT_NAMES = [
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
]
DEFAULT_SEMANTIC_CACHE_NAME = "semantic_embeddings.npz"
SINGLE_OBSERVER_MAP = {
    "semantic": ["motion"],
    "motion": ["semantic"],
    "uniform": ["semantic", "motion"],
}
FIXED_BUDGET_RULE_KEYS = (
    "relocate_radius",
    "min_gap",
    "snap_radius",
    "density_eps",
    "density_alpha",
    "density_beta",
    "smooth_window",
)


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
    ap.add_argument("--smooth_window", type=int, default=5)
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
    deploy_schedule_path = segment_dir / "deploy_schedule.json"
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
    deploy_schedule = _read_json(deploy_schedule_path) if deploy_schedule_path.is_file() else None
    exp_meta = _read_json(exp_meta_path) if exp_meta_path.is_file() else {}
    return {
        "frames_dir": frames_dir,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "keyframes_meta": keyframes_meta,
        "step_meta": step_meta,
        "preprocess_meta": preprocess_meta,
        "deploy_schedule": deploy_schedule,
        "exp_meta": exp_meta,
    }


def _resolve_keyframe_sets(keyframes_meta):
    uniform_indices = list(keyframes_meta.get("uniform_base_indices") or keyframes_meta.get("keyframe_indices") or [])
    final_indices = list(keyframes_meta.get("keyframe_indices") or uniform_indices)
    uniform_set = set(int(x) for x in uniform_indices)
    final_set = set(int(x) for x in final_indices)
    return uniform_indices, final_indices, uniform_set, final_set


def _mark_uniform_keyframes(rows, uniform_keyframe_indices):
    keyframe_set = set(int(x) for x in uniform_keyframe_indices)
    for row in rows:
        row["is_uniform_keyframe"] = bool(int(row["frame_idx"]) in keyframe_set)
    return keyframe_set


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


def _remove_legacy_outputs(analysis_dir):
    for name in LEGACY_OUTPUT_NAMES:
        path = analysis_dir / name
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception:
            continue


def _policy_name(keyframes_meta):
    return normalize_policy_name(keyframes_meta.get("policy_name", "") or "uniform")


def _signal_prefix(policy_name):
    policy_name = normalize_policy_name(policy_name)
    if policy_name == "semantic":
        return "semantic"
    if policy_name == "motion":
        return "motion"
    return ""


def _observer_policies(policy_name):
    return list(SINGLE_OBSERVER_MAP.get(str(policy_name), []))


def _needed_signal_policies(policy_name):
    names = set(_observer_policies(policy_name))
    if normalize_policy_name(policy_name) in ("semantic", "motion"):
        names.add(str(policy_name))
    return names


def _resolve_semantic_cache_dir(segment_dir, keyframes_meta):
    policy_name = normalize_policy_name(keyframes_meta.get("policy_name", "") or "")
    if policy_name:
        policy_cache_dir = segment_dir / ".segment_cache" / policy_name
        if (policy_cache_dir / DEFAULT_SEMANTIC_CACHE_NAME).is_file():
            return policy_cache_dir
    return segment_dir / ".segment_cache" / "segment_analyze"


def _used_frame_count(keyframes_meta, total_rows):
    value = int(keyframes_meta.get("frame_count_used", 0) or 0)
    if value <= 0:
        return int(total_rows)
    return int(min(value, int(total_rows)))


def _used_rows(rows, keyframes_meta):
    return list(rows[:_used_frame_count(keyframes_meta, len(rows))])


def _mean(values):
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _max(values):
    if not values:
        return 0.0
    return float(max(float(v) for v in values))


def _series(rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _rankdata(values):
    indexed = [(float(value), idx) for idx, value in enumerate(values)]
    indexed.sort(key=lambda item: (item[0], item[1]))
    ranks = [0.0 for _ in indexed]
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        avg_rank = (float(pos + 1) + float(end)) / 2.0
        for cur in range(pos, end):
            ranks[indexed[cur][1]] = float(avg_rank)
        pos = end
    return ranks


def _pearson(values_a, values_b):
    if not values_a or not values_b:
        return 0.0
    count = min(len(values_a), len(values_b))
    if count < 2:
        return 0.0
    xs = [float(values_a[idx]) for idx in range(count)]
    ys = [float(values_b[idx]) for idx in range(count)]
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for idx in range(count):
        dx = float(xs[idx] - mean_x)
        dy = float(ys[idx] - mean_y)
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
    if denom <= 1e-12:
        return 0.0
    return float(num / denom)


def _spearman(values_a, values_b):
    if not values_a or not values_b:
        return 0.0
    count = min(len(values_a), len(values_b))
    if count < 2:
        return 0.0
    return float(_pearson(_rankdata(values_a[:count]), _rankdata(values_b[:count])))


def _lagged_pair(values_a, values_b, lag):
    lag = int(lag)
    if lag > 0:
        if lag >= len(values_a) or lag >= len(values_b):
            return [], []
        return values_a[:-lag], values_b[lag:]
    if lag < 0:
        lag_abs = abs(lag)
        if lag_abs >= len(values_a) or lag_abs >= len(values_b):
            return [], []
        return values_a[lag_abs:], values_b[:-lag_abs]
    return list(values_a), list(values_b)


def _best_lagged_density_corr(values_a, values_b, max_abs_lag):
    best_lag = 0
    best_corr = -1e9
    for lag in range(-int(max_abs_lag), int(max_abs_lag) + 1):
        left, right = _lagged_pair(values_a, values_b, lag)
        corr = _pearson(left, right)
        if corr > best_corr:
            best_corr = float(corr)
            best_lag = int(lag)
    if best_corr < -1e8:
        best_corr = 0.0
    return int(best_lag), float(best_corr)


def _nearest_distance(value, sorted_values):
    if not sorted_values:
        return 0.0
    pos = bisect_left(sorted_values, int(value))
    best = None
    if pos < len(sorted_values):
        best = abs(int(sorted_values[pos]) - int(value))
    if pos > 0:
        cand = abs(int(sorted_values[pos - 1]) - int(value))
        if best is None or cand < best:
            best = cand
    return float(best or 0.0)


def _direction(value):
    value = int(value)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _signed_anchor_shifts(base_indices, final_indices):
    count = min(len(base_indices), len(final_indices))
    shifts = []
    for idx in range(count):
        shifts.append(int(final_indices[idx]) - int(base_indices[idx]))
    return shifts


def _allocation_alignment(active_indices, observer_indices, base_indices):
    active_sorted = sorted(int(idx) for idx in active_indices)
    observer_sorted = sorted(int(idx) for idx in observer_indices)
    if not active_sorted:
        return {
            "exact_overlap_ratio": 0.0,
            "tolerance_overlap_ratio_at_3": 0.0,
            "mean_nearest_keyframe_distance": 0.0,
            "anchor_shift_corr": 0.0,
            "anchor_shift_direction_agreement": 0.0,
        }

    overlap_count = 0
    tolerance_hits = 0
    nearest_distances = []
    observer_set = set(observer_sorted)
    for frame_idx in active_sorted:
        if frame_idx in observer_set:
            overlap_count += 1
        distance = _nearest_distance(frame_idx, observer_sorted)
        nearest_distances.append(float(distance))
        if distance <= 3.0:
            tolerance_hits += 1

    active_shifts = _signed_anchor_shifts(base_indices, active_sorted)
    observer_shifts = _signed_anchor_shifts(base_indices, observer_sorted)
    directional_hits = 0
    directional_total = 0
    for idx in range(min(len(active_shifts), len(observer_shifts))):
        left_dir = _direction(active_shifts[idx])
        right_dir = _direction(observer_shifts[idx])
        if left_dir == 0 and right_dir == 0:
            continue
        directional_total += 1
        if left_dir == right_dir:
            directional_hits += 1

    if directional_total <= 0:
        direction_agreement = 1.0
    else:
        direction_agreement = float(directional_hits) / float(directional_total)

    return {
        "exact_overlap_ratio": float(overlap_count) / float(len(active_sorted)),
        "tolerance_overlap_ratio_at_3": float(tolerance_hits) / float(len(active_sorted)),
        "mean_nearest_keyframe_distance": float(_mean(nearest_distances)),
        "anchor_shift_corr": float(_pearson(active_shifts, observer_shifts)),
        "anchor_shift_direction_agreement": float(direction_agreement),
    }


def _signal_alignment(active_bundle, observer_bundle):
    best_lag, lagged_corr = _best_lagged_density_corr(
        active_bundle["density"],
        observer_bundle["density"],
        3,
    )
    return {
        "density_pearson": float(_pearson(active_bundle["density"], observer_bundle["density"])),
        "density_spearman": float(_spearman(active_bundle["density"], observer_bundle["density"])),
        "action_pearson": float(_pearson(active_bundle["action"], observer_bundle["action"])),
        "action_spearman": float(_spearman(active_bundle["action"], observer_bundle["action"])),
        "best_lag": int(best_lag),
        "lagged_density_corr": float(lagged_corr),
    }


def _observer_summary(policy_name, signal_bundle, final_indices):
    prefix = _signal_prefix(policy_name)
    base_indices = list(signal_bundle.get("base_indices", []))
    shifts = _signed_anchor_shifts(base_indices, final_indices)
    relocated_shifts = [abs(int(shift)) for shift in shifts if int(shift) != 0]
    summary = {
        "policy_name": str(policy_name),
        "signal_prefix": str(prefix),
        "uniform_base_count": int(len(base_indices)),
        "final_keyframe_count": int(len(final_indices)),
        "observer_final_indices": list(int(idx) for idx in final_indices),
        "observer_relocated_count": int(len(relocated_shifts)),
        "observer_avg_abs_shift": float(_mean(relocated_shifts)),
        "observer_max_abs_shift": int(max(relocated_shifts) if relocated_shifts else 0),
    }
    if prefix:
        summary["{}_density_mean".format(prefix)] = float(_mean(signal_bundle["density"]))
        summary["{}_density_max".format(prefix)] = float(_max(signal_bundle["density"]))
        summary["{}_action_total".format(prefix)] = float(signal_bundle["action"][-1]) if signal_bundle["action"] else 0.0
    return summary


def _resolve_semantic_cache_dir_for_policy(segment_dir, keyframes_meta, policy_name):
    policy_name = normalize_policy_name(policy_name)
    active_policy_name = _policy_name(keyframes_meta)
    if policy_name and policy_name == active_policy_name:
        return _resolve_semantic_cache_dir(segment_dir, keyframes_meta)
    if policy_name:
        return segment_dir / ".segment_cache" / policy_name
    return segment_dir / ".segment_cache" / "segment_analyze"


def _allocator_rules(keyframes_meta):
    kf_gap = int(keyframes_meta.get("kf_gap", 1) or 1)
    rules = build_fixed_budget_rules(kf_gap)
    stored_rules = dict((keyframes_meta.get("policy_meta", {}) or {}).get("rules", {}) or {})
    for key in FIXED_BUDGET_RULE_KEYS:
        value = stored_rules.get(key)
        if isinstance(value, (int, float)):
            if key in ("relocate_radius", "min_gap", "snap_radius", "smooth_window"):
                rules[key] = int(value)
            else:
                rules[key] = float(value)
    return rules


def _signal_bundle_from_rows(rows, keyframes_meta, policy_name, meta):
    prefix = _signal_prefix(policy_name)
    used_rows = _used_rows(rows, keyframes_meta)
    return {
        "policy_name": str(policy_name),
        "prefix": str(prefix),
        "meta": dict(meta or {}),
        "base_indices": list(_resolve_keyframe_sets(keyframes_meta)[0]),
        "rows": list(used_rows),
        "density": _series(used_rows, "{}_density".format(prefix)),
        "action": _series(used_rows, "{}_action".format(prefix)),
        "displacement": _series(used_rows, "{}_displacement".format(prefix)),
        "velocity": _series(used_rows, "{}_velocity".format(prefix)),
        "velocity_smooth": _series(used_rows, "{}_velocity_smooth".format(prefix)),
        "acceleration": _series(used_rows, "{}_acceleration".format(prefix)),
        "acceleration_smooth": _series(used_rows, "{}_acceleration_smooth".format(prefix)),
    }


def _active_allocation_summary(keyframes_meta):
    base_indices, final_indices, _uniform_set, final_set = _resolve_keyframe_sets(keyframes_meta)
    shift_values = _signed_anchor_shifts(base_indices, final_indices)
    relocated_shifts = [abs(int(shift)) for shift in shift_values if int(shift) != 0]
    selected_order_map = {}
    for pos, frame_idx in enumerate(final_indices):
        selected_order_map[int(frame_idx)] = int(pos)
    return {
        "policy_name": str(_policy_name(keyframes_meta)),
        "base_indices": list(base_indices),
        "final_indices": list(final_indices),
        "final_set": set(final_set),
        "selected_order_map": selected_order_map,
        "shift_values": list(shift_values),
        "relocated_count": int(len(relocated_shifts)),
        "avg_abs_shift": float(_mean(relocated_shifts)),
        "max_abs_shift": int(max(relocated_shifts) if relocated_shifts else 0),
    }


def _observer_allocation(signal_bundle, keyframes_meta, rules):
    base_indices = list(signal_bundle.get("base_indices", []))
    if not base_indices:
        return {
            "policy_name": str(signal_bundle.get("policy_name", "")),
            "final_indices": [],
            "final_set": set(),
            "selected_order_map": {},
            "shift_values": [],
            "relocated_count": 0,
            "avg_abs_shift": 0.0,
            "max_abs_shift": 0,
        }

    if len(base_indices) <= 2:
        final_indices = list(base_indices)
    else:
        allocation = allocate_fixed_budget(
            base_indices=base_indices,
            used_last_idx=max(0, _used_frame_count(keyframes_meta, len(signal_bundle.get("rows", []))) - 1),
            density_values=signal_bundle["density"],
            action_values=signal_bundle["action"],
            rules=rules,
        )
        final_indices = list(allocation["final_indices"])

    shift_values = _signed_anchor_shifts(base_indices, final_indices)
    relocated_shifts = [abs(int(shift)) for shift in shift_values if int(shift) != 0]
    selected_order_map = {}
    for pos, frame_idx in enumerate(final_indices):
        selected_order_map[int(frame_idx)] = int(pos)
    return {
        "policy_name": str(signal_bundle.get("policy_name", "")),
        "final_indices": list(final_indices),
        "final_set": set(int(idx) for idx in final_indices),
        "selected_order_map": selected_order_map,
        "shift_values": list(shift_values),
        "relocated_count": int(len(relocated_shifts)),
        "avg_abs_shift": float(_mean(relocated_shifts)),
        "max_abs_shift": int(max(relocated_shifts) if relocated_shifts else 0),
    }


def _build_comparison_payload(active_policy, active_allocation, observer_entries):
    payload = {
        "active_policy": str(active_policy),
        "active_keyframes": list(active_allocation["final_indices"]),
        "uniform_indices": list(active_allocation["base_indices"]),
        "observers": {},
    }
    for policy_name, entry in observer_entries.items():
        payload["observers"][policy_name] = {
            "policy_name": str(policy_name),
            "signal_prefix": str(_signal_prefix(policy_name)),
            "keyframes": list(entry["allocation"]["final_indices"]),
        }
    return payload


def _keyframe_maps(keyframes_meta):
    uniform_indices, final_indices, uniform_set, final_set = _resolve_keyframe_sets(keyframes_meta)
    uniform_anchor_idx_map = {}
    for pos, frame_idx in enumerate(uniform_indices):
        uniform_anchor_idx_map[int(frame_idx)] = int(pos)

    selected_order_map = {}
    for pos, frame_idx in enumerate(final_indices):
        selected_order_map[int(frame_idx)] = int(pos)

    relocated_set = set()
    for item in keyframes_meta.get("keyframes", []):
        if bool(item.get("is_relocated", False)):
            relocated_set.add(int(item.get("frame_idx", 0)))

    return {
        "uniform_indices": list(uniform_indices),
        "final_indices": list(final_indices),
        "uniform_set": uniform_set,
        "final_set": final_set,
        "uniform_anchor_idx_map": uniform_anchor_idx_map,
        "selected_order_map": selected_order_map,
        "relocated_set": relocated_set,
    }


def _official_signal_rows(data, segment_dir, keyframes_meta, smooth_window):
    policy_name = _policy_name(keyframes_meta)
    needed_policies = _needed_signal_policies(policy_name)
    semantic_rows = None
    motion_rows = None
    signal_meta = {}

    if "semantic" in needed_policies:
        cache_dir = _resolve_semantic_cache_dir_for_policy(segment_dir, keyframes_meta, "semantic")
        cache_dir.mkdir(parents=True, exist_ok=True)
        semantic_rows, signal_meta["semantic"] = compute_semantic_rows(
            data["frame_paths"],
            cache_dir,
            smooth_window=int(smooth_window),
            timestamps=data["timestamps"],
        )

    if "motion" in needed_policies:
        motion_rows, signal_meta["motion"] = compute_motion_rows(
            data["frame_paths"],
            smooth_window=int(smooth_window),
            timestamps=data["timestamps"],
        )

    rows, signal_meta["frame_signals"] = compute_frame_signal_rows(
        data["frame_paths"],
        data["timestamps"],
        semantic_rows=semantic_rows,
        motion_rows=motion_rows,
    )
    return rows, signal_meta


def _signal_summary_from_rows(rows, prefix):
    if not prefix:
        return {}

    def _values(key):
        return [float(row.get(key, 0.0) or 0.0) for row in rows]

    displacement = _values("{}_displacement".format(prefix))
    velocity = _values("{}_velocity".format(prefix))
    acceleration = _values("{}_acceleration".format(prefix))
    density = _values("{}_density".format(prefix))
    action = _values("{}_action".format(prefix))

    return {
        "{}_displacement_mean".format(prefix): float(_mean(displacement)),
        "{}_displacement_max".format(prefix): float(_max(displacement)),
        "{}_velocity_mean".format(prefix): float(_mean(velocity)),
        "{}_velocity_max".format(prefix): float(_max(velocity)),
        "{}_acceleration_mean".format(prefix): float(_mean(acceleration)),
        "{}_acceleration_max".format(prefix): float(_max(acceleration)),
        "{}_density_mean".format(prefix): float(_mean(density)),
        "{}_density_max".format(prefix): float(_max(density)),
        "{}_action_total".format(prefix): float(action[-1]) if action else 0.0,
    }


def _build_official_comparison(rows, keyframes_meta, signal_meta):
    policy_name = _policy_name(keyframes_meta)
    observer_policies = _observer_policies(policy_name)
    active_allocation = _active_allocation_summary(keyframes_meta)
    observer_entries = {}
    rules = _allocator_rules(keyframes_meta)

    if policy_name in ("semantic", "motion"):
        active_prefix = _signal_prefix(policy_name)
        active_meta = signal_meta.get(active_prefix, {})
        active_bundle = _signal_bundle_from_rows(rows, keyframes_meta, policy_name, active_meta)
    else:
        active_bundle = None

    for observer_policy in observer_policies:
        observer_prefix = _signal_prefix(observer_policy)
        observer_bundle = _signal_bundle_from_rows(
            rows,
            keyframes_meta,
            observer_policy,
            signal_meta.get(observer_prefix, {}),
        )
        observer_allocation = _observer_allocation(observer_bundle, keyframes_meta, rules)
        observer_entries[str(observer_policy)] = {
            "bundle": observer_bundle,
            "allocation": observer_allocation,
            "summary": _observer_summary(observer_policy, observer_bundle, observer_allocation["final_indices"]),
        }

    if policy_name == "uniform":
        left_name = "semantic"
        right_name = "motion"
        left_entry = observer_entries.get(left_name)
        right_entry = observer_entries.get(right_name)
        comparison_block = {
            "observers": {},
        }
        for observer_policy in observer_policies:
            comparison_block["observers"][observer_policy] = {
                "observer_summary": dict(observer_entries[observer_policy]["summary"]),
            }
        if left_entry is not None and right_entry is not None:
            comparison_block["observer_pair_alignment"] = {
                "reference_policy": str(left_name),
                "compare_policy": str(right_name),
                "signal_alignment": _signal_alignment(left_entry["bundle"], right_entry["bundle"]),
                "allocation_alignment": _allocation_alignment(
                    left_entry["allocation"]["final_indices"],
                    right_entry["allocation"]["final_indices"],
                    active_allocation["base_indices"],
                ),
            }
    else:
        observer_policy = observer_policies[0] if observer_policies else ""
        observer_entry = observer_entries.get(observer_policy)
        comparison_block = {
            "observer_policy": str(observer_policy),
            "signal_alignment": _signal_alignment(active_bundle, observer_entry["bundle"]) if observer_entry else {},
            "allocation_alignment": _allocation_alignment(
                active_allocation["final_indices"],
                observer_entry["allocation"]["final_indices"] if observer_entry else [],
                active_allocation["base_indices"],
            ),
            "observer_summary": dict(observer_entry["summary"]) if observer_entry else {},
        }

    return {
        "active_allocation": active_allocation,
        "observer_entries": observer_entries,
        "summary_block": comparison_block,
        "plot_payload": _build_comparison_payload(policy_name, active_allocation, observer_entries),
    }


def _projection_maps(deploy_schedule):
    if not isinstance(deploy_schedule, dict):
        return {
            "raw_boundary_order": {},
            "deploy_boundary_order": {},
            "raw_segments": {},
            "deploy_segments": {},
        }

    raw_boundary_order = {}
    for pos, frame_idx in enumerate(list(deploy_schedule.get("raw_keyframe_indices") or [])):
        raw_boundary_order[int(frame_idx)] = int(pos)

    deploy_boundary_order = {}
    for pos, frame_idx in enumerate(list(deploy_schedule.get("deploy_keyframe_indices") or [])):
        deploy_boundary_order[int(frame_idx)] = int(pos)

    raw_segments = {}
    deploy_segments = {}
    for item in list(deploy_schedule.get("segments") or []):
        segment_id = int(item.get("segment_id", 0) or 0)
        raw_end_idx = int(item.get("raw_end_idx", 0) or 0)
        deploy_end_idx = int(item.get("deploy_end_idx", 0) or 0)
        raw_segments[raw_end_idx] = {
            "projection_segment_id": segment_id,
            "projection_raw_gap": int(item.get("raw_gap", 0) or 0),
            "projection_deploy_gap": int(item.get("deploy_gap", 0) or 0),
            "projection_gap_error": int(item.get("gap_error", 0) or 0),
            "projection_boundary_shift": int(item.get("boundary_shift", 0) or 0),
        }
        deploy_segments[deploy_end_idx] = {
            "projection_segment_id": segment_id,
            "projection_raw_gap": int(item.get("raw_gap", 0) or 0),
            "projection_deploy_gap": int(item.get("deploy_gap", 0) or 0),
            "projection_gap_error": int(item.get("gap_error", 0) or 0),
            "projection_boundary_shift": int(item.get("boundary_shift", 0) or 0),
        }

    return {
        "raw_boundary_order": raw_boundary_order,
        "deploy_boundary_order": deploy_boundary_order,
        "raw_segments": raw_segments,
        "deploy_segments": deploy_segments,
    }


def _official_csv_rows(rows, keyframes_meta, comparison, deploy_schedule):
    policy_name = _policy_name(keyframes_meta)
    prefix = _signal_prefix(policy_name)
    maps = _keyframe_maps(keyframes_meta)
    observer_entries = dict(comparison.get("observer_entries", {}) or {})
    projection = _projection_maps(deploy_schedule)
    csv_rows = []

    for row in rows:
        frame_idx = int(row.get("frame_idx", 0))
        csv_row = {
            "frame_idx": int(frame_idx),
            "is_uniform_anchor": bool(frame_idx in maps["uniform_set"]),
            "is_selected_keyframe": bool(frame_idx in maps["final_set"]),
            "is_relocated_keyframe": bool(frame_idx in maps["relocated_set"]),
            "is_active_keyframe": bool(frame_idx in maps["final_set"]),
            "selected_order": maps["selected_order_map"].get(frame_idx),
            "active_selected_order": maps["selected_order_map"].get(frame_idx),
            "uniform_anchor_idx": maps["uniform_anchor_idx_map"].get(frame_idx),
            "is_raw_boundary": bool(frame_idx in projection["raw_boundary_order"]),
            "raw_boundary_order": projection["raw_boundary_order"].get(frame_idx),
            "is_deploy_boundary": bool(frame_idx in projection["deploy_boundary_order"]),
            "deploy_boundary_order": projection["deploy_boundary_order"].get(frame_idx),
        }
        raw_projection = projection["raw_segments"].get(frame_idx, {})
        deploy_projection = projection["deploy_segments"].get(frame_idx, {})
        projection_row = raw_projection or deploy_projection
        for key in (
            "projection_segment_id",
            "projection_raw_gap",
            "projection_deploy_gap",
            "projection_gap_error",
            "projection_boundary_shift",
        ):
            csv_row[key] = projection_row.get(key)
        if prefix:
            csv_row["{}_displacement".format(prefix)] = float(row.get("{}_displacement".format(prefix), 0.0) or 0.0)
            csv_row["{}_velocity".format(prefix)] = float(row.get("{}_velocity".format(prefix), 0.0) or 0.0)
            csv_row["{}_velocity_smooth".format(prefix)] = float(row.get("{}_velocity_smooth".format(prefix), 0.0) or 0.0)
            csv_row["{}_acceleration".format(prefix)] = float(row.get("{}_acceleration".format(prefix), 0.0) or 0.0)
            csv_row["{}_acceleration_smooth".format(prefix)] = float(row.get("{}_acceleration_smooth".format(prefix), 0.0) or 0.0)
            csv_row["{}_density".format(prefix)] = float(row.get("{}_density".format(prefix), 0.0) or 0.0)
            csv_row["{}_action".format(prefix)] = float(row.get("{}_action".format(prefix), 0.0) or 0.0)
            csv_row["active_density"] = float(row.get("{}_density".format(prefix), 0.0) or 0.0)
            csv_row["active_action"] = float(row.get("{}_action".format(prefix), 0.0) or 0.0)
        for observer_policy, observer_entry in observer_entries.items():
            observer_prefix = _signal_prefix(observer_policy)
            observer_key = "observer_{}".format(observer_policy)
            csv_row["is_{}_keyframe".format(observer_key)] = bool(
                frame_idx in observer_entry["allocation"]["final_set"]
            )
            csv_row["{}_selected_order".format(observer_key)] = observer_entry["allocation"]["selected_order_map"].get(frame_idx)
            csv_row["{}_density".format(observer_policy)] = float(
                row.get("{}_density".format(observer_prefix), 0.0) or 0.0
            )
            csv_row["{}_action".format(observer_policy)] = float(
                row.get("{}_action".format(observer_prefix), 0.0) or 0.0
            )
        if policy_name != "uniform" and observer_entries:
            observer_policy = sorted(observer_entries.keys())[0]
            observer_prefix = _signal_prefix(observer_policy)
            csv_row["observer_policy"] = str(observer_policy)
            csv_row["observer_density"] = float(row.get("{}_density".format(observer_prefix), 0.0) or 0.0)
            csv_row["observer_action"] = float(row.get("{}_action".format(observer_prefix), 0.0) or 0.0)
            csv_row["is_observer_keyframe"] = bool(
                frame_idx in observer_entries[observer_policy]["allocation"]["final_set"]
            )
        csv_rows.append(csv_row)
    return csv_rows


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


def _projection_summary(deploy_schedule):
    if not isinstance(deploy_schedule, dict):
        return {}
    projection_stats = dict(deploy_schedule.get("projection_stats", {}) or {})
    return {
        "mean_abs_boundary_shift": float(projection_stats.get("mean_abs_boundary_shift", 0.0) or 0.0),
        "max_abs_boundary_shift": int(projection_stats.get("max_abs_boundary_shift", 0) or 0),
        "mean_abs_gap_error": float(projection_stats.get("mean_abs_gap_error", 0.0) or 0.0),
        "max_abs_gap_error": int(projection_stats.get("max_abs_gap_error", 0) or 0),
    }


def _official_analysis_summary(exp_dir, data, rows, keyframes_meta, comparison):
    exp_info = _experiment_info(exp_dir, data["exp_meta"], data["step_meta"])
    policy_name = _policy_name(keyframes_meta)
    prefix = _signal_prefix(policy_name)
    keyframe_summary = dict(keyframes_meta.get("summary", {}) or {})
    projection = _projection_summary(data.get("deploy_schedule"))
    allocation = {
        "policy_name": str(policy_name),
        "uniform_base_count": int(len(keyframes_meta.get("uniform_base_indices", []) or [])),
        "final_keyframe_count": int(len(keyframes_meta.get("keyframe_indices", []) or [])),
        "fixed_budget": bool(keyframe_summary.get("fixed_budget", policy_name != "uniform")),
        "relocated_count": int(keyframe_summary.get("relocated_count", 0) or 0),
        "avg_abs_shift": float(keyframe_summary.get("avg_abs_shift", 0.0) or 0.0),
        "max_abs_shift": int(keyframe_summary.get("max_abs_shift", 0) or 0),
        "keyframe_bytes_sum": int(keyframes_meta.get("keyframe_bytes_sum", 0) or 0),
        "extra_kf_ratio": float(keyframe_summary.get("extra_kf_ratio", 0.0) or 0.0),
    }
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
        "allocation": allocation,
    }

    if projection:
        summary["projection"] = projection

    comparison_block = dict(comparison.get("summary_block", {}) or {})
    if policy_name == "uniform":
        if comparison_block:
            summary["comparison"] = comparison_block
        return summary

    observer_policy = str(comparison_block.get("observer_policy", "") or "")
    if observer_policy:
        summary["alignment"] = dict(comparison_block.get("signal_alignment", {}) or {})
        summary["alignment"].update(dict(comparison_block.get("allocation_alignment", {}) or {}))
        summary["alignment"]["observer_policy"] = observer_policy

    signal_stats = _signal_summary_from_rows(rows, prefix)
    if signal_stats:
        summary["signals"] = signal_stats
    return summary


def _official_log(signal_meta, keyframes_meta, csv_rows, analysis_dir, comparison):
    policy_name = _policy_name(keyframes_meta)
    maps = _keyframe_maps(keyframes_meta)
    log_info("analysis_dir: {}".format(analysis_dir))
    log_info("segment_summary.json written")
    log_info("segment_timeseries.csv rows: {}".format(len(csv_rows)))
    log_info("uniform base keyframes: {}".format(len(maps["uniform_indices"])))
    log_info("final keyframes: {}".format(len(maps["final_indices"])))
    log_info("policy analyzed: {}".format(policy_name))
    observer_entries = dict(comparison.get("observer_entries", {}) or {})
    if observer_entries:
        log_info("observer policies: {}".format(", ".join(sorted(observer_entries.keys()))))
    semantic_meta = dict(signal_meta.get("semantic", {}) or {})
    if semantic_meta:
        log_info("semantic cache hit: {}".format(bool(semantic_meta.get("cache_hit", False))))
        log_info("semantic cache path: {}".format(semantic_meta.get("cache_path", "")))


def _risk_kf_gap(data, args):
    keyframes_meta = dict(data.get("keyframes_meta", {}) or {})
    step_meta = dict(data.get("step_meta", {}) or {})
    step_params = dict(step_meta.get("params", {}) or {})
    candidates = [
        keyframes_meta.get("kf_gap"),
        step_params.get("kf_gap"),
        getattr(args, "kf_gap", 0),
    ]
    for value in candidates:
        try:
            gap = int(value)
        except Exception:
            gap = 0
        if gap > 0:
            return int(gap)
    return 0


def _risk_log(risk_bundle, analysis_dir):
    payload = risk_bundle_to_dict(risk_bundle)
    hardest_window = dict(payload.get("hardest_window") or {})
    if hardest_window:
        hardest_range = "[{}:{}]".format(
            int(hardest_window.get("expanded_start_frame", 0) or 0),
            int(hardest_window.get("expanded_end_frame", 0) or 0),
        )
    else:
        hardest_range = "none"
    log_info("risk bundle version: {}".format(payload.get("metadata", {}).get("version", "")))
    log_info("risk frame rows: {}".format(len(payload.get("frame_rows", []) or [])))
    log_info("risk expanded windows: {}".format(len(payload.get("expanded_windows", []) or [])))
    log_info("risk hardest window range: {}".format(hardest_range))
    log_info("risk_bundle.json: {}".format(analysis_dir / "risk_bundle.json"))
    log_info("risk_windows.csv: {}".format(analysis_dir / "risk_windows.csv"))


def _run_risk_analysis(rows, data, analysis_dir, args):
    maps = _keyframe_maps(data["keyframes_meta"])
    risk_bundle = compute_risk_bundle(
        rows=rows,
        uniform_keyframe_indices=maps["uniform_indices"],
        final_keyframe_indices=maps["final_indices"],
        smooth_window=int(args.smooth_window),
        kf_gap=_risk_kf_gap(data, args),
    )
    risk_summary = build_risk_summary(risk_bundle)

    risk_bundle_path = analysis_dir / "risk_bundle.json"
    risk_summary_path = analysis_dir / "risk_summary.json"
    risk_timeseries_path = analysis_dir / "risk_timeseries.csv"
    risk_windows_path = analysis_dir / "risk_windows.csv"
    risk_curve_path = analysis_dir / "risk_curve.png"
    risk_anchor_path = analysis_dir / "risk_anchor_overview.png"

    write_json_atomic(str(risk_bundle_path), risk_bundle_to_dict(risk_bundle), indent=2)
    write_json_atomic(str(risk_summary_path), risk_summary, indent=2)
    _write_csv(risk_timeseries_path, risk_frame_rows_to_dicts(risk_bundle))
    _write_csv(risk_windows_path, risk_windows_to_dicts(risk_bundle))
    save_risk_curve(risk_bundle, risk_curve_path)
    save_risk_anchor_overview(risk_bundle, risk_anchor_path)
    _risk_log(risk_bundle, analysis_dir)


def _run_official_analysis(exp_dir, segment_dir, analysis_dir, data, args):
    rows, signal_meta = _official_signal_rows(data, segment_dir, data["keyframes_meta"], args.smooth_window)
    maps = _keyframe_maps(data["keyframes_meta"])
    _mark_uniform_keyframes(rows, maps["uniform_indices"])
    comparison = _build_official_comparison(rows, data["keyframes_meta"], signal_meta)
    csv_rows = _official_csv_rows(rows, data["keyframes_meta"], comparison, data.get("deploy_schedule"))
    analysis_summary = _official_analysis_summary(exp_dir, data, rows, data["keyframes_meta"], comparison)
    final_keyframe_set = set(maps["final_indices"])

    csv_path = analysis_dir / "segment_timeseries.csv"
    summary_path = analysis_dir / "segment_summary.json"
    comparison_overview_path = analysis_dir / "comparison_overview.png"
    allocation_overview_path = analysis_dir / "allocation_overview.png"
    kinematics_overview_path = analysis_dir / "kinematics_overview.png"
    projection_overview_path = analysis_dir / "projection_overview.png"

    _write_csv(csv_path, csv_rows)
    write_json_atomic(str(summary_path), analysis_summary, indent=2)
    save_comparison_overview(
        rows,
        comparison_overview_path,
        sorted(final_keyframe_set),
        policy_name=_policy_name(data["keyframes_meta"]),
        uniform_indices=maps["uniform_indices"],
        keyframe_items=data["keyframes_meta"].get("keyframes", []),
        comparison_payload=comparison.get("plot_payload", {}),
    )
    save_allocation_overview(
        rows,
        allocation_overview_path,
        sorted(final_keyframe_set),
        policy_name=_policy_name(data["keyframes_meta"]),
        uniform_indices=maps["uniform_indices"],
        keyframe_items=data["keyframes_meta"].get("keyframes", []),
    )
    save_kinematics_overview(
        rows,
        kinematics_overview_path,
        sorted(final_keyframe_set),
        keyframe_items=data["keyframes_meta"].get("keyframes", []),
        policy_name=_policy_name(data["keyframes_meta"]),
        uniform_indices=maps["uniform_indices"],
    )
    deploy_schedule = data.get("deploy_schedule") or {}
    save_projection_overview(
        projection_overview_path,
        policy_name=_policy_name(data["keyframes_meta"]),
        raw_keyframe_indices=list(deploy_schedule.get("raw_keyframe_indices") or maps["final_indices"]),
        deploy_keyframe_indices=list(deploy_schedule.get("deploy_keyframe_indices") or maps["final_indices"]),
        segments=list(deploy_schedule.get("segments") or []),
        uniform_indices=maps["uniform_indices"],
    )

    _run_risk_analysis(rows, data, analysis_dir, args)
    _official_log(signal_meta, data["keyframes_meta"], csv_rows, analysis_dir, comparison)


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
    policy_name = _policy_name(data["keyframes_meta"])
    if policy_name not in OFFICIAL_POLICY_NAMES:
        log_warn("segment analyze skipped for unsupported historical policy: {}".format(policy_name))
        return

    _run_official_analysis(exp_dir, segment_dir, analysis_dir, data, args)
    log_prog("segment analyze done: {}".format(analysis_dir))


if __name__ == "__main__":
    run_segment_analyze()
