#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
from pathlib import Path

import numpy as np

from _common import log_warn, write_json_atomic


STAT_KEYS = ["mean", "median", "std", "min", "max"]


def empty_stats():
    return {key: None for key in STAT_KEYS}


def float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def metric_stats(values):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return empty_stats()
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def append_warning(metrics_obj, msg):
    msg_text = str(msg)
    warnings_list = metrics_obj.setdefault("warnings", [])
    if msg_text not in warnings_list:
        warnings_list.append(msg_text)
    log_warn(msg_text)


def write_json(path, obj, indent=2):
    write_json_atomic(path, obj, indent=indent)


def write_text(path, text):
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(str(text), encoding="utf-8")
    os.replace(str(tmp_path), str(out_path))


def write_csv(path, fieldnames, rows):
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    os.replace(str(tmp_path), str(out_path))


def read_json(path):
    json_path = Path(path).resolve()
    if not json_path.is_file():
        return None
    try:
        import json

        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def read_timestamps(path):
    ts_path = Path(path).resolve()
    if not ts_path.is_file():
        return []
    out = []
    for line in ts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = str(line).strip()
        if not text:
            continue
        try:
            out.append(float(text))
        except Exception:
            out.append(text)
    return out


def fmt_value(value, unit=""):
    if value is None:
        return "n/a"
    text = "{:.6f}".format(float(value))
    unit_text = str(unit or "").strip()
    if unit_text:
        return "{} {}".format(text, unit_text)
    return text


def resolve_image_eval_inputs(exp_dir):
    exp_root = Path(exp_dir).resolve()
    return {
        "segment_frames_dir": exp_root / "segment" / "frames",
        "merge_frames_dir": exp_root / "merge" / "frames",
        "merge_timestamps_path": exp_root / "merge" / "timestamps.txt",
        "runs_plan_path": exp_root / "infer" / "runs_plan.json",
        "merge_meta_path": exp_root / "merge" / "merge_meta.json",
    }


def resolve_slam_eval_inputs(exp_dir):
    exp_root = Path(exp_dir).resolve()
    return {
        "segment_frames_dir": exp_root / "segment" / "frames",
        "segment_timestamps_path": exp_root / "segment" / "timestamps.txt",
        "segment_calib_path": exp_root / "segment" / "calib.txt",
        "deploy_schedule_path": exp_root / "segment" / "deploy_schedule.json",
        "merge_frames_dir": exp_root / "merge" / "frames",
        "merge_timestamps_path": exp_root / "merge" / "timestamps.txt",
        "merge_calib_path": exp_root / "merge" / "calib.txt",
        "runs_plan_path": exp_root / "infer" / "runs_plan.json",
        "merge_meta_path": exp_root / "merge" / "merge_meta.json",
        "slam_ori_tum_path": exp_root / "slam" / "ori" / "traj_est.tum",
        "slam_ori_npz_path": exp_root / "slam" / "ori" / "traj_est.npz",
        "slam_ori_run_meta_path": exp_root / "slam" / "ori" / "run_meta.json",
        "slam_gt_tum_path": exp_root / "slam" / "gt" / "traj_est.tum",
        "slam_gt_npz_path": exp_root / "slam" / "gt" / "traj_est.npz",
        "slam_gt_run_meta_path": exp_root / "slam" / "gt" / "run_meta.json",
    }


def _final_keyframe_warning(metrics_obj, warning_prefix, detail):
    if metrics_obj is None:
        return
    prefix = str(warning_prefix or "").strip()
    if prefix:
        append_warning(metrics_obj, "{}: {}".format(prefix, detail))
        return
    append_warning(metrics_obj, detail)


def _candidate_exp_roots(path_candidates):
    seen = set()
    out = []
    for raw_path in list(path_candidates or []):
        if raw_path is None:
            continue
        text = str(raw_path).strip()
        if not text:
            continue
        try:
            path_obj = Path(text).resolve()
        except Exception:
            continue
        for candidate in [path_obj] + list(path_obj.parents):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def resolve_eval_exp_root(path_candidates):
    for candidate in _candidate_exp_roots(path_candidates):
        meta_path = candidate / "segment" / "keyframes" / "keyframes_meta.json"
        if meta_path.is_file():
            return candidate
    return None


def _segment_timestamp_map(exp_root):
    if exp_root is None:
        return {}
    timestamps = read_timestamps(Path(exp_root).resolve() / "segment" / "timestamps.txt")
    out = {}
    for idx, value in enumerate(timestamps):
        try:
            out[int(idx)] = float(value)
        except Exception:
            continue
    return out


def load_final_keyframe_context(path_candidates, metrics_obj=None, warning_prefix=""):
    exp_root = resolve_eval_exp_root(path_candidates)
    empty_context = {
        "exp_root": None,
        "meta_path": "",
        "frame_indices": [],
        "timestamps_by_frame": {},
    }
    if exp_root is None:
        _final_keyframe_warning(
            metrics_obj,
            warning_prefix,
            "final keyframes unavailable: missing segment/keyframes/keyframes_meta.json",
        )
        return empty_context

    meta_path = Path(exp_root).resolve() / "segment" / "keyframes" / "keyframes_meta.json"
    meta_obj = read_json(meta_path)
    if meta_obj is None:
        _final_keyframe_warning(
            metrics_obj,
            warning_prefix,
            "final keyframes unavailable: failed to read {}".format(meta_path),
        )
        return empty_context

    frame_indices = []
    seen = set()
    for value in list(meta_obj.get("keyframe_indices") or []):
        try:
            frame_idx = int(value)
        except Exception:
            continue
        if frame_idx < 0 or frame_idx in seen:
            continue
        seen.add(frame_idx)
        frame_indices.append(frame_idx)
    frame_indices.sort()
    if not frame_indices:
        _final_keyframe_warning(
            metrics_obj,
            warning_prefix,
            "final keyframes unavailable: keyframe_indices missing or empty in {}".format(meta_path),
        )
        return empty_context

    timestamps_by_frame = {}
    for item in list(meta_obj.get("keyframes") or []):
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_idx"))
            ts_sec = float(item.get("ts_sec"))
        except Exception:
            continue
        timestamps_by_frame[int(frame_idx)] = float(ts_sec)

    if len(timestamps_by_frame) < len(frame_indices):
        fallback_map = _segment_timestamp_map(exp_root)
        for frame_idx in frame_indices:
            if frame_idx in timestamps_by_frame:
                continue
            if frame_idx in fallback_map:
                timestamps_by_frame[frame_idx] = float(fallback_map[frame_idx])

    return {
        "exp_root": Path(exp_root).resolve(),
        "meta_path": str(meta_path),
        "frame_indices": list(frame_indices),
        "timestamps_by_frame": dict(timestamps_by_frame),
    }
