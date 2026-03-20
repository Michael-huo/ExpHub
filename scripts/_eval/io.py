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
