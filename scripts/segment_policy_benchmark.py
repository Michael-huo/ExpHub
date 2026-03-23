#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark existing segment policies from experiment directories."""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
else:
    _REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts._common import ensure_dir, log_info, log_warn, write_json_atomic
from scripts._segment.policies.naming import OFFICIAL_POLICY_NAMES, normalize_policy_name


_EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<tag>.+?)_(?P<w>\d+)x(?P<h>\d+)_t(?P<start>[^_]+)_dur(?P<dur>[^_]+)_fps(?P<fps>[^_]+)_gap(?P<gap>[^_]+)$"
)
_SEGMENT_KEYFRAME_META_REL = Path("segment") / "keyframes" / "keyframes_meta.json"
_POLICY_ORDER = {
    "uniform60": 0,
    "uniform24": 1,
    "semantic": 2,
    "motion": 3,
    "risk": 4,
}
_POLICY_COLORS = {
    "uniform60": "#4e79a7",
    "uniform24": "#a0cbe8",
    "semantic": "#59a14f",
    "motion": "#f28e2b",
    "risk": "#e15759",
}
_CSV_FIELD_ORDER = [
    "exp_dir",
    "dataset",
    "sequence",
    "tag",
    "mode",
    "segment_policy",
    "policy_label",
    "fps",
    "duration",
    "kf_gap",
    "width",
    "height",
    "uniform_base_count",
    "final_keyframe_count",
    "reduction_vs_teacher",
    "risk_window_count",
    "all_risky_windows_protected",
    "worst_window_gap",
    "hardest_window_frame_range",
    "hardest_window_peak_score",
    "compression_ratio_frames",
    "compression_ratio_bytes",
    "compression_reduction_bytes",
    "ori_frame_count",
    "keyframes_frame_count",
    "ori_bytes",
    "keyframes_bytes",
    "prompt_bytes",
    "infer_segments",
    "infer_frames",
    "infer_init_sec",
    "infer_run_sec",
    "infer_avg_frame_sec",
    "ape_rmse",
    "rpe_trans_rmse",
    "matched_count",
    "image_psnr_mean",
    "image_ms_ssim_mean",
    "image_lpips_mean",
    "image_frame_count",
    "slam_inlier_ratio_mean",
    "slam_pose_success_rate",
    "slam_reference_source",
    "traj_eval_status",
    "image_eval_status",
    "slam_eval_status",
    "sources_present",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark existing experiment directories across segment policies without modifying the main pipeline."
    )
    parser.add_argument(
        "--exp_dir",
        action="append",
        default=[],
        help="existing experiment directory; can be specified multiple times",
    )
    parser.add_argument(
        "--experiments_root",
        default="",
        help="scan experiments under this root and collect directories with segment/keyframes/keyframes_meta.json",
    )
    parser.add_argument("--dataset", default="", help="optional dataset exact-match filter when scanning")
    parser.add_argument("--sequence", default="", help="optional sequence exact-match filter when scanning")
    parser.add_argument("--tag", action="append", default=[], help="optional tag exact-match filter; repeatable")
    parser.add_argument(
        "--tag_contains",
        action="append",
        default=[],
        help="optional tag substring filter; repeatable",
    )
    parser.add_argument(
        "--policy_label",
        action="append",
        default=[],
        help="optional normalized policy label filter after extraction; repeatable",
    )
    parser.add_argument(
        "--out_dir",
        default="policy_benchmark_out",
        help="output directory for csv/json/markdown/png",
    )
    return parser


def _read_json(path):
    json_path = Path(path).resolve()
    if not json_path.is_file():
        return None
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _coerce_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    return None


def _get_nested(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _pick_first(obj, paths, caster=None):
    for path in list(paths or []):
        value = _get_nested(obj, path)
        if caster is not None:
            value = caster(value)
        if value is not None:
            return value
    return None


def _mean(values):
    items = [float(v) for v in list(values or []) if v is not None]
    if not items:
        return None
    return float(sum(items) / float(len(items)))


def _format_num(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        fmt = "{:." + str(int(digits)) + "f}"
        return fmt.format(float(value))
    return str(value)


def _format_ratio(value):
    if value is None:
        return ""
    return "{:.2f}%".format(float(value) * 100.0)


def _format_bool_text(value):
    if value is None:
        return ""
    return "yes" if bool(value) else "no"


def _safe_slug(text):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return slug.strip("_") or "unknown"


def _parse_experiment_name(exp_name):
    match = _EXPERIMENT_NAME_RE.match(str(exp_name or "").strip())
    if not match:
        return {}
    groups = match.groupdict()
    return {
        "tag": str(groups.get("tag", "") or ""),
        "width": _coerce_int(groups.get("w")),
        "height": _coerce_int(groups.get("h")),
        "start_sec": groups.get("start"),
        "duration": groups.get("dur"),
        "fps": _coerce_float(groups.get("fps")),
        "kf_gap": _coerce_int(groups.get("gap")),
    }


def _infer_exp_identity(exp_dir, exp_meta, segment_step_meta, keyframes_meta):
    exp_name = str(Path(exp_dir).resolve().name)
    parsed = _parse_experiment_name(exp_name)
    exp_params = dict((exp_meta or {}).get("params", {}) or {})
    step_params = dict((segment_step_meta or {}).get("params", {}) or {})

    dataset = str((exp_meta or {}).get("dataset", "") or "")
    sequence = str((exp_meta or {}).get("sequence", "") or "")
    tag = str((exp_meta or {}).get("tag", "") or "") or str(parsed.get("tag", "") or "")

    if (not dataset or not sequence) and len(Path(exp_dir).resolve().parts) >= 3:
        sequence = sequence or str(Path(exp_dir).resolve().parent.name)
        dataset = dataset or str(Path(exp_dir).resolve().parent.parent.name)

    width = _pick_first(
        {"exp": {"params": exp_params}, "step": {"params": step_params}, "parsed": parsed},
        [
            ["step", "params", "w"],
            ["exp", "params", "w"],
            ["parsed", "width"],
        ],
        caster=_coerce_int,
    )
    height = _pick_first(
        {"exp": {"params": exp_params}, "step": {"params": step_params}, "parsed": parsed},
        [
            ["step", "params", "h"],
            ["exp", "params", "h"],
            ["parsed", "height"],
        ],
        caster=_coerce_int,
    )
    fps = _pick_first(
        {"exp": {"params": exp_params}, "step": {"params": step_params}, "parsed": parsed},
        [
            ["step", "params", "fps"],
            ["exp", "params", "fps"],
            ["parsed", "fps"],
        ],
        caster=_coerce_float,
    )
    duration = _pick_first(
        {"exp": {"params": exp_params}, "step": {"params": step_params}, "parsed": parsed},
        [
            ["step", "params", "dur"],
            ["exp", "params", "dur"],
            ["parsed", "duration"],
        ],
    )
    kf_gap = _pick_first(
        {"exp": {"params": exp_params}, "step": {"params": step_params}, "parsed": parsed, "kf": keyframes_meta or {}},
        [
            ["kf", "kf_gap"],
            ["step", "params", "kf_gap"],
            ["exp", "params", "kf_gap"],
            ["parsed", "kf_gap"],
        ],
        caster=_coerce_int,
    )
    segment_policy = (
        str((keyframes_meta or {}).get("policy_name", "") or "")
        or str(step_params.get("segment_policy", "") or "")
        or str(exp_params.get("segment_policy", "") or "")
        or str((keyframes_meta or {}).get("policy", "") or "")
    )

    mode = str((keyframes_meta or {}).get("mode_actual", "") or (keyframes_meta or {}).get("mode_requested", "") or "")
    return {
        "dataset": dataset,
        "sequence": sequence,
        "tag": tag,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "kf_gap": kf_gap,
        "segment_policy": segment_policy,
        "mode": mode,
        "exp_name": exp_name,
    }


def _normalize_policy_alias(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    alias_map = {
        "sematic": "semantic",
        "semantics": "semantic",
        "uniform_gap24": "uniform24",
        "uniform_gap60": "uniform60",
    }
    if raw in alias_map:
        return alias_map[raw]
    if raw.startswith("semantic"):
        return "semantic"
    if raw.startswith("motion"):
        return "motion"
    if raw.startswith("risk"):
        return "risk"
    if raw.startswith("uniform60"):
        return "uniform60"
    if raw.startswith("uniform24"):
        return "uniform24"
    if raw.startswith("uniform"):
        return "uniform"
    return raw


def _normalize_policy_label(segment_policy, tag, kf_gap, exp_name):
    candidates = [
        _normalize_policy_alias(segment_policy),
        _normalize_policy_alias(tag),
        _normalize_policy_alias(exp_name),
    ]
    for candidate in candidates:
        if candidate in ("uniform24", "uniform60", "semantic", "motion", "risk"):
            return candidate

    policy_name = ""
    for candidate in candidates:
        if candidate in OFFICIAL_POLICY_NAMES:
            policy_name = candidate
            break

    if policy_name == "uniform":
        if int(kf_gap or 0) == 60:
            return "uniform60"
        if int(kf_gap or 0) == 24:
            return "uniform24"
        if kf_gap is not None:
            return "uniform_gap{}".format(int(kf_gap))
        return "uniform"
    if policy_name in ("semantic", "motion", "risk"):
        return policy_name

    if int(kf_gap or 0) == 60 and ("uniform" in str(tag or "").lower() or "uniform" in str(exp_name or "").lower()):
        return "uniform60"
    if int(kf_gap or 0) == 24 and ("uniform" in str(tag or "").lower() or "uniform" in str(exp_name or "").lower()):
        return "uniform24"

    fallback = _normalize_policy_alias(tag) or _normalize_policy_alias(segment_policy) or _normalize_policy_alias(exp_name)
    return fallback or str(tag or segment_policy or exp_name or "")


def _parse_infer_log_details(log_path):
    path_obj = Path(log_path).resolve()
    out = {}
    if not path_obj.is_file():
        return out

    init_re = re.compile(
        r"Initialization completed in ([0-9.]+)s \(Loading: ([0-9.]+)s, Quantization: ([0-9.]+)s\)"
    )
    done_re = re.compile(
        r"done: segments=(\d+) frames=(\d+) init=([0-9.]+)s infer_sum=([0-9.]+)s "
        r"avg_infer=([0-9.]+)s avg_frame=([0-9.]+)s total=([0-9.]+)s"
    )
    try:
        lines = path_obj.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for raw_line in lines:
        line = str(raw_line or "").strip()
        if line.startswith("[INFO] "):
            line = line[7:]
        init_match = init_re.search(line)
        if init_match:
            out["infer.init"] = float(init_match.group(1))
            out["infer.load"] = float(init_match.group(2))
            out["infer.quant"] = float(init_match.group(3))
        done_match = done_re.search(line)
        if done_match:
            out["infer.segments"] = int(done_match.group(1))
            out["infer.frames"] = int(done_match.group(2))
            out["infer.init"] = float(done_match.group(3))
            out["infer.run"] = float(done_match.group(4))
            out["infer.avg"] = float(done_match.group(5))
            out["infer.avg_fr"] = float(done_match.group(6))
            out["infer.total"] = float(done_match.group(7))
    return out


def _compression_summary(stats_report, stats_legacy, segment_step_meta):
    report_comp = dict((stats_report or {}).get("compression", {}) or {})
    legacy_ori = dict((stats_legacy or {}).get("ori", {}) or {})
    legacy_comp = dict((stats_legacy or {}).get("compressed", {}) or {})
    legacy_ratios = dict((stats_legacy or {}).get("ratios", {}) or {})
    outputs = dict((segment_step_meta or {}).get("outputs", {}) or {})

    ori_frames = _pick_first(
        {"report": report_comp, "legacy_ori": legacy_ori, "outputs": outputs},
        [
            ["report", "ori_frames"],
            ["outputs", "ori", "frame_count"],
            ["outputs", "frame_count"],
            ["legacy_ori", "frame_count"],
        ],
        caster=_coerce_int,
    )
    keyframes_frames = _pick_first(
        {"report": report_comp, "legacy_comp": legacy_comp, "outputs": outputs},
        [
            ["report", "keyframes_frames"],
            ["outputs", "keyframes", "frame_count"],
            ["outputs", "keyframe_count"],
            ["legacy_comp", "keyframe_count"],
        ],
        caster=_coerce_int,
    )
    ori_bytes = _pick_first(
        {"report": report_comp, "legacy_ori": legacy_ori, "outputs": outputs},
        [
            ["report", "ori_bytes"],
            ["outputs", "ori", "bytes_sum"],
            ["outputs", "bytes_sum"],
            ["legacy_ori", "bytes_sum"],
        ],
        caster=_coerce_int,
    )
    keyframes_bytes = _pick_first(
        {"report": report_comp, "legacy_comp": legacy_comp, "outputs": outputs},
        [
            ["report", "keyframes_bytes"],
            ["outputs", "keyframes", "bytes_sum"],
            ["outputs", "keyframe_bytes_sum"],
            ["legacy_comp", "keyframe_bytes_sum"],
        ],
        caster=_coerce_int,
    )
    prompt_bytes = _pick_first(
        {"report": report_comp, "legacy_comp": legacy_comp},
        [
            ["report", "prompt_bytes"],
            ["legacy_comp", "prompt_bytes_sum"],
        ],
        caster=_coerce_int,
    )
    ratio_frames = _pick_first(
        {"report": report_comp, "legacy_ratios": legacy_ratios},
        [
            ["report", "ratio_frames"],
            ["legacy_ratios", "frames"],
        ],
        caster=_coerce_float,
    )
    ratio_bytes = _pick_first(
        {"report": report_comp, "legacy_ratios": legacy_ratios},
        [
            ["report", "ratio_bytes"],
            ["legacy_ratios", "bytes"],
        ],
        caster=_coerce_float,
    )

    if ratio_frames is None and ori_frames and keyframes_frames is not None and int(ori_frames) > 0:
        ratio_frames = float(keyframes_frames) / float(ori_frames)
    if ratio_bytes is None and ori_bytes and keyframes_bytes is not None and prompt_bytes is not None and int(ori_bytes) > 0:
        ratio_bytes = float(int(keyframes_bytes) + int(prompt_bytes)) / float(ori_bytes)

    reduction_bytes = None
    if ratio_bytes is not None:
        reduction_bytes = 1.0 - float(ratio_bytes)

    return {
        "ori_frames": ori_frames,
        "keyframes_frames": keyframes_frames,
        "ori_bytes": ori_bytes,
        "keyframes_bytes": keyframes_bytes,
        "prompt_bytes": prompt_bytes,
        "ratio_frames": ratio_frames,
        "ratio_bytes": ratio_bytes,
        "reduction_bytes": reduction_bytes,
    }


def _risk_summary_fields(risk_summary, risk_bundle, segment_step_meta):
    outputs = dict((segment_step_meta or {}).get("outputs", {}) or {})
    keyframe_policy = dict(outputs.get("keyframe_policy", {}) or {})

    summary_obj = dict(risk_summary or {})
    coverage = dict(summary_obj.get("keyframe_coverage", {}) or {})
    hardest_window = dict(coverage.get("hardest_window", {}) or {})
    worst_window = dict(coverage.get("worst_window", {}) or {})

    if not hardest_window and isinstance(risk_bundle, dict):
        hardest_window = dict(risk_bundle.get("hardest_window", {}) or {})
    if not worst_window and isinstance(risk_bundle, dict):
        worst_window = dict(risk_bundle.get("worst_window", {}) or {})

    risk_windows = list(summary_obj.get("risk_windows", []) or [])
    if not risk_windows and isinstance(risk_bundle, dict):
        risk_windows = list(risk_bundle.get("risk_windows", []) or [])

    hardest_start = _pick_first(
        {"window": hardest_window},
        [["window", "expanded_start_frame"], ["window", "raw_start_frame"]],
        caster=_coerce_int,
    )
    hardest_end = _pick_first(
        {"window": hardest_window},
        [["window", "expanded_end_frame"], ["window", "raw_end_frame"]],
        caster=_coerce_int,
    )
    hardest_range = ""
    if hardest_start is not None and hardest_end is not None:
        hardest_range = "{}-{}".format(int(hardest_start), int(hardest_end))

    worst_window_gap = _pick_first(
        {"coverage": coverage, "worst": worst_window, "hardest": hardest_window},
        [
            ["worst", "final_span_across_window"],
            ["hardest", "final_span_across_window"],
            ["coverage", "max_gap_in_windows"],
        ],
        caster=_coerce_int,
    )

    return {
        "reduction_vs_teacher": _pick_first(
            {"policy": keyframe_policy},
            [["policy", "reduction_vs_teacher"]],
            caster=_coerce_float,
        ),
        "risk_window_count": _pick_first(
            {"policy": keyframe_policy, "summary": summary_obj},
            [["policy", "risk_window_count"]],
            caster=_coerce_int,
        )
        or (len(risk_windows) if risk_windows else None),
        "all_risky_windows_protected": _pick_first(
            {"policy": keyframe_policy},
            [["policy", "all_risky_windows_protected"]],
            caster=_coerce_bool,
        ),
        "worst_window_gap": worst_window_gap,
        "hardest_window_frame_range": hardest_range or None,
        "hardest_window_peak_score": _pick_first(
            {"hardest": hardest_window},
            [["hardest", "peak_score"]],
            caster=_coerce_float,
        ),
    }


def _collect_source_flags(source_map):
    names = []
    for name in sorted(source_map.keys()):
        if source_map[name]:
            names.append(name)
    return ";".join(names)


def _benchmark_row(exp_dir):
    exp_path = Path(exp_dir).resolve()
    keyframes_meta_path = exp_path / "segment" / "keyframes" / "keyframes_meta.json"
    if not keyframes_meta_path.is_file():
        log_warn("skip exp without segment keyframes meta: {}".format(exp_path))
        return None

    source_map = OrderedDict()
    exp_meta = _read_json(exp_path / "exp_meta.json")
    source_map["exp_meta"] = exp_meta is not None
    segment_step_meta = _read_json(exp_path / "segment" / "step_meta.json")
    source_map["segment_step_meta"] = segment_step_meta is not None
    keyframes_meta = _read_json(keyframes_meta_path) or {}
    source_map["keyframes_meta"] = True
    risk_summary = _read_json(exp_path / "segment" / "analysis" / "risk_summary.json")
    source_map["risk_summary"] = risk_summary is not None
    risk_bundle = _read_json(exp_path / "segment" / "analysis" / "risk_bundle.json")
    source_map["risk_bundle"] = risk_bundle is not None
    infer_step_meta = _read_json(exp_path / "infer" / "step_meta.json")
    source_map["infer_step_meta"] = infer_step_meta is not None
    traj_metrics = _read_json(exp_path / "eval" / "traj_metrics.json")
    source_map["traj_metrics"] = traj_metrics is not None
    image_metrics = _read_json(exp_path / "eval" / "image_metrics.json")
    source_map["image_metrics"] = image_metrics is not None
    slam_metrics = _read_json(exp_path / "eval" / "slam_metrics.json")
    source_map["slam_metrics"] = slam_metrics is not None
    stats_report = _read_json(exp_path / "stats" / "report.json")
    source_map["stats_report"] = stats_report is not None
    stats_compression = _read_json(exp_path / "stats" / "compression.json")
    source_map["stats_compression"] = stats_compression is not None
    infer_log_details = _parse_infer_log_details(exp_path / "logs" / "infer.log")
    source_map["infer_log"] = bool(infer_log_details)

    identity = _infer_exp_identity(exp_path, exp_meta, segment_step_meta, keyframes_meta)
    segment_policy = normalize_policy_name(identity["segment_policy"] or "")
    policy_label = _normalize_policy_label(segment_policy, identity["tag"], identity["kf_gap"], identity["exp_name"])

    uniform_base_count = len(list(keyframes_meta.get("uniform_base_indices") or []))
    if uniform_base_count <= 0:
        uniform_base_count = _pick_first(
            {"step": dict((segment_step_meta or {}).get("outputs", {}) or {})},
            [["step", "keyframe_policy", "uniform_base_count"]],
            caster=_coerce_int,
        ) or 0

    final_keyframe_count = len(list(keyframes_meta.get("keyframe_indices") or []))
    if final_keyframe_count <= 0:
        final_keyframe_count = _pick_first(
            {
                "kf": keyframes_meta,
                "step": dict((segment_step_meta or {}).get("outputs", {}) or {}),
            },
            [
                ["kf", "keyframe_count"],
                ["step", "keyframe_policy", "final_keyframe_count"],
                ["step", "keyframe_count"],
            ],
            caster=_coerce_int,
        ) or 0

    compression = _compression_summary(stats_report, stats_compression, segment_step_meta)
    risk_fields = _risk_summary_fields(risk_summary, risk_bundle, segment_step_meta)

    row = OrderedDict()
    row["exp_dir"] = str(exp_path)
    row["dataset"] = identity["dataset"]
    row["sequence"] = identity["sequence"]
    row["tag"] = identity["tag"]
    row["mode"] = identity["mode"]
    row["segment_policy"] = segment_policy
    row["policy_label"] = policy_label
    row["fps"] = identity["fps"]
    row["duration"] = identity["duration"]
    row["kf_gap"] = identity["kf_gap"]
    row["width"] = identity["width"]
    row["height"] = identity["height"]

    row["uniform_base_count"] = uniform_base_count or None
    row["final_keyframe_count"] = final_keyframe_count or None
    row["reduction_vs_teacher"] = risk_fields["reduction_vs_teacher"]
    row["risk_window_count"] = risk_fields["risk_window_count"]
    row["all_risky_windows_protected"] = risk_fields["all_risky_windows_protected"]
    row["worst_window_gap"] = risk_fields["worst_window_gap"]
    row["hardest_window_frame_range"] = risk_fields["hardest_window_frame_range"]
    row["hardest_window_peak_score"] = risk_fields["hardest_window_peak_score"]

    row["compression_ratio_frames"] = compression["ratio_frames"]
    row["compression_ratio_bytes"] = compression["ratio_bytes"]
    row["compression_reduction_bytes"] = compression["reduction_bytes"]
    row["ori_frame_count"] = compression["ori_frames"]
    row["keyframes_frame_count"] = compression["keyframes_frames"]
    row["ori_bytes"] = compression["ori_bytes"]
    row["keyframes_bytes"] = compression["keyframes_bytes"]
    row["prompt_bytes"] = compression["prompt_bytes"]

    row["infer_segments"] = _coerce_int(infer_log_details.get("infer.segments"))
    if row["infer_segments"] is None:
        row["infer_segments"] = _pick_first(
            {"infer": infer_step_meta or {}},
            [["infer", "segments"], ["infer", "execution_plan_segments"]],
            caster=_coerce_int,
        )
    row["infer_frames"] = _coerce_int(infer_log_details.get("infer.frames"))
    if row["infer_frames"] is None:
        row["infer_frames"] = _pick_first(
            {"infer": infer_step_meta or {}},
            [["infer", "used_frames"]],
            caster=_coerce_int,
        )
    row["infer_init_sec"] = _coerce_float(infer_log_details.get("infer.init"))
    row["infer_run_sec"] = _coerce_float(infer_log_details.get("infer.run"))
    row["infer_avg_frame_sec"] = _coerce_float(infer_log_details.get("infer.avg_fr"))

    row["ape_rmse"] = _pick_first({"traj": traj_metrics or {}}, [["traj", "ape_trans", "rmse"]], caster=_coerce_float)
    row["rpe_trans_rmse"] = _pick_first(
        {"traj": traj_metrics or {}},
        [["traj", "rpe_trans", "rmse"]],
        caster=_coerce_float,
    )
    row["matched_count"] = _pick_first(
        {"traj": traj_metrics or {}},
        [["traj", "matched_pose_count"]],
        caster=_coerce_int,
    )
    row["image_psnr_mean"] = _pick_first(
        {"image": image_metrics or {}},
        [["image", "psnr", "mean"]],
        caster=_coerce_float,
    )
    row["image_ms_ssim_mean"] = _pick_first(
        {"image": image_metrics or {}},
        [["image", "ms_ssim", "mean"]],
        caster=_coerce_float,
    )
    row["image_lpips_mean"] = _pick_first(
        {"image": image_metrics or {}},
        [["image", "lpips", "mean"]],
        caster=_coerce_float,
    )
    row["image_frame_count"] = _pick_first(
        {"image": image_metrics or {}},
        [["image", "frame_count"]],
        caster=_coerce_int,
    )
    row["slam_inlier_ratio_mean"] = _pick_first(
        {"slam": slam_metrics or {}},
        [["slam", "inlier_ratio", "mean"]],
        caster=_coerce_float,
    )
    row["slam_pose_success_rate"] = _pick_first(
        {"slam": slam_metrics or {}},
        [["slam", "pose_success_rate"]],
        caster=_coerce_float,
    )
    row["slam_reference_source"] = str((slam_metrics or {}).get("reference_source", "") or "")
    row["traj_eval_status"] = str((traj_metrics or {}).get("eval_status", "") or "")
    row["image_eval_status"] = str((image_metrics or {}).get("eval_status", "") or "")
    row["slam_eval_status"] = str((slam_metrics or {}).get("eval_status", "") or "")
    row["sources_present"] = _collect_source_flags(source_map)

    return {
        "row": row,
        "source_map": source_map,
    }


def _iter_experiment_dirs(experiments_root):
    root = ensure_dir(experiments_root, name="experiments root")
    seen = set()
    for current_root, dirnames, filenames in os.walk(str(root)):
        current_path = Path(current_root)
        meta_path = current_path / _SEGMENT_KEYFRAME_META_REL
        if meta_path.is_file():
            key = str(current_path.resolve())
            if key not in seen:
                seen.add(key)
                yield current_path.resolve()
            dirnames[:] = []


def _filters_match(row, args):
    if args.dataset and str(row.get("dataset", "") or "") != str(args.dataset):
        return False
    if args.sequence and str(row.get("sequence", "") or "") != str(args.sequence):
        return False

    tags_exact = [str(item) for item in list(args.tag or []) if str(item).strip()]
    if tags_exact and str(row.get("tag", "") or "") not in tags_exact:
        return False

    tag_contains = [str(item).lower() for item in list(args.tag_contains or []) if str(item).strip()]
    if tag_contains:
        haystack = "{} {}".format(str(row.get("tag", "") or ""), str(Path(row.get("exp_dir", "")).name)).lower()
        matched = False
        for needle in tag_contains:
            if needle in haystack:
                matched = True
                break
        if not matched:
            return False

    policy_labels = [str(item) for item in list(args.policy_label or []) if str(item).strip()]
    if policy_labels and str(row.get("policy_label", "") or "") not in policy_labels:
        return False
    return True


def _sort_rows(rows):
    def _row_key(row):
        policy_label = str(row.get("policy_label", "") or "")
        return (
            int(_POLICY_ORDER.get(policy_label, 99)),
            policy_label,
            str(row.get("dataset", "") or ""),
            str(row.get("sequence", "") or ""),
            str(row.get("tag", "") or ""),
            str(row.get("exp_dir", "") or ""),
        )

    return sorted(rows, key=_row_key)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path, rows):
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(_CSV_FIELD_ORDER))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((field, _csv_value(row.get(field))) for field in _CSV_FIELD_ORDER))


def _policy_groups(rows):
    groups = OrderedDict()
    for row in rows:
        label = str(row.get("policy_label", "") or "unknown")
        if label not in groups:
            groups[label] = []
        groups[label].append(row)
    return groups


def _group_summary(label, rows):
    bool_values = [row.get("all_risky_windows_protected") for row in rows if row.get("all_risky_windows_protected") is not None]
    protected_status = None
    if bool_values:
        if all(bool(v) for v in bool_values):
            protected_status = "all_yes"
        elif not any(bool(v) for v in bool_values):
            protected_status = "all_no"
        else:
            protected_status = "mixed"

    return OrderedDict(
        [
            ("policy_label", label),
            ("exp_count", len(rows)),
            ("datasets", sorted(set(str(row.get("dataset", "") or "") for row in rows if row.get("dataset")))),
            ("sequences", sorted(set(str(row.get("sequence", "") or "") for row in rows if row.get("sequence")))),
            ("final_keyframe_count_mean", _mean([row.get("final_keyframe_count") for row in rows])),
            ("final_keyframe_count_min", min([row.get("final_keyframe_count") for row in rows if row.get("final_keyframe_count") is not None]) if [row.get("final_keyframe_count") for row in rows if row.get("final_keyframe_count") is not None] else None),
            ("final_keyframe_count_max", max([row.get("final_keyframe_count") for row in rows if row.get("final_keyframe_count") is not None]) if [row.get("final_keyframe_count") for row in rows if row.get("final_keyframe_count") is not None] else None),
            ("ape_rmse_mean", _mean([row.get("ape_rmse") for row in rows])),
            ("rpe_trans_rmse_mean", _mean([row.get("rpe_trans_rmse") for row in rows])),
            ("matched_count_mean", _mean([row.get("matched_count") for row in rows])),
            ("risk_window_count_mean", _mean([row.get("risk_window_count") for row in rows])),
            ("all_risky_windows_protected_status", protected_status),
            ("compression_ratio_frames_mean", _mean([row.get("compression_ratio_frames") for row in rows])),
            ("compression_ratio_bytes_mean", _mean([row.get("compression_ratio_bytes") for row in rows])),
            ("infer_avg_frame_sec_mean", _mean([row.get("infer_avg_frame_sec") for row in rows])),
        ]
    )


def _markdown_table(headers, rows):
    if not rows:
        return ["_No data_"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _summary_observations(rows, group_summaries):
    observations = []
    if rows:
        keyframe_rows = [row for row in rows if row.get("final_keyframe_count") is not None]
        if keyframe_rows:
            best = min(keyframe_rows, key=lambda item: int(item.get("final_keyframe_count")))
            observations.append(
                "{} has the fewest final keyframes ({}) in this benchmark.".format(
                    str(best.get("policy_label") or best.get("tag") or "unknown"),
                    int(best.get("final_keyframe_count")),
                )
            )

        ape_rows = [row for row in rows if row.get("ape_rmse") is not None]
        if ape_rows:
            best_ape = min(ape_rows, key=lambda item: float(item.get("ape_rmse")))
            observations.append(
                "{} has the best APE RMSE among available eval results ({:.4f} m).".format(
                    str(best_ape.get("policy_label") or best_ape.get("tag") or "unknown"),
                    float(best_ape.get("ape_rmse")),
                )
            )
        else:
            observations.append("No available trajectory eval metrics were found, so APE/RPE comparison is skipped.")

        risk_rows = [row for row in rows if str(row.get("policy_label") or "") == "risk"]
        if risk_rows:
            risk_row = risk_rows[0]
            protected = risk_row.get("all_risky_windows_protected")
            risk_windows = risk_row.get("risk_window_count")
            reduction = risk_row.get("reduction_vs_teacher")
            if protected is not None or risk_windows is not None or reduction is not None:
                observations.append(
                    "risk protection status: protected={} risk_windows={} reduction_vs_teacher={}.".format(
                        _format_bool_text(protected) or "n/a",
                        _format_num(risk_windows, digits=0) or "n/a",
                        _format_ratio(reduction) or "n/a",
                    )
                )

    return observations[:3]


def _write_summary_md(path, rows, group_summaries, benchmark_title, filters_summary):
    experiment_rows = []
    for row in rows:
        experiment_rows.append(
            [
                str(row.get("policy_label", "") or ""),
                str(row.get("tag", "") or ""),
                _format_num(row.get("final_keyframe_count"), digits=0),
                _format_num(row.get("uniform_base_count"), digits=0),
                _format_num(row.get("risk_window_count"), digits=0),
                _format_bool_text(row.get("all_risky_windows_protected")),
                _format_num(row.get("ape_rmse")),
                _format_num(row.get("rpe_trans_rmse")),
                _format_num(row.get("matched_count"), digits=0),
            ]
        )

    group_rows = []
    for summary in group_summaries:
        protected_status = str(summary.get("all_risky_windows_protected_status") or "")
        group_rows.append(
            [
                str(summary.get("policy_label", "") or ""),
                _format_num(summary.get("exp_count"), digits=0),
                _format_num(summary.get("final_keyframe_count_mean")),
                _format_num(summary.get("ape_rmse_mean")),
                _format_num(summary.get("rpe_trans_rmse_mean")),
                _format_num(summary.get("matched_count_mean")),
                _format_num(summary.get("risk_window_count_mean")),
                protected_status,
            ]
        )

    observations = _summary_observations(rows, group_summaries)

    lines = [
        "# Policy Benchmark Summary",
        "",
        "- title: {}".format(benchmark_title),
        "- generated_at: {}".format(datetime.now().isoformat(timespec="seconds")),
        "- experiment_count: {}".format(len(rows)),
        "- filters: {}".format(filters_summary or "none"),
        "",
        "## Experiment Comparison",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "policy_label",
                "tag",
                "final_keyframe_count",
                "uniform_base_count",
                "risk_window_count",
                "protected",
                "APE RMSE",
                "RPE trans RMSE",
                "matched",
            ],
            experiment_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Policy Aggregate",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "policy_label",
                "exp_count",
                "final_keyframe_count_mean",
                "APE RMSE mean",
                "RPE trans RMSE mean",
                "matched mean",
                "risk_window_count_mean",
                "protected_status",
            ],
            group_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Observations",
            "",
        ]
    )
    if observations:
        for item in observations:
            lines.append("- {}".format(item))
    else:
        lines.append("- No stable observation could be generated from the current input set.")

    Path(path).resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")


def _import_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _group_plot_payload(rows):
    payload = []
    for label, group_rows in _policy_groups(rows).items():
        summary = _group_summary(label, group_rows)
        payload.append(
            {
                "policy_label": label,
                "exp_count": len(group_rows),
                "final_keyframe_count_mean": summary.get("final_keyframe_count_mean"),
                "ape_rmse_mean": summary.get("ape_rmse_mean"),
                "risk_window_count_mean": summary.get("risk_window_count_mean"),
                "protected_status": summary.get("all_risky_windows_protected_status"),
            }
        )
    payload.sort(key=lambda item: (_POLICY_ORDER.get(str(item.get("policy_label", "")), 99), str(item.get("policy_label", ""))))
    return payload


def _plot_overview(path, rows, benchmark_title):
    plt = _import_matplotlib()
    payload = _group_plot_payload(rows)
    if not payload:
        raise SystemExit("[ERR] no rows available for overview plot")

    has_ape = any(item.get("ape_rmse_mean") is not None for item in payload)
    subplot_count = 2 if has_ape else 1
    fig, axes = plt.subplots(subplot_count, 1, figsize=(12, 5.5 + (2.4 if has_ape else 0.0)))
    if subplot_count == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    labels = [str(item.get("policy_label") or "") for item in payload]
    xs = list(range(len(payload)))
    colors = [_POLICY_COLORS.get(label, "#7f7f7f") for label in labels]

    keyframe_ax = axes[0]
    keyframe_values = [item.get("final_keyframe_count_mean") or 0.0 for item in payload]
    bars = keyframe_ax.bar(xs, keyframe_values, color=colors, edgecolor="#4a4a4a", linewidth=0.7)
    keyframe_ax.set_title("{}: final keyframe count".format(benchmark_title), fontsize=13)
    keyframe_ax.set_ylabel("final_keyframe_count")
    keyframe_ax.set_xticks(xs)
    keyframe_ax.set_xticklabels(labels, rotation=20, ha="right")
    keyframe_ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.9)
    keyframe_ax.set_axisbelow(True)

    max_value = max(keyframe_values) if keyframe_values else 0.0
    text_offset = 0.03 * max(1.0, max_value)
    for idx, bar in enumerate(bars):
        payload_item = payload[idx]
        value = payload_item.get("final_keyframe_count_mean")
        label_lines = [_format_num(value, digits=2)]
        risk_windows = payload_item.get("risk_window_count_mean")
        protected_status = str(payload_item.get("protected_status") or "")
        if risk_windows is not None or protected_status:
            if risk_windows is not None:
                label_lines.append("risk_w={}".format(_format_num(risk_windows, digits=1)))
            if protected_status:
                label_lines.append("prot={}".format(protected_status))
        label_lines.append("n={}".format(int(payload_item.get("exp_count") or 0)))
        keyframe_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + text_offset,
            "\n".join(label_lines),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    if has_ape:
        ape_ax = axes[1]
        ape_values = [item.get("ape_rmse_mean") if item.get("ape_rmse_mean") is not None else float("nan") for item in payload]
        ape_bars = ape_ax.bar(xs, ape_values, color=colors, edgecolor="#4a4a4a", linewidth=0.7)
        ape_ax.set_title("{}: APE RMSE".format(benchmark_title), fontsize=13)
        ape_ax.set_ylabel("APE RMSE (m)")
        ape_ax.set_xticks(xs)
        ape_ax.set_xticklabels(labels, rotation=20, ha="right")
        ape_ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.9)
        ape_ax.set_axisbelow(True)
        finite_ape = [float(v) for v in ape_values if not math.isnan(v)]
        ape_offset = 0.04 * max(finite_ape) if finite_ape else 0.02
        for idx, bar in enumerate(ape_bars):
            value = ape_values[idx]
            if math.isnan(value):
                ape_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.02,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#444444",
                )
            else:
                ape_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + ape_offset,
                    _format_num(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#222222",
                )

    fig.tight_layout()
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _filters_summary(args):
    parts = []
    if args.dataset:
        parts.append("dataset={}".format(args.dataset))
    if args.sequence:
        parts.append("sequence={}".format(args.sequence))
    if args.tag:
        parts.append("tag={}".format(",".join(str(item) for item in args.tag)))
    if args.tag_contains:
        parts.append("tag_contains={}".format(",".join(str(item) for item in args.tag_contains)))
    if args.policy_label:
        parts.append("policy_label={}".format(",".join(str(item) for item in args.policy_label)))
    return "; ".join(parts)


def _resolve_exp_dirs(args):
    ordered = []
    seen = set()

    for raw in list(args.exp_dir or []):
        if not str(raw or "").strip():
            continue
        path_obj = Path(raw).resolve()
        key = str(path_obj)
        if key not in seen:
            seen.add(key)
            ordered.append(path_obj)

    if args.experiments_root:
        for exp_path in _iter_experiment_dirs(args.experiments_root):
            key = str(exp_path)
            if key not in seen:
                seen.add(key)
                ordered.append(exp_path)

    return ordered


def main():
    args = build_arg_parser().parse_args()
    if not args.exp_dir and not args.experiments_root:
        raise SystemExit("[ERR] provide at least one --exp_dir or --experiments_root")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_dirs = _resolve_exp_dirs(args)
    if not resolved_dirs:
        raise SystemExit("[ERR] no experiment directories were resolved")

    log_info("policy benchmark collect start: candidates={}".format(len(resolved_dirs)))

    rows = []
    records = []
    for exp_path in resolved_dirs:
        record = _benchmark_row(exp_path)
        if record is None:
            continue
        row = record["row"]
        if not _filters_match(row, args):
            continue
        rows.append(row)
        records.append(
            OrderedDict(
                [
                    ("exp_dir", str(exp_path)),
                    ("policy_label", row.get("policy_label")),
                    ("row", row),
                    ("sources", record.get("source_map", {})),
                ]
            )
        )

    rows = _sort_rows(rows)
    records = sorted(
        records,
        key=lambda item: (
            _POLICY_ORDER.get(str(item.get("policy_label", "") or ""), 99),
            str(item.get("policy_label", "") or ""),
            str(item.get("exp_dir", "") or ""),
        ),
    )

    if not rows:
        raise SystemExit("[ERR] no experiment matched the current inputs and filters")

    benchmark_title = "policy benchmark"
    dataset_set = sorted(set(str(row.get("dataset", "") or "") for row in rows if row.get("dataset")))
    sequence_set = sorted(set(str(row.get("sequence", "") or "") for row in rows if row.get("sequence")))
    if len(dataset_set) == 1 and len(sequence_set) == 1:
        benchmark_title = "{} / {}".format(dataset_set[0], sequence_set[0])

    group_summaries = [_group_summary(label, group_rows) for label, group_rows in _policy_groups(rows).items()]

    csv_path = out_dir / "policy_benchmark.csv"
    json_path = out_dir / "policy_benchmark.json"
    summary_md_path = out_dir / "policy_benchmark_summary.md"
    overview_png_path = out_dir / "policy_benchmark_overview.png"

    _write_csv(csv_path, rows)
    write_json_atomic(
        json_path,
        OrderedDict(
            [
                ("created_at", datetime.now().isoformat(timespec="seconds")),
                ("benchmark_title", benchmark_title),
                ("filters", _filters_summary(args)),
                ("experiment_count", len(rows)),
                ("rows", rows),
                ("policy_groups", group_summaries),
                ("records", records),
                (
                    "outputs",
                    {
                        "csv": str(csv_path),
                        "json": str(json_path),
                        "summary_md": str(summary_md_path),
                        "overview_png": str(overview_png_path),
                    },
                ),
            ]
        ),
        indent=2,
    )
    _write_summary_md(summary_md_path, rows, group_summaries, benchmark_title, _filters_summary(args))
    _plot_overview(overview_png_path, rows, benchmark_title)

    log_info("policy benchmark rows: {}".format(len(rows)))
    log_info("policy benchmark csv: {}".format(csv_path))
    log_info("policy benchmark json: {}".format(json_path))
    log_info("policy benchmark summary: {}".format(summary_md_path))
    log_info("policy benchmark overview: {}".format(overview_png_path))


if __name__ == "__main__":
    main()
