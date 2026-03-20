#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
from pathlib import Path

from _common import log_err, log_info, log_prog, log_warn, read_step_meta, write_json_atomic


def _as_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _get_nested(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _pick_int(meta, candidate_paths):
    for path in candidate_paths:
        raw = _get_nested(meta, path)
        val = _as_int_or_none(raw)
        if val is not None:
            return val
    return None


def _meta_created_at(meta):
    if isinstance(meta, dict):
        return meta.get("created_at")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    stats_dir = exp_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    segment_meta_path = exp_dir / "segment" / "step_meta.json"
    prompt_meta_path = exp_dir / "prompt" / "step_meta.json"
    infer_meta_path = exp_dir / "infer" / "step_meta.json"
    merge_meta_path = exp_dir / "merge" / "step_meta.json"

    segment_meta = read_step_meta(segment_meta_path)
    prompt_meta = read_step_meta(prompt_meta_path)
    infer_meta = read_step_meta(infer_meta_path)
    merge_meta = read_step_meta(merge_meta_path)

    if not segment_meta_path.is_file():
        msg = "missing segment/step_meta.json; segment compression fields set to null"
        warnings.append(msg)
        log_warn(msg)
    elif not segment_meta:
        msg = "invalid or empty segment/step_meta.json; segment compression fields set to null"
        warnings.append(msg)
        log_warn(msg)

    if not prompt_meta_path.is_file():
        msg = "missing prompt/step_meta.json; prompt bytes set to null"
        warnings.append(msg)
        log_warn(msg)
    elif not prompt_meta:
        msg = "invalid or empty prompt/step_meta.json; prompt bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    ori_frames = _pick_int(
        segment_meta,
        [
            ("outputs", "ori", "frame_count"),
            ("outputs", "frame_count"),
            ("frame_count",),
            ("frames_count",),
        ],
    )
    if ori_frames is None:
        msg = "missing segment frame_count in segment/step_meta.json; ori_frames set to null"
        warnings.append(msg)
        log_warn(msg)

    ori_bytes = _pick_int(
        segment_meta,
        [
            ("outputs", "ori", "bytes_sum"),
            ("outputs", "bytes_sum"),
            ("outputs", "ori_bytes"),
            ("bytes_sum",),
            ("ori_bytes",),
        ],
    )
    if ori_bytes is None:
        msg = "missing segment bytes_sum in segment/step_meta.json; ori_bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    keyframes_frames = _pick_int(
        segment_meta,
        [
            ("outputs", "keyframes", "frame_count"),
            ("outputs", "keyframes_frames"),
            ("outputs", "keyframe_count"),
            ("keyframes_frames",),
            ("keyframe_count",),
        ],
    )
    if keyframes_frames is None:
        msg = "missing keyframes frame_count in segment/step_meta.json; keyframes_frames set to null"
        warnings.append(msg)
        log_warn(msg)

    keyframes_bytes = _pick_int(
        segment_meta,
        [
            ("outputs", "keyframes", "bytes_sum"),
            ("outputs", "keyframes_bytes"),
            ("outputs", "keyframe_bytes_sum"),
            ("keyframes_bytes",),
            ("keyframe_bytes_sum",),
        ],
    )
    if keyframes_bytes is None:
        msg = "missing keyframes bytes_sum in segment/step_meta.json; keyframes_bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    prompt_bytes = _pick_int(
        prompt_meta,
        [
            ("outputs", "prompt", "bytes_sum"),
            ("outputs", "prompt_bytes"),
            ("outputs", "bytes_sum"),
            ("prompt_bytes",),
            ("manifest_size",),
            ("outputs", "manifest_size"),
        ],
    )
    if prompt_bytes is None:
        msg = "missing prompt bytes_sum in prompt/step_meta.json; prompt_bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    ratio_bytes = None
    if ori_bytes is not None and ori_bytes > 0 and keyframes_bytes is not None and prompt_bytes is not None:
        ratio_bytes = float(keyframes_bytes + prompt_bytes) / float(ori_bytes)

    ratio_frames = None
    if ori_frames is not None and ori_frames > 0 and keyframes_frames is not None:
        ratio_frames = float(keyframes_frames) / float(ori_frames)

    segment_frames = exp_dir / "segment" / "frames"
    keyframes_dir = exp_dir / "segment" / "keyframes"
    prompt_profile = exp_dir / "prompt" / "profile.json"
    prompt_final = exp_dir / "prompt" / "final_prompt.json"

    inputs = {
        "segment_step_meta": str(segment_meta_path.resolve()) if segment_meta_path.is_file() else None,
        "prompt_step_meta": str(prompt_meta_path.resolve()) if prompt_meta_path.is_file() else None,
        "infer_step_meta": str(infer_meta_path.resolve()) if infer_meta_path.is_file() else None,
        "merge_step_meta": str(merge_meta_path.resolve()) if merge_meta_path.is_file() else None,
    }

    timing = {
        "segment_created_at": _meta_created_at(segment_meta),
        "prompt_created_at": _meta_created_at(prompt_meta),
        "infer_created_at": _meta_created_at(infer_meta),
        "merge_created_at": _meta_created_at(merge_meta),
    }

    report = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(exp_dir),
        "compression": {
            "ori_frames": ori_frames,
            "ori_bytes": ori_bytes,
            "keyframes_frames": keyframes_frames,
            "keyframes_bytes": keyframes_bytes,
            "prompt_bytes": prompt_bytes,
            "ratio_bytes": ratio_bytes,
            "ratio_frames": ratio_frames,
        },
        "timing": timing,
        "inputs": inputs,
        "warnings": warnings,
        "notes": [
            "TODO: add trajectory/quality similarity metrics in later PR."
        ],
    }

    # Keep legacy compression.json for backward compatibility.
    legacy = {
        "ori": {
            "frames_dir": str(segment_frames.resolve()),
            "frame_count": ori_frames,
            "bytes_sum": ori_bytes,
        },
        "compressed": {
            "keyframes_dir": str(keyframes_dir.resolve()),
            "keyframe_count": keyframes_frames,
            "keyframe_bytes_sum": keyframes_bytes,
            "prompt_files": [
                str(path.resolve())
                for path in [prompt_profile, prompt_final]
                if prompt_bytes is not None and path.is_file()
            ],
            "prompt_file_count": len(
                [path for path in [prompt_profile, prompt_final] if prompt_bytes is not None and path.is_file()]
            ),
            "prompt_bytes_sum": prompt_bytes,
            "total_bytes_sum": (keyframes_bytes + prompt_bytes) if (keyframes_bytes is not None and prompt_bytes is not None) else None,
        },
        "ratios": {
            "bytes": ratio_bytes,
            "frames": ratio_frames,
        },
    }

    report_path = stats_dir / "report.json"
    comp_path = stats_dir / "compression.json"
    write_json_atomic(report_path, report, indent=2)
    write_json_atomic(comp_path, legacy, indent=2)
    log_prog("stats summary: report generated")
    log_info("stats report written: {}".format(report_path))
    log_info("legacy compression written: {}".format(comp_path))


if __name__ == "__main__":
    main()
