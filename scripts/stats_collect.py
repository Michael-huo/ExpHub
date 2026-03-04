#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
from pathlib import Path

from _common import log_info, log_warn, write_json_atomic


def _sum_png_bytes(dir_path):
    p = Path(dir_path).resolve()
    if not p.is_dir():
        return None, None
    n = 0
    b = 0
    for fp in sorted(p.glob("*.png")):
        if not fp.is_file():
            continue
        n += 1
        try:
            b += int(fp.stat().st_size)
        except Exception:
            pass
    return int(n), int(b)


def _file_size(path):
    p = Path(path).resolve()
    if not p.is_file():
        return None
    try:
        return int(p.stat().st_size)
    except Exception:
        return None


def _read_created_at(path):
    p = Path(path).resolve()
    if not p.is_file():
        return None
    try:
        import json
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("created_at")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    stats_dir = exp_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    segment_frames = exp_dir / "segment" / "frames"
    keyframes_dir = exp_dir / "segment" / "keyframes"
    prompt_manifest = exp_dir / "prompt" / "manifest.json"

    ori_frames, ori_bytes = _sum_png_bytes(segment_frames)
    if ori_frames is None:
        msg = "missing segment/frames; ori compression fields set to null"
        warnings.append(msg)
        log_warn(msg)

    keyframes_frames, keyframes_bytes = _sum_png_bytes(keyframes_dir)
    if keyframes_frames is None:
        msg = "missing segment/keyframes; keyframes compression fields set to null"
        warnings.append(msg)
        log_warn(msg)

    prompt_bytes = _file_size(prompt_manifest)
    if prompt_bytes is None:
        msg = "missing prompt/manifest.json; prompt bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    ratio_bytes = None
    if ori_bytes is not None and ori_bytes > 0 and keyframes_bytes is not None and prompt_bytes is not None:
        ratio_bytes = float(keyframes_bytes + prompt_bytes) / float(ori_bytes)

    ratio_frames = None
    if ori_frames is not None and ori_frames > 0 and keyframes_frames is not None:
        ratio_frames = float(keyframes_frames) / float(ori_frames)

    inputs = {
        "segment_step_meta": str((exp_dir / "segment" / "step_meta.json").resolve()) if (exp_dir / "segment" / "step_meta.json").is_file() else None,
        "prompt_step_meta": str((exp_dir / "prompt" / "step_meta.json").resolve()) if (exp_dir / "prompt" / "step_meta.json").is_file() else None,
        "infer_step_meta": str((exp_dir / "infer" / "step_meta.json").resolve()) if (exp_dir / "infer" / "step_meta.json").is_file() else None,
        "merge_step_meta": str((exp_dir / "merge" / "step_meta.json").resolve()) if (exp_dir / "merge" / "step_meta.json").is_file() else None,
    }

    timing = {
        "segment_created_at": _read_created_at(exp_dir / "segment" / "step_meta.json"),
        "prompt_created_at": _read_created_at(exp_dir / "prompt" / "step_meta.json"),
        "infer_created_at": _read_created_at(exp_dir / "infer" / "step_meta.json"),
        "merge_created_at": _read_created_at(exp_dir / "merge" / "step_meta.json"),
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
            "prompt_files": [str(prompt_manifest.resolve())] if prompt_bytes is not None else [],
            "prompt_file_count": 1 if prompt_bytes is not None else 0,
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
    log_info("stats report written: {}".format(report_path))
    log_info("legacy compression written: {}".format(comp_path))


if __name__ == "__main__":
    main()
