#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
from scripts._common import list_frames_sorted, log_err, log_warn, write_json_atomic

try:
    import cv2
except Exception as e:
    log_err("import cv2 failed: {}".format(e))
    sys.exit(2)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _dir_file_stats(dir_path):
    """Return (file_count, bytes_sum) for direct files under a directory."""
    if not os.path.isdir(dir_path):
        return 0, 0

    file_count = 0
    bytes_sum = 0
    for name in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, name)
        if not os.path.isfile(fp):
            continue
        file_count += 1
        try:
            bytes_sum += int(os.path.getsize(fp))
        except Exception:
            pass
    return int(file_count), int(bytes_sum)


def _write_preprocess_artifacts(root_dir, preprocess_meta, calib_out):
    with open(os.path.join(root_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)

    np.savetxt(
        os.path.join(root_dir, "calib.txt"),
        np.array(calib_out, dtype=np.float64).reshape(1, -1),
        fmt="%.10f",
    )


def materialize_frames(args, ctx, frame_items):
    ensure_dir(ctx["frames_dir"])
    ensure_dir(ctx["root_dir"])

    ts_lines = []
    wrote_meta = False
    for item in frame_items:
        if (not wrote_meta) and ctx.get("preprocess_meta") is not None and ctx.get("calib_out") is not None:
            _write_preprocess_artifacts(ctx["root_dir"], ctx["preprocess_meta"], ctx["calib_out"])
            wrote_meta = True

        out_path = os.path.join(ctx["frames_dir"], "{:06d}.png".format(int(item["index"])))
        cv2.imwrite(out_path, item["image"], [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compress)])
        ts_lines.append(item["timestamp_line"])

    with open(os.path.join(ctx["root_dir"], "timestamps.txt"), "w", encoding="utf-8") as f:
        f.writelines(ts_lines)

    return ts_lines


def materialize_keyframes(root_dir, frames_dir, plan, keyframes_mode):
    keyframes_dir = os.path.join(root_dir, "keyframes")
    ensure_dir(keyframes_dir)

    req_mode = str(keyframes_mode)
    actual_mode = req_mode

    def _make_one(src_path, dst_path):
        nonlocal actual_mode
        try:
            if os.path.lexists(dst_path):
                os.remove(dst_path)
        except Exception:
            pass

        if actual_mode == "symlink":
            try:
                rel = os.path.relpath(src_path, start=os.path.dirname(dst_path))
                os.symlink(rel, dst_path)
                return
            except Exception:
                actual_mode = "hardlink"

        if actual_mode == "hardlink":
            try:
                os.link(src_path, dst_path)
                return
            except Exception:
                actual_mode = "copy"

        shutil.copy2(src_path, dst_path)

    bytes_sum = 0
    for idx in plan["keyframe_indices"]:
        src = os.path.join(frames_dir, "{:06d}.png".format(int(idx)))
        dst = os.path.join(keyframes_dir, "{:06d}.png".format(int(idx)))
        if not os.path.exists(src):
            log_warn("keyframe source missing: {}".format(src))
            continue
        _make_one(src, dst)
        try:
            bytes_sum += int(os.path.getsize(src))
        except Exception:
            pass

    kf_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kf_gap": int(plan["kf_gap"]),
        "mode_requested": req_mode,
        "mode_actual": actual_mode,
        "frame_count_total": int(plan["frame_count_total"]),
        "frame_count_used": int(plan["frame_count_used"]),
        "tail_drop": int(plan["tail_drop"]),
        "keyframe_count": int(plan["keyframe_count"]),
        "keyframe_indices": list(plan["keyframe_indices"]),
        "keyframe_bytes_sum": int(bytes_sum),
        "note": "Keyframes are anchors sampled every kf_gap frames (including 0). If tail_drop>0, the last few frames are not used by infer segmentation.",
    }
    with open(os.path.join(keyframes_dir, "keyframes_meta.json"), "w", encoding="utf-8") as f:
        json.dump(kf_meta, f, ensure_ascii=False, indent=2)

    return kf_meta


def write_step_meta(args, root_dir, frames_dir, ts_lines):
    actual_frame_count = len(list_frames_sorted(frames_dir))
    _, frames_bytes_sum = _dir_file_stats(frames_dir)
    keyframes_dir = os.path.join(root_dir, "keyframes")
    keyframes_file_count, keyframes_bytes_sum = _dir_file_stats(keyframes_dir)
    keyframes_frame_count = len(list_frames_sorted(keyframes_dir)) if os.path.isdir(keyframes_dir) else 0

    step_meta = {
        "step": "segment",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "bag": os.path.abspath(args.bag),
            "topic": args.topic,
        },
        "params": {
            "w": int(args.width),
            "h": int(args.height),
            "fps": float(args.fps),
            "dur": float(args.duration),
            "start_sec": float(args.start_sec),
            "start_idx": int(args.start_idx),
            "kf_gap": int(args.kf_gap),
        },
        "outputs": {
            "frame_count": int(actual_frame_count),
            "bytes_sum": int(frames_bytes_sum),
            "timestamps_count": int(len(ts_lines)),
            "keyframes_frame_count": int(keyframes_frame_count),
            "keyframes_bytes_sum": int(keyframes_bytes_sum),
            "keyframes_file_count": int(keyframes_file_count),
            "keyframe_count": int(keyframes_frame_count),
            "keyframe_bytes_sum": int(keyframes_bytes_sum),
            "ori": {
                "frame_count": int(actual_frame_count),
                "bytes_sum": int(frames_bytes_sum),
            },
            "keyframes": {
                "frame_count": int(keyframes_frame_count),
                "bytes_sum": int(keyframes_bytes_sum),
            },
        },
    }
    write_json_atomic(os.path.join(root_dir, "step_meta.json"), step_meta, indent=2)
