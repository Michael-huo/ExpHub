#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path


def materialize_keyframes(frames_dir, keyframes_dir, keyframe_indices, mode_requested):
    frames_dir = Path(frames_dir).resolve()
    keyframes_dir = Path(keyframes_dir).resolve()
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    actual_mode = str(mode_requested)

    def _make_one(src_path, dst_path):
        nonlocal actual_mode
        try:
            if os.path.lexists(str(dst_path)):
                os.remove(str(dst_path))
        except Exception:
            pass

        if actual_mode == "symlink":
            try:
                rel = os.path.relpath(str(src_path), start=str(dst_path.parent))
                os.symlink(rel, str(dst_path))
                return
            except Exception:
                actual_mode = "hardlink"

        if actual_mode == "hardlink":
            try:
                os.link(str(src_path), str(dst_path))
                return
            except Exception:
                actual_mode = "copy"

        shutil.copy2(str(src_path), str(dst_path))

    bytes_sum = 0
    for frame_idx in keyframe_indices:
        src_path = frames_dir / "{:06d}.png".format(int(frame_idx))
        dst_path = keyframes_dir / "{:06d}.png".format(int(frame_idx))
        if not src_path.exists():
            continue
        _make_one(src_path, dst_path)
        try:
            bytes_sum += int(src_path.stat().st_size)
        except Exception:
            pass

    return actual_mode, int(bytes_sum)
