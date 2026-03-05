#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ExpHub launcher for VideoX-Fun (Wan2.2) keyframe-interpolation inference.

Design goals (theory-first):
- Dataset defines the *time grid* (timestamps) and frame indices.
- We keep only sparse keyframes as anchors and use i2v to recover missing frames.
- Adjacent segments share the boundary anchor (tail == next head), so the merged
  sequence can be made 1:1 aligned with the original dataset indices.
- Inference outputs for each segment remain "complete" (includes both anchors).
  De-duplication is handled in the merge stage.

Key parameters:
- fps: output FPS (also used as dataset_fps for index/time mapping)
- kf_gap: keyframe gap in frames (must be >0; recommended kf_gap%4==0 for r=4)

Outputs:
  <exp_dir>/infer/runs/<run_xxx>/...
  <exp_dir>/infer/runs_plan.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import subprocess
from collections import deque
from pathlib import Path
from datetime import datetime
from _common import ensure_dir, ensure_file, list_frames_sorted, write_json_atomic


ALLOW_PREFIX = ("[PROG]", "[INFO]", "[WARN]", "[ERR]", "[BAR]", "[PROMPT]")


def _samefile_safe(src: Path, dst: Path) -> bool:
    """Python3.7-safe samefile check with abspath fallback."""
    src_s = os.path.abspath(str(src))
    dst_s = os.path.abspath(str(dst))
    try:
        return os.path.samefile(src_s, dst_s)
    except Exception:
        return src_s == dst_s


def _resolve_frames_dir(segment_dir: Path) -> Path:
    # Accept either dataset root (contains frames/) or frames/ directly.
    if segment_dir.name == "frames":
        return segment_dir
    frames = segment_dir / "frames"
    if frames.is_dir():
        return frames
    return segment_dir


def _torchrun_cmd(gpus: int) -> list:
    if gpus <= 1:
        return []
    import shutil

    tr = shutil.which("torchrun") or "torchrun"
    return [tr, "--nproc_per_node", str(gpus)]


def _run_filtered(cmd: list, cwd: Path, env: dict) -> int:
    """Run command, forward only selected lines to terminal."""
    p = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None

    tail = deque(maxlen=250)
    for line in p.stdout:
        tail.append(line)
        clean_line = line.strip()
        if any(clean_line.startswith(p) for p in ALLOW_PREFIX):
            sys.stdout.write(clean_line + "\n")
            sys.stdout.flush()  # CRITICAL: Force flush the pipe to the outer Runner

    rc = p.wait()
    if rc != 0:
        sys.stderr.write(f"[ERR] infer failed (rc={rc}). Showing last {len(tail)} lines (unfiltered):\n")
        for line in tail:
            sys.stderr.write(line)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment_dir", required=True, help="ExpHub segment dir (contains frames/) or frames dir")
    ap.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (will create infer/ subdir inside)")
    ap.add_argument("--videox_root", default="/data/hx/VideoX-Fun", help="VideoX-Fun repo root")
    ap.add_argument("--python", default="python3", help="Python executable (run inside videox env)")

    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--fps", type=int, default=24, help="Target FPS (also used as dataset_fps)")
    ap.add_argument(
        "--kf_gap",
        type=int,
        default=0,
        help=(
            "Keyframe gap in frames. If 0, choose the largest multiple of 4 <= fps (r=4 safe default). "
            "Example: fps=24 -> kf_gap=24 (1s)."
        ),
    )
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--num_segments", type=int, default=0, help="If >0, cap segments to this value")
    ap.add_argument("--seed_base", type=int, default=43)
    ap.add_argument(
        "--prompt_manifest",
        default="",
        help="Optional prompt manifest json; archived under <exp_dir>/prompt/manifest.json",
    )

    # forward extras after `--`
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    segment_dir = Path(args.segment_dir).resolve()
    ensure_dir(segment_dir, "segment_dir")
    frames_dir = _resolve_frames_dir(segment_dir)
    ensure_dir(frames_dir, "segment/frames")

    frames_avail = len(list_frames_sorted(frames_dir))
    if frames_avail <= 0:
        raise SystemExit(f"[ERR] segment has no frames: {frames_dir}")

    fps = int(args.fps)
    if fps <= 0:
        raise SystemExit("[ERR] --fps must be > 0")

    # r=4 safe default: choose gap as multiple of 4.
    kf_gap = int(args.kf_gap)
    if kf_gap <= 0:
        kf_gap = fps - (fps % 4)
        if kf_gap <= 0:
            raise SystemExit("[ERR] cannot auto-pick kf_gap; provide --kf_gap")

    if kf_gap <= 0:
        raise SystemExit("[ERR] --kf_gap must be > 0")
    if kf_gap % 4 != 0:
        sys.stdout.write(f"[WARN] kf_gap={kf_gap} is not divisible by 4 (r=4); video_length may be truncated by the model\n")

    base_idx = int(args.base_idx)
    if base_idx < 0:
        raise SystemExit("[ERR] --base_idx must be >= 0")
    if base_idx >= frames_avail:
        raise SystemExit(f"[ERR] base_idx={base_idx} out of range (frames_avail={frames_avail})")

    # Derive max segments that can be formed with anchors [s, s+kf_gap] within available frames.
    max_segments = (frames_avail - 1 - base_idx) // kf_gap
    if max_segments <= 0:
        raise SystemExit(
            f"[ERR] not enough frames for even 1 segment: frames_avail={frames_avail} base_idx={base_idx} kf_gap={kf_gap}"
        )

    segments = max_segments
    if int(args.num_segments) > 0:
        segments = min(segments, int(args.num_segments))

    used_end_idx = base_idx + segments * kf_gap
    used_frames = used_end_idx - base_idx + 1
    tail_drop = frames_avail - (base_idx + used_frames)
    if tail_drop < 0:
        tail_drop = 0

    sys.stdout.write(
        f"[WARN] segment has {frames_avail} frames (base_idx={base_idx}); plan uses {used_frames} frames, tail_drop={tail_drop}\n"
        if tail_drop > 0
        else f"[INFO] segment has {frames_avail} frames (base_idx={base_idx}); plan uses {used_frames} frames\n"
    )

    # Keep metadata consistent: one segment spans (kf_gap / fps) seconds on the dataset time grid.
    segment_seconds = float(kf_gap) / float(fps)

    exp_dir = Path(args.exp_dir).resolve()
    prompt_dir = exp_dir / "prompt"
    prompt_manifest_std = prompt_dir / "manifest.json"
    infer_dir = exp_dir / "infer"
    runs_root = infer_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    videox_root = Path(args.videox_root).resolve()
    ensure_dir(videox_root, "videox_root")
    this_dir = Path(__file__).resolve().parent
    predict_script = this_dir / "_infer_i2v_impl.py"
    ensure_file(predict_script, "predict_script")

    if args.prompt_manifest:
        src_manifest = Path(args.prompt_manifest).resolve()
        ensure_file(src_manifest, "prompt_manifest")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        if _samefile_safe(src_manifest, prompt_manifest_std):
            sys.stdout.write(
                "[INFO] prompt_manifest already at standard path, skip copy: {}\n".format(prompt_manifest_std)
            )
        else:
            shutil.copy2(str(src_manifest), str(prompt_manifest_std))
            sys.stdout.write(
                "[INFO] prompt_manifest copied to standard path: {} -> {}\n".format(src_manifest, prompt_manifest_std)
            )

    if not prompt_manifest_std.is_file():
        raise SystemExit(
            "[ERR] missing prompt manifest: {}. Run prompt step first or provide --prompt_manifest.".format(
                prompt_manifest_std
            )
        )

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(videox_root) + (os.pathsep + old_pp if old_pp else "")

    # Build cmd
    base = _torchrun_cmd(args.gpus)
    if base:
        cmd = base + [str(predict_script)]
    else:
        cmd = [args.python, str(predict_script)]

    cmd += [
        "--gpus",
        str(args.gpus),
        "--batch",
        "--frames_dir",
        str(frames_dir),
        "--dataset_fps",
        str(fps),
        "--fps",
        str(fps),
        "--kf_gap",
        str(kf_gap),
        "--segment_seconds",
        f"{segment_seconds:.9f}",
        "--base_idx",
        str(base_idx),
        "--num_segments",
        str(segments),
        "--seed_base",
        str(int(args.seed_base)),
    ]

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    # ExpHub layout: runs live under <exp_dir>/infer/runs; plan lives under <exp_dir>/infer/runs_plan.json
    cmd += ["--runs_parent", str(infer_dir), "--exp_name", "runs"]

    t0 = time.time()
    sys.stdout.write(
        f"[PROG] infer start: segments={segments} fps={fps} kf_gap={kf_gap} used_frames={used_frames} gpus={args.gpus}\n"
    )
    rc = _run_filtered(cmd, cwd=videox_root, env=env)
    if rc != 0:
        raise SystemExit(rc)
    dt = time.time() - t0

    runs_plan = ensure_file(infer_dir / "runs_plan.json", "runs_plan")
    plan_bytes = runs_plan.read_bytes()
    try:
        plan_obj = json.loads(plan_bytes.decode("utf-8"))
        plan_segments = int(len(plan_obj.get("segments", []) or []))
    except Exception:
        plan_segments = int(segments)
    step_meta = {
        "step": "infer_i2v",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": int(args.gpus),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "frames_avail": int(frames_avail),
        "segments": int(plan_segments),
        "used_frames": int(used_frames),
        "runs_plan_path": str(runs_plan),
        "runs_plan_size": int(len(plan_bytes)),
        "runs_plan_sha1": hashlib.sha1(plan_bytes).hexdigest(),
    }
    write_json_atomic(infer_dir / "step_meta.json", step_meta, indent=2)

    sys.stdout.write(f"[TIME] infer finished: {dt:.2f}s\n")
    sys.stdout.write(f"[INFO] step meta written: {infer_dir / 'step_meta.json'}\n")


if __name__ == "__main__":
    main()
