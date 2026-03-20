#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ExpHub merger for Wan2.2(VideoX-Fun) batch runs.

Goals:
- Only depends on ExpHub experiment layout:
    <exp_dir>/infer/runs/<run_xxx>/{frames/, params.json}
- Produce a "dataset-like" merged directory for downstream SLAM:
    <exp_dir>/merge/frames/*.png
    <exp_dir>/merge/calib.txt
    <exp_dir>/merge/timestamps.txt
    <exp_dir>/merge/merge_meta.json

Notes:
- This script does NOT call VideoX-Fun/my_ws and does not import VideoX-Fun modules.
- It merges runs strictly by a runs plan file (default: <exp_dir>/infer/runs_plan.json), then concatenating frames in order.
This prevents stale runs from previous executions from being accidentally merged.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from _common import ensure_cmd, ensure_dir, ensure_file, list_frames_sorted, log_err, log_info, log_prog, log_warn, write_json_atomic


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _try_ffmpeg_make_video(frames_dir: Path, fps: int, out_mp4: Path) -> bool:
    """
    Try: ffmpeg -r <fps> -i %06d.png -pix_fmt yuv420p out.mp4
    Works if merged frames are named 000000.png ...
    """
    ffmpeg = ensure_cmd("ffmpeg", required=False)
    if not ffmpeg:
        return False

    # Only supports our naming
    pattern = str(frames_dir / "%06d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        str(fps),
        "-i",
        pattern,
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
        )
        if proc.returncode != 0:
            details = ""
            if proc.stdout:
                lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
                if lines:
                    details = " | ".join(lines[-3:])
            if details:
                log_warn("ffmpeg preview failed rc={} details={}".format(proc.returncode, details))
            else:
                log_warn("ffmpeg preview failed rc={}".format(proc.returncode))
            return False
        return out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception:
        return False


def _python_make_video(frames_dir: Path, fps: int, out_mp4: Path) -> bool:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        return False

    frames = list_frames_sorted(frames_dir)
    if not frames:
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(str(out_mp4), fps=fps)
        try:
            for p in frames:
                writer.append_data(imageio.imread(str(p)))
        finally:
            writer.close()
        return out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception:
        return False

def _guard_safe_out_dir(out_dir: Path, exp_dir: Path, runs_root: Path) -> None:
    """Protect merge cleanup from deleting outside the expected experiment scope."""
    merge_root = (exp_dir / "merge").resolve()
    out_dir_r = out_dir.resolve()
    runs_root_r = runs_root.resolve()

    try:
        out_dir_r.relative_to(merge_root)
    except ValueError:
        log_err(
            "unsafe out_dir: {} is outside expected scope {}. Refusing to delete.".format(
                out_dir_r, merge_root
            )
        )
        raise SystemExit(2)

    try:
        out_dir_r.relative_to(runs_root_r)
        log_err(
            "unsafe out_dir: {} overlaps runs_root {}; refusing to delete.".format(
                out_dir_r, runs_root_r
            )
        )
        raise SystemExit(2)
    except ValueError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment_dir", required=True, help="Source ExpHub segment (for calib/timestamps copy)")
    ap.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (expects infer/runs inside)")
    ap.add_argument("--runs_root", default="", help="Override runs_root (default: <exp_dir>/infer/runs)")
    ap.add_argument("--plan", default="", help="Runs plan json (default: <exp_dir>/infer/runs_plan.json)")
    ap.add_argument("--out_dir", default="", help="Override merged out_dir (default: <exp_dir>/merge)")
    ap.add_argument("--fps", type=int, default=0, help="Override output fps (0 => use runs' target_fps)")
    ap.add_argument("--no_preview", action="store_true", help="Skip preview mp4 encoding")
    args = ap.parse_args()

    segment_dir = Path(args.segment_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()

    infer_dir = exp_dir / "infer"
    runs_root = Path(args.runs_root).resolve() if args.runs_root else (infer_dir / "runs")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "merge")

    ensure_dir(runs_root, "runs_root")


    # ---- load runs plan (no legacy scan) ----
    plan_path = Path(args.plan).resolve() if args.plan else (infer_dir / "runs_plan.json")
    ensure_file(plan_path, "runs plan")

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as e:
        log_err("failed to read runs plan: {} ({})".format(plan_path, e))
        raise SystemExit(2)

    segs = plan.get("segments") or []
    if not isinstance(segs, list) or len(segs) == 0:
        log_err("invalid runs plan (no segments): {}".format(plan_path))
        raise SystemExit(2)

    # segments are the source of truth (start/end indices define overlap)
    for s in segs:
        rn = (s or {}).get("run_name", "")
        if not rn:
            log_err("invalid segment entry (missing run_name): {}".format(s))
            raise SystemExit(2)
        if "start_idx" not in s or "end_idx" not in s:
            log_err("invalid segment entry (missing start_idx/end_idx): {}".format(s))
            raise SystemExit(2)

    # output fps
    if args.fps > 0:
        out_fps = args.fps
    else:
        out_fps = _safe_int(plan.get("fps", 0), 0)
        if out_fps <= 0:
            # fallback to first run params
            pp = runs_root / str(segs[0]["run_name"]) / "params.json"
            if pp.exists():
                try:
                    out_fps = _safe_int(json.loads(pp.read_text(encoding="utf-8")).get("target_fps", 25), 25)
                except Exception:
                    out_fps = 25
            else:
                out_fps = 25

    out_frames = out_dir / "frames"
    preview_mp4 = out_dir / "preview.mp4"

    # overwrite (wipe merged dir to avoid stale artifacts)
    _guard_safe_out_dir(out_dir, exp_dir, runs_root)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_frames.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge frames (segment order) with de-duplication based on index overlap
    # Overlap in frames = prev_end_idx - cur_start_idx + 1 (>=0)
    g = 0
    prev_end = None
    merged_start_idx = _safe_int(plan.get("base_idx", 0), 0)
    merged_end_idx = None

    for s in segs:
        rn = str(s["run_name"])
        cur_start = _safe_int(s.get("start_idx", 0), 0)
        cur_end = _safe_int(s.get("end_idx", 0), 0)
        merged_end_idx = cur_end

        run_dir = runs_root / rn
        frames_dir = run_dir / "frames"
        ensure_dir(frames_dir, "frames_dir")
        frames = list_frames_sorted(frames_dir)
        if not frames:
            log_err("no frames in {} (run={})".format(frames_dir, run_dir))
            raise SystemExit(2)

        skip = 0
        if prev_end is not None:
            overlap = prev_end - cur_start + 1
            if overlap > 0:
                skip = overlap

        if skip >= len(frames):
            log_err(
                "overlap too large: skip={} >= frames={} for run={} (prev_end={}, cur_start={})".format(
                    skip, len(frames), rn, prev_end, cur_start
                )
            )
            raise SystemExit(2)

        for src in frames[skip:]:
            dst = out_frames / f"{g:06d}.png"
            shutil.copy2(src, dst)
            g += 1

        prev_end = cur_end

    # calib (required)
    src_calib = segment_dir / "calib.txt"
    ensure_file(src_calib, "calib.txt")
    (out_dir / "calib.txt").write_text(src_calib.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    # timestamps: strictly slice from dataset timestamps to keep 1:1 alignment.
    merged_count = len(list_frames_sorted(out_frames))
    src_ts = segment_dir / "timestamps.txt"
    dst_ts = out_dir / "timestamps.txt"
    ensure_file(src_ts, "timestamps.txt")

    lines = [x for x in src_ts.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    if merged_end_idx is None:
        log_err("cannot infer merged_end_idx from plan: {}".format(plan_path))
        raise SystemExit(2)
    expected_count = merged_end_idx - merged_start_idx + 1
    if expected_count != merged_count:
        log_err(
            "merged_count mismatch: merged_count={} but plan implies {} (start={}, end={}).".format(
                merged_count, expected_count, merged_start_idx, merged_end_idx
            )
        )
        raise SystemExit(2)
    if len(lines) < merged_start_idx + merged_count:
        log_err(
            "dataset timestamps too short: have={} need>={} (start={}, count={})".format(
                len(lines), merged_start_idx + merged_count, merged_start_idx, merged_count
            )
        )
        raise SystemExit(2)

    slice_lines = lines[merged_start_idx : merged_start_idx + merged_count]
    # Re-zero to 0.0 for downstream SLAM convenience.
    try:
        t0 = float(slice_lines[0])
        out_lines = [f"{(float(x) - t0):.9f}" for x in slice_lines]
    except Exception:
        # Fallback: keep as-is
        out_lines = slice_lines
    dst_ts.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # preview
    preview_ok = False
    if not args.no_preview and merged_count > 0:
        preview_ok = _try_ffmpeg_make_video(out_frames, out_fps, preview_mp4)
        if not preview_ok:
            preview_ok = _python_make_video(out_frames, out_fps, preview_mp4)

    meta = {
        "segment_dir": str(segment_dir),
        "runs_root": str(runs_root),
        "plan": str(plan_path),
        "out_dir": str(out_dir),
        "fps": int(out_fps),
        "frame_count": int(merged_count),
        "runs": int(len(segs)),
        "merged_start_idx": int(merged_start_idx),
        "merged_end_idx": int(merged_end_idx) if merged_end_idx is not None else None,
        "schedule_source": str(plan.get("schedule_source", "")),
        "execution_backend": str(plan.get("execution_backend", "")),
        "preview": str(preview_mp4) if preview_ok else "",
        "created_at": datetime.now().isoformat(),
    }
    (out_dir / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    plan_bytes = plan_path.read_bytes()
    step_meta = {
        "step": "merge_seq",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merged_frame_count": int(merged_count),
        "timestamps_count": int(len(out_lines)),
        "merged_segments": int(len(segs)),
        "schedule_source": str(plan.get("schedule_source", "")),
        "execution_backend": str(plan.get("execution_backend", "")),
        "runs_plan_path": str(plan_path),
        "runs_plan_sha1": hashlib.sha1(plan_bytes).hexdigest(),
    }
    write_json_atomic(out_dir / "step_meta.json", step_meta, indent=2)

    log_prog("merge summary: merged_frames={} segments={}".format(int(merged_count), int(len(segs))))
    log_info("out_dir: {}".format(out_dir))
    log_info("frames: {} (count={})".format(out_frames, merged_count))
    log_info("root_files: {}/*.txt/*.json".format(out_dir))
    log_info("step_meta: {}".format(out_dir / "step_meta.json"))
    if preview_ok:
        log_info("preview_mp4: {}".format(preview_mp4))


if __name__ == "__main__":
    main()
