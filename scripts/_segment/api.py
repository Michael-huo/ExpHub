#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os

from scripts._common import log_info, log_prog, log_warn

from . import extract, materialize
from .policies.uniform import build_uniform_keyframe_plan


SUPPORTED_POLICIES = ("uniform",)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Make standardized dataset from rosbag (Image/CompressedImage).")

    ap.add_argument("--bag", required=True, help="rosbag path")
    ap.add_argument("--topic", default="/camera/rgb/image_raw/compressed", help="image topic (sensor_msgs/Image or sensor_msgs/CompressedImage)")
    ap.add_argument("--out_root", required=True, help="output root, e.g. /data/hx/ExpHub/datasets/scand")

    ap.add_argument("--name", default="", help="segment folder name, e.g. scand_seq01_dur16s_w768_h480_fps25_v1")
    ap.add_argument("--dataset", default="scand", help="dataset name (used only when --name not set)")
    ap.add_argument("--seq", default="seq01", help="sequence label (used only when --name not set)")
    ap.add_argument("--version", default="v1", help="version suffix (used only when --name not set)")

    ap.add_argument("--duration", type=float, default=16.0, help="segment duration in seconds")
    ap.add_argument("--fps", type=float, default=25.0, help="target fps for resampling (Hz)")
    ap.add_argument("--start_idx", type=int, default=-1, help="start index in the image topic (0-based). If set, overrides --start_sec")
    ap.add_argument("--start_sec", type=float, default=0.0, help="start time offset (sec) from FIRST image msg time")
    ap.add_argument("--start_abs", type=float, default=None, help="absolute start time in seconds (ROS time). Overrides start_idx/start_sec")

    ap.add_argument("--width", type=int, required=True, help="target width")
    ap.add_argument("--height", type=int, required=True, help="target height")
    ap.add_argument("--interpolation", default="auto", choices=["auto", "area", "lanczos", "linear"], help="resize interpolation")

    ap.add_argument("--calib_in", default="", help="raw calib file (fx fy cx cy [dist...]) for original images")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--dist", nargs="*", default=None, help="optional distortion coeffs")

    ap.add_argument("--png_compress", type=int, default=1, help="0-9, smaller is faster/bigger file")
    ap.add_argument("--strategy", default="nearest", choices=["nearest"], help="resample strategy (v1: nearest)")

    ap.add_argument("--kf_gap", type=int, default=0, help="keyframe gap in frames. if >0, will create keyframes/ folder under dataset root")
    ap.add_argument(
        "--keyframes_mode",
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="how to materialize keyframes in keyframes/: symlink (default, no duplication), hardlink, or copy",
    )
    ap.add_argument(
        "--segment_policy",
        default="uniform",
        choices=list(SUPPORTED_POLICIES),
        help="internal segment policy hook; default keeps legacy uniform behavior",
    )

    ap.add_argument("--dry_run", action="store_true", help="print plan and exit")
    ap.add_argument("--quiet", action="store_true", help="less logs")
    return ap


def _build_segment_name(args):
    dur_tag = int(round(args.duration))
    fps_tag = int(round(args.fps))
    if args.name:
        return args.name
    return "{}_{}_dur{}s_w{}_h{}_fps{}_{}".format(
        args.dataset,
        args.seq,
        dur_tag,
        args.width,
        args.height,
        fps_tag,
        args.version,
    )


def _resolve_output_paths(args):
    ds_name = _build_segment_name(args)
    out_dir = os.path.join(args.out_root, ds_name)
    frames_dir = os.path.join(out_dir, "frames")
    root_dir = out_dir
    return out_dir, frames_dir, root_dir


def _run_dry_run(args, out_dir):
    log_prog("dry run: out_dir={}".format(out_dir))
    log_info("dry run: topic={}".format(args.topic))
    out_cnt = int(math.floor(float(args.duration) * float(args.fps) + 1e-9)) + 1
    log_info("dry run: duration={} fps={} -> frames={}".format(args.duration, args.fps, out_cnt))



def run_segment_make(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.width % 16 != 0 or args.height % 16 != 0:
        log_warn("target size {}x{} is not divisible by 16 (may affect diffusion models)".format(args.width, args.height))

    out_dir, frames_dir, root_dir = _resolve_output_paths(args)
    if args.dry_run:
        _run_dry_run(args, out_dir)
        return

    materialize.ensure_dir(frames_dir)
    materialize.ensure_dir(root_dir)

    ctx = extract.prepare_extraction(args, out_dir, frames_dir, root_dir)
    ts_lines = materialize.materialize_frames(args, ctx, extract.iter_processed_frames(args, ctx))

    if int(args.kf_gap) > 0:
        if args.segment_policy != "uniform":
            raise ValueError("unsupported segment_policy: {}".format(args.segment_policy))
        plan = build_uniform_keyframe_plan(ctx["out_count"], int(args.kf_gap))
        plan["kf_gap"] = int(args.kf_gap)
        materialize.materialize_keyframes(root_dir, frames_dir, plan, args.keyframes_mode)

    materialize.write_step_meta(args, root_dir, frames_dir, ts_lines)

    log_prog("wrote dataset: {}".format(out_dir))
    log_info("frames: {}".format(frames_dir))
    log_info("root: {}".format(root_dir))
    log_info("count: {}".format(ctx["out_count"]))
