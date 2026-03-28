#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the formal state segmentation pipeline."""

import argparse

from scripts._common import log_info, log_prog
from scripts._segment.state_segmentation import (
    DEFAULT_ENTER_TH,
    DEFAULT_EXIT_TH,
    DEFAULT_GLITCH_MERGE_LEN,
    DEFAULT_MIN_HIGH_LEN,
    DEFAULT_MIN_LOW_LEN,
    DEFAULT_NORMALIZATION_METHOD,
    DEFAULT_SMOOTHING_WINDOW,
    run_state_segmentation,
    save_state_segmentation_plots,
    write_state_segmentation_outputs,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build the current formal state segmentation output from motion_velocity and semantic_velocity. "
            "The default output is a low_state/high_state interval sequence produced by the high-risk interval detector."
        )
    )
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument(
        "--normalization_method",
        default=DEFAULT_NORMALIZATION_METHOD,
        help="fallback only; persisted formal state inputs are reused when available",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="fallback only; persisted formal state inputs are reused when available",
    )
    parser.add_argument("--enter_th", type=float, default=DEFAULT_ENTER_TH, help="detector score threshold for entering a high-risk interval")
    parser.add_argument("--exit_th", type=float, default=DEFAULT_EXIT_TH, help="detector score threshold for leaving a high-risk interval")
    parser.add_argument("--min_high_len", type=int, default=DEFAULT_MIN_HIGH_LEN, help="minimum high-risk interval duration in frames")
    parser.add_argument("--min_low_len", type=int, default=DEFAULT_MIN_LOW_LEN, help="minimum trailing baseline context in frames when detector statistics are built")
    parser.add_argument("--glitch_merge_len", type=int, default=DEFAULT_GLITCH_MERGE_LEN, help="merge short detector gaps between nearby high-risk intervals")
    parser.add_argument("--motion_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--feature_weight", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def run_state_segmentation_cli(argv=None):
    args = build_arg_parser().parse_args(argv)
    log_prog("state segmentation start: exp_dir={}".format(args.exp_dir))
    result = run_state_segmentation(
        exp_dir=args.exp_dir,
        normalization_method=args.normalization_method,
        smoothing_window=args.smoothing_window,
        enter_th=args.enter_th,
        exit_th=args.exit_th,
        min_high_len=args.min_high_len,
        min_low_len=args.min_low_len,
        glitch_merge_len=args.glitch_merge_len,
    )
    io_paths = write_state_segmentation_outputs(result)
    plot_paths = save_state_segmentation_plots(
        output_dir=result["output_dir"],
        frame_rows=result["frame_rows"],
        segments=result["segments"],
        enter_th=args.enter_th,
        exit_th=args.exit_th,
    )
    log_info("state segmentation json: {}".format(io_paths["json_path"]))
    log_info("state segmentation report: {}".format(io_paths["report_path"]))
    log_info("state segmentation overview: {}".format(plot_paths["overview_path"]))
    log_prog("state segmentation done: {}".format(result["output_dir"]))
    return {
        "output_dir": str(result["output_dir"]),
        "json_path": str(io_paths["json_path"]),
        "report_path": str(io_paths["report_path"]),
        "overview_path": str(plot_paths["overview_path"]),
    }


if __name__ == "__main__":
    run_state_segmentation_cli()
