#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a minimal baseline state segmentation on top of signal extraction outputs."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scripts._common import log_info, log_prog
from scripts._segment.state_segmentation import (
    DEFAULT_ENTER_TH,
    DEFAULT_EXIT_TH,
    DEFAULT_GLITCH_MERGE_LEN,
    DEFAULT_MIN_HIGH_LEN,
    DEFAULT_MIN_LOW_LEN,
    DEFAULT_NORMALIZATION_METHOD,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_WEIGHTS,
    run_state_segmentation,
    save_state_segmentation_plots,
    write_state_segmentation_outputs,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build a simple, explainable state segmentation baseline from signal_extraction outputs."
    )
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (expects segment/signal_extraction/signal_timeseries.csv)")
    parser.add_argument(
        "--normalization_method",
        default=DEFAULT_NORMALIZATION_METHOD,
        choices=["robust_zscore_per_signal", "zscore_per_signal"],
        help="per-signal normalization method before smoothing and weighted fusion",
    )
    parser.add_argument("--smoothing_window", type=int, default=DEFAULT_SMOOTHING_WINDOW, help="moving-average window used only for state segmentation")
    parser.add_argument("--enter_th", type=float, default=DEFAULT_ENTER_TH, help="enter threshold for low->high transition")
    parser.add_argument("--exit_th", type=float, default=DEFAULT_EXIT_TH, help="exit threshold for high->low transition")
    parser.add_argument("--min_high_len", type=int, default=DEFAULT_MIN_HIGH_LEN, help="consecutive frames required above enter_th before entering high_state")
    parser.add_argument("--min_low_len", type=int, default=DEFAULT_MIN_LOW_LEN, help="consecutive frames required below exit_th before leaving high_state")
    parser.add_argument("--glitch_merge_len", type=int, default=DEFAULT_GLITCH_MERGE_LEN, help="minimum segment length after segmentation; shorter runs are merged away")
    parser.add_argument("--motion_weight", type=float, default=DEFAULT_WEIGHTS["motion_velocity"], help="weight for motion_velocity in state_score")
    parser.add_argument("--feature_weight", type=float, default=DEFAULT_WEIGHTS["feature_motion"], help="weight for feature_motion in state_score")
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
        weights={
            "motion_velocity": args.motion_weight,
            "feature_motion": args.feature_weight,
        },
    )
    io_paths = write_state_segmentation_outputs(result)
    plot_paths = save_state_segmentation_plots(
        output_dir=result["output_dir"],
        frame_rows=result["frame_rows"],
        segments=result["segments"],
        enter_th=args.enter_th,
        exit_th=args.exit_th,
    )
    log_info("state segmentation csv: {}".format(io_paths["csv_path"]))
    log_info("state segmentation json: {}".format(io_paths["json_path"]))
    log_info("state segmentation meta: {}".format(io_paths["meta_path"]))
    log_info("state segmentation overview: {}".format(plot_paths["overview_path"]))
    log_info("state segmentation overlay: {}".format(plot_paths["overlay_path"]))
    log_prog("state segmentation done: {}".format(result["output_dir"]))
    return {
        "output_dir": str(result["output_dir"]),
        "csv_path": str(io_paths["csv_path"]),
        "json_path": str(io_paths["json_path"]),
        "meta_path": str(io_paths["meta_path"]),
        "overview_path": str(plot_paths["overview_path"]),
        "overlay_path": str(plot_paths["overlay_path"]),
    }


if __name__ == "__main__":
    run_state_segmentation_cli()
