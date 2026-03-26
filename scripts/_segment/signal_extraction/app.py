#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract and visualize candidate raw signals for segment-side manual inspection."""

import argparse

from scripts._common import log_info


_DEFAULT_PLOT_SMOOTH_WINDOW = 5


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract and visualize raw segment signals for an existing experiment.")
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (expects segment/frames and segment/timestamps.txt)")
    parser.add_argument("--plot_smooth_window", type=int, default=_DEFAULT_PLOT_SMOOTH_WINDOW, help="moving-average window used only for plotted curves")
    return parser


def run_signal_extraction(argv=None):
    args = build_arg_parser().parse_args(argv)
    from scripts._segment.signal_extraction import extract_signal_timeseries

    payload = extract_signal_timeseries(args.exp_dir, plot_smooth_window=args.plot_smooth_window)
    log_info("signal extraction done: output_dir={}".format(payload["output_dir"]))
    return {
        "csv_path": str(payload["csv_path"]),
        "meta_path": str(payload["meta_path"]),
        "output_dir": str(payload["output_dir"]),
    }


if __name__ == "__main__":
    run_signal_extraction()
