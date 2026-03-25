#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract and visualize candidate raw signals for segment-side manual inspection."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scripts._common import log_info
from scripts._segment.signal_extraction import (
    DEFAULT_PLOT_SMOOTH_WINDOW,
    extract_signal_timeseries,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract and visualize raw segment signals for an existing experiment.")
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir (expects segment/frames and segment/timestamps.txt)")
    parser.add_argument("--plot_smooth_window", type=int, default=DEFAULT_PLOT_SMOOTH_WINDOW, help="moving-average window used only for plotted curves")
    return parser


def run_signal_extraction(argv=None):
    args = build_arg_parser().parse_args(argv)
    payload = extract_signal_timeseries(args.exp_dir, plot_smooth_window=args.plot_smooth_window)
    log_info("signal extraction done: output_dir={}".format(payload["output_dir"]))
    return {
        "csv_path": str(payload["csv_path"]),
        "meta_path": str(payload["meta_path"]),
        "output_dir": str(payload["output_dir"]),
    }


if __name__ == "__main__":
    run_signal_extraction()
