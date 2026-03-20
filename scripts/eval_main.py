#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from _eval.api import run_eval
from _eval.traj_eval import add_traj_eval_args


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    add_traj_eval_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_eval(
        exp_dir=Path(args.exp_dir).resolve(),
        reference=args.reference,
        estimate=args.estimate,
        out_dir=args.out_dir,
        reference_name=args.reference_name,
        estimate_name=args.estimate_name,
        alignment_mode=args.alignment_mode,
        delta=args.delta,
        delta_unit=args.delta_unit,
        t_max_diff=args.t_max_diff,
        t_offset=args.t_offset,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
