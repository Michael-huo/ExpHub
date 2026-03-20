#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from _common import log_err, log_warn
from _eval.image_eval import run_image_eval
from _eval.slam_eval import run_slam_eval
from _eval.summary import log_eval_terminal_summary, write_eval_summary
from _eval.traj_eval import run_traj_eval


def _traj_namespace(
    reference,
    estimate,
    out_dir,
    reference_name,
    estimate_name,
    alignment_mode,
    delta,
    delta_unit,
    t_max_diff,
    t_offset,
    skip_plots,
):
    return argparse.Namespace(
        reference=str(reference),
        estimate=str(estimate),
        out_dir=str(out_dir),
        reference_name=str(reference_name),
        estimate_name=str(estimate_name),
        alignment_mode=str(alignment_mode),
        delta=float(delta),
        delta_unit=str(delta_unit),
        t_max_diff=float(t_max_diff),
        t_offset=float(t_offset),
        skip_plots=bool(skip_plots),
    )


def run_eval(
    exp_dir,
    reference,
    estimate,
    out_dir,
    reference_name="ori",
    estimate_name="gen",
    alignment_mode="se3",
    delta=1.0,
    delta_unit="frames",
    t_max_diff=0.01,
    t_offset=0.0,
    skip_plots=False,
):
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    traj_result = None
    image_result = None
    slam_result = None
    try:
        traj_result = run_traj_eval(
            _traj_namespace(
                reference=reference,
                estimate=estimate,
                out_dir=out_path,
                reference_name=reference_name,
                estimate_name=estimate_name,
                alignment_mode=alignment_mode,
                delta=delta,
                delta_unit=delta_unit,
                t_max_diff=t_max_diff,
                t_offset=t_offset,
                skip_plots=skip_plots,
            ),
            emit_terminal_summary=False,
        )
    except Exception as exc:
        log_err("unexpected traj eval backend failure: {}".format(exc))

    try:
        image_result = run_image_eval(exp_dir=exp_dir, out_dir=out_path)
    except Exception as exc:
        log_warn("image eval backend failure: {}".format(exc))

    try:
        slam_result = run_slam_eval(exp_dir=exp_dir, out_dir=out_path)
    except Exception as exc:
        log_warn("slam-friendly eval backend failure: {}".format(exc))

    write_eval_summary(
        out_path,
        (traj_result or {}).get("metrics", {}),
        (image_result or {}).get("metrics", {}),
        (slam_result or {}).get("metrics", {}),
    )
    log_eval_terminal_summary(
        (traj_result or {}).get("metrics", {}),
        (image_result or {}).get("metrics", {}),
        (slam_result or {}).get("metrics", {}),
        out_path,
    )

    return {
        "traj": traj_result,
        "image": image_result,
        "slam": slam_result,
    }
