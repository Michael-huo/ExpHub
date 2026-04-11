from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import read_json_dict, write_json_atomic, write_text_atomic
from exphub.eval.report_build import (
    build_summary_text,
    log_eval_terminal_summary,
    save_metrics_overview,
    write_eval_details,
)
from exphub.eval.trajectory_eval import run_traj_eval


def _resolve_traj_inputs(exp_dir, slam_report_path):
    slam_report = read_json_dict(slam_report_path)
    tracks = dict(slam_report.get("tracks") or {})

    reference_rel = str(slam_report.get("reference_trajectory_path", "") or "")
    estimate_rel = str(slam_report.get("primary_trajectory_path", "") or "")
    if not reference_rel and isinstance(tracks.get("ori"), dict):
        reference_rel = str(tracks["ori"].get("traj_path", "") or "")
    if not estimate_rel and isinstance(tracks.get("gen"), dict):
        estimate_rel = str(tracks["gen"].get("traj_path", "") or "")

    reference_path = (Path(exp_dir).resolve() / reference_rel).resolve() if reference_rel else None
    estimate_path = (Path(exp_dir).resolve() / estimate_rel).resolve() if estimate_rel else None
    return {
        "reference": reference_path,
        "estimate": estimate_path,
        "reference_name": str(slam_report.get("reference_track", "") or "ori"),
        "estimate_name": str(slam_report.get("primary_track", "") or "gen"),
    }


def run_metrics_substage(args):
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slam_report_path = Path(args.slam_report).resolve()
    traj_inputs = _resolve_traj_inputs(exp_dir, slam_report_path)
    traj_result = run_traj_eval(
        {
            "reference": str(traj_inputs["reference"]) if traj_inputs["reference"] is not None else "",
            "estimate": str(traj_inputs["estimate"]) if traj_inputs["estimate"] is not None else "",
            "out_dir": str(out_dir),
            "reference_name": str(traj_inputs["reference_name"]),
            "estimate_name": str(traj_inputs["estimate_name"]),
            "alignment_mode": str(args.alignment_mode),
            "delta": float(args.delta),
            "delta_unit": str(args.delta_unit),
            "t_max_diff": float(args.t_max_diff),
            "t_offset": float(args.t_offset),
            "skip_plots": bool(args.skip_plots),
        }
    )

    traj_metrics = dict((traj_result or {}).get("metrics") or {})
    summary_text = build_summary_text(traj_metrics)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    traj_metrics_path = metrics_dir / "traj_eval.json"
    write_json_atomic(traj_metrics_path, traj_metrics, indent=2)
    details_path = write_eval_details(out_dir, (traj_result or {}).get("records", []))
    metrics_overview_path = save_metrics_overview(out_dir, (traj_result or {}).get("overview", {}))
    summary_path = out_dir / "summary.txt"
    write_text_atomic(summary_path, str(summary_text or "") + "\n")
    log_eval_terminal_summary(traj_metrics, out_dir)

    return {
        "traj_result": traj_result,
        "traj_metrics": traj_metrics,
        "summary_text": summary_text,
        "artifacts": {
            "traj_metrics_path": traj_metrics_path,
            "details_path": details_path,
            "metrics_overview_path": metrics_overview_path,
            "traj_plot_path": Path(str((traj_result or {}).get("artifacts", {}).get("traj_xy_plot", "") or "")),
            "summary_path": summary_path,
        },
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--slam_report", required=True)
    parser.add_argument("--alignment_mode", default="se3", choices=["none", "se3", "sim3", "origin"])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--delta_unit", default="frames", choices=["frames", "meters", "seconds"])
    parser.add_argument("--t_max_diff", type=float, default=0.01)
    parser.add_argument("--t_offset", type=float, default=0.0)
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("eval metrics helper requires --run-formal-mainline")
    run_metrics_substage(args)


if __name__ == "__main__":
    main()
