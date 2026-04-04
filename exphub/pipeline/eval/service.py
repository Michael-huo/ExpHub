from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_file, read_json_dict
from exphub.common.logging import debug_info, log_warn
from exphub.contracts import eval as eval_contract


def _resolve_traj_inputs(exp_dir):
    slam_report = read_json_dict(Path(exp_dir).resolve() / "slam" / "report.json")
    tracks = dict(slam_report.get("tracks") or {})

    reference_rel = str(slam_report.get("reference_trajectory_path", "") or "")
    estimate_rel = str(slam_report.get("primary_trajectory_path", "") or "")
    if not reference_rel and isinstance(tracks.get("ori"), dict):
        reference_rel = str(tracks["ori"].get("traj_path", "") or "")
    if not estimate_rel and isinstance(tracks.get("gen"), dict):
        estimate_rel = str(tracks["gen"].get("traj_path", "") or "")

    reference_path = (Path(exp_dir).resolve() / reference_rel).resolve() if reference_rel else None
    estimate_path = (Path(exp_dir).resolve() / estimate_rel).resolve() if estimate_rel else None
    reference_name = str(slam_report.get("reference_track", "") or "ori")
    estimate_name = str(slam_report.get("primary_track", "") or "gen")
    return {
        "reference": reference_path,
        "estimate": estimate_path,
        "reference_name": reference_name,
        "estimate_name": estimate_name,
    }


def _run_formal_mainline(args):
    from exphub.pipeline.eval.reporting import build_summary_text, log_eval_terminal_summary, write_eval_artifacts
    from exphub.pipeline.eval.traj_eval import run_traj_eval

    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_inputs = _resolve_traj_inputs(exp_dir)
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
    summary_text = build_summary_text(dict((traj_result or {}).get("metrics") or {}))
    artifacts = write_eval_artifacts(out_dir, traj_result, summary_text)
    log_eval_terminal_summary(dict((traj_result or {}).get("metrics") or {}), out_dir)
    return artifacts


def run(runtime):
    contract = eval_contract.build_contract(runtime.paths)
    ensure_file(contract.artifacts[eval_contract.SLAM_REPORT], "slam report")

    runtime.remove_in_exp(runtime.paths.eval_dir)
    runtime.paths.eval_dir.mkdir(parents=True, exist_ok=True)

    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "eval" / "service.py").resolve()
    cmd = [
        "python",
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--out_dir",
        str(runtime.paths.eval_dir),
        "--alignment_mode",
        "se3",
        "--delta",
        "1.0",
        "--delta_unit",
        "frames",
        "--t_max_diff",
        "0.01",
        "--t_offset",
        "0.0",
    ]
    if runtime.args.no_viz:
        cmd.append("--skip_plots")

    runtime.step_runner.run_env_python(
        cmd,
        phase_name="slam",
        log_name="eval.log",
        cwd=runtime.exphub_root,
        check=False,
    )

    required_artifacts = [
        (eval_contract.REPORT, "eval report"),
        (eval_contract.SUMMARY, "eval summary"),
        (eval_contract.DETAILS, "eval details"),
        (eval_contract.TRAJ_METRICS, "eval traj metrics"),
        (eval_contract.METRICS_OVERVIEW_PLOT, "eval metrics overview plot"),
    ]
    for artifact_key, label in required_artifacts:
        artifact_path = Path(contract.artifacts[artifact_key])
        if artifact_path.is_file():
            debug_info("STEP eval: {}={}".format(label, artifact_path))
        else:
            log_warn("{} missing: {}".format(label, artifact_path))

    traj_plot_path = Path(contract.artifacts[eval_contract.TRAJ_PLOT])
    if traj_plot_path.is_file():
        debug_info("STEP eval: eval traj plot={}".format(traj_plot_path))
    elif not runtime.args.no_viz:
        log_warn("eval traj plot missing: {}".format(traj_plot_path))
    return contract.root


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
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
        raise SystemExit("eval service helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
