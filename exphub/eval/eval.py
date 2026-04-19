from __future__ import annotations

"""Native eval mainline: SLAM -> trajectory metrics -> summary artifacts."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, remove_path
from exphub.common.logging import log_info, log_prog, log_warn
from exphub.eval.slam_run import run_slam
from exphub.eval.summary_build import build_eval_summary
from exphub.eval.trajectory_eval import run_trajectory_eval


_STALE_EVAL_OUTPUTS = (
    "eval_slam_report.json",
    "eval_traj_report.json",
    "eval_compression_report.json",
    "eval_summary.txt",
    "eval_details.csv",
    "eval_traj_xy.png",
    "eval_metrics_overview.png",
)


def _clean_eval_outputs(out_dir):
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in _STALE_EVAL_OUTPUTS:
        remove_path(root / name)


def _validate_native_inputs(args):
    return {
        "prepare_result": ensure_file(args.prepare_result, "prepare result"),
        "prepare_frames_dir": ensure_dir(args.prepare_frames_dir, "prepare frames dir"),
        "generation_units": ensure_file(args.generation_units, "generation units"),
        "prompts": ensure_file(args.prompts, "prompts"),
        "encode_result": ensure_file(args.encode_result, "encode result"),
        "decode_frames_dir": ensure_dir(args.decode_frames_dir, "decode frames dir"),
        "decode_calib": ensure_file(args.decode_calib, "decode calib"),
        "decode_timestamps": ensure_file(args.decode_timestamps, "decode timestamps"),
        "decode_report": ensure_file(args.decode_report, "decode report"),
        "decode_merge_report": ensure_file(args.decode_merge_report, "decode merge report"),
    }


def run_native_mainline(args):
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    inputs = _validate_native_inputs(args)
    _clean_eval_outputs(out_dir)

    log_prog("eval native mainline: slam")
    slam_result = run_slam(
        {
            "exp_dir": str(exp_dir),
            "out_dir": str(out_dir),
            "prepare_result": str(inputs["prepare_result"]),
            "prepare_frames_dir": str(inputs["prepare_frames_dir"]),
            "decode_frames_dir": str(inputs["decode_frames_dir"]),
            "decode_calib": str(inputs["decode_calib"]),
            "decode_timestamps": str(inputs["decode_timestamps"]),
            "decode_report": str(inputs["decode_report"]),
            "decode_merge_report": str(inputs["decode_merge_report"]),
            "generation_units": str(inputs["generation_units"]),
            "encode_result": str(inputs["encode_result"]),
            "seq": str(args.seq),
            "droid_repo": str(args.droid_repo),
            "weights": str(args.weights),
            "fps": float(args.fps),
            "disable_vis": bool(args.disable_vis),
            "t0": 0,
            "stride": 1,
            "max_frames": 0,
            "undistort_mode": "auto",
            "resize_interp": "linear",
            "intr_scale_mode": "demo",
            "buffer": 512,
            "image_size": [240, 320],
            "beta": 0.3,
            "filter_thresh": 1.5,
            "warmup": 12,
            "keyframe_thresh": 2.0,
            "frontend_thresh": 12.0,
            "frontend_window": 25,
            "frontend_radius": 2,
            "frontend_nms": 1,
            "backend_thresh": 20.0,
            "backend_radius": 2,
            "backend_nms": 3,
            "upsample": False,
            "no_tqdm": False,
            "stereo": False,
        }
    )

    log_prog("eval native mainline: trajectory")
    traj_result = run_trajectory_eval(
        {
            "exp_dir": str(exp_dir),
            "out_dir": str(out_dir),
            "prepare_result": str(inputs["prepare_result"]),
            "generation_units": str(inputs["generation_units"]),
            "reference": str((out_dir / "ori" / "traj_est.tum").resolve()),
            "estimate": str((out_dir / "gen" / "traj_est.tum").resolve()),
            "reference_name": "ori",
            "estimate_name": "gen",
            "alignment_mode": "se3",
            "delta": 1.0,
            "delta_unit": "frames",
            "t_max_diff": 0.01,
            "t_offset": 0.0,
            "skip_plots": bool(args.skip_plots),
        }
    )

    log_prog("eval native mainline: summary")
    summary_result = build_eval_summary(
        {
            "exp_dir": str(exp_dir),
            "out_dir": str(out_dir),
            "prepare_result": str(inputs["prepare_result"]),
            "generation_units": str(inputs["generation_units"]),
            "prompts": str(inputs["prompts"]),
            "encode_result": str(inputs["encode_result"]),
            "prepare_frames_dir": str(inputs["prepare_frames_dir"]),
            "decode_report": str(inputs["decode_report"]),
            "decode_merge_report": str(inputs["decode_merge_report"]),
            "slam_report": str(out_dir / "eval_slam_report.json"),
            "traj_report": str(out_dir / "eval_traj_report.json"),
            "traj_records": list((traj_result or {}).get("records") or []),
            "traj_overview": dict((traj_result or {}).get("overview") or {}),
        }
    )

    log_info("eval native mainline complete: {}".format(out_dir))
    return {
        "slam": slam_result,
        "trajectory": traj_result,
        "summary": summary_result,
        "out_dir": out_dir,
    }


def run(runtime):
    ensure_file(runtime.paths.prepare_result_path, "prepare result")
    ensure_dir(runtime.paths.prepare_frames_dir, "prepare frames dir")
    ensure_file(runtime.paths.encode_generation_units_path, "generation units")
    ensure_file(runtime.paths.encode_prompts_path, "prompts")
    ensure_file(runtime.paths.encode_result_path, "encode result")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_file(runtime.paths.decode_merge_report_path, "decode merge report")

    runtime.paths.eval_dir.mkdir(parents=True, exist_ok=True)
    for name in _STALE_EVAL_OUTPUTS:
        runtime.remove_in_exp(runtime.paths.eval_dir / name)

    cmd = [
        "-m",
        "exphub.eval.eval",
        "--run-native-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--out_dir",
        str(runtime.paths.eval_dir),
        "--prepare_result",
        str(runtime.paths.prepare_result_path),
        "--prepare_frames_dir",
        str(runtime.paths.prepare_frames_dir),
        "--generation_units",
        str(runtime.paths.encode_generation_units_path),
        "--prompts",
        str(runtime.paths.encode_prompts_path),
        "--encode_result",
        str(runtime.paths.encode_result_path),
        "--decode_frames_dir",
        str(runtime.paths.decode_frames_dir),
        "--decode_calib",
        str(runtime.paths.decode_calib_path),
        "--decode_timestamps",
        str(runtime.paths.decode_timestamps_path),
        "--decode_report",
        str(runtime.paths.decode_report_path),
        "--decode_merge_report",
        str(runtime.paths.decode_merge_report_path),
        "--seq",
        str(runtime.args.droid_seq),
        "--droid_repo",
        str(runtime.args.droid_repo),
        "--weights",
        str(runtime.args.droid_weights),
        "--fps",
        runtime.fps_arg,
    ]
    if not runtime.viz_enable:
        cmd.append("--disable_vis")
    if runtime.args.no_viz:
        cmd.append("--skip_plots")

    runtime.step_runner.run_env_python(
        cmd,
        phase_name="slam",
        log_name="eval.log",
        cwd=runtime.exphub_root,
    )

    required_artifacts = [
        (runtime.paths.eval_slam_report_path, "eval slam report"),
        (runtime.paths.eval_traj_report_path, "eval traj report"),
        (runtime.paths.eval_compression_report_path, "eval compression report"),
        (runtime.paths.eval_summary_path, "eval summary"),
        (runtime.paths.eval_details_path, "eval details"),
        (runtime.paths.eval_traj_plot_path, "eval traj plot"),
        (runtime.paths.eval_metrics_overview_path, "eval metrics overview plot"),
    ]
    for artifact_path, label in required_artifacts:
        if label == "eval traj plot" and runtime.args.no_viz:
            if not Path(artifact_path).is_file():
                log_warn("eval traj plot skipped by --no_viz")
                continue
        ensure_file(artifact_path, label)

    return runtime.paths.eval_dir


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-native-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--decode_frames_dir", required=True)
    parser.add_argument("--decode_calib", required=True)
    parser.add_argument("--decode_timestamps", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--decode_merge_report", required=True)
    parser.add_argument("--seq", default="both", choices=["auto", "ori", "gen", "both"])
    parser.add_argument("--droid_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_native_mainline:
        raise SystemExit("eval helper requires --run-native-mainline")
    run_native_mainline(args)


if __name__ == "__main__":
    main()
