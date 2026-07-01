from __future__ import annotations

"""Eval mainline: SLAM -> evo_ape trajectory metrics -> summary artifacts."""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, remove_path
from exphub.common.logging import log_info, log_prog
from exphub.config import get_platform_config
from exphub.eval.evo_eval import run_evo_eval
from exphub.eval.slam_run import run_slam
from exphub.eval.summary_build import build_eval_summary


_STALE_EVAL_OUTPUTS = (
    "eval_slam_report.json",
    "internal/eval_slam_report.json",
    "evo_ori_ape.zip",
    "evo_rec_ape.zip",
    "ori/evo_ape.zip",
    "rec/evo_ape.zip",
    "ori/traj_est.npz",
    "rec/traj_est.npz",
    "evo_ori_stdout.txt",
    "evo_ori_stderr.txt",
    "evo_rec_stdout.txt",
    "evo_rec_stderr.txt",
    "evo_failure.log",
    "eval_metrics_overview.png",
    "trajectory_plot_data.json",
    "compression_downstream",
    "compression_benchmark",
    "gen",
)


def _clean_eval_outputs(out_dir):
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in _STALE_EVAL_OUTPUTS:
        remove_path(root / name)


def _validate_eval_inputs(args):
    return {
        "prepare_result": ensure_file(args.prepare_result, "prepare result"),
        "prepare_frames_dir": ensure_dir(args.prepare_frames_dir, "prepare frames dir"),
        "generation_units": ensure_file(args.generation_units, "generation units"),
        "prompts": ensure_file(args.prompts, "prompts"),
        "encode_result": Path(args.encode_result).resolve() if str(args.encode_result or "").strip() else None,
        "decode_frames_dir": ensure_dir(args.decode_frames_dir, "decode frames dir"),
        "decode_calib": ensure_file(args.decode_calib, "decode calib"),
        "decode_timestamps": ensure_file(args.decode_timestamps, "decode timestamps"),
        "decode_report": ensure_file(args.decode_report, "decode report"),
    }


def run_eval_mainline(args):
    eval_started = time.time()
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    inputs = _validate_eval_inputs(args)
    gt_traj_path = Path(args.gt_traj).resolve()
    if not gt_traj_path.is_file():
        raise RuntimeError("missing prepare/gt_traj.tum; rerun prepare or manually materialize the known GT artifact")
    _clean_eval_outputs(out_dir)

    log_prog("eval mainline: slam")
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
            "generation_units": str(inputs["generation_units"]),
            "encode_result": str(inputs["encode_result"] or ""),
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

    log_prog("eval mainline: evo_ape")
    evo_result = run_evo_eval(
        {
            "out_dir": str(out_dir),
            "exp_dir": str(exp_dir),
            "gt_traj": str(gt_traj_path),
            "ori_traj": str((out_dir / "ori" / "traj_est.tum").resolve()),
            "rec_traj": str((out_dir / "rec" / "traj_est.tum").resolve()),
            "t_max_diff": float(args.t_max_diff),
            "skip_plots": bool(args.skip_plots),
            "fps": float(args.fps),
            "prepare_result": str(inputs["prepare_result"]),
            "generation_units": str(inputs["generation_units"]),
            "decode_report": str(inputs["decode_report"]),
        }
    )

    log_prog("eval mainline: summary")
    summary_result = build_eval_summary(
        {
            "exp_dir": str(exp_dir),
            "out_dir": str(out_dir),
            "prepare_result": str(inputs["prepare_result"]),
            "generation_units": str(inputs["generation_units"]),
            "prompts": str(inputs["prompts"]),
            "encode_result": str(inputs["encode_result"] or ""),
            "prepare_frames_dir": str(inputs["prepare_frames_dir"]),
            "decode_report": str(inputs["decode_report"]),
            "ori_run_meta": str(out_dir / "ori" / "run_meta.json"),
            "rec_run_meta": str(out_dir / "rec" / "run_meta.json"),
            "evo_result": dict(evo_result.get("summary") or {}),
            "eval_runtime_sec": float(time.time() - eval_started),
            "stage_times": json.loads(str(args.stage_times_json or "{}")),
            "complete_main_chain": bool(args.complete_main_chain),
        }
    )

    log_info("eval mainline complete: {}".format(out_dir))
    return {
        "slam": slam_result,
        "evo": evo_result,
        "summary": summary_result,
        "out_dir": out_dir,
    }


def run(runtime, *, droid_live_viewer: bool = False):
    ensure_file(runtime.paths.prepare_result_path, "prepare result")
    ensure_dir(runtime.paths.prepare_frames_dir, "prepare frames dir")
    ensure_file(runtime.paths.encode_generation_units_path, "generation units")
    ensure_file(runtime.paths.encode_prompts_path, "prompts")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    ensure_file(runtime.paths.decode_report_path, "decode report")

    gt_traj_path = runtime.paths.prepare_gt_traj_path
    if not gt_traj_path.is_file():
        raise RuntimeError("missing prepare/gt_traj.tum; rerun prepare or manually materialize the known GT artifact")

    platform_cfg = get_platform_config(exphub_root=runtime.exphub_root)
    droid_repo = str(platform_cfg.get("repos", {}).get("droid_slam", "") or "")
    droid_weights = str(platform_cfg.get("models", {}).get("droid", {}).get("path", "") or "")

    runtime.paths.eval_dir.mkdir(parents=True, exist_ok=True)
    for name in _STALE_EVAL_OUTPUTS:
        runtime.remove_in_exp(runtime.paths.eval_dir / name)

    cmd = [
        "-m",
        "exphub.eval.eval",
        "--run-eval-mainline",
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
        str(runtime.paths.encode_result_path if runtime.paths.encode_result_path.is_file() else ""),
        "--decode_frames_dir",
        str(runtime.paths.decode_frames_dir),
        "--decode_calib",
        str(runtime.paths.decode_calib_path),
        "--decode_timestamps",
        str(runtime.paths.decode_timestamps_path),
        "--decode_report",
        str(runtime.paths.decode_report_path),
        "--gt_traj",
        str(gt_traj_path),
        "--seq",
        "both",
        "--droid_repo",
        droid_repo,
        "--weights",
        droid_weights,
        "--fps",
        runtime.fps_arg,
        "--stage_times_json",
        json.dumps(dict(runtime.step_times), sort_keys=True),
    ]
    if runtime.execution_plan.mode == "infer" and tuple(runtime.execution_plan.stages) == (
        "prepare",
        "encode",
        "decode",
        "eval",
    ):
        cmd.append("--complete_main_chain")
    if not bool(droid_live_viewer):
        cmd.append("--disable_vis")

    runtime.step_runner.run_env_python(
        cmd,
        phase_name="slam",
        log_name="eval.log",
        cwd=runtime.exphub_root,
    )

    required_artifacts = [
        (runtime.paths.eval_canonical_summary_path, "canonical eval summary"),
        (runtime.paths.eval_canonical_summary_csv_path, "canonical eval summary csv"),
        (runtime.paths.eval_trajectory_overlay_path, "trajectory overlay"),
        (runtime.paths.eval_trajectory_interactive_path, "interactive trajectory overlay"),
        (runtime.paths.eval_ori_traj_path, "ORI trajectory"),
        (runtime.paths.eval_rec_traj_path, "REC trajectory"),
        (runtime.paths.eval_ori_run_meta_path, "ORI run meta"),
        (runtime.paths.eval_rec_run_meta_path, "REC run meta"),
        (runtime.paths.eval_evo_ori_ape_path, "ORI evo APE result"),
        (runtime.paths.eval_evo_rec_ape_path, "REC evo APE result"),
    ]
    for artifact_path, label in required_artifacts:
        ensure_file(artifact_path, label)

    return runtime.paths.eval_dir


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-eval-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--encode_result", default="")
    parser.add_argument("--decode_frames_dir", required=True)
    parser.add_argument("--decode_calib", required=True)
    parser.add_argument("--decode_timestamps", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--gt_traj", required=True)
    parser.add_argument("--seq", default="both", choices=["auto", "ori", "rec", "both"])
    parser.add_argument("--droid_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--stage_times_json", default="{}")
    parser.add_argument("--complete_main_chain", action="store_true")
    parser.add_argument("--t_max_diff", type=float, default=0.03)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_eval_mainline:
        raise SystemExit("eval helper requires --run-eval-mainline")
    run_eval_mainline(args)


if __name__ == "__main__":
    main()
