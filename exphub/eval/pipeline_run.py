from __future__ import annotations

"""Unified eval stage for slam + metrics + diagnostics."""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file
from exphub.common.logging import debug_info, log_warn
from exphub.eval.summary_build import run_diagnostics_substage
from exphub.eval.metrics_build import run_metrics_substage
from exphub.eval.slam_run import run_slam_substage


def _build_slam_args(args):
    return argparse.Namespace(
        exp_dir=args.exp_dir,
        out_dir=str(Path(args.out_dir).resolve()),
        segment_dir=args.segment_dir,
        prepare_result=args.prepare_result,
        generation_units=args.generation_units,
        encode_result=args.encode_result,
        infer_dir=args.infer_dir,
        infer_report=args.infer_report,
        merge_dir=args.merge_dir,
        merge_report=args.merge_report,
        merge_manifest=args.merge_manifest,
        seq=args.seq,
        droid_repo=args.droid_repo,
        weights=args.weights,
        t0=0,
        stride=1,
        fps=float(args.fps),
        undistort_mode="auto",
        resize_interp="linear",
        intr_scale_mode="demo",
        buffer=512,
        image_size=[240, 320],
        disable_vis=bool(args.disable_vis),
        beta=0.3,
        filter_thresh=1.5,
        warmup=12,
        keyframe_thresh=2.0,
        frontend_thresh=12.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=20.0,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
        max_frames=0,
        no_tqdm=False,
        stereo=False,
    )


def _build_metrics_args(args, slam_report_path):
    return argparse.Namespace(
        exp_dir=args.exp_dir,
        out_dir=args.out_dir,
        slam_report=str(slam_report_path),
        alignment_mode="se3",
        delta=1.0,
        delta_unit="frames",
        t_max_diff=0.01,
        t_offset=0.0,
        skip_plots=bool(args.skip_plots),
    )


def _build_diagnostics_args(args, slam_report_path, metrics_result):
    artifacts = dict((metrics_result or {}).get("artifacts") or {})
    return argparse.Namespace(
        exp_dir=args.exp_dir,
        out_dir=args.out_dir,
        slam_report=str(slam_report_path),
        prepare_result=str(args.prepare_result),
        generation_units=str(args.generation_units),
        prompts=str(args.prompts),
        encode_result=str(args.encode_result),
        infer_report=str(args.infer_report),
        merge_report=str(args.merge_report),
        merge_manifest=str(args.merge_manifest),
        traj_metrics=str(artifacts.get("traj_metrics_path")),
        summary=str(artifacts.get("summary_path")),
        details=str(artifacts.get("details_path")),
    )


def _run_formal_mainline(args):
    slam_result = run_slam_substage(_build_slam_args(args))
    slam_report_path = Path(slam_result["report_path"]).resolve()
    metrics_result = run_metrics_substage(_build_metrics_args(args, slam_report_path))
    diagnostics_result = run_diagnostics_substage(_build_diagnostics_args(args, slam_report_path, metrics_result))
    return {
        "slam": slam_result,
        "metrics": metrics_result,
        "diagnostics": diagnostics_result,
    }


def run(runtime):
    ensure_dir(runtime.paths.prepare_dir, "prepare dir")
    ensure_dir(runtime.paths.decode_dir, "decode dir")
    ensure_file(runtime.paths.prepare_result_path, "prepare result")
    ensure_file(runtime.paths.encode_generation_units_path, "generation units")
    ensure_file(runtime.paths.encode_prompts_path, "prompts")
    ensure_file(runtime.paths.encode_result_path, "encode result")
    ensure_file(runtime.paths.decode_plan_path, "decode plan")
    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_file(runtime.paths.decode_merge_report_path, "decode merge report")

    runtime.remove_in_exp(runtime.paths.eval_dir)
    runtime.paths.eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "-m",
        "exphub.eval.pipeline_run",
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--out_dir",
        str(runtime.paths.eval_dir),
        "--segment_dir",
        str(runtime.paths.prepare_dir),
        "--prepare_result",
        str(runtime.paths.prepare_result_path),
        "--generation_units",
        str(runtime.paths.encode_generation_units_path),
        "--prompts",
        str(runtime.paths.encode_prompts_path),
        "--encode_result",
        str(runtime.paths.encode_result_path),
        "--infer_dir",
        str(runtime.paths.decode_dir),
        "--infer_report",
        str(runtime.paths.decode_report_path),
        "--merge_dir",
        str(runtime.paths.decode_dir),
        "--merge_report",
        str(runtime.paths.decode_merge_report_path),
        "--merge_manifest",
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
        (runtime.paths.eval_slam_primary_traj_path, "eval slam primary trajectory"),
        (runtime.paths.eval_report_path, "eval report"),
        (runtime.paths.eval_compression_report_path, "eval compression report"),
        (runtime.paths.eval_summary_path, "eval summary"),
        (runtime.paths.eval_details_path, "eval details"),
        (runtime.paths.eval_traj_report_path, "eval traj report"),
        (runtime.paths.eval_metrics_overview_path, "eval metrics overview plot"),
    ]
    for artifact_path, label in required_artifacts:
        artifact_path = Path(artifact_path)
        ensure_file(artifact_path, label)
        debug_info("STEP eval: {}={}".format(label, artifact_path))

    traj_plot_path = Path(runtime.paths.eval_traj_plot_path)
    if traj_plot_path.is_file():
        debug_info("STEP eval: eval traj plot={}".format(traj_plot_path))
    elif not runtime.args.no_viz:
        log_warn("eval traj plot missing: {}".format(traj_plot_path))
    return runtime.paths.eval_dir


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--infer_dir", required=True)
    parser.add_argument("--infer_report", required=True)
    parser.add_argument("--merge_dir", required=True)
    parser.add_argument("--merge_report", required=True)
    parser.add_argument("--merge_manifest", required=True)
    parser.add_argument("--seq", default="both")
    parser.add_argument("--droid_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("eval service helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
