from __future__ import annotations

from .common import StageContract


INPUT_RUNS_PLAN = "runs_plan"
INPUT_MERGE_MANIFEST = "merge_manifest"
INPUT_SLAM_REPORT = "slam_report"
REPORT = "report"
METRICS_DIR = "metrics_dir"
PLOTS_DIR = "plots_dir"
DETAILS = "details"
TRAJ_METRICS = "traj_metrics"
IMAGE_METRICS = "image_metrics"
SLAM_METRICS = "slam_metrics"


def build_contract(paths):
    return StageContract(
        stage="eval",
        root=paths.eval_dir,
        artifacts={
            INPUT_RUNS_PLAN: paths.infer_runs_plan_path,
            INPUT_MERGE_MANIFEST: paths.merge_manifest_path,
            INPUT_SLAM_REPORT: paths.slam_report_path,
            REPORT: paths.eval_report_path,
            METRICS_DIR: paths.eval_metrics_dir,
            PLOTS_DIR: paths.eval_plots_dir,
            DETAILS: paths.eval_details_path,
            TRAJ_METRICS: paths.eval_metrics_dir / "traj_eval.json",
            IMAGE_METRICS: paths.eval_metrics_dir / "image_eval.json",
            SLAM_METRICS: paths.eval_metrics_dir / "slam_eval.json",
        },
    )
