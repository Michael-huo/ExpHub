from __future__ import annotations

from .common import StageContract


INPUT_SLAM_REPORT = "slam_report"
REPORT = "report"
METRICS_DIR = "metrics_dir"
PLOTS_DIR = "plots_dir"
DETAILS = "details"
TRAJ_METRICS = "traj_metrics"


def build_contract(paths):
    return StageContract(
        stage="eval",
        root=paths.eval_dir,
        artifacts={
            INPUT_SLAM_REPORT: paths.slam_report_path,
            REPORT: paths.eval_report_path,
            METRICS_DIR: paths.eval_metrics_dir,
            PLOTS_DIR: paths.eval_plots_dir,
            DETAILS: paths.eval_details_path,
            TRAJ_METRICS: paths.eval_metrics_dir / "traj_eval.json",
        },
    )
