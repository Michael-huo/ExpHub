from __future__ import annotations

from .common import StageContract


SLAM_REPORT = "slam_report"
REPORT = "report"
SUMMARY = "summary"
DETAILS = "details"
TRAJ_METRICS = "traj_metrics"
TRAJ_PLOT = "traj_plot"
METRICS_OVERVIEW_PLOT = "metrics_overview_plot"


def build_contract(paths):
    return StageContract(
        stage="eval",
        root=paths.eval_dir,
        artifacts={
            SLAM_REPORT: paths.slam_report_path,
            REPORT: paths.eval_report_path,
            SUMMARY: paths.eval_summary_path,
            DETAILS: paths.eval_details_path,
            TRAJ_METRICS: paths.eval_traj_metrics_path,
            TRAJ_PLOT: paths.eval_traj_plot_path,
            METRICS_OVERVIEW_PLOT: paths.eval_metrics_overview_path,
        },
    )
