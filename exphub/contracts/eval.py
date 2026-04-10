from __future__ import annotations

from .common import StageContract


SLAM_REPORT = "slam_report"
SLAM_PRIMARY_TRAJECTORY = "slam_primary_trajectory"
REPORT = "report"
COMPRESSION = "compression"
SUMMARY = "summary"
DETAILS = "details"
TRAJ_METRICS = "traj_metrics"
TRAJ_PLOT = "traj_plot"
METRICS_OVERVIEW_PLOT = "metrics_overview_plot"


def build_contract(paths, eval_source="aligned"):
    return StageContract(
        stage="eval",
        root=paths.eval_source_dir(eval_source),
        artifacts={
            SLAM_REPORT: paths.eval_slam_report_source_path(eval_source),
            SLAM_PRIMARY_TRAJECTORY: paths.eval_slam_primary_traj_source_path(eval_source),
            REPORT: paths.eval_report_source_path(eval_source),
            COMPRESSION: paths.eval_compression_source_path(eval_source),
            SUMMARY: paths.eval_summary_source_path(eval_source),
            DETAILS: paths.eval_details_source_path(eval_source),
            TRAJ_METRICS: paths.eval_traj_metrics_source_path(eval_source),
            TRAJ_PLOT: paths.eval_traj_plot_source_path(eval_source),
            METRICS_OVERVIEW_PLOT: paths.eval_metrics_overview_source_path(eval_source),
        },
    )
