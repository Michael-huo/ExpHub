from __future__ import annotations

from .common import StageContract


SEGMENT_REPORT = "segment_report"
PROMPT_REPORT = "prompt_report"
INFER_REPORT = "infer_report"
MERGE_REPORT = "merge_report"
SLAM_REPORT = "slam_report"
EVAL_REPORT = "eval_report"
FINAL_REPORT = "final_report"
COMPRESSION = "compression"


def build_contract(paths):
    return StageContract(
        stage="stats",
        root=paths.stats_dir,
        artifacts={
            SEGMENT_REPORT: paths.segment_report_path,
            PROMPT_REPORT: paths.prompt_report_path,
            INFER_REPORT: paths.infer_report_path,
            MERGE_REPORT: paths.merge_report_path,
            SLAM_REPORT: paths.slam_report_path,
            EVAL_REPORT: paths.eval_report_path,
            FINAL_REPORT: paths.stats_report_path,
            COMPRESSION: paths.stats_compression_path,
        },
    )
