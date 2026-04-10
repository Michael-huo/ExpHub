from __future__ import annotations

from .common import StageContract


RUNS_PLAN = "runs_plan"
RUNS_DIR = "runs_dir"
REPORT = "report"


def build_contract(paths, decode_source="aligned"):
    return StageContract(
        stage="infer",
        root=paths.infer_source_dir(decode_source),
        artifacts={
            RUNS_DIR: paths.infer_runs_source_dir(decode_source),
            RUNS_PLAN: paths.infer_runs_plan_source_path(decode_source),
            REPORT: paths.infer_report_source_path(decode_source),
        },
    )
