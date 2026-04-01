from __future__ import annotations

from .common import StageContract


RUNS_PLAN = "runs_plan"
RUNS_DIR = "runs_dir"
REPORT = "report"

FORMAL_INFER_ARTIFACT_KEYS = (
    RUNS_DIR,
    RUNS_PLAN,
    REPORT,
)


def build_contract(paths):
    return StageContract(
        stage="infer",
        root=paths.infer_dir,
        artifacts={
            RUNS_DIR: paths.infer_runs_dir,
            RUNS_PLAN: paths.infer_runs_plan_path,
            REPORT: paths.infer_report_path,
        },
    )
