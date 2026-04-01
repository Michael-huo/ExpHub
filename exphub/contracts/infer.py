from __future__ import annotations

from .common import StageContract


RUNS_PLAN = "runs_plan"


def build_contract(paths):
    return StageContract(
        stage="infer",
        root=paths.infer_dir,
        artifacts={
            "runs_dir": paths.infer_runs_dir,
            "runs_plan": paths.infer_runs_plan_path,
            "report": paths.infer_report_path,
        },
    )
