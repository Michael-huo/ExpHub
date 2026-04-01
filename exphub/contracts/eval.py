from __future__ import annotations

from .common import StageContract


EVAL_REPORT = "eval_report"


def build_contract(paths):
    return StageContract(
        stage="eval",
        root=paths.eval_dir,
        artifacts={
            "report": paths.eval_artifact_path("report.json"),
            "details": paths.eval_artifact_path("details.csv"),
            "traj_xy_plot": paths.eval_dir / "plots" / "traj_xy.png",
            "metrics_overview_plot": paths.eval_dir / "plots" / "metrics_overview.png",
        },
    )
