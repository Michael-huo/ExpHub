from __future__ import annotations

from .common import StageContract


FINAL_REPORT = "final_report"


def build_contract(paths):
    return StageContract(
        stage="stats",
        root=paths.stats_dir,
        artifacts={
            "report": paths.stats_report_path,
            "compression": paths.stats_compression_path,
        },
    )
