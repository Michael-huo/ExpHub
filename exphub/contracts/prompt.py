from __future__ import annotations

from .common import StageContract


PROMPT_MANIFEST = "prompt_manifest"
REPORT = "report"


def build_contract(paths):
    return StageContract(
        stage="prompt",
        root=paths.prompt_dir,
        artifacts={
            PROMPT_MANIFEST: paths.prompt_manifest_path,
            REPORT: paths.prompt_report_path,
        },
    )
