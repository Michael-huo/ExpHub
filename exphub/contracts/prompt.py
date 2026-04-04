from __future__ import annotations

from .common import StageContract


BASE_PROMPT = "base_prompt"
STATE_PROMPT_MANIFEST = "state_prompt_manifest"
RUNTIME_PROMPT_PLAN = "runtime_prompt_plan"
REPORT = "report"


def build_contract(paths):
    return StageContract(
        stage="prompt",
        root=paths.prompt_dir,
        artifacts={
            BASE_PROMPT: paths.prompt_base_path,
            STATE_PROMPT_MANIFEST: paths.prompt_state_manifest_path,
            RUNTIME_PROMPT_PLAN: paths.prompt_runtime_plan_path,
            REPORT: paths.prompt_report_path,
        },
    )
