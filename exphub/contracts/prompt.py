from __future__ import annotations

from .common import StageContract


BASE_PROMPT = "base_prompt"
STATE_PROMPT_MANIFEST = "state_prompt_manifest"
RUNTIME_PROMPT_PLAN = "runtime_prompt_plan"
REPORT = "report"

PROMPT_INTERNAL_ARTIFACT_KEYS = (
    BASE_PROMPT,
    STATE_PROMPT_MANIFEST,
)

FORMAL_PROMPT_DOWNSTREAM_ARTIFACT_KEYS = (
    RUNTIME_PROMPT_PLAN,
)

FORMAL_PROMPT_ARTIFACT_KEYS = (
    RUNTIME_PROMPT_PLAN,
    REPORT,
)

PROMPT_STAGE_ARTIFACT_KEYS = (
    BASE_PROMPT,
    STATE_PROMPT_MANIFEST,
    RUNTIME_PROMPT_PLAN,
    REPORT,
)


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
