from __future__ import annotations

from .common import StageContract


BASE_PROMPT = "base_prompt"
STATE_PROMPT_MANIFEST = "state_prompt_manifest"
RUNTIME_PROMPT_PLAN = "runtime_prompt_plan"


def build_contract(paths):
    return StageContract(
        stage="prompt",
        root=paths.prompt_dir,
        artifacts={
            "base_prompt": paths.prompt_base_path,
            "state_prompt_manifest": paths.prompt_dir / "state_prompt_manifest.json",
            "runtime_prompt_plan": paths.prompt_runtime_plan_path,
            "report": paths.prompt_report_path,
        },
    )
