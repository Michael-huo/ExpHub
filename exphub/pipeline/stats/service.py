from __future__ import annotations

from exphub.common.io import ensure_file
from exphub.contracts import stats as stats_contract


def run(runtime):
    """Formal stats stage entry; implementation remains in the legacy script for now."""
    contract = stats_contract.build_contract(runtime.paths)
    cmd = [
        "python",
        str(runtime.script_path("stats_collect.py")),
        "--exp_dir",
        str(runtime.paths.exp_dir),
    ]
    runtime.step_runner.run_env_python(cmd, phase_name="prompt", log_name="stats.log", cwd=runtime.exphub_root)
    ensure_file(contract.artifacts["report"], "stats report")
    return contract.artifacts["report"]
