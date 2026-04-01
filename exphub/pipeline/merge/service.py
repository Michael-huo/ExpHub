from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file
from exphub.contracts import merge as merge_contract


def run(runtime):
    """Formal merge stage entry with temporary command-level compatibility."""
    contract = merge_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.infer_runs_dir, "infer runs dir")
    ensure_file(runtime.paths.infer_runs_plan_path, "infer runs plan")

    runtime.remove_in_exp(runtime.paths.merge_dir)
    infer_phase = runtime.infer_phase_name()

    cmd = [
        "python",
        str(runtime.script_path("merge_seq.py")),
        "--segment_dir",
        str(runtime.paths.segment_dir),
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--runs_root",
        str(runtime.paths.infer_runs_dir),
        "--plan",
        str(runtime.paths.infer_runs_plan_path),
        "--out_dir",
        str(runtime.paths.merge_dir),
    ]
    runtime.step_runner.run_env_python(cmd, phase_name=infer_phase, log_name="merge.log", cwd=runtime.exphub_root)

    ensure_dir(contract.artifacts["frames_dir"], "merge frames dir")
    ensure_file(contract.artifacts["calib"], "merge calib")
    ensure_file(contract.artifacts["timestamps"], "merge timestamps")
    return contract.root
