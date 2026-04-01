from __future__ import annotations

import shlex

from exphub.common.io import ensure_dir, ensure_file
from exphub.contracts import infer as infer_contract


def run(runtime):
    """Formal infer stage entry with legacy backend bridging kept inside."""
    contract = infer_contract.build_contract(runtime.paths)
    ensure_file(runtime.paths.prompt_runtime_plan_path, "runtime prompt plan")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(runtime.paths.infer_dir)
    infer_phase = runtime.infer_phase_name()
    runtime.phase_python(infer_phase)

    cmd = [
        "python",
        str(runtime.script_path("infer_i2v.py")),
        "--segment_dir",
        str(runtime.paths.segment_dir),
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--videox_root",
        str(runtime.args.videox_root),
        "--gpus",
        str(runtime.args.gpus),
        "--fps",
        runtime.fps_arg,
        "--kf_gap",
        str(runtime.spec.kf_gap),
        "--base_idx",
        str(runtime.args.base_idx),
        "--seed_base",
        str(runtime.args.seed_base),
        "--prompt_file",
        str(runtime.paths.prompt_runtime_plan_path),
        "--infer_backend",
        str(runtime.args.infer_backend),
        "--infer_model_dir",
        str(runtime.args.infer_model_dir),
        "--backend_python_phase",
        str(infer_phase),
    ]
    if runtime.args.infer_extra:
        cmd.extend(shlex.split(runtime.args.infer_extra))

    runtime.step_runner.run_env_python(cmd, phase_name=infer_phase, log_name="infer.log", cwd=runtime.exphub_root)

    ensure_dir(contract.artifacts["runs_dir"], "infer runs dir")
    ensure_file(contract.artifacts["runs_plan"], "infer runs plan")
    ensure_file(contract.artifacts["report"], "infer report")
    return contract.artifacts["report"]
