from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file
from exphub.contracts import prompt as prompt_contract


def run(runtime):
    """Formal prompt stage entry with a temporary legacy-script bridge."""
    contract = prompt_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.paths.prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_phase = runtime.prompt_phase_name()
    if not runtime.phase_python(prompt_phase):
        raise RuntimeError("missing prompt phase config in config/platform.yaml")

    cmd = [
        "python",
        str(runtime.script_path("prompt_gen.py")),
        "--frames_dir",
        str(runtime.paths.segment_frames_dir),
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--fps",
        runtime.fps_arg,
        "--backend",
        str(runtime.args.prompt_backend),
        "--model_dir",
        str(runtime.prompt_model_ref()),
        "--dtype",
        str(runtime.args.prompt_dtype),
        "--sample_mode",
        str(runtime.args.prompt_sample_mode),
        "--num_images",
        str(runtime.args.prompt_num_images),
        "--backend_python_phase",
        str(prompt_phase),
    ]
    runtime.step_runner.run_env_python(cmd, phase_name=prompt_phase, log_name="prompt.log", cwd=runtime.exphub_root)

    ensure_file(contract.artifacts["report"], "prompt report")
    ensure_file(contract.artifacts["base_prompt"], "prompt base_prompt")
    ensure_file(contract.artifacts["state_prompt_manifest"], "state prompt manifest")
    ensure_file(contract.artifacts["runtime_prompt_plan"], "runtime prompt plan")
    return contract.artifacts["report"]
