from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file, read_json_dict
from exphub.common.logging import log_info, log_prog
from exphub.config import get_platform_config

from .comfyui_client import run_comfyui_decode_tasks
from .task_build import build_generation_tasks
from .unit_merge import merge_units


def _ensure_native_inputs(runtime):
    ensure_file(runtime.paths.prepare_result_path, "prepare result")
    ensure_dir(runtime.paths.prepare_frames_dir, "prepare frames dir")
    ensure_file(runtime.paths.encode_generation_units_path, "generation units")
    ensure_file(runtime.paths.encode_prompts_path, "prompts")
    ensure_file(runtime.paths.encode_result_path, "encode result")


def _remove_decode_outputs(runtime):
    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    if runtime.paths.decode_dir.exists():
        runtime.remove_in_exp(runtime.paths.decode_dir)
    runtime.paths.decode_dir.mkdir(parents=True, exist_ok=True)
    runtime.paths.decode_runs_dir.mkdir(parents=True, exist_ok=True)


def run(runtime):
    _ensure_native_inputs(runtime)
    _remove_decode_outputs(runtime)

    log_prog("decode native mainline: build generation tasks")
    tasks_payload = build_generation_tasks(runtime)
    log_info(
        "decode generation tasks: count={} range={}..{}".format(
            int(tasks_payload["summary"]["task_count"]),
            int(tasks_payload["summary"]["start_idx"]),
            int(tasks_payload["summary"]["end_idx"]),
        )
    )

    run_comfyui_decode_tasks(tasks_payload, runtime, get_platform_config(exphub_root=runtime.exphub_root))

    decode_report = read_json_dict(runtime.paths.decode_report_path)
    if not decode_report:
        raise RuntimeError("invalid decode report: {}".format(runtime.paths.decode_report_path))
    merge_units(runtime, tasks_payload, decode_report)

    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_file(runtime.paths.decode_merge_report_path, "decode merge report")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    return runtime.paths.decode_dir
