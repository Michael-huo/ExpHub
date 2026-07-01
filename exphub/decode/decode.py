from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file, read_json_dict
from exphub.common.logging import log_info, log_prog
from exphub.config import get_platform_config

from .compression_benchmark import CompressionBenchmarkDecode
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
    runtime.paths.decode_dir.mkdir(parents=True, exist_ok=True)
    for child in list(runtime.paths.decode_dir.iterdir()):
        if child.resolve() == runtime.paths.decode_preview_path.resolve():
            continue
        runtime.remove_in_exp(child)
    runtime.paths.decode_runs_dir.mkdir(parents=True, exist_ok=True)


def _cleanup_successful_runs_dir(runtime):
    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    ensure_file(runtime.paths.decode_preview_path, "decode preview")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    runtime.remove_in_exp(runtime.paths.decode_runs_dir)


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

    run_comfyui_decode_tasks(
        tasks_payload,
        runtime,
        get_platform_config(exphub_root=runtime.exphub_root),
        decode_profile=runtime.config.decode_profile,
        seed_base=runtime.config.seed,
    )

    decode_report = read_json_dict(runtime.paths.decode_report_path)
    if not decode_report:
        raise RuntimeError("invalid decode report: {}".format(runtime.paths.decode_report_path))
    merge_units(runtime, tasks_payload, decode_report)

    _cleanup_successful_runs_dir(runtime)

    return runtime.paths.decode_dir


def run_compression_benchmark_decode_extra(runtime):
    ensure_file(
        runtime.paths.encode_compression_report_path,
        "compression benchmark encode report",
    )
    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    log_prog("decode compression benchmark: materialize frame sources")
    report = CompressionBenchmarkDecode(
        exp_dir=runtime.paths.exp_dir,
        output_dir=runtime.paths.decode_compression_dir,
        encode_report_path=runtime.paths.encode_compression_report_path,
        prepare_frames_dir=runtime.paths.prepare_frames_dir,
        native_decode_frames_dir=runtime.paths.decode_frames_dir,
        native_decode_report_path=runtime.paths.decode_report_path,
        fps=runtime.spec.fps,
        exphub_root=runtime.exphub_root,
    ).run()
    ensure_file(
        runtime.paths.decode_compression_report_path,
        "compression benchmark decode report",
    )
    return report
