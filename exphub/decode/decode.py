from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog

from .task_build import build_decode_tasks
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


def _write_task_snapshot(runtime, tasks_payload):
    snapshot = dict(tasks_payload)
    snapshot.pop("_raw", None)
    write_json_atomic(runtime.paths.decode_dir / "decode_tasks.json", snapshot, indent=2)
    return runtime.paths.decode_dir / "decode_tasks.json"


def _run_frame_generation(runtime, tasks_path):
    cmd = [
        "-m",
        "exphub.decode.frames_generate",
        "--run-native-tasks",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--prepare_frames_dir",
        str(runtime.paths.prepare_frames_dir),
        "--tasks",
        str(tasks_path),
        "--run_id",
        str(runtime.spec.exp_name),
        "--videox_root",
        str(runtime.args.videox_root),
        "--gpus",
        str(runtime.args.gpus),
        "--fps",
        runtime.fps_arg,
        "--kf_gap",
        str(runtime.spec.kf_gap),
        "--seed_base",
        str(runtime.args.seed_base),
        "--infer_backend",
        str(runtime.args.infer_backend),
        "--infer_model_dir",
        str(runtime.args.infer_model_dir),
        "--backend_python_phase",
        str(runtime.infer_phase_name()),
    ]
    if runtime.args.infer_extra:
        cmd.extend(["--infer_extra", str(runtime.args.infer_extra)])
    runtime.step_runner.run_env_python(
        cmd,
        phase_name=runtime.infer_phase_name(),
        log_name="infer.log",
        cwd=runtime.exphub_root,
    )


def run(runtime):
    _ensure_native_inputs(runtime)
    _remove_decode_outputs(runtime)

    log_prog("decode native mainline: build tasks")
    tasks_payload = build_decode_tasks(runtime)
    tasks_path = _write_task_snapshot(runtime, tasks_payload)
    log_info(
        "decode tasks: count={} range={}..{}".format(
            int(tasks_payload["summary"]["task_count"]),
            int(tasks_payload["summary"]["start_idx"]),
            int(tasks_payload["summary"]["end_idx"]),
        )
    )

    _run_frame_generation(runtime, tasks_path)
    decode_report = read_json_dict(runtime.paths.decode_report_path)
    if not decode_report:
        raise RuntimeError("invalid decode report: {}".format(runtime.paths.decode_report_path))
    merge_units(runtime, tasks_payload, decode_report)

    ensure_file(runtime.paths.decode_report_path, "decode report")
    ensure_file(runtime.paths.decode_plan_path, "decode eval compatibility plan")
    ensure_file(runtime.paths.decode_merge_report_path, "decode merge report")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    return runtime.paths.decode_dir
