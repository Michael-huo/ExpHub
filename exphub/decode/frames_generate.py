from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from exphub.common.io import ensure_dir, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog


REPORT_FILENAME = "decode_report.json"


@dataclass
class WanTaskRequest(object):
    frames_dir: Path
    exp_dir: Path
    prompt_file_path: Path
    execution_plan_path: Path
    runs_parent: Path
    fps: int
    kf_gap: int
    base_idx: int
    num_segments: int
    seed_base: int
    gpus: int
    schedule_source: str
    execution_backend: str
    execution_segments: List[Dict[str, object]] = field(default_factory=list)
    infer_extra: List[str] = field(default_factory=list)


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _normalize_extra(extra_args):
    text = str(extra_args or "").strip()
    if not text:
        return []
    extra = shlex.split(text)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _task_payload(tasks_payload, backend_name):
    tasks = list(tasks_payload.get("tasks") or [])
    segments = []
    for idx, task in enumerate(tasks):
        start_idx = int(task["start_idx"])
        end_idx = int(task["end_idx"])
        length = int(task["length"])
        segments.append(
            {
                "seg": int(idx),
                "segment_id": int(idx),
                "unit_id": str(task["unit_id"]),
                "source_unit_id": str(task["unit_id"]),
                "source_span_id": str(dict(task.get("source_prompt_ref") or {}).get("span_id", "")),
                "source_prompt_ref": dict(task.get("source_prompt_ref") or {}),
                "run_id": "run_{:03d}".format(idx),
                "run_name": str(task["run_name"]),
                "schedule_source": "decode.native_tasks",
                "execution_backend": str(backend_name),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_start_idx": int(start_idx),
                "raw_end_idx": int(end_idx),
                "desired_start_idx": int(start_idx),
                "desired_end_idx": int(end_idx),
                "desired_num_frames": int(length),
                "aligned_start_idx": int(start_idx),
                "aligned_end_idx": int(end_idx),
                "aligned_num_frames": int(length),
                "deploy_start_idx": int(start_idx),
                "deploy_end_idx": int(end_idx),
                "raw_gap": int(end_idx - start_idx),
                "deploy_gap": int(end_idx - start_idx),
                "num_frames": int(length),
                "target_num_frames": int(length),
                "align_reason": str(task.get("align_reason", "generation_unit_shared_anchor") or "generation_unit_shared_anchor"),
                "is_valid_for_decode": True,
                "is_valid_for_export": bool(task.get("is_valid_for_export", False)),
                "state_label": str(task.get("scene_label", "") or ""),
                "motion_label": str(task.get("motion_label", "") or ""),
                "prompt_source": str(task.get("prompt_source", "prompts.assembled_prompt") or "prompts.assembled_prompt"),
                "base_prompt": str(task.get("base_prompt", "") or ""),
                "resolved_prompt": str(task["prompt"]),
                "negative_prompt": str(task.get("negative_prompt", "") or ""),
                "prompt": str(task["prompt"]),
                "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
                "num_inference_steps": task.get("num_inference_steps"),
                "guidance_scale": task.get("guidance_scale"),
            }
        )
    base_prompt = str(tasks[0].get("base_prompt", "") or tasks[0].get("prompt", "")) if tasks else ""
    negative_prompt = str(tasks[0].get("negative_prompt", "") or "") if tasks else ""
    return {
        "version": 1,
        "schema": "decode_tasks_runtime.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_strategy": "prompts",
        "source": "decode.native_tasks",
        "base_prompt": base_prompt,
        "negative_prompt": negative_prompt,
        "execution_backend": str(backend_name),
        "schedule_source": "decode.native_tasks",
        "source_inputs": dict(tasks_payload.get("source_inputs") or {}),
        "segments": segments,
        "tasks": tasks,
        "summary": dict(tasks_payload.get("summary") or {}),
    }


def _execution_segments(tasks_payload, backend_name):
    return list(_task_payload(tasks_payload, backend_name).get("segments") or [])


def _write_backend_task_payload(decode_dir, tasks_payload, backend_name):
    payload = _task_payload(tasks_payload, backend_name)
    path = Path(decode_dir).resolve() / "decode_tasks_runtime.json"
    write_json_atomic(path, payload, indent=2)
    return path, payload


def _validate_run_output(task, run_dir):
    run_root = ensure_dir(run_dir, "decode unit run dir")
    params_path = run_root / "params.json"
    if not params_path.is_file():
        raise RuntimeError("WAN backend did not write params.json for {}".format(task["unit_id"]))
    frames_dir = ensure_dir(run_root / "frames", "decode unit frames dir")
    frames = list_frames_sorted(frames_dir)
    if not frames:
        raise RuntimeError("WAN backend produced zero frames for {}".format(task["unit_id"]))
    params = read_json_dict(params_path)
    expected = int(params.get("video_length_run", task["length"]) or 0)
    if expected != len(frames):
        raise RuntimeError(
            "decode unit frame count mismatch for {}: params.video_length_run={} files={}".format(
                task["unit_id"],
                expected,
                len(frames),
            )
        )
    if int(params.get("start_idx", task["start_idx"]) or 0) != int(task["start_idx"]):
        raise RuntimeError("decode unit {} params start_idx mismatch".format(task["unit_id"]))
    if int(params.get("end_idx", task["end_idx"]) or 0) != int(task["end_idx"]):
        raise RuntimeError("decode unit {} params end_idx mismatch".format(task["unit_id"]))
    return {
        "params_path": params_path,
        "frames_dir": frames_dir,
        "num_frames": int(len(frames)),
        "params": params,
    }


def _build_decode_plan(exp_dir, tasks_payload, report, fps, kf_gap, seed_base):
    exp_root = Path(exp_dir).resolve()
    per_unit = {str(item.get("unit_id", "")): dict(item) for item in list(report.get("units") or [])}
    segments = []
    for idx, task in enumerate(list(tasks_payload.get("tasks") or [])):
        status = per_unit.get(str(task["unit_id"]), {})
        start_idx = int(task["start_idx"])
        end_idx = int(task["end_idx"])
        length = int(task["length"])
        segments.append(
            {
                "seg": int(idx),
                "segment_id": int(idx),
                "schedule_source": "decode.native_tasks",
                "execution_backend": str(report.get("backend_name", "")),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_start_idx": int(start_idx),
                "raw_end_idx": int(end_idx),
                "desired_start_idx": int(start_idx),
                "desired_end_idx": int(end_idx),
                "desired_num_frames": int(length),
                "aligned_start_idx": int(start_idx),
                "aligned_end_idx": int(end_idx),
                "aligned_num_frames": int(length),
                "actual_saved_start_idx": int(start_idx),
                "actual_saved_end_idx": int(end_idx),
                "actual_saved_frames": int(status.get("num_frames", length) or length),
                "deploy_start_idx": int(start_idx),
                "deploy_end_idx": int(end_idx),
                "raw_gap": int(end_idx - start_idx),
                "deploy_gap": int(end_idx - start_idx),
                "num_frames": int(length),
                "run_id": "run_{:03d}".format(idx),
                "run_name": str(task["run_name"]),
                "source_unit_id": str(task["unit_id"]),
                "source_span_id": str(dict(task.get("source_prompt_ref") or {}).get("span_id", "")),
                "source_prompt_ref": dict(task.get("source_prompt_ref") or {}),
                "target_num_frames": int(length),
                "seed": int(task["seed"]),
                "prompt": str(task["prompt"]),
                "negative_prompt": str(task.get("negative_prompt", "") or ""),
                "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
                "prompt_source": str(task.get("prompt_source", "") or ""),
                "base_prompt": str(task.get("base_prompt", "") or ""),
                "resolved_prompt": str(task["prompt"]),
                "state_label": str(task.get("scene_label", "") or ""),
                "motion_label": str(task.get("motion_label", "") or ""),
                "align_reason": str(task.get("align_reason", "") or ""),
                "is_valid_for_decode": True,
                "is_valid_for_export": bool(task.get("is_valid_for_export", False)),
                "run_params_path": _relative_path(exp_root, status.get("params_path", "")),
                "run_frames_dir": _relative_path(exp_root, status.get("frames_dir", "")),
            }
        )
    return {
        "version": 1,
        "schema": "decode_plan.eval_compat.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_strategy": "prompts",
        "compatibility_only": True,
        "derived_from": "decode.native_tasks",
        "runs_parent": str((exp_root / "decode").resolve()),
        "runs_root": str((exp_root / "decode" / "runs").resolve()),
        "exp_name": "runs",
        "task": "generation_unit",
        "fps": int(fps),
        "dataset_fps": int(fps),
        "kf_gap": int(kf_gap),
        "base_idx": int(segments[0]["start_idx"]) if segments else 0,
        "num_segments": int(len(segments)),
        "seed_base": int(seed_base),
        "schedule_source": "decode.native_tasks",
        "execution_backend": str(report.get("backend_name", "")),
        "source_inputs": dict(tasks_payload.get("source_inputs") or {}),
        "segments": segments,
    }


def generate_tasks(runtime, tasks_payload):
    decode_dir = runtime.paths.decode_dir
    runs_dir = runtime.paths.decode_runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    backend_name = str(runtime.args.infer_backend or "wan_fun_5b_inp").strip().lower()
    if backend_name != "wan_fun_5b_inp":
        raise RuntimeError("decode backend supports only wan_fun_5b_inp: {}".format(backend_name))

    payload_path, task_runtime_payload = _write_backend_task_payload(decode_dir, tasks_payload, backend_name)
    execution_segments = _execution_segments(tasks_payload, backend_name)
    if not execution_segments:
        raise RuntimeError("decode task builder produced zero executable tasks")

    from ._wan_fun_5b_inp import WanFun5BInpBackend

    backend = WanFun5BInpBackend(
        videox_root=str(runtime.args.videox_root),
        model_ref=str(runtime.args.infer_model_dir or ""),
        backend_python_phase=str(runtime.infer_phase_name()),
    )
    backend.load()
    backend_meta = dict(backend.meta() or {})
    request = WanTaskRequest(
        frames_dir=runtime.paths.prepare_frames_dir,
        exp_dir=runtime.paths.exp_dir,
        prompt_file_path=payload_path,
        execution_plan_path=payload_path,
        runs_parent=runtime.paths.decode_dir,
        fps=int(float(runtime.fps_arg)),
        kf_gap=int(runtime.spec.kf_gap),
        base_idx=int(execution_segments[0]["start_idx"]),
        num_segments=int(len(execution_segments)),
        seed_base=int(runtime.args.seed_base),
        gpus=int(runtime.args.gpus),
        schedule_source="decode.native_tasks",
        execution_backend=backend_name,
        execution_segments=execution_segments,
        infer_extra=_normalize_extra(runtime.args.infer_extra),
    )

    log_prog(
        "decode generate: backend={} tasks={} fps={} gpus={}".format(
            backend_name,
            len(execution_segments),
            int(request.fps),
            int(request.gpus),
        )
    )
    started = time.time()
    backend_result = dict(backend.run(request) or {})
    elapsed = float(time.time() - started)

    unit_reports = []
    for task in list(tasks_payload.get("tasks") or []):
        run_dir = Path(task["output_dir"]).resolve()
        actual = _validate_run_output(task, run_dir)
        unit_reports.append(
            {
                "unit_id": str(task["unit_id"]),
                "status": "success",
                "start_idx": int(task["start_idx"]),
                "end_idx": int(task["end_idx"]),
                "length": int(task["length"]),
                "seed": int(task["seed"]),
                "output_dir": _relative_path(runtime.paths.exp_dir, run_dir),
                "frames_dir": _relative_path(runtime.paths.exp_dir, actual["frames_dir"]),
                "params_path": _relative_path(runtime.paths.exp_dir, actual["params_path"]),
                "num_frames": int(actual["num_frames"]),
                "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
            }
        )

    report = {
        "schema": "decode_report.v1",
        "stage": "decode",
        "substage": "frames_generate",
        "status": "success",
        "run_id": str(runtime.spec.exp_name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_strategy": "prompts",
        "backend_name": backend_name,
        "backend_meta": backend_meta,
        "backend_result": backend_result,
        "num_tasks": int(len(unit_reports)),
        "source_inputs": dict(tasks_payload.get("source_inputs") or {}),
        "task_summary": dict(tasks_payload.get("summary") or {}),
        "units": unit_reports,
        "outputs": {
            "runs_dir": _relative_path(runtime.paths.exp_dir, runs_dir),
            "report": "decode/{}".format(REPORT_FILENAME),
        },
        "total_runtime_sec": float(elapsed),
    }
    report_path = runtime.paths.decode_report_path
    write_json_atomic(report_path, report, indent=2)

    decode_plan = _build_decode_plan(
        runtime.paths.exp_dir,
        tasks_payload,
        report,
        fps=int(float(runtime.fps_arg)),
        kf_gap=int(runtime.spec.kf_gap),
        seed_base=int(runtime.args.seed_base),
    )
    write_json_atomic(runtime.paths.decode_plan_path, decode_plan, indent=2)
    log_info("decode generate report: {}".format(report_path))
    return report


def _runtime_from_args(args):
    exp_dir = Path(args.exp_dir).resolve()
    paths = SimpleNamespace(
        exp_dir=exp_dir,
        prepare_frames_dir=Path(args.prepare_frames_dir).resolve(),
        decode_dir=(exp_dir / "decode").resolve(),
        decode_runs_dir=(exp_dir / "decode" / "runs").resolve(),
        decode_report_path=(exp_dir / "decode" / "decode_report.json").resolve(),
        decode_plan_path=(exp_dir / "decode" / "decode_plan.json").resolve(),
    )
    spec = SimpleNamespace(
        exp_name=str(args.run_id),
        kf_gap=int(args.kf_gap),
    )
    runtime_args = SimpleNamespace(
        infer_backend=str(args.infer_backend),
        videox_root=str(args.videox_root),
        infer_model_dir=str(args.infer_model_dir or ""),
        seed_base=int(args.seed_base),
        gpus=int(args.gpus),
        infer_extra=str(args.infer_extra or ""),
    )
    return SimpleNamespace(
        paths=paths,
        spec=spec,
        args=runtime_args,
        fps_arg=str(args.fps),
        infer_phase_name=lambda: str(args.backend_python_phase or "infer_fun_5b"),
    )


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub native decode frame generation.")
    parser.add_argument("--run-native-tasks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--videox_root", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--kf_gap", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=43)
    parser.add_argument("--infer_backend", default="wan_fun_5b_inp", choices=["wan_fun_5b_inp"])
    parser.add_argument("--infer_model_dir", default="")
    parser.add_argument("--backend_python_phase", default="infer_fun_5b")
    parser.add_argument("--infer_extra", default="")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_native_tasks:
        raise SystemExit("[ERR] use --run-native-tasks")
    tasks_payload = read_json_dict(args.tasks)
    if not tasks_payload:
        raise RuntimeError("invalid decode tasks payload: {}".format(args.tasks))
    generate_tasks(_runtime_from_args(args), tasks_payload)


if __name__ == "__main__":
    main()
