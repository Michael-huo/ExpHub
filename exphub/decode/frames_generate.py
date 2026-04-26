from __future__ import annotations

import argparse
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from exphub.common.io import ensure_dir, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog
from exphub.config import get_platform_config


REPORT_FILENAME = "decode_report.json"
COMFYUI_BACKEND = "comfyui_wan2_2_5b_inp"
WAN_BACKEND = "wan_fun_5b_inp"


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
        "schema": "generation_task_runtime.v1",
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


def _write_backend_task_payload(work_dir, tasks_payload, backend_name):
    payload = _task_payload(tasks_payload, backend_name)
    path = Path(work_dir).resolve() / "generation_task_runtime.json"
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


def _image_size(path):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to read ComfyUI decode input frame size") from exc
    with Image.open(str(path)) as image_obj:
        width, height = image_obj.size
    return int(width), int(height)


def _comfyui_cfg_from_platform():
    platform_path = Path(__file__).resolve().parents[2] / "config" / "platform.yaml"
    cfg = get_platform_config()
    services = cfg.get("services", {})
    if not isinstance(services, dict):
        services = {}
    comfyui = services.get("comfyui", {})
    if not isinstance(comfyui, dict):
        comfyui = {}

    def require(key):
        value = comfyui.get(key)
        if value is None or str(value).strip() == "":
            raise RuntimeError("Missing services.comfyui.{} in config/platform.yaml".format(key))
        return value

    return {
        "base_url": str(require("base_url")).strip(),
        "workflow_json": Path(str(require("workflow_json"))).expanduser().resolve(),
        "output_root": Path(str(require("output_root"))).expanduser().resolve(),
        "timeout_sec": int(comfyui.get("timeout_sec", 1800) or 1800),
        "poll_interval_sec": float(comfyui.get("poll_interval_sec", 2.0) or 2.0),
        "platform_config": platform_path.resolve(),
    }


def _task_steps(task):
    value = task.get("num_inference_steps")
    if value is None or str(value).strip() == "":
        return 20
    return int(value)


def _task_cfg(task):
    value = task.get("guidance_scale")
    if value is None or str(value).strip() == "":
        return 6.0
    return float(value)


def _write_comfyui_params(task, run_dir, result, width, height, fps, steps, cfg, backend_name):
    params = {
        "task": str(task["unit_id"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": "runs",
        "experiment_root": str(Path(run_dir).parent),
        "dataset_fps": int(fps),
        "target_fps": int(fps),
        "width": int(width),
        "height": int(height),
        "video_length_desired": int(task["length"]),
        "video_length_run": int(task["length"]),
        "saved_frame_count": int(result.output_frames),
        "start_idx": int(task["start_idx"]),
        "end_idx": int(task["end_idx"]),
        "start_path": str(task["start_frame_path"]),
        "end_path": str(task["end_frame_path"]),
        "batch": True,
        "source_frames_dir": str(Path(task["start_frame_path"]).parent),
        "base_idx": int(task["start_idx"]),
        "num_segments": 1,
        "segment_seconds": float(max(0, int(task["length"]) - 1)) / float(max(int(fps), 1)),
        "schedule_source": "decode.native_tasks",
        "execution_backend": str(backend_name),
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg),
        "seed": int(result.actual_seed),
        "prompt": str(task["prompt"]),
        "negative_prompt": str(task.get("negative_prompt", "") or ""),
        "prompt_source": str(task.get("prompt_source", "prompts.assembled_prompt") or "prompts.assembled_prompt"),
        "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
        "output_dir": str(run_dir),
        "output_video": Path(result.video_paths[0]).name if result.video_paths else "",
        "frames_dir": "frames",
        "frame_ext": "png",
        "comfyui_prompt_id": str(result.prompt_id),
    }
    write_json_atomic(Path(run_dir) / "params.json", params, indent=2)
    return params


def _run_comfyui_tasks(runtime, tasks_payload, backend_name, execution_segments):
    from .comfyui_decode_client import DecodeRequest, run_comfyui_decode

    cfg = _comfyui_cfg_from_platform()
    unit_reports = []
    started = time.time()

    for idx, task in enumerate(list(tasks_payload.get("tasks") or [])):
        run_dir = Path(task["output_dir"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        width, height = _image_size(task["start_frame_path"])
        steps = _task_steps(task)
        cfg_scale = _task_cfg(task)
        requested_seed = -1 if int(runtime.args.seed_base) == -1 else int(task["seed"])

        log_prog(
            "decode generate comfyui: unit={}/{} id={} frames={} seed={}".format(
                idx + 1,
                len(tasks_payload.get("tasks") or []),
                task["unit_id"],
                int(task["length"]),
                int(requested_seed),
            )
        )

        req = DecodeRequest(
            segment_id=str(task["unit_id"]),
            start_frame=str(task["start_frame_path"]),
            end_frame=str(task["end_frame_path"]),
            positive_prompt=str(task["prompt"]),
            negative_prompt=str(task.get("negative_prompt", "") or ""),
            width=int(width),
            height=int(height),
            length=int(task["length"]),
            fps=int(float(runtime.fps_arg)),
            seed=int(requested_seed),
            steps=int(steps),
            cfg=float(cfg_scale),
        )
        result = run_comfyui_decode(
            req,
            comfy_url=cfg["base_url"],
            workflow_json=cfg["workflow_json"],
            comfy_output_root=cfg["output_root"],
            exp_output_dir=run_dir,
            timeout_sec=int(cfg["timeout_sec"]),
            poll_interval_sec=float(cfg["poll_interval_sec"]),
            platform_config=cfg["platform_config"],
        )

        _write_comfyui_params(
            task=task,
            run_dir=run_dir,
            result=result,
            width=width,
            height=height,
            fps=int(float(runtime.fps_arg)),
            steps=steps,
            cfg=cfg_scale,
            backend_name=backend_name,
        )
        actual = _validate_run_output(task, run_dir)
        unit_reports.append(
            {
                "unit_id": str(task["unit_id"]),
                "status": "success",
                "start_idx": int(task["start_idx"]),
                "end_idx": int(task["end_idx"]),
                "length": int(task["length"]),
                "seed": int(result.actual_seed),
                "output_dir": _relative_path(runtime.paths.exp_dir, run_dir),
                "frames_dir": _relative_path(runtime.paths.exp_dir, actual["frames_dir"]),
                "params_path": _relative_path(runtime.paths.exp_dir, actual["params_path"]),
                "decode_meta_path": _relative_path(runtime.paths.exp_dir, run_dir / "decode_meta.json"),
                "num_frames": int(actual["num_frames"]),
                "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
            }
        )

    elapsed = float(time.time() - started)
    return unit_reports, {
        "backend": str(backend_name),
        "mode": "direct_python_comfyui_client",
        "workflow_json": str(cfg["workflow_json"]),
        "comfy_url": str(cfg["base_url"]),
        "comfy_output_root": str(cfg["output_root"]),
        "platform_config": str(cfg["platform_config"]),
        "execution_segments": int(len(execution_segments)),
        "total_runtime_sec": float(elapsed),
    }, elapsed


def generate_tasks(runtime, tasks_payload):
    runs_dir = runtime.paths.decode_runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    backend_name = str(runtime.args.infer_backend or WAN_BACKEND).strip().lower()
    if backend_name not in (WAN_BACKEND, COMFYUI_BACKEND):
        raise RuntimeError("decode backend supports only {}, {}: {}".format(WAN_BACKEND, COMFYUI_BACKEND, backend_name))

    execution_segments = _execution_segments(tasks_payload, backend_name)
    if not execution_segments:
        raise RuntimeError("generation task builder produced zero executable tasks")

    if backend_name == COMFYUI_BACKEND:
        log_prog(
            "decode generate: backend={} tasks={} fps={}".format(
                backend_name,
                len(execution_segments),
                int(float(runtime.fps_arg)),
            )
        )
        unit_reports, backend_result, elapsed = _run_comfyui_tasks(runtime, tasks_payload, backend_name, execution_segments)
        backend_meta = {
            "backend": COMFYUI_BACKEND,
            "client": "exphub.decode.comfyui_decode_client",
            "call_mode": "direct_python",
        }
    else:
        from ._wan_fun_5b_inp import WanFun5BInpBackend

        backend = WanFun5BInpBackend(
            videox_root=str(runtime.args.videox_root),
            model_ref=str(runtime.args.infer_model_dir or ""),
            backend_python_phase=str(runtime.infer_phase_name()),
        )
        backend.load()
        backend_meta = dict(backend.meta() or {})
        with tempfile.TemporaryDirectory(prefix="exphub_generation_runtime_") as tmp_dir:
            payload_path, _ = _write_backend_task_payload(tmp_dir, tasks_payload, backend_name)
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
    parser.add_argument("--infer_backend", default=WAN_BACKEND, choices=[WAN_BACKEND, COMFYUI_BACKEND])
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
