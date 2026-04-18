from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, write_json_atomic
from exphub.common.logging import log_info, log_prog
from .plans_build import (
    ImageGenRequest,
    build_execution_plan,
    build_image_gen_runtime,
    build_prompt_resolution,
    load_segment_manifest,
    merge_prompt_resolution_into_runs_plan,
    write_backend_runtime_files,
)
from .runtime_manage import create_backend


REPORT_FILENAME = "decode_report.json"


def run(runtime):
    decode_root = runtime.paths.decode_dir
    ensure_dir(runtime.paths.input_frames_dir, "input frames dir")
    ensure_file(runtime.paths.decode_manifest_path, "decode manifest")
    ensure_file(runtime.paths.encode_plan_path, "encode plan")
    ensure_file(runtime.paths.prompt_spans_path, "prompt spans")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(decode_root)

    infer_phase = runtime.infer_phase_name()
    cmd = [
        "-m",
        "exphub.decode.frames_generate",
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--frames_dir",
        str(runtime.paths.input_frames_dir),
        "--segment_manifest",
        str(runtime.paths.decode_manifest_path),
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
        str(infer_phase),
    ]
    if runtime.args.infer_extra:
        cmd.extend(["--infer_extra", str(runtime.args.infer_extra)])

    runtime.step_runner.run_env_python(
        cmd,
        phase_name=infer_phase,
        log_name="infer.log",
        cwd=runtime.exphub_root,
    )

    ensure_dir(runtime.paths.decode_runs_dir, "image gen runs dir")
    ensure_file(runtime.paths.decode_plan_path, "decode plan")
    ensure_file(runtime.paths.decode_report_path, "decode report")
    return runtime.paths.decode_report_path


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _list_frame_count(frames_dir):
    count = 0
    for item in Path(frames_dir).resolve().iterdir():
        if item.is_file():
            count += 1
    return int(count)


def _normalize_extra(extra_args):
    text = str(extra_args or "").strip()
    if not text:
        return []
    extra = shlex.split(text)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _validate_execution_segments(frames_avail, execution_segments):
    if frames_avail <= 0:
        raise RuntimeError("segment has no frames")
    for idx, item in enumerate(list(execution_segments or [])):
        start_idx = int(item.get("start_idx", 0) or 0)
        end_idx = int(item.get("end_idx", 0) or 0)
        if start_idx < 0 or end_idx < start_idx:
            raise RuntimeError("invalid execution segment range at index {}".format(idx))
        if end_idx >= int(frames_avail):
            raise RuntimeError(
                "execution segment {} exceeds frames_dir range: end_idx={} frames_avail={}".format(
                    idx,
                    int(end_idx),
                    int(frames_avail),
                )
            )


def _sha1_bytes(payload_bytes):
    import hashlib

    return hashlib.sha1(payload_bytes).hexdigest()


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _segment_summary(plan_segments):
    aligned_frames = []
    actual_frames = []
    prompt_sources = {}
    start_idx = None
    end_idx = None
    for item in list(plan_segments or []):
        if not isinstance(item, dict):
            continue
        aligned_num_frames = int(item.get("aligned_num_frames", item.get("num_frames", 0)) or 0)
        if aligned_num_frames > 0:
            aligned_frames.append(aligned_num_frames)
        actual_saved_frames = int(item.get("actual_saved_frames", 0) or 0)
        if actual_saved_frames > 0:
            actual_frames.append(actual_saved_frames)
        prompt_source = str(item.get("prompt_source", "") or "").strip()
        if prompt_source:
            prompt_sources[prompt_source] = int(prompt_sources.get(prompt_source, 0)) + 1
        try:
            item_start = int(item.get("aligned_start_idx", item.get("start_idx")))
            item_end = int(item.get("aligned_end_idx", item.get("end_idx")))
        except Exception:
            continue
        start_idx = item_start if start_idx is None else min(start_idx, item_start)
        end_idx = item_end if end_idx is None else max(end_idx, item_end)
    return {
        "segment_count": int(len(list(plan_segments or []))),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "aligned_num_frames_min": min(aligned_frames) if aligned_frames else None,
        "aligned_num_frames_max": max(aligned_frames) if aligned_frames else None,
        "aligned_num_frames_sum": int(sum(aligned_frames)) if aligned_frames else 0,
        "actual_saved_frames_sum": int(sum(actual_frames)) if actual_frames else 0,
        "prompt_source_counts": prompt_sources,
        "segment_preview": [
            {
                "seg": item.get("seg"),
                "segment_id": item.get("segment_id"),
                "raw_start_idx": item.get("raw_start_idx"),
                "raw_end_idx": item.get("raw_end_idx"),
                "aligned_start_idx": item.get("aligned_start_idx", item.get("start_idx")),
                "aligned_end_idx": item.get("aligned_end_idx", item.get("end_idx")),
                "aligned_num_frames": item.get("aligned_num_frames", item.get("num_frames")),
                "actual_saved_frames": item.get("actual_saved_frames"),
                "prompt_source": item.get("prompt_source"),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
            }
            for item in list(plan_segments or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _load_run_actuals(run_dir):
    run_root = Path(run_dir).resolve()
    params_path = ensure_file(run_root / "params.json", "image gen run params")
    params_obj = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(params_obj, dict):
        raise RuntimeError("invalid image gen run params: {}".format(params_path))

    frames_dir = ensure_dir(run_root / "frames", "image gen run frames")
    saved_frame_count = _list_frame_count(frames_dir)
    if saved_frame_count <= 0:
        raise RuntimeError("image gen run produced zero saved frames: {}".format(run_root))

    video_length_run = int(params_obj.get("video_length_run", 0) or 0)
    if video_length_run <= 0:
        raise RuntimeError("image gen run params missing video_length_run: {}".format(params_path))
    if video_length_run != saved_frame_count:
        raise RuntimeError(
            "image gen run frame count mismatch: run_dir={} params.video_length_run={} saved_frames={}".format(
                run_root,
                int(video_length_run),
                int(saved_frame_count),
            )
        )

    return {
        "params_path": params_path,
        "frames_dir": frames_dir,
        "saved_frame_count": int(saved_frame_count),
        "start_idx": int(params_obj.get("start_idx", 0) or 0),
        "end_idx": int(params_obj.get("end_idx", 0) or 0),
    }


def _augment_runs_plan_with_saved_frames(infer_dir, plan_obj):
    infer_root = Path(infer_dir).resolve()
    plan = dict(plan_obj or {})
    segments = list(plan.get("segments") or [])
    if not segments:
        raise RuntimeError("runs plan contains zero segments")

    for idx, raw_item in enumerate(segments):
        if not isinstance(raw_item, dict):
            raise RuntimeError("invalid runs plan segment at index {}: not an object".format(idx))
        aligned_start_idx = raw_item.get("aligned_start_idx")
        aligned_end_idx = raw_item.get("aligned_end_idx")
        aligned_num_frames = raw_item.get("aligned_num_frames")
        if aligned_start_idx is None or aligned_end_idx is None or aligned_num_frames is None:
            raise RuntimeError("runs plan missing aligned contract fields at segment {}".format(idx))

        run_name = str(raw_item.get("run_name", "") or "").strip()
        if not run_name:
            raise RuntimeError("runs plan missing run_name at segment {}".format(idx))
        actual = _load_run_actuals(infer_root / "runs" / run_name)
        if int(actual["saved_frame_count"]) != int(aligned_num_frames):
            raise RuntimeError(
                "aligned/actual frame count mismatch for run {}: aligned_num_frames={} actual_saved_frames={}".format(
                    run_name,
                    int(aligned_num_frames),
                    int(actual["saved_frame_count"]),
                )
            )
        if int(actual["start_idx"]) != int(aligned_start_idx) or int(actual["end_idx"]) != int(aligned_end_idx):
            raise RuntimeError(
                "aligned/params range mismatch for run {}: aligned={}..{} params={}..{}".format(
                    run_name,
                    int(aligned_start_idx),
                    int(aligned_end_idx),
                    int(actual["start_idx"]),
                    int(actual["end_idx"]),
                )
            )
        raw_item["actual_saved_frames"] = int(actual["saved_frame_count"])
        raw_item["actual_saved_start_idx"] = int(aligned_start_idx)
        raw_item["actual_saved_end_idx"] = int(aligned_end_idx)
        raw_item["run_params_path"] = _relative_path(infer_root.parent, actual["params_path"])
        raw_item["run_frames_dir"] = _relative_path(infer_root.parent, actual["frames_dir"])

    plan["segments"] = segments
    return plan


def build_image_gen_report(exp_dir, infer_dir, runs_plan_obj, prompt_resolution, backend_meta, backend_result, runtime_summary):
    infer_dir = Path(infer_dir).resolve()
    exp_dir = Path(exp_dir).resolve()
    runs_plan_path = (infer_dir / "decode_plan.json").resolve()
    runs_plan_bytes = runs_plan_path.read_bytes()
    plan_segments = list((runs_plan_obj or {}).get("segments", []) or [])

    source_files = dict((prompt_resolution or {}).get("prompt_resolution", {}).get("source_files", {}) or {})
    return {
        "report_schema_version": "image_gen_report.v1",
        "step": "decode",
        "substage": "image_gen",
        "created_at": str((runtime_summary or {}).get("created_at", "") or ""),
        "image_gen_status": "success",
        "planner": "generation_units",
        "prompt_strategy": "prompt_spans",
        "infer_backend": str((runtime_summary or {}).get("execution_backend", "") or ""),
        "gpus": int((runtime_summary or {}).get("gpus", 0) or 0),
        "fps": int((runtime_summary or {}).get("fps", 0) or 0),
        "kf_gap": int((runtime_summary or {}).get("kf_gap", 0) or 0),
        "frames_avail": int((runtime_summary or {}).get("frames_avail", 0) or 0),
        "segments": int((runtime_summary or {}).get("segments", 0) or 0),
        "used_frames": int((runtime_summary or {}).get("used_frames", 0) or 0),
        "used_start_idx": int((runtime_summary or {}).get("used_start_idx", 0) or 0),
        "used_end_idx": int((runtime_summary or {}).get("used_end_idx", 0) or 0),
        "schedule_source": str((runtime_summary or {}).get("schedule_source", "") or ""),
        "mean_deploy_gap": (runtime_summary or {}).get("mean_deploy_gap"),
        "runs_plan_path": str(runs_plan_path),
        "runs_plan_size": int(len(runs_plan_bytes)),
        "runs_plan_sha1": _sha1_bytes(runs_plan_bytes),
        "state_prompt_enabled": bool((runtime_summary or {}).get("state_prompt_enabled", False)),
        "state_prompt_segment_count": int((runtime_summary or {}).get("state_prompt_segment_count", 0) or 0),
        "matched_execution_segment_count": int((runtime_summary or {}).get("matched_execution_segment_count", 0) or 0),
        "image_gen_runtime_version": int((runtime_summary or {}).get("image_gen_runtime_version", 0) or 0),
        "image_gen_runtime_schema": str((runtime_summary or {}).get("image_gen_runtime_schema", "") or ""),
        "image_gen_runtime_source": str((runtime_summary or {}).get("image_gen_runtime_source", "") or ""),
        "prompt_source_counts": dict((runtime_summary or {}).get("prompt_source_counts", {}) or {}),
        "state_label_counts": dict((runtime_summary or {}).get("state_label_counts", {}) or {}),
        "outputs": {
            "bytes_sum": 0,
            "report_bytes_sum": 0,
            "runs_plan_bytes_sum": int(len(runs_plan_bytes)),
            "report_file_count": 1,
            "runs_plan_file_count": 1,
        },
        "backend_meta": dict(backend_meta or {}),
        "backend_result": dict(backend_result or {}),
        "backend_summary": {
            "infer_backend": str((backend_meta or {}).get("infer_backend", "") or ""),
            "backend_entry_type": str((backend_meta or {}).get("backend_entry_type", "") or ""),
            "backend_python_phase": str((backend_meta or {}).get("backend_python_phase", "") or ""),
            "videox_root": str((backend_meta or {}).get("videox_root", "") or ""),
            "model_dir": str((backend_meta or {}).get("model_dir", "") or ""),
            "model_id": str((backend_meta or {}).get("model_id", "") or ""),
            "config_path": str((backend_meta or {}).get("config_path", "") or ""),
        },
        "prompt_resolution_summary": {
            "state_prompt_enabled": bool((runtime_summary or {}).get("state_prompt_enabled", False)),
            "state_prompt_segment_count": int((runtime_summary or {}).get("state_prompt_segment_count", 0) or 0),
            "matched_execution_segment_count": int((runtime_summary or {}).get("matched_execution_segment_count", 0) or 0),
            "prompt_source_counts": dict((runtime_summary or {}).get("prompt_source_counts", {}) or {}),
            "state_label_counts": dict((runtime_summary or {}).get("state_label_counts", {}) or {}),
            "warnings": list((prompt_resolution or {}).get("warnings", []) or []),
        },
        "prompt_resolution": dict((prompt_resolution or {}).get("prompt_resolution", {}) or {}),
        "execution_segments_summary": _segment_summary(plan_segments),
        "skipped_units": list((runs_plan_obj or {}).get("skipped_units", []) or []),
        "source_files": {
            "decode_plan": _relative_path(exp_dir, runs_plan_path),
            "input_report": str(source_files.get("input_report", "") or ""),
            "encode_plan": str(source_files.get("encode_plan", "") or ""),
            "prompt_spans": str(source_files.get("prompt_spans", "") or ""),
        },
        "artifact_contract": {
            "formal_files": ["decode_plan.json", REPORT_FILENAME],
            "formal_prompt_inputs": ["prepare/prepare_result.json", "prepare/frames/", "encode/legacy_segment_manifest.json", "encode/encode_plan.json", "encode/prompt_spans.json"],
            "transitional_files": [],
        },
    }


def write_image_gen_report(infer_dir, report):
    infer_dir = Path(infer_dir).resolve()
    report_path = infer_dir / REPORT_FILENAME
    report_obj = dict(report or {})
    report_obj["report_path"] = str(report_path)
    last_size = None
    for _ in range(3):
        write_json_atomic(report_path, report_obj, indent=2)
        report_bytes = report_path.read_bytes()
        report_size = int(len(report_bytes))
        outputs = dict(report_obj.get("outputs", {}) or {})
        outputs["report_bytes_sum"] = report_size
        outputs["bytes_sum"] = int(int(outputs.get("report_bytes_sum", 0) or 0) + int(outputs.get("runs_plan_bytes_sum", 0) or 0))
        report_obj["outputs"] = outputs
        report_obj["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, report_obj, indent=2)
    return report_path


def _run_formal_mainline(args):
    frames_dir = ensure_dir(args.frames_dir, "input frames dir")
    exp_dir = Path(args.exp_dir).resolve()
    infer_dir = (exp_dir / "decode").resolve()
    infer_dir.mkdir(parents=True, exist_ok=True)

    frames_avail = _list_frame_count(frames_dir)
    segment_manifest = load_segment_manifest(str(args.segment_manifest))
    image_gen_runtime = build_image_gen_runtime(
        segment_manifest=segment_manifest,
        infer_backend=str(args.infer_backend),
    )
    execution_plan = build_execution_plan(image_gen_runtime)
    execution_segments = list(execution_plan.get("segments") or [])
    if not execution_segments:
        raise RuntimeError("image gen runtime resolved to zero execution segments")
    _validate_execution_segments(frames_avail, execution_segments)
    prompt_resolution = build_prompt_resolution(image_gen_runtime, execution_segments, exp_dir=exp_dir)

    runtime_payload_path, execution_plan_path = write_backend_runtime_files(infer_dir, image_gen_runtime, execution_plan)
    gpus = int(args.gpus)
    fps = int(float(args.fps))
    kf_gap = int(args.kf_gap)
    used_start_idx = min([int(seg["start_idx"]) for seg in execution_segments])
    used_end_idx = max([int(seg["end_idx"]) for seg in execution_segments])
    used_frames = int(sum([int(seg.get("aligned_num_frames", seg.get("num_frames", 0)) or 0) for seg in execution_segments]))
    mean_deploy_gap = float(_mean([int(seg.get("aligned_num_frames", seg.get("num_frames", 0)) or 0) for seg in execution_segments]))

    request = ImageGenRequest(
        frames_dir=frames_dir,
        exp_dir=exp_dir,
        prompt_file_path=runtime_payload_path,
        execution_plan_path=execution_plan_path,
        runs_parent=infer_dir,
        fps=int(fps),
        kf_gap=int(kf_gap),
        base_idx=int(used_start_idx),
        num_segments=int(len(execution_segments)),
        seed_base=int(args.seed_base),
        gpus=int(gpus),
        schedule_source=str(execution_plan.get("schedule_source", "") or ""),
        execution_backend=str(execution_plan.get("execution_backend", "") or ""),
        execution_segments=list(execution_segments),
        infer_extra=_normalize_extra(args.infer_extra),
    )

    backend = create_backend(
        backend_name=str(args.infer_backend),
        videox_root=str(args.videox_root),
        model_ref=str(args.infer_model_dir or ""),
        backend_python_phase=str(args.backend_python_phase or "infer"),
    )
    backend.load()
    backend_meta = dict(backend.meta() or {})

    log_prog(
        "image gen config: backend={} segments={} fps={} gpus={}".format(
            backend_meta.get("infer_backend", args.infer_backend),
            int(len(execution_segments)),
            int(fps),
            int(gpus),
        )
    )
    log_info(
        "image gen detail: schedule_source={} used_frames={}".format(
            str(execution_plan.get("schedule_source", "") or "segment.generation_units"),
            int(used_frames),
        )
    )

    t0 = time.time()
    try:
        backend_result = dict(backend.run(request) or {})
    finally:
        for temp_path in [runtime_payload_path, execution_plan_path]:
            try:
                if Path(temp_path).is_file():
                    Path(temp_path).unlink()
            except Exception:
                pass
    dt = float(time.time() - t0)

    runs_plan_path = infer_dir / "decode_plan.json"
    if runs_plan_path.is_file():
        plan_obj = json.loads(runs_plan_path.read_text(encoding="utf-8"))
    else:
        plan_obj = dict(execution_plan)
    if not isinstance(plan_obj, dict):
        raise RuntimeError("invalid decode_plan.json: {}".format(runs_plan_path))
    plan_obj["planner"] = "generation_units"
    plan_obj["prompt_strategy"] = "prompt_spans"
    plan_obj["schedule_source"] = str(execution_plan.get("schedule_source", "") or plan_obj.get("schedule_source", "") or "")
    plan_obj["skipped_units"] = list(image_gen_runtime.get("skipped_units") or [])
    plan_obj = merge_prompt_resolution_into_runs_plan(plan_obj, prompt_resolution.get("segment_resolutions", []))
    plan_obj = _augment_runs_plan_with_saved_frames(infer_dir, plan_obj)
    plan_obj["planner"] = "generation_units"
    plan_obj["prompt_strategy"] = "prompt_spans"
    plan_obj["schedule_source"] = str(execution_plan.get("schedule_source", "") or plan_obj.get("schedule_source", "") or "")
    plan_obj["image_gen_runtime_version"] = int(prompt_resolution.get("image_gen_runtime_version", 1) or 1)
    plan_obj["image_gen_runtime_schema"] = str(prompt_resolution.get("image_gen_runtime_schema", "") or "image_gen_runtime.v1")
    plan_obj["image_gen_runtime_source"] = str(prompt_resolution.get("image_gen_runtime_source", "") or "decode.image_gen.runtime")
    plan_obj["prompt_source_counts"] = dict(prompt_resolution.get("prompt_source_counts", {}) or {})
    plan_obj["state_label_counts"] = dict(prompt_resolution.get("state_label_counts", {}) or {})
    write_json_atomic(runs_plan_path, plan_obj, indent=2)

    runtime_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": int(gpus),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "frames_avail": int(frames_avail),
        "segments": int(len(list(plan_obj.get("segments", []) or []))),
        "used_frames": int(used_frames),
        "used_start_idx": int(used_start_idx),
        "used_end_idx": int(used_end_idx),
        "schedule_source": str(execution_plan.get("schedule_source", "") or ""),
        "execution_backend": str(execution_plan.get("execution_backend", "") or ""),
        "mean_deploy_gap": float(mean_deploy_gap),
        "state_prompt_enabled": bool(prompt_resolution.get("state_prompt_enabled", False)),
        "state_prompt_segment_count": int(prompt_resolution.get("state_prompt_segment_count", 0) or 0),
        "matched_execution_segment_count": int(prompt_resolution.get("matched_execution_segment_count", 0) or 0),
        "image_gen_runtime_version": int(prompt_resolution.get("image_gen_runtime_version", 1) or 1),
        "image_gen_runtime_schema": str(prompt_resolution.get("image_gen_runtime_schema", "") or "image_gen_runtime.v1"),
        "image_gen_runtime_source": str(prompt_resolution.get("image_gen_runtime_source", "") or "decode.image_gen.runtime"),
        "prompt_source_counts": dict(prompt_resolution.get("prompt_source_counts", {}) or {}),
        "state_label_counts": dict(prompt_resolution.get("state_label_counts", {}) or {}),
    }
    image_gen_report = build_image_gen_report(
        exp_dir=exp_dir,
        infer_dir=infer_dir,
        runs_plan_obj=plan_obj,
        prompt_resolution=prompt_resolution,
        backend_meta=backend_meta,
        backend_result=backend_result,
        runtime_summary=runtime_summary,
    )
    report_path = write_image_gen_report(infer_dir, image_gen_report)

    log_info("image gen finished: {:.2f}s".format(dt))
    log_info("report written: {}".format(report_path))
    return report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub decode.image_gen mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--frames_dir", required=True, help="segment frames dir")
    parser.add_argument("--segment_manifest", required=True, help="formal input_report.json path")
    parser.add_argument("--videox_root", required=True, help="VideoX-Fun repo root")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--kf_gap", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=43)
    parser.add_argument("--infer_backend", default="wan_fun_5b_inp", choices=["wan_fun_5b_inp"])
    parser.add_argument("--infer_model_dir", default="")
    parser.add_argument("--backend_python_phase", default="infer")
    parser.add_argument("--infer_extra", default="")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
