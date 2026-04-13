from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic

from .boundaries_build import build_candidate_boundaries_payload, write_encode_segmentation_overview
from .plan_build import build_generation_units_payload
from .prompts_build import build_prompt_spans_payload
from .signals_build import build_generation_risk_payload, build_motion_score_payload, build_semantic_shift_payload


_PROMPT_PHASE = "prompt_smol"


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _relative_to_exp(exp_dir, target_path):
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _build_scene_split_cmd(runtime):
    dataset = runtime.dataset()
    segment_python = runtime.phase_python("segment")
    dist_args = []
    if dataset.dist:
        dist_args = ["--dist"] + [str(item) for item in dataset.dist]

    return [
        str(segment_python),
        "-m",
        "exphub.encode.boundaries_build",
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--bag",
        str(dataset.bag),
        "--topic",
        dataset.topic,
        "--duration",
        str(runtime.spec.dur),
        "--fps",
        runtime.fps_arg,
        "--kf_gap",
        str(runtime.spec.kf_gap),
        "--keyframes_mode",
        str(runtime.args.keyframes_mode),
        "--segment_policy",
        str(runtime.args.segment_policy),
        "--start_idx",
        str(runtime.args.start_idx),
        "--start_sec",
        str(runtime.spec.start_sec),
        "--width",
        str(runtime.spec.w),
        "--height",
        str(runtime.spec.h),
        "--fx",
        str(dataset.fx),
        "--fy",
        str(dataset.fy),
        "--cx",
        str(dataset.cx),
        "--cy",
        str(dataset.cy),
    ] + dist_args


def _build_text_gen_cmd(runtime, temp_prompt_manifest_path):
    cmd = [
        "-m",
        "exphub.encode.prompts_build",
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--input_report",
        str(runtime.paths.input_report_path),
        "--out_path",
        str(temp_prompt_manifest_path),
    ]
    prompt_model_dir = str(runtime.args.prompt_model_dir or "").strip()
    if prompt_model_dir:
        cmd.extend(["--prompt_model_dir", prompt_model_dir])
    return cmd


def _build_encode_plan(input_report, motion_score, semantic_shift, generation_risk, candidate_boundaries, generation_units):
    frames_meta = dict(_as_dict(input_report).get("frames") or {})
    inputs_meta = dict(_as_dict(input_report).get("inputs") or {})
    keyframes_meta = dict(_as_dict(input_report).get("keyframes") or {})
    state_summary = dict(_as_dict(input_report).get("summary") or {})
    units = list(_as_dict(generation_units).get("units") or [])
    selected_boundaries = []
    for idx, unit in enumerate(units):
        if idx == 0:
            selected_boundaries.append(int(unit.get("anchor_start_idx", 0) or 0))
        selected_boundaries.append(int(unit.get("anchor_end_idx", 0) or 0))
    return {
        "schema": "encode_plan.v1",
        "stage": "encode",
        "planner": "generation_units",
        "prompt_strategy": "prompt_spans",
        "video": {
            "frame_count": int(frames_meta.get("frame_count", 0) or 0),
            "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
            "tail_drop": int(frames_meta.get("tail_drop", 0) or 0),
            "fps": inputs_meta.get("fps"),
            "duration": inputs_meta.get("duration"),
            "width": inputs_meta.get("width"),
            "height": inputs_meta.get("height"),
        },
        "signals": {
            "motion_score": dict(motion_score or {}),
            "semantic_shift": dict(semantic_shift or {}),
            "generation_risk": dict(generation_risk or {}),
        },
        "boundaries": {
            "candidates": list(_as_dict(candidate_boundaries).get("boundaries") or []),
            "selected": list(selected_boundaries),
        },
        "sequence_range": dict(_as_dict(generation_units).get("sequence_range") or {}),
        "units": list(units),
        "summary": {
            "frame_count": int(frames_meta.get("frame_count", 0) or 0),
            "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
            "keyframe_count": int(keyframes_meta.get("count", 0) or 0),
            "state_segment_count": int(state_summary.get("state_segment_count", 0) or 0),
            "scene_group_count": int(_as_dict(semantic_shift).get("summary", {}).get("scene_group_count", 0) or 0),
            "candidate_boundary_count": int(len(list(_as_dict(candidate_boundaries).get("boundaries") or []))),
            "unit_count": int(_as_dict(generation_units).get("summary", {}).get("unit_count", 0) or 0),
            "decode_valid_unit_count": int(_as_dict(generation_units).get("summary", {}).get("decode_valid_unit_count", 0) or 0),
            "export_valid_unit_count": int(_as_dict(generation_units).get("summary", {}).get("export_valid_unit_count", 0) or 0),
        },
    }


def _build_encode_report(runtime, input_report, encode_plan, prompt_spans, prompt_manifest, prompt_sec, plan_sec):
    exp_dir = runtime.paths.exp_dir
    input_summary = dict(_as_dict(input_report).get("summary") or {})
    prompt_summary = dict(_as_dict(prompt_spans).get("summary") or {})
    encode_summary = dict(_as_dict(encode_plan).get("summary") or {})
    input_timings = dict(_as_dict(input_report).get("timings_sec") or {})
    backend_meta = dict(_as_dict(prompt_manifest).get("backend_meta") or {})
    return {
        "schema": "encode_report.v1",
        "stage": "encode",
        "status": "success",
        "timings_sec": {
            "input_prepare": float(input_timings.get("total", 0.0) or 0.0),
            "prompt_prepare": float(prompt_sec),
            "plan_build": float(plan_sec),
            "encode_total": float(float(input_timings.get("total", 0.0) or 0.0) + float(prompt_sec) + float(plan_sec)),
        },
        "counts": {
            "frames": int(input_summary.get("frame_count", 0) or 0),
            "keyframes": int(input_summary.get("keyframe_count", 0) or 0),
            "state_segments": int(input_summary.get("state_segment_count", 0) or 0),
            "units": int(encode_summary.get("unit_count", 0) or 0),
            "prompt_spans": int(prompt_summary.get("span_count", 0) or 0),
        },
        "config": {
            "segment_policy": str(runtime.args.segment_policy),
            "keyframes_mode": str(runtime.args.keyframes_mode),
            "prompt_model_dir": str(runtime.args.prompt_model_dir or ""),
            "prompt_backend": str(_as_dict(prompt_manifest).get("scene_encoding_backend", "") or ""),
            "planner": "generation_units",
            "prompt_strategy": "prompt_spans",
        },
        "prompt_backend": {
            "backend": str(_as_dict(prompt_manifest).get("scene_encoding_backend", "") or ""),
            "processor_load_sec": backend_meta.get("processor_load_sec"),
            "model_load_sec": backend_meta.get("model_load_sec"),
        },
        "artifacts": {
            "input_report": _relative_to_exp(exp_dir, runtime.paths.input_report_path),
            "encode_plan": _relative_to_exp(exp_dir, runtime.paths.encode_plan_path),
            "prompt_spans": _relative_to_exp(exp_dir, runtime.paths.prompt_spans_path),
            "encode_segmentation_overview": _relative_to_exp(exp_dir, runtime.paths.encode_segmentation_overview_path),
            "encode_report": _relative_to_exp(exp_dir, runtime.paths.encode_report_path),
        },
    }


def run_scene_split(runtime):
    from exphub.encode.boundaries_build import require_formal_segment_policy

    require_formal_segment_policy(runtime.args.segment_policy)
    runtime.ensure_clean_exp_dir()
    runtime.write_meta_snapshot()

    runtime.step_runner.run_ros(
        _build_scene_split_cmd(runtime),
        log_name="encode.log",
        cwd=runtime.exphub_root,
    )

    ensure_dir(runtime.paths.input_frames_dir, "input frames dir")
    ensure_file(runtime.paths.input_report_path, "input report")
    return runtime.paths.input_report_path


def _load_prompt_manifest(runtime):
    ensure_file(runtime.paths.input_report_path, "input report")
    runtime.paths.encode_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_path_text = tempfile.mkstemp(prefix="prompt_manifest_", suffix=".json", dir=str(runtime.paths.encode_dir))
    os.close(fd)
    temp_path = Path(temp_path_text).resolve()
    if temp_path.exists():
        temp_path.unlink()
    started = time.time()
    try:
        runtime.step_runner.run_env_python(
            _build_text_gen_cmd(runtime, temp_path),
            phase_name=_PROMPT_PHASE,
            log_name="encode.log",
            cwd=runtime.exphub_root,
        )
        ensure_file(temp_path, "temporary prompt manifest")
        prompt_manifest = read_json_dict(temp_path)
        if not prompt_manifest:
            raise RuntimeError("invalid temporary prompt manifest: {}".format(temp_path))
        return prompt_manifest, float(time.time() - started)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def run_generation_unit_planner(runtime):
    ensure_file(runtime.paths.input_report_path, "input report")

    input_report = read_json_dict(runtime.paths.input_report_path)
    if not input_report:
        raise RuntimeError("invalid input report: {}".format(runtime.paths.input_report_path))

    prompt_manifest, prompt_sec = _load_prompt_manifest(runtime)

    started = time.time()
    motion_score = build_motion_score_payload(input_report)
    semantic_shift = build_semantic_shift_payload(input_report, prompt_manifest)
    generation_risk = build_generation_risk_payload(motion_score, semantic_shift)
    candidate_boundaries = build_candidate_boundaries_payload(motion_score, semantic_shift, generation_risk)

    frame_meta = dict(_as_dict(input_report).get("frames") or {})
    frame_count_used = int(frame_meta.get("frame_count_used", 0) or 0)
    if frame_count_used <= 0:
        raise RuntimeError("input report has invalid frame_count_used for generation units")

    generation_units = build_generation_units_payload(
        motion_score_payload=motion_score,
        semantic_shift_payload=semantic_shift,
        generation_risk_payload=generation_risk,
        candidate_boundaries_payload=candidate_boundaries,
        sequence_start_idx=0,
        sequence_end_idx=int(frame_count_used - 1),
    )
    prompt_spans = build_prompt_spans_payload(prompt_manifest, generation_units)
    encode_plan = _build_encode_plan(
        input_report=input_report,
        motion_score=motion_score,
        semantic_shift=semantic_shift,
        generation_risk=generation_risk,
        candidate_boundaries=candidate_boundaries,
        generation_units=generation_units,
    )
    plan_sec = float(time.time() - started)

    ensure_dir(runtime.paths.encode_dir, "encode dir")
    write_json_atomic(runtime.paths.encode_plan_path, encode_plan, indent=2)
    write_json_atomic(runtime.paths.prompt_spans_path, prompt_spans, indent=2)
    write_encode_segmentation_overview(
        runtime.paths.encode_segmentation_overview_path,
        input_report=input_report,
        encode_plan=encode_plan,
        source_path=runtime.paths.input_dir / "state_overview.png",
    )
    write_json_atomic(
        runtime.paths.encode_report_path,
        _build_encode_report(runtime, input_report, encode_plan, prompt_spans, prompt_manifest, prompt_sec, plan_sec),
        indent=2,
    )

    ensure_file(runtime.paths.encode_plan_path, "encode plan")
    ensure_file(runtime.paths.prompt_spans_path, "prompt spans")
    ensure_file(runtime.paths.encode_segmentation_overview_path, "encode segmentation overview")
    ensure_file(runtime.paths.encode_report_path, "encode report")
    return runtime.paths.encode_report_path


def run(runtime):
    run_scene_split(runtime)
    return run_generation_unit_planner(runtime)
