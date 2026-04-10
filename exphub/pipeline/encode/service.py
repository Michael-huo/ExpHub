from __future__ import annotations

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic
from exphub.contracts import prompt as prompt_contract
from exphub.contracts import segment as segment_contract
from exphub.pipeline.encode.candidate_boundaries import build_candidate_boundaries_payload
from exphub.pipeline.encode.state_analysis import (
    build_generation_risk_payload,
    build_motion_score_payload,
    build_semantic_shift_payload,
)
from exphub.pipeline.encode.text_gen.prompt_spans import build_prompt_spans_payload
from exphub.pipeline.encode.unit_planner import build_generation_units_payload


_PROMPT_PHASE = "prompt_smol"


def _scene_split_helper_path(runtime):
    return (runtime.exphub_root / "exphub" / "pipeline" / "encode" / "scene_split" / "core.py").resolve()


def _text_gen_helper_path(runtime):
    return (runtime.exphub_root / "exphub" / "pipeline" / "encode" / "text_gen" / "core.py").resolve()


def _build_scene_split_cmd(runtime):
    dataset = runtime.dataset()
    segment_python = runtime.phase_python("segment")
    dist_args = []
    if dataset.dist:
        dist_args = ["--dist"] + [str(item) for item in dataset.dist]

    return [
        str(segment_python),
        str(_scene_split_helper_path(runtime)),
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


def _build_text_gen_cmd(runtime):
    cmd = [
        str(_text_gen_helper_path(runtime)),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--segment_manifest",
        str(runtime.paths.segment_manifest_path),
        "--fps",
        runtime.fps_arg,
        "--backend_python_phase",
        _PROMPT_PHASE,
    ]
    prompt_model_dir = str(runtime.args.prompt_model_dir or "").strip()
    if prompt_model_dir:
        cmd.extend(["--prompt_model_dir", prompt_model_dir])
    return cmd


def run_scene_split(runtime):
    contract = segment_contract.build_contract(runtime.paths)
    segment_contract.require_formal_segment_policy(runtime.args.segment_policy)
    runtime.ensure_clean_exp_dir()
    runtime.write_meta_snapshot()

    runtime.step_runner.run_ros(
        _build_scene_split_cmd(runtime),
        log_name="segment.log",
        cwd=runtime.exphub_root,
    )

    ensure_dir(contract.artifacts["frames_dir"], "segment frames dir")
    ensure_dir(contract.artifacts["keyframes_dir"], "segment keyframes dir")
    ensure_file(contract.artifacts["manifest"], "segment manifest")
    ensure_file(contract.artifacts["aligned_plan"], "aligned segment plan")
    ensure_file(contract.artifacts["report"], "segment report")
    ensure_file(contract.artifacts["overview"], "segment state overview")
    ensure_file(contract.artifacts["calib"], "segment calib")
    ensure_file(contract.artifacts["timestamps"], "segment timestamps")
    return contract.artifacts["manifest"]


def run_text_gen(runtime):
    contract = prompt_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")
    ensure_file(runtime.paths.segment_manifest_path, "segment manifest")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(runtime.paths.prompt_dir)
    runtime.step_runner.run_env_python(
        _build_text_gen_cmd(runtime),
        phase_name=_PROMPT_PHASE,
        log_name="prompt.log",
        cwd=runtime.exphub_root,
    )

    ensure_file(contract.artifacts[prompt_contract.REPORT], "prompt report")
    ensure_file(contract.artifacts[prompt_contract.PROMPT_MANIFEST], "prompt manifest")
    return contract.artifacts[prompt_contract.REPORT]


def run_generation_unit_planner(runtime):
    ensure_file(runtime.paths.segment_manifest_path, "segment manifest")
    ensure_file(runtime.paths.prompt_manifest_path, "prompt manifest")

    segment_manifest = read_json_dict(runtime.paths.segment_manifest_path)
    prompt_manifest = read_json_dict(runtime.paths.prompt_manifest_path)
    if not segment_manifest:
        raise RuntimeError("invalid segment manifest: {}".format(runtime.paths.segment_manifest_path))
    if not prompt_manifest:
        raise RuntimeError("invalid prompt manifest: {}".format(runtime.paths.prompt_manifest_path))

    motion_score = build_motion_score_payload(segment_manifest)
    semantic_shift = build_semantic_shift_payload(segment_manifest, prompt_manifest)
    generation_risk = build_generation_risk_payload(motion_score, semantic_shift)
    candidate_boundaries = build_candidate_boundaries_payload(motion_score, semantic_shift, generation_risk)

    frame_meta = dict(segment_manifest.get("frames") or {})
    frame_count_used = int(frame_meta.get("frame_count_used", 0) or 0)
    if frame_count_used <= 0:
        raise RuntimeError("segment manifest has invalid frame_count_used for generation units")

    generation_units = build_generation_units_payload(
        motion_score_payload=motion_score,
        semantic_shift_payload=semantic_shift,
        generation_risk_payload=generation_risk,
        candidate_boundaries_payload=candidate_boundaries,
        sequence_start_idx=0,
        sequence_end_idx=int(frame_count_used - 1),
    )
    prompt_spans = build_prompt_spans_payload(prompt_manifest, generation_units)

    write_json_atomic(runtime.paths.segment_motion_score_path, motion_score, indent=2)
    write_json_atomic(runtime.paths.segment_semantic_shift_path, semantic_shift, indent=2)
    write_json_atomic(runtime.paths.segment_generation_risk_path, generation_risk, indent=2)
    write_json_atomic(runtime.paths.segment_candidate_boundaries_path, candidate_boundaries, indent=2)
    write_json_atomic(runtime.paths.segment_generation_units_path, generation_units, indent=2)
    write_json_atomic(runtime.paths.prompt_spans_path, prompt_spans, indent=2)

    ensure_file(runtime.paths.segment_motion_score_path, "motion score")
    ensure_file(runtime.paths.segment_semantic_shift_path, "semantic shift")
    ensure_file(runtime.paths.segment_generation_risk_path, "generation risk")
    ensure_file(runtime.paths.segment_candidate_boundaries_path, "candidate boundaries")
    ensure_file(runtime.paths.segment_generation_units_path, "generation units")
    ensure_file(runtime.paths.prompt_spans_path, "prompt spans")
    return runtime.paths.segment_generation_units_path


def run(runtime):
    run_scene_split(runtime)
    run_text_gen(runtime)
    return run_generation_unit_planner(runtime)
