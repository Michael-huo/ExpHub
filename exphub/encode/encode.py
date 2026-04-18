from __future__ import annotations

import time
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict
from exphub.common.logging import log_info

from .export_router import export_encode_outputs
from .generation_unit import build_generation_units
from .motion_segment import build_motion_segments
from .synthetic_prompt import build_prompts


def _semantic_anchor_cmd(runtime, motion_segments_path, semantic_anchors_path):
    return [
        "-m",
        "exphub.encode.semantic_anchor",
        "--run-mainline",
        "--prepare_result",
        str(runtime.paths.prepare_result_path),
        "--motion_segments",
        str(motion_segments_path),
        "--frames_dir",
        str(runtime.paths.prepare_frames_dir),
        "--out_path",
        str(semantic_anchors_path),
    ]


def run(runtime):
    total_started = time.time()
    ensure_file(runtime.paths.prepare_result_path, "prepare result")
    ensure_dir(runtime.paths.prepare_frames_dir, "prepare frames dir")
    runtime.write_meta_snapshot()

    runtime.remove_in_exp(runtime.paths.encode_dir)
    runtime.paths.encode_dir.mkdir(parents=True, exist_ok=True)

    prepare_result = runtime.prepare_result()
    motion_segments_path = runtime.paths.encode_dir / "motion_segments.json"
    semantic_anchors_path = runtime.paths.encode_dir / "semantic_anchors.json"
    generation_units_path = runtime.paths.encode_dir / "generation_units.json"
    prompts_path = runtime.paths.encode_dir / "prompts.json"

    log_info("encode pass1 motion_segment start")
    motion_segments = build_motion_segments(
        prepare_result=prepare_result,
        frames_dir=runtime.paths.prepare_frames_dir,
        out_path=motion_segments_path,
    )

    log_info("encode pass1 semantic_anchor start backend=semantic_openclip")
    runtime.step_runner.run_env_python(
        _semantic_anchor_cmd(runtime, motion_segments_path, semantic_anchors_path),
        phase_name="semantic_openclip",
        log_name="encode.log",
        cwd=runtime.exphub_root,
    )
    ensure_file(semantic_anchors_path, "semantic anchors")
    semantic_anchors = read_json_dict(semantic_anchors_path)
    if not semantic_anchors:
        raise RuntimeError("invalid semantic anchors: {}".format(semantic_anchors_path))

    log_info("encode pass1 generation_unit start")
    generation_units = build_generation_units(
        prepare_result=prepare_result,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        out_path=generation_units_path,
    )

    log_info("encode pass1 synthetic_prompt start")
    prompts = build_prompts(
        generation_units=generation_units,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        frames_dir=runtime.paths.prepare_frames_dir,
        prompt_model_dir=str(runtime.args.prompt_model_dir or ""),
        out_path=prompts_path,
    )

    log_info("encode pass1 export_router start")
    result_path = export_encode_outputs(
        runtime=runtime,
        prepare_result=prepare_result,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        generation_units=generation_units,
        prompts=prompts,
        elapsed_sec=float(time.time() - total_started),
    )
    for path, label in [
        (runtime.paths.encode_dir / "motion_segments.json", "motion segments"),
        (runtime.paths.encode_dir / "semantic_anchors.json", "semantic anchors"),
        (runtime.paths.encode_dir / "generation_units.json", "generation units"),
        (runtime.paths.encode_dir / "prompts.json", "prompts"),
        (runtime.paths.encode_dir / "encode_result.json", "encode result"),
        (runtime.paths.encode_dir / "encode_overview.png", "encode overview"),
        (runtime.paths.encode_legacy_manifest_path, "legacy segment manifest"),
        (runtime.paths.decode_manifest_path, "decode manifest"),
        (runtime.paths.encode_plan_path, "transition encode plan"),
        (runtime.paths.prompt_spans_path, "transition prompt spans"),
        (runtime.paths.encode_report_path, "transition encode report"),
    ]:
        ensure_file(path, label)
    log_info("encode pass1 done: {}".format(Path(result_path).resolve()))
    return result_path
