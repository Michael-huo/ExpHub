from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic
from exphub.common.logging import log_info

from .generation_unit import build_generation_units
from .motion_segment import build_motion_segments
from .result_writer import write_encode_outputs
from .synthetic_prompt import build_prompts
from .train_export import TrainExportSession


@dataclass(frozen=True)
class _EncodePaths:
    exp_dir: Path
    prepare_result_path: Path
    prepare_frames_dir: Path
    encode_dir: Path
    encode_motion_segments_path: Path
    encode_semantic_anchors_path: Path
    encode_generation_units_path: Path
    encode_prompts_path: Path
    encode_result_path: Path
    encode_overview_path: Path


def _infer_encode_paths(runtime):
    return _EncodePaths(
        exp_dir=runtime.paths.exp_dir,
        prepare_result_path=runtime.paths.prepare_result_path,
        prepare_frames_dir=runtime.paths.prepare_frames_dir,
        encode_dir=runtime.paths.encode_dir,
        encode_motion_segments_path=runtime.paths.encode_motion_segments_path,
        encode_semantic_anchors_path=runtime.paths.encode_semantic_anchors_path,
        encode_generation_units_path=runtime.paths.encode_generation_units_path,
        encode_prompts_path=runtime.paths.encode_prompts_path,
        encode_result_path=runtime.paths.encode_result_path,
        encode_overview_path=runtime.paths.encode_overview_path,
    )


def _train_sequence_encode_paths(runtime, sequence):
    return _EncodePaths(
        exp_dir=runtime.paths.exp_dir,
        prepare_result_path=runtime.paths.prepare_sequence_result_path(sequence),
        prepare_frames_dir=runtime.paths.prepare_sequence_frames_dir(sequence),
        encode_dir=runtime.paths.encode_sequence_dir(sequence),
        encode_motion_segments_path=runtime.paths.encode_sequence_motion_segments_path(sequence),
        encode_semantic_anchors_path=runtime.paths.encode_sequence_semantic_anchors_path(sequence),
        encode_generation_units_path=runtime.paths.encode_sequence_generation_units_path(sequence),
        encode_prompts_path=runtime.paths.encode_sequence_prompts_path(sequence),
        encode_result_path=runtime.paths.encode_sequence_result_path(sequence),
        encode_overview_path=runtime.paths.encode_sequence_overview_path(sequence),
    )


def _semantic_anchor_cmd(paths, motion_segments_path, semantic_anchors_path):
    return [
        "-m",
        "exphub.encode.semantic_anchor",
        "--run-mainline",
        "--prepare_result",
        str(paths.prepare_result_path),
        "--motion_segments",
        str(motion_segments_path),
        "--frames_dir",
        str(paths.prepare_frames_dir),
        "--out_path",
        str(semantic_anchors_path),
    ]


def _run_single_encode(runtime, paths, log_name="encode.log"):
    total_started = time.time()
    ensure_file(paths.prepare_result_path, "prepare result")
    ensure_dir(paths.prepare_frames_dir, "prepare frames dir")
    runtime.write_meta_snapshot()

    runtime.remove_in_exp(paths.encode_dir)
    paths.encode_dir.mkdir(parents=True, exist_ok=True)

    prepare_result = read_json_dict(paths.prepare_result_path)
    if not prepare_result:
        raise RuntimeError("invalid prepare result: {}".format(paths.prepare_result_path))
    if paths.prepare_result_path == runtime.paths.prepare_result_path:
        runtime._prepare_result_cache = dict(prepare_result)
    motion_segments_path = paths.encode_motion_segments_path
    semantic_anchors_path = paths.encode_semantic_anchors_path
    generation_units_path = paths.encode_generation_units_path
    prompts_path = paths.encode_prompts_path

    log_info("encode pass1 motion_state start")
    motion_segments = build_motion_segments(
        prepare_result=prepare_result,
        frames_dir=paths.prepare_frames_dir,
        out_path=motion_segments_path,
    )

    log_info("encode pass1 visual anchor tracking start backend=openclip_image_embedding")
    runtime.step_runner.run_env_python(
        _semantic_anchor_cmd(
            paths,
            motion_segments_path,
            semantic_anchors_path,
        ),
        phase_name="semantic_openclip",
        log_name=log_name,
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
    log_info(
        "generation units generated count={} unit_length_guard_count={}".format(
            int(len(list(dict(generation_units).get("units") or []))),
            int(
                dict(dict(generation_units).get("summary") or {}).get(
                    "unit_length_guard_count",
                    0,
                )
                or 0
            ),
        )
    )

    log_info("encode pass1 synthetic_prompt start")
    prompts = build_prompts(
        generation_units=generation_units,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        frames_dir=paths.prepare_frames_dir,
        out_path=prompts_path,
    )

    log_info("encode pass1 result_writer start")
    result_path = write_encode_outputs(
        runtime=runtime,
        prepare_result=prepare_result,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        generation_units=generation_units,
        prompts=prompts,
        elapsed_sec=float(time.time() - total_started),
        paths=paths,
    )
    for path, label in [
        (paths.encode_motion_segments_path, "motion segments"),
        (paths.encode_semantic_anchors_path, "semantic anchors"),
        (paths.encode_generation_units_path, "generation units"),
        (paths.encode_prompts_path, "prompts"),
        (paths.encode_result_path, "encode result"),
        (paths.encode_overview_path, "encode overview"),
    ]:
        ensure_file(path, label)
    log_info("encode pass1 done: {}".format(Path(result_path).resolve()))
    return {
        "prepare_result": prepare_result,
        "motion_segments": motion_segments,
        "semantic_anchors": semantic_anchors,
        "generation_units": generation_units,
        "prompts": prompts,
        "result_path": Path(result_path).resolve(),
    }


def _write_train_encode_index(runtime, sequences):
    entries = list(sequences or [])
    ok_count = len([item for item in entries if item.get("status") == "ok"])
    failed_count = len([item for item in entries if item.get("status") == "failed"])
    skipped_count = len([item for item in entries if item.get("status") == "skipped"])
    payload = {
        "version": 1,
        "mode": "train",
        "scope": str(runtime.paths.scope),
        "dataset": str(runtime.spec.dataset),
        "run_id": str(runtime.spec.exp_name),
        "sequence_count": int(len(entries)),
        "ok_count": int(ok_count),
        "failed_count": int(failed_count),
        "skipped_count": int(skipped_count),
        "sequences": entries,
    }
    write_json_atomic(runtime.paths.encode_dataset_index_path, payload, indent=2)
    return payload


def _train_encode_ok_entry(sequence, paths, result):
    generation_units = dict(result.get("generation_units") or {})
    units = list(generation_units.get("units") or [])
    return {
        "sequence": str(sequence),
        "status": "ok",
        "error_message": "",
        "encode_result_path": str(Path(paths.encode_result_path).resolve()),
        "generation_units_path": str(Path(paths.encode_generation_units_path).resolve()),
        "prompts_path": str(Path(paths.encode_prompts_path).resolve()),
        "unit_count": int(len(units)),
    }


def _train_encode_failed_entry(sequence, error):
    return {
        "sequence": str(sequence),
        "status": "failed",
        "error_message": str(error),
        "encode_result_path": "",
        "generation_units_path": "",
        "prompts_path": "",
        "unit_count": 0,
    }


def _train_encode_skipped_entry(sequence, reason):
    return {
        "sequence": str(sequence),
        "status": "skipped",
        "error_message": str(reason),
        "encode_result_path": "",
        "generation_units_path": "",
        "prompts_path": "",
        "unit_count": 0,
    }


def _run_train_encode(runtime):
    prepare_index = read_json_dict(runtime.paths.prepare_dataset_index_path)
    if not prepare_index:
        raise RuntimeError("train encode requires prepare index: {}".format(runtime.paths.prepare_dataset_index_path))
    if str(prepare_index.get("mode", "") or "") != "train":
        raise RuntimeError("invalid train prepare index mode: {}".format(runtime.paths.prepare_dataset_index_path))
    if str(prepare_index.get("scope", "") or "") != str(runtime.paths.scope):
        raise RuntimeError(
            "train prepare index scope mismatch: index={} runtime={}".format(
                prepare_index.get("scope"),
                runtime.paths.scope,
            )
        )

    runtime.write_meta_snapshot()
    runtime.remove_in_exp(runtime.paths.encode_dir)
    runtime.paths.encode_dir.mkdir(parents=True, exist_ok=True)
    runtime.paths.encode_sequences_dir.mkdir(parents=True, exist_ok=True)
    train_export = TrainExportSession(runtime)
    train_export.prepare_output_dirs()

    entries = []
    successful_sequences = 0
    first_error = None

    for prepare_entry in list(prepare_index.get("sequences") or []):
        sequence = str(prepare_entry.get("sequence", "") or "")
        if not sequence:
            continue
        if str(prepare_entry.get("status", "") or "") != "ok":
            reason = str(prepare_entry.get("error_message", "") or "prepare sequence did not complete successfully")
            entries.append(_train_encode_skipped_entry(sequence, reason))
            if first_error is None:
                first_error = RuntimeError(reason)
            continue

        paths = _train_sequence_encode_paths(runtime, sequence)
        try:
            result = _run_single_encode(runtime, paths, log_name="encode_{}.log".format(sequence))
            train_export.export_sequence(
                sequence=sequence,
                prepare_result=result["prepare_result"],
                prepare_frames_dir=paths.prepare_frames_dir,
                generation_units=result["generation_units"],
                prompts=result["prompts"],
                generation_units_path=paths.encode_generation_units_path,
                prompts_path=paths.encode_prompts_path,
            )
            successful_sequences += 1
            entries.append(_train_encode_ok_entry(sequence, paths, result))
        except Exception as exc:
            entries.append(_train_encode_failed_entry(sequence, exc))
            if first_error is None:
                first_error = exc
            if runtime.paths.scope == "sequence":
                _write_train_encode_index(runtime, entries)
                raise

    index_payload = _write_train_encode_index(runtime, entries)
    train_export.write_indexes(sequence_count=successful_sequences)
    failed_or_skipped = int(index_payload.get("failed_count", 0) or 0) + int(index_payload.get("skipped_count", 0) or 0)
    if failed_or_skipped > 0:
        raise RuntimeError(
            "train encode completed with failed/skipped sequences: count={} first_error={}".format(
                failed_or_skipped,
                first_error,
            )
        )
    return runtime.paths.trainset_dir


def run(runtime):
    mode = str(runtime.args.mode).strip().lower()
    if mode == "infer":
        return _run_single_encode(runtime, _infer_encode_paths(runtime))["result_path"]
    if mode == "train":
        return _run_train_encode(runtime)
    raise RuntimeError("unsupported encode mode: {}".format(mode))
