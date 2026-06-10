from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_warn

from .compression_benchmark import CompressionBenchmark
from .generation_unit import build_generation_units
from .motion_benchmark import run_motion_benchmark
from .motion_segment import build_motion_segments
from .payload_writer import write_hvm_payload
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


def _patch_encode_result_profile(result_path, encode_profile):
    payload = read_json_dict(result_path)
    if not payload:
        return
    incoming_profile = dict(encode_profile or {})
    existing_profile = payload.get("profile")
    existing_motion = {}
    if isinstance(existing_profile, dict) and isinstance(existing_profile.get("motion"), dict):
        for key in [
            "version",
            "read_gray_sec",
            "motion_estimation_sec",
            "phase_correlation_sec",
            "orb_tracking_sec",
            "optical_flow_sec",
            "motion_benchmark_sec",
            "write_json_sec",
            "total_sec",
        ]:
            if key in existing_profile["motion"]:
                existing_motion[key] = existing_profile["motion"][key]
    profile = {}
    for key in [
        "version",
        "total_sec",
        "formal_encode_sec_without_benchmark",
        "motion_segment_sec",
        "semantic_anchor_sec",
        "result_writer_sec",
        "benchmark_enabled",
        "motion_benchmark_sec",
        "motion_benchmark_report",
        "benchmark_overhead_note",
    ]:
        if key in incoming_profile:
            profile[key] = incoming_profile[key]
    if existing_motion:
        profile["motion"] = existing_motion
    payload["profile"] = profile
    write_json_atomic(result_path, payload, indent=2)


def _fmt_benchmark_value(value, digits=3):
    try:
        if value is None:
            return "n/a"
        return "{:.{digits}f}".format(float(value), digits=int(digits))
    except Exception:
        return "n/a"


def _log_motion_benchmark_report(report):
    report = dict(report or {})
    methods = dict(report.get("methods") or {})
    relative = dict(dict(report.get("relative_cost") or {}).get("pc_as_1x") or {})
    rows = [
        ("phase_correlation.runtime_sec", _fmt_benchmark_value(dict(methods.get("phase_correlation") or {}).get("runtime_sec"))),
        ("phase_correlation.ms_per_pair", _fmt_benchmark_value(dict(methods.get("phase_correlation") or {}).get("time_per_pair_ms"))),
        ("phase_correlation.valid_rate", _fmt_benchmark_value(dict(methods.get("phase_correlation") or {}).get("valid_rate"))),
        ("orb.runtime_sec", _fmt_benchmark_value(dict(methods.get("orb") or {}).get("runtime_sec"))),
        ("orb.ms_per_pair", _fmt_benchmark_value(dict(methods.get("orb") or {}).get("time_per_pair_ms"))),
        ("orb.valid_rate", _fmt_benchmark_value(dict(methods.get("orb") or {}).get("valid_rate"))),
        ("optical_flow.runtime_sec", _fmt_benchmark_value(dict(methods.get("optical_flow") or {}).get("runtime_sec"))),
        ("optical_flow.ms_per_pair", _fmt_benchmark_value(dict(methods.get("optical_flow") or {}).get("time_per_pair_ms"))),
        ("optical_flow.valid_rate", _fmt_benchmark_value(dict(methods.get("optical_flow") or {}).get("valid_rate"))),
        ("relative_cost.pc_as_1x.orb", _fmt_benchmark_value(relative.get("orb"))),
        ("relative_cost.pc_as_1x.of", _fmt_benchmark_value(relative.get("optical_flow"))),
    ]
    width = max(len(key) for key, _ in rows)
    log_info("[Motion Benchmark]")
    for key, value in rows:
        log_info("{:<{w}} : {}".format(key, value, w=width))


def _is_infer_runtime(runtime):
    return str(getattr(runtime.args, "mode", "") or "").strip().lower() == "infer"


def _run_infer_payload_hooks(runtime, paths, generation_units, prompts, formal_hvm_algorithmic_sec):
    if not _is_infer_runtime(runtime):
        return

    payload_dir = paths.encode_dir / "hvm_payload"
    runtime.remove_in_exp(payload_dir)
    runtime.remove_in_exp(paths.encode_dir / "hvm_payload.zip")
    payload_started = time.perf_counter()
    payload_report = write_hvm_payload(
        frames_dir=paths.prepare_frames_dir,
        generation_units=generation_units,
        prompts=prompts,
        payload_dir=payload_dir,
    )
    payload_write_sec = float(time.perf_counter() - payload_started)
    ensure_dir(payload_dir, "Ours payload")
    log_info(
        "Ours payload done: frames={} dir={}".format(
            int(payload_report.get("frame_count", 0) or 0),
            _relative_to_exp(runtime, payload_dir),
        )
    )

    if not bool(getattr(runtime.args, "compression_benchmark", False)):
        return

    benchmark_dir = runtime.paths.encode_compression_benchmark_dir
    runtime.remove_in_exp(benchmark_dir)
    benchmark_report = CompressionBenchmark(
        frames_dir=paths.prepare_frames_dir,
        output_dir=benchmark_dir,
        fps=getattr(runtime.args, "fps", runtime.spec.fps),
        bitrate=getattr(runtime.args, "video_bitrate", "10M"),
        hvm_payload_dir=payload_dir,
        hvm_algorithmic_time=float(formal_hvm_algorithmic_sec) + float(payload_write_sec),
        exphub_root=runtime.exphub_root,
        exp_dir=paths.exp_dir,
    ).run()
    ensure_file(benchmark_report["raw_zip"], "compression benchmark raw zip")
    ensure_file(benchmark_report["h265_video"], "compression benchmark H.265 video")
    ensure_file(benchmark_report["hvm_payload_zip"], "compression benchmark Ours payload zip")
    ensure_file(benchmark_report["benchmark_report"], "compression benchmark report")
    log_info(
        "compression benchmark encode stage done: frames={} fps={} bitrate={} zip={} h265={} vlmem={}".format(
            int(benchmark_report.get("frame_count", 0) or 0),
            int(benchmark_report.get("fps", 0) or 0),
            str(benchmark_report.get("bitrate", "")),
            _relative_to_exp(runtime, benchmark_report["raw_zip"]),
            _relative_to_exp(runtime, benchmark_report["h265_video"]),
            _relative_to_exp(runtime, benchmark_report["hvm_payload_zip"]),
        )
    )


def _run_single_encode(runtime, paths, log_name="encode.log"):
    total_started_wall = time.time()
    total_started_perf = time.perf_counter()
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

    log_info("encode motion state start")
    phase_started = time.perf_counter()
    motion_segments = build_motion_segments(
        prepare_result=prepare_result,
        frames_dir=paths.prepare_frames_dir,
        out_path=motion_segments_path,
    )
    motion_segment_sec = float(time.perf_counter() - phase_started)

    log_info("encode visual anchor tracking start backend=openclip_image_embedding")
    phase_started = time.perf_counter()
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
    semantic_anchor_sec = float(time.perf_counter() - phase_started)
    ensure_file(semantic_anchors_path, "semantic anchors")
    semantic_anchors = read_json_dict(semantic_anchors_path)
    if not semantic_anchors:
        raise RuntimeError("invalid semantic anchors: {}".format(semantic_anchors_path))

    log_info("encode generation unit start")
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

    log_info("encode synthetic prompt start")
    prompts = build_prompts(
        generation_units=generation_units,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        frames_dir=paths.prepare_frames_dir,
        out_path=prompts_path,
        prompt_manifest_path=runtime.exphub_root / "config" / "prompt_manifest.json",
    )

    log_info("encode result writer start")
    encode_profile = {
        "version": 1,
        "total_sec": float(time.perf_counter() - total_started_perf),
        "motion_segment_sec": float(motion_segment_sec),
        "semantic_anchor_sec": float(semantic_anchor_sec),
        "result_writer_sec": 0.0,
    }
    phase_started = time.perf_counter()
    result_path = write_encode_outputs(
        runtime=runtime,
        prepare_result=prepare_result,
        motion_segments=motion_segments,
        semantic_anchors=semantic_anchors,
        generation_units=generation_units,
        prompts=prompts,
        elapsed_sec=float(time.time() - total_started_wall),
        paths=paths,
        encode_profile=encode_profile,
    )
    result_writer_sec = float(time.perf_counter() - phase_started)
    total_sec = float(time.perf_counter() - total_started_perf)
    encode_profile["result_writer_sec"] = float(result_writer_sec)
    encode_profile["total_sec"] = float(total_sec)
    _patch_encode_result_profile(result_path, encode_profile)
    formal_hvm_algorithmic_sec = float(total_sec)
    for path, label in [
        (paths.encode_motion_segments_path, "motion segments"),
        (paths.encode_semantic_anchors_path, "semantic anchors"),
        (paths.encode_generation_units_path, "generation units"),
        (paths.encode_prompts_path, "prompts"),
        (paths.encode_result_path, "encode result"),
        (paths.encode_overview_path, "encode overview"),
    ]:
        ensure_file(path, label)

    benchmark_enabled = bool(getattr(runtime.args, "encode_motion_benchmark", False))
    if benchmark_enabled:
        formal_encode_sec = float(total_sec)
        benchmark_report = run_motion_benchmark(
            prepare_result=prepare_result,
            frames_dir=paths.prepare_frames_dir,
            encode_dir=paths.encode_dir,
            exp_dir=paths.exp_dir,
            formal_encode_sec_without_benchmark=formal_encode_sec,
        )
        _log_motion_benchmark_report(benchmark_report)
        motion_benchmark_sec = float(dict(benchmark_report).get("motion_benchmark_sec", 0.0) or 0.0)
        total_sec = float(time.perf_counter() - total_started_perf)
        encode_profile["benchmark_enabled"] = True
        encode_profile["motion_benchmark_sec"] = float(motion_benchmark_sec)
        encode_profile["formal_encode_sec_without_benchmark"] = float(formal_encode_sec)
        encode_profile["benchmark_overhead_note"] = "motion_benchmark_sec is opt-in experimental overhead, not default PC-only inference cost"
        encode_profile["motion_benchmark_report"] = "encode/motion_benchmark_report.json"
        encode_profile["total_sec"] = float(total_sec)
        _patch_encode_result_profile(result_path, encode_profile)
        for path, label in [
            (paths.encode_dir / "motion_benchmark_report.json", "motion benchmark report"),
            (paths.encode_dir / "motion_benchmark.csv", "motion benchmark csv"),
            (paths.encode_dir / "motion_benchmark_overview.png", "motion benchmark overview"),
        ]:
            ensure_file(path, label)

    _run_infer_payload_hooks(runtime, paths, generation_units, prompts, formal_hvm_algorithmic_sec)

    log_info(
        "encode profile: motion={:.2f}s semantic={:.2f}s writer={:.2f}s total={:.2f}s".format(
            float(motion_segment_sec),
            float(semantic_anchor_sec),
            float(result_writer_sec),
            float(total_sec),
        )
    )
    log_info("encode done: {}".format(Path(result_path).resolve()))
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
    clip_count = sum(int(item.get("clip_count", 0) or 0) for item in entries)
    payload = {
        "mode": "train",
        "scope": str(runtime.paths.scope),
        "dataset": str(runtime.spec.dataset),
        "run_id": str(runtime.spec.exp_name),
        "sequence_count": int(len(entries)),
        "clip_count": int(clip_count),
        "ok_count": int(ok_count),
        "failed_count": int(failed_count),
        "skipped_count": int(skipped_count),
        "sequences": entries,
        "trainset_dir": "trainset",
        "train_metadata_path": "trainset/train_metadata.json",
        "stats_path": "trainset/stats.json",
    }
    write_json_atomic(runtime.paths.encode_dataset_index_path, payload, indent=2)
    return payload


def _relative_to_exp(runtime, path):
    target = Path(path).resolve()
    try:
        return target.relative_to(runtime.paths.exp_dir.resolve()).as_posix()
    except Exception:
        return str(target)


def _train_encode_ok_entry(runtime, sequence, paths, result, export_stats):
    generation_units = dict(result.get("generation_units") or {})
    units = list(generation_units.get("units") or [])
    return {
        "sequence": str(sequence),
        "status": "ok",
        "error_message": "",
        "encode_result_path": _relative_to_exp(runtime, paths.encode_result_path),
        "generation_units_path": _relative_to_exp(runtime, paths.encode_generation_units_path),
        "prompts_path": _relative_to_exp(runtime, paths.encode_prompts_path),
        "unit_count": int(len(units)),
        "clip_count": int(dict(export_stats or {}).get("clip_count", 0) or 0),
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
        "clip_count": 0,
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
        "clip_count": 0,
    }


def _run_train_encode(runtime):
    if not runtime.paths.prepare_dataset_index_path.is_file():
        raise RuntimeError(
            "prepare dataset index not found, run train prepare first: {}".format(
                runtime.paths.prepare_dataset_index_path
            )
        )
    prepare_index = read_json_dict(runtime.paths.prepare_dataset_index_path)
    if not prepare_index:
        raise RuntimeError("invalid prepare dataset index: {}".format(runtime.paths.prepare_dataset_index_path))
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
            _write_train_encode_index(runtime, entries)
            runtime.write_meta_snapshot()
            raise first_error

        paths = _train_sequence_encode_paths(runtime, sequence)
        try:
            result = _run_single_encode(runtime, paths, log_name="encode_{}.log".format(sequence))
            export_stats = train_export.export_sequence(
                sequence=sequence,
                prepare_result=result["prepare_result"],
                prepare_frames_dir=paths.prepare_frames_dir,
                generation_units=result["generation_units"],
                prompts=result["prompts"],
                generation_units_path=paths.encode_generation_units_path,
                prompts_path=paths.encode_prompts_path,
                encode_result_path=paths.encode_result_path,
            )
            successful_sequences += 1
            entries.append(_train_encode_ok_entry(runtime, sequence, paths, result, export_stats))
        except Exception as exc:
            entries.append(_train_encode_failed_entry(sequence, exc))
            if first_error is None:
                first_error = exc
            _write_train_encode_index(runtime, entries)
            runtime.write_meta_snapshot()
            raise

    index_payload = _write_train_encode_index(runtime, entries)
    if successful_sequences <= 0:
        raise RuntimeError("train encode produced no successful sequences")
    train_export.write_indexes(sequence_count=successful_sequences)
    runtime.write_meta_snapshot()
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
        if bool(getattr(runtime.args, "compression_benchmark", False)):
            log_warn("compression benchmark is infer-only; skipping for train mode")
        return _run_train_encode(runtime)
    raise RuntimeError("unsupported encode mode: {}".format(mode))
