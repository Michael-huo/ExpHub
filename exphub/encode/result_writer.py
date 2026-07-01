from __future__ import annotations

from pathlib import Path

from exphub.common.io import list_frames_sorted, write_json_atomic


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _motion_label_counts(motion_segments):
    counts = {}
    for item in list(_as_dict(motion_segments).get("motion_states") or []):
        label = str(_as_dict(item).get("motion_label", "") or "unknown")
        counts[label] = int(counts.get(label, 0)) + 1
    return counts


def _build_profile(encode_profile, motion_segments):
    profile = {}
    if isinstance(encode_profile, dict):
        for key in ["version", "total_sec", "motion_segment_sec", "semantic_anchor_sec", "result_writer_sec"]:
            if key in encode_profile:
                profile[key] = encode_profile[key]
    if profile or isinstance(encode_profile, dict):
        profile["version"] = int(profile.get("version", 1) or 1)

    motion_profile = _as_dict(_as_dict(_as_dict(motion_segments).get("summary")).get("profile"))
    if motion_profile:
        profile["motion"] = dict(motion_profile)
    return profile


def _artifact_rel(paths, attr_name):
    if paths is None:
        defaults = {
            "encode_motion_segments_path": "encode/motion_segments.json",
            "encode_semantic_anchors_path": "encode/semantic_anchors.json",
            "encode_generation_units_path": "encode/generation_units.json",
            "encode_prompts_path": "encode/prompts.json",
            "encode_result_path": "encode/encode_result.json",
        }
        return defaults.get(str(attr_name), str(attr_name))
    path = Path(getattr(paths, attr_name)).resolve()
    exp_dir = Path(getattr(paths, "exp_dir", path.parent.parent)).resolve()
    try:
        return path.relative_to(exp_dir).as_posix()
    except Exception:
        return path.name


def _tree_size(path):
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += int(item.stat().st_size)
        except Exception:
            pass
    return int(total)


def _frame_bytes(frames_dir):
    frames = list_frames_sorted(frames_dir)
    total = 0
    for frame in frames:
        try:
            total += int(frame.stat().st_size)
        except Exception:
            pass
    return int(total), int(len(frames))


def _payload_metrics(paths, generation_units, payload_report):
    report = _as_dict(payload_report)
    if not report:
        return {}
    payload_dir = Path(report.get("payload_dir", ""))
    raw_bytes, raw_frame_count = _frame_bytes(paths.prepare_frames_dir)
    payload_bytes = _tree_size(payload_dir)
    boundary_frame_bytes = _tree_size(payload_dir / "frames")
    json_payload_bytes = payload_bytes - boundary_frame_bytes
    ratio = float(payload_bytes) / float(raw_bytes) if raw_bytes > 0 else None
    reduction = (100.0 * (1.0 - ratio)) if ratio is not None else None
    units = list(_as_dict(generation_units).get("units") or [])
    metrics = {
        "raw_bytes": int(raw_bytes),
        "payload_bytes": int(payload_bytes),
        "raw_frame_count": int(raw_frame_count),
        "transmitted_frame_count": int(report.get("frame_count", 0) or 0),
        "generation_unit_count": int(len(units)),
        "unit_boundary_count": int(len(list(report.get("boundary_indices") or []))),
        "boundary_frame_bytes": int(boundary_frame_bytes),
        "json_payload_bytes": int(max(0, json_payload_bytes)),
        "payload_ratio": ratio,
        "reduction_pct": reduction,
    }
    metrics["payload"] = dict(metrics)
    metrics["payload"]["payload_dir"] = "encode/hvm_payload"
    return metrics


def _build_encode_result(
    motion_segments,
    semantic_anchors,
    generation_units,
    prompts,
    paths=None,
    encode_profile=None,
    payload_report=None,
):
    units = list(_as_dict(generation_units).get("units") or [])
    prompt_units = list(_as_dict(prompts).get("units") or [])
    semantic_summary = _as_dict(_as_dict(semantic_anchors).get("summary"))
    semantic_policy = _as_dict(_as_dict(semantic_anchors).get("policy"))
    unit_summary = _as_dict(_as_dict(generation_units).get("summary"))
    result = {
        "source": "encode.result",
        "num_motion_states": int(len(list(_as_dict(motion_segments).get("motion_states") or []))),
        "num_semantic_states": int(semantic_summary.get("semantic_state_count", 0) or 0),
        "visual_anchor_count": int(semantic_summary.get("visual_anchor_count", 0) or 0),
        "coverage_gap_count": int(semantic_summary.get("coverage_gap_count", 0) or 0),
        "num_generation_units": int(len(units)),
        "num_prompt_units": int(len(prompt_units)),
        "motion_labels": _motion_label_counts(motion_segments),
        "unit_lengths": [int(item.get("length", item.get("duration_frames", 0)) or 0) for item in units],
        "unit_length_guard_count": int(unit_summary.get("unit_length_guard_count", 0) or 0),
        "max_unit_span_frames": int(semantic_policy.get("max_unit_span_frames", 0) or 0),
        "prompt_schema": "prompts",
        "prompt_profile": str(_as_dict(prompts).get("prompt_profile", "base_motion_prompt") or ""),
        "prompt_source": "prompts.prompt_positive",
        "semantic_state_source": str(_as_dict(semantic_anchors).get("source", "") or ""),
        "anchor_backend": "image_embedding_visual_anchor",
        "artifacts": {
            "motion_segments": _artifact_rel(paths, "encode_motion_segments_path"),
            "semantic_anchors": _artifact_rel(paths, "encode_semantic_anchors_path"),
            "generation_units": _artifact_rel(paths, "encode_generation_units_path"),
            "prompts": _artifact_rel(paths, "encode_prompts_path"),
            "encode_result": _artifact_rel(paths, "encode_result_path"),
            "motion_overview": _artifact_rel(paths, "encode_motion_overview_path"),
            "payload_dir": "encode/hvm_payload",
        },
    }
    if paths is not None:
        result.update(_payload_metrics(paths, generation_units, payload_report))
    profile = _build_profile(encode_profile, motion_segments)
    if profile:
        result["profile"] = profile
    return result


def write_encode_outputs(
    runtime,
    prepare_result,
    motion_segments,
    semantic_anchors,
    generation_units,
    prompts,
    elapsed_sec=0.0,
    paths=None,
    encode_profile=None,
    payload_report=None,
):
    del prepare_result, elapsed_sec
    out_paths = paths or runtime.paths
    out_paths.encode_dir.mkdir(parents=True, exist_ok=True)

    encode_result = _build_encode_result(
        motion_segments,
        semantic_anchors,
        generation_units,
        prompts,
        paths=out_paths,
        encode_profile=encode_profile,
        payload_report=payload_report,
    )

    write_json_atomic(out_paths.encode_motion_segments_path, motion_segments, indent=2)
    write_json_atomic(out_paths.encode_semantic_anchors_path, semantic_anchors, indent=2)
    write_json_atomic(out_paths.encode_generation_units_path, generation_units, indent=2)
    write_json_atomic(out_paths.encode_prompts_path, prompts, indent=2)
    write_json_atomic(out_paths.encode_result_path, encode_result, indent=2)
    return out_paths.encode_result_path
