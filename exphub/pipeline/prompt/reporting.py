from __future__ import annotations

import hashlib
from pathlib import Path

from exphub.common.io import write_json_atomic


REPORT_FILENAME = "report.json"
OBSOLETE_PROMPT_OUTPUT_NAMES = [
    "profile.json",
    "step_meta.json",
    "final" "_prompt.json",
    "deploy_to_state" "_prompt_map.json",
]


def _sha1_bytes(payload_bytes):
    # type: (bytes) -> str
    return hashlib.sha1(payload_bytes).hexdigest()


def _count_by_key(items, key):
    # type: (list, str) -> dict
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get(key, "") or "").strip() or "unknown"
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _relative_path(base_dir, target_path):
    # type: (Path, Path) -> str
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _state_prompt_statistics(state_prompt_manifest, runtime_prompt_plan):
    # type: (dict, dict) -> dict
    state_segments = list((state_prompt_manifest or {}).get("state_segments") or [])
    runtime_segments = list((runtime_prompt_plan or {}).get("segments") or [])
    return {
        "state_segment_count": int(len(state_segments)),
        "deploy_segment_count": int(len(runtime_segments)),
        "state_label_counts": dict(_count_by_key(state_segments, "state_label")),
        "motion_trend_counts": dict(_count_by_key(state_segments, "motion_trend")),
        "deploy_match_source_counts": dict(_count_by_key(runtime_segments, "match_source")),
        "runtime_state_label_counts": dict(_count_by_key(runtime_segments, "state_label")),
    }


def build_prompt_report(
    prompt_dir,
    aggregated_profile,
    base_prompt_payload,
    state_prompt_manifest,
    runtime_prompt_plan,
    backend_meta,
    backend_name,
    backend_python_phase,
    model_record,
    dtype,
    sample_mode,
    num_images_requested,
    selected_paths,
    frame_files_count,
    fps,
    frame_records,
    errors,
    total_sec,
    avg_prompt_sec,
):
    # type: (...) -> dict
    prompt_dir = Path(prompt_dir).resolve()
    base_prompt_path = (prompt_dir / "base_prompt.json").resolve()
    state_prompt_manifest_path = (prompt_dir / "state_prompt_manifest.json").resolve()
    runtime_prompt_plan_path = (prompt_dir / "runtime_prompt_plan.json").resolve()

    base_prompt_bytes = base_prompt_path.read_bytes()
    state_prompt_manifest_bytes = state_prompt_manifest_path.read_bytes()
    runtime_prompt_plan_bytes = runtime_prompt_plan_path.read_bytes()

    return {
        "report_schema_version": "prompt_report.v1",
        "step": "prompt",
        "prompt_status": "success",
        "backend": str(backend_name),
        "model_dir": str((backend_meta or {}).get("model_dir", "") or ""),
        "model_id": str((backend_meta or {}).get("model_id", "") or ""),
        "dtype": str((backend_meta or {}).get("dtype", "") or dtype or ""),
        "processor_load_sec": float((backend_meta or {}).get("processor_load_sec", 0.0) or 0.0),
        "model_load_sec": float((backend_meta or {}).get("model_load_sec", 0.0) or 0.0),
        "prompt_gen_total_sec": float(total_sec),
        "avg_prompt_sec_per_frame": float(avg_prompt_sec),
        "backend_python_phase": str(backend_python_phase or ""),
        "sample_mode": str(sample_mode),
        "num_images_requested": int(num_images_requested),
        "num_images_used": int(len(list(selected_paths or []))),
        "frames_count": int(frame_files_count),
        "fps": int(fps) if fps is not None else None,
        "profile_version": int((aggregated_profile or {}).get("version", 0) or 0),
        "base_prompt_source": str((base_prompt_payload or {}).get("source", "") or ""),
        "representative_frames": [Path(path_text).name for path_text in list(selected_paths or [])],
        "representative_indices": [
            int(item.get("frame_idx"))
            for item in list(frame_records or [])
            if isinstance(item, dict) and item.get("frame_idx") is not None
        ],
        "frame_candidates": list(frame_records or []),
        "errors": list(errors or []),
        "error_count": int(len(list(errors or []))),
        "base_prompt_path": str(base_prompt_path),
        "base_prompt_size": int(len(base_prompt_bytes)),
        "base_prompt_sha1": _sha1_bytes(base_prompt_bytes),
        "state_prompt_manifest_path": str(state_prompt_manifest_path),
        "state_prompt_manifest_size": int(len(state_prompt_manifest_bytes)),
        "state_prompt_manifest_sha1": _sha1_bytes(state_prompt_manifest_bytes),
        "runtime_prompt_plan_path": str(runtime_prompt_plan_path),
        "runtime_prompt_plan_size": int(len(runtime_prompt_plan_bytes)),
        "runtime_prompt_plan_sha1": _sha1_bytes(runtime_prompt_plan_bytes),
        "outputs": {
            "bytes_sum": 0,
            "report_bytes_sum": 0,
            "base_prompt_bytes_sum": int(len(base_prompt_bytes)),
            "state_prompt_manifest_bytes_sum": int(len(state_prompt_manifest_bytes)),
            "runtime_prompt_plan_bytes_sum": int(len(runtime_prompt_plan_bytes)),
            "report_file_count": 1,
            "base_prompt_file_count": 1,
            "state_prompt_manifest_file_count": 1,
            "runtime_prompt_plan_file_count": 1,
        },
        "profile": dict(aggregated_profile or {}),
        "rules_hit": list((base_prompt_payload or {}).get("rules_hit", []) or []),
        "base_prompt_preview": str((base_prompt_payload or {}).get("base_prompt", "") or ""),
        "negative_prompt_preview": str((base_prompt_payload or {}).get("negative_prompt", "") or ""),
        "state_prompt_summary": dict((state_prompt_manifest or {}).get("summary", {}) or {}),
        "state_prompt_statistics": _state_prompt_statistics(state_prompt_manifest, runtime_prompt_plan),
        "model": str(model_record or ""),
        "frames_dir": "",
        "backend_summary": {
            "backend": str(backend_name),
            "model": str(model_record or ""),
            "model_dir": str((backend_meta or {}).get("model_dir", "") or ""),
            "model_id": str((backend_meta or {}).get("model_id", "") or ""),
            "dtype": str((backend_meta or {}).get("dtype", "") or dtype or ""),
            "backend_python_phase": str(backend_python_phase or ""),
            "processor_load_sec": float((backend_meta or {}).get("processor_load_sec", 0.0) or 0.0),
            "model_load_sec": float((backend_meta or {}).get("model_load_sec", 0.0) or 0.0),
        },
        "sampling_summary": {
            "sample_mode": str(sample_mode),
            "num_images_requested": int(num_images_requested),
            "num_images_used": int(len(list(selected_paths or []))),
            "frames_count": int(frame_files_count),
            "fps": int(fps) if fps is not None else None,
            "representative_frames": [Path(path_text).name for path_text in list(selected_paths or [])],
            "representative_indices": [
                int(item.get("frame_idx"))
                for item in list(frame_records or [])
                if isinstance(item, dict) and item.get("frame_idx") is not None
            ],
        },
        "prompt_generation_summary": {
            "profile_version": int((aggregated_profile or {}).get("version", 0) or 0),
            "base_prompt_source": str((base_prompt_payload or {}).get("source", "") or ""),
            "rules_hit": list((base_prompt_payload or {}).get("rules_hit", []) or []),
            "prompt_gen_total_sec": float(total_sec),
            "avg_prompt_sec_per_frame": float(avg_prompt_sec),
            "error_count": int(len(list(errors or []))),
            "base_prompt_preview": str((base_prompt_payload or {}).get("base_prompt", "") or ""),
            "negative_prompt_preview": str((base_prompt_payload or {}).get("negative_prompt", "") or ""),
        },
        "source_files": {
            "base_prompt": _relative_path(prompt_dir.parent, base_prompt_path),
            "state_prompt_manifest": _relative_path(prompt_dir.parent, state_prompt_manifest_path),
            "runtime_prompt_plan": _relative_path(prompt_dir.parent, runtime_prompt_plan_path),
        },
        "artifact_contract": {
            "formal_files": [
                "base_prompt.json",
                "state_prompt_manifest.json",
                "runtime_prompt_plan.json",
                REPORT_FILENAME,
            ],
            "obsolete_outputs_pruned": True,
            "obsolete_output_count": int(len(OBSOLETE_PROMPT_OUTPUT_NAMES)),
        },
    }


def write_prompt_report(prompt_dir, report):
    # type: (Path, dict) -> Path
    prompt_dir = Path(prompt_dir).resolve()
    report_path = prompt_dir / REPORT_FILENAME
    report_obj = dict(report or {})
    report_obj["report_path"] = str(report_path)
    last_size = None
    for _ in range(3):
        write_json_atomic(report_path, report_obj, indent=2)
        report_bytes = report_path.read_bytes()
        report_size = int(len(report_bytes))
        outputs = dict(report_obj.get("outputs", {}) or {})
        outputs["report_bytes_sum"] = report_size
        outputs["bytes_sum"] = int(
            int(outputs.get("report_bytes_sum", 0) or 0)
            + int(outputs.get("base_prompt_bytes_sum", 0) or 0)
            + int(outputs.get("state_prompt_manifest_bytes_sum", 0) or 0)
            + int(outputs.get("runtime_prompt_plan_bytes_sum", 0) or 0)
        )
        report_obj["outputs"] = outputs
        report_obj["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, report_obj, indent=2)
    return report_path


def cleanup_legacy_prompt_outputs(prompt_dir):
    # type: (Path) -> None
    prompt_dir = Path(prompt_dir).resolve()
    for rel_name in OBSOLETE_PROMPT_OUTPUT_NAMES:
        target = (prompt_dir / rel_name).resolve()
        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
        except Exception:
            continue
