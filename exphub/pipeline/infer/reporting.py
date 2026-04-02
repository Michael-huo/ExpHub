from __future__ import annotations

import hashlib
from pathlib import Path

from exphub.common.io import write_json_atomic


REPORT_FILENAME = "report.json"
OBSOLETE_INFER_OUTPUT_NAMES = [
    "execution_plan.json",
    "prompt_manifest_resolved.json",
    "prompt_resolution.json",
    "step_meta.json",
]


def _sha1_bytes(payload_bytes):
    # type: (bytes) -> str
    return hashlib.sha1(payload_bytes).hexdigest()


def _relative_path(base_dir, target_path):
    # type: (Path, Path) -> str
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _segment_summary(plan_segments):
    # type: (list) -> dict
    deploy_gaps = []
    prompt_sources = {}
    motion_trends = {}
    start_idx = None
    end_idx = None
    for item in list(plan_segments or []):
        if not isinstance(item, dict):
            continue
        try:
            deploy_gap = int(item.get("deploy_gap", 0) or 0)
        except Exception:
            deploy_gap = 0
        if deploy_gap > 0:
            deploy_gaps.append(deploy_gap)
        name = str(item.get("prompt_source", "") or "").strip()
        if name:
            prompt_sources[name] = int(prompt_sources.get(name, 0)) + 1
        trend = str(item.get("motion_trend", "") or "").strip()
        if trend:
            motion_trends[trend] = int(motion_trends.get(trend, 0)) + 1
        try:
            item_start = int(item.get("start_idx"))
            item_end = int(item.get("end_idx"))
        except Exception:
            continue
        start_idx = item_start if start_idx is None else min(start_idx, item_start)
        end_idx = item_end if end_idx is None else max(end_idx, item_end)
    return {
        "segment_count": int(len(list(plan_segments or []))),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "deploy_gap_min": min(deploy_gaps) if deploy_gaps else None,
        "deploy_gap_max": max(deploy_gaps) if deploy_gaps else None,
        "deploy_gap_mean": (float(sum(deploy_gaps)) / float(len(deploy_gaps)) if deploy_gaps else None),
        "prompt_source_counts": prompt_sources,
        "motion_trend_counts": motion_trends,
        "segment_preview": [
            {
                "seg": item.get("seg"),
                "start_idx": item.get("start_idx"),
                "end_idx": item.get("end_idx"),
                "deploy_gap": item.get("deploy_gap"),
                "prompt_source": item.get("prompt_source"),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
                "motion_trend": item.get("motion_trend"),
            }
            for item in list(plan_segments or [])[:5]
            if isinstance(item, dict)
        ],
    }


def build_infer_report(infer_dir, runs_plan_obj, prompt_resolution, backend_meta, backend_result, runtime_summary):
    # type: (...) -> dict
    infer_dir = Path(infer_dir).resolve()
    exp_dir = infer_dir.parent.resolve()
    runs_plan_path = (infer_dir / "runs_plan.json").resolve()

    runs_plan_bytes = runs_plan_path.read_bytes()
    plan_segments = list((runs_plan_obj or {}).get("segments", []) or [])
    segment_summary = _segment_summary(plan_segments)

    return {
        "report_schema_version": "infer_report.v1",
        "step": "infer",
        "created_at": str((runtime_summary or {}).get("created_at", "") or ""),
        "infer_status": "success",
        "gpus": int((runtime_summary or {}).get("gpus", 0) or 0),
        "fps": int((runtime_summary or {}).get("fps", 0) or 0),
        "kf_gap": int((runtime_summary or {}).get("kf_gap", 0) or 0),
        "frames_avail": int((runtime_summary or {}).get("frames_avail", 0) or 0),
        "segments": int((runtime_summary or {}).get("segments", 0) or 0),
        "used_frames": int((runtime_summary or {}).get("used_frames", 0) or 0),
        "used_start_idx": int((runtime_summary or {}).get("used_start_idx", 0) or 0),
        "used_end_idx": int((runtime_summary or {}).get("used_end_idx", 0) or 0),
        "schedule_source": str((runtime_summary or {}).get("schedule_source", "") or ""),
        "execution_backend": str((runtime_summary or {}).get("execution_backend", "") or ""),
        "mean_deploy_gap": (runtime_summary or {}).get("mean_deploy_gap"),
        "runs_plan_path": str(runs_plan_path),
        "runs_plan_size": int(len(runs_plan_bytes)),
        "runs_plan_sha1": _sha1_bytes(runs_plan_bytes),
        "runtime_prompt_plan_path": str((prompt_resolution or {}).get("runtime_prompt_plan_path", "") or ""),
        "state_prompt_enabled": bool((runtime_summary or {}).get("state_prompt_enabled", False)),
        "state_prompt_segment_count": int((runtime_summary or {}).get("state_prompt_segment_count", 0) or 0),
        "matched_execution_segment_count": int((runtime_summary or {}).get("matched_execution_segment_count", 0) or 0),
        "prompt_file_version": int((runtime_summary or {}).get("prompt_file_version", 0) or 0),
        "prompt_file_source": str((runtime_summary or {}).get("prompt_file_source", "") or ""),
        "prompt_source_counts": dict((runtime_summary or {}).get("prompt_source_counts", {}) or {}),
        "state_motion_trend_counts": dict((runtime_summary or {}).get("state_motion_trend_counts", {}) or {}),
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
            "model_dir": str((backend_meta or {}).get("model_dir", "") or (backend_meta or {}).get("infer_model_dir", "") or ""),
            "model_id": str((backend_meta or {}).get("model_id", "") or (backend_meta or {}).get("infer_model_id", "") or ""),
            "config_path": str((backend_meta or {}).get("config_path", "") or (backend_meta or {}).get("infer_config_path", "") or ""),
        },
        "schedule_summary": {
            "schedule_source": str((runtime_summary or {}).get("schedule_source", "") or ""),
            "execution_backend": str((runtime_summary or {}).get("execution_backend", "") or ""),
            "frames_avail": int((runtime_summary or {}).get("frames_avail", 0) or 0),
            "used_frames": int((runtime_summary or {}).get("used_frames", 0) or 0),
            "used_start_idx": int((runtime_summary or {}).get("used_start_idx", 0) or 0),
            "used_end_idx": int((runtime_summary or {}).get("used_end_idx", 0) or 0),
            "mean_deploy_gap": (runtime_summary or {}).get("mean_deploy_gap"),
            "segment_count": int((runtime_summary or {}).get("segments", 0) or 0),
        },
        "prompt_resolution_summary": {
            "state_prompt_enabled": bool((runtime_summary or {}).get("state_prompt_enabled", False)),
            "state_prompt_segment_count": int((runtime_summary or {}).get("state_prompt_segment_count", 0) or 0),
            "matched_execution_segment_count": int((runtime_summary or {}).get("matched_execution_segment_count", 0) or 0),
            "prompt_source_counts": dict((runtime_summary or {}).get("prompt_source_counts", {}) or {}),
            "state_motion_trend_counts": dict((runtime_summary or {}).get("state_motion_trend_counts", {}) or {}),
            "state_label_counts": dict((runtime_summary or {}).get("state_label_counts", {}) or {}),
            "warnings": list((prompt_resolution or {}).get("warnings", []) or []),
        },
        "prompt_resolution": dict((prompt_resolution or {}).get("prompt_resolution", {}) or {}),
        "execution_segments_summary": segment_summary,
        "source_files": {
            "runs_plan": _relative_path(exp_dir, runs_plan_path),
            "runtime_prompt_plan": (
                _relative_path(exp_dir, Path(prompt_resolution.get("runtime_prompt_plan_path")).resolve())
                if (prompt_resolution or {}).get("runtime_prompt_plan_path")
                else ""
            ),
        },
        "artifact_contract": {
            "formal_files": ["runs_plan.json", REPORT_FILENAME],
            "obsolete_outputs_pruned": True,
            "obsolete_output_count": int(len(OBSOLETE_INFER_OUTPUT_NAMES)),
        },
    }


def write_infer_report(infer_dir, report):
    # type: (Path, dict) -> Path
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


def cleanup_legacy_infer_outputs(infer_dir):
    # type: (Path) -> None
    infer_dir = Path(infer_dir).resolve()
    for rel_name in OBSOLETE_INFER_OUTPUT_NAMES:
        target = (infer_dir / rel_name).resolve()
        try:
            if target.is_file() or target.is_symlink():
                target.unlink()
        except Exception:
            continue
