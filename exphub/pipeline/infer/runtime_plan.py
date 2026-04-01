from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from exphub.common.io import ensure_file, write_json_atomic


PROMPT_SOURCE_RUNTIME_PLAN = "runtime_prompt_plan"


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def _safe_int(value, default=None):
    # type: (object, object) -> object
    try:
        return int(value)
    except Exception:
        return default


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _prompt_preview(text, limit=160):
    # type: (str, int) -> str
    collapsed = _collapse_ws(text)
    if len(collapsed) <= int(limit):
        return collapsed
    return collapsed[: max(0, int(limit) - 3)].rstrip() + "..."


def _count_by_key(items, key):
    # type: (List[Dict[str, object]], str) -> Dict[str, int]
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = _collapse_ws(item.get(key, "")) or "unknown"
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


def _normalize_runtime_segment(item, idx, default_prompt, default_negative_prompt, default_source, default_backend):
    # type: (Dict[str, object], int, str, str, str, str) -> Dict[str, object]
    deploy_segment_id = _safe_int(item.get("deploy_segment_id", item.get("segment_id", item.get("seg", idx))), idx)
    start_idx = _safe_int(item.get("start_idx", item.get("start_frame")), None)
    end_idx = _safe_int(item.get("end_idx", item.get("end_frame")), None)
    if start_idx is None or end_idx is None:
        raise ValueError("runtime prompt segment missing start/end for deploy segment {}".format(int(deploy_segment_id)))
    if int(end_idx) < int(start_idx):
        raise ValueError("runtime prompt segment has invalid range for deploy segment {}".format(int(deploy_segment_id)))

    raw_start_idx = _safe_int(item.get("raw_start_idx", item.get("raw_start_frame", start_idx)), start_idx)
    raw_end_idx = _safe_int(item.get("raw_end_idx", item.get("raw_end_frame", end_idx)), end_idx)
    deploy_gap = _safe_int(item.get("deploy_gap"), int(end_idx) - int(start_idx))
    raw_gap = _safe_int(item.get("raw_gap"), int(raw_end_idx) - int(raw_start_idx))
    num_frames = _safe_int(item.get("num_frames"), int(end_idx) - int(start_idx) + 1)

    resolved_prompt = (
        _collapse_ws(item.get("resolved_prompt", ""))
        or _collapse_ws(item.get("prompt", ""))
        or _collapse_ws(item.get("final_prompt", ""))
        or str(default_prompt)
    )
    negative_prompt = (
        _collapse_ws(item.get("negative_prompt", ""))
        or _collapse_ws(item.get("final_neg_prompt", ""))
        or str(default_negative_prompt)
    )
    prompt_source = _collapse_ws(item.get("prompt_source", "")) or str(default_source)
    execution_backend = _collapse_ws(item.get("execution_backend", "")) or str(default_backend)

    return {
        "seg": int(_safe_int(item.get("seg"), int(deploy_segment_id))),
        "segment_id": int(_safe_int(item.get("segment_id"), int(deploy_segment_id))),
        "deploy_segment_id": int(deploy_segment_id),
        "schedule_source": _collapse_ws(item.get("schedule_source", "")) or PROMPT_SOURCE_RUNTIME_PLAN,
        "execution_backend": str(execution_backend),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_frame": int(start_idx),
        "end_frame": int(end_idx),
        "raw_start_idx": int(raw_start_idx),
        "raw_end_idx": int(raw_end_idx),
        "raw_start_frame": int(raw_start_idx),
        "raw_end_frame": int(raw_end_idx),
        "deploy_start_idx": int(_safe_int(item.get("deploy_start_idx"), int(start_idx))),
        "deploy_end_idx": int(_safe_int(item.get("deploy_end_idx"), int(end_idx))),
        "raw_gap": int(raw_gap),
        "deploy_gap": int(deploy_gap),
        "num_frames": int(num_frames),
        "boundary_shift": int(_safe_int(item.get("boundary_shift"), 0)),
        "gap_error": int(_safe_int(item.get("gap_error"), 0)),
        "state_segment_id": _safe_int(item.get("state_segment_id"), None),
        "state_label": _collapse_ws(item.get("state_label", "")) or None,
        "motion_trend": _collapse_ws(item.get("motion_trend", "")) or None,
        "match_source": _collapse_ws(item.get("match_source", "")) or None,
        "prompt_source": str(prompt_source),
        "base_prompt": _collapse_ws(item.get("base_prompt", "")) or str(default_prompt),
        "local_prompt": _collapse_ws(item.get("local_prompt", "")),
        "resolved_prompt": str(resolved_prompt),
        "negative_prompt": str(negative_prompt),
        "prompt_strength": float(item.get("prompt_strength", 0.5) or 0.5),
        "final_prompt": str(resolved_prompt),
        "final_neg_prompt": str(negative_prompt),
        "num_inference_steps": item.get("num_inference_steps"),
        "guidance_scale": item.get("guidance_scale"),
    }


def load_runtime_prompt_plan(path, default_prompt="", default_negative_prompt=""):
    # type: (str, str, str) -> Dict[str, object]
    prompt_path = ensure_file(path, "runtime prompt plan")
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime prompt plan must be a JSON object")

    base_prompt = _collapse_ws(payload.get("base_prompt", "")) or _collapse_ws(payload.get("prompt", ""))
    negative_prompt = _collapse_ws(payload.get("negative_prompt", ""))
    if not base_prompt:
        base_prompt = _collapse_ws(default_prompt)
    if not negative_prompt:
        negative_prompt = _collapse_ws(default_negative_prompt)
    if not base_prompt:
        raise ValueError("runtime prompt plan must contain base_prompt")

    source = str(payload.get("source", "") or "").strip() or PROMPT_SOURCE_RUNTIME_PLAN
    execution_backend = str(payload.get("execution_backend", "") or "").strip()
    schedule_source = str(payload.get("schedule_source", "") or "").strip() or PROMPT_SOURCE_RUNTIME_PLAN

    segments = []
    for idx, raw_item in enumerate(list(payload.get("segments") or [])):
        item = _as_dict(raw_item)
        if not item:
            continue
        segments.append(
            _normalize_runtime_segment(
                item=item,
                idx=idx,
                default_prompt=base_prompt,
                default_negative_prompt=negative_prompt,
                default_source=source,
                default_backend=execution_backend,
            )
        )

    if not segments:
        raise ValueError("runtime prompt plan must contain at least one segment")

    segments.sort(key=lambda item: (int(item.get("seg", 0)), int(item.get("start_idx", 0))))
    return {
        "version": int(payload.get("version", 1) or 1),
        "schema": str(payload.get("schema", "") or "runtime_prompt_plan.v1"),
        "prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "source": str(source),
        "schedule_source": str(schedule_source),
        "execution_backend": str(execution_backend),
        "segments": segments,
        "segment_overrides": list(segments),
        "source_files": dict(payload.get("source_files", {}) or {}),
        "_raw": payload,
        "_path": prompt_path,
    }


def load_runtime_prompt_plan_for_infer(path, default_prompt="", default_negative_prompt=""):
    # type: (str, str, str) -> dict
    plan = load_runtime_prompt_plan(path, default_prompt=default_prompt, default_negative_prompt=default_negative_prompt)
    return {
        "version": int(plan.get("version", 1) or 1),
        "schema": str(plan.get("schema", "") or "runtime_prompt_plan.v1"),
        "prompt": str(plan.get("prompt", "")),
        "negative_prompt": str(plan.get("negative_prompt", "")),
        "source": str(plan.get("source", "")),
        "segment_overrides": list(plan.get("segment_overrides", []) or []),
        "_raw": dict(plan.get("_raw", {}) or {}),
    }


def resolve_segment_overrides(
    manifest_info,
    num_segments,
    default_prompt,
    default_negative_prompt,
    default_num_inference_steps,
    default_guidance_scale,
):
    # type: (dict, int, str, str, int, float) -> List[dict]
    count = max(0, int(num_segments))
    prompt = _collapse_ws(manifest_info.get("prompt", "")) or _collapse_ws(default_prompt)
    negative_prompt = _collapse_ws(manifest_info.get("negative_prompt", "")) or _collapse_ws(default_negative_prompt)
    source = str(manifest_info.get("source", "") or PROMPT_SOURCE_RUNTIME_PLAN)
    override_map = {}
    for raw_item in list(manifest_info.get("segment_overrides") or []):
        item = _as_dict(raw_item)
        seg = _safe_int(item.get("seg"), -1)
        if seg is None or int(seg) < 0:
            continue
        override_map[int(seg)] = item

    resolved = []
    for seg in range(count):
        override_item = _as_dict(override_map.get(int(seg), {}))
        num_inference_steps = override_item.get("num_inference_steps", default_num_inference_steps)
        if num_inference_steps in (None, ""):
            num_inference_steps = default_num_inference_steps
        guidance_scale = override_item.get("guidance_scale", default_guidance_scale)
        if guidance_scale in (None, ""):
            guidance_scale = default_guidance_scale
        final_prompt = _collapse_ws(override_item.get("final_prompt", "")) or _collapse_ws(override_item.get("resolved_prompt", "")) or str(prompt)
        final_neg_prompt = _collapse_ws(override_item.get("final_neg_prompt", "")) or _collapse_ws(override_item.get("negative_prompt", "")) or str(negative_prompt)
        resolved.append(
            {
                "seg": int(seg),
                "final_prompt": str(final_prompt),
                "final_neg_prompt": str(final_neg_prompt),
                "prompt_source": _collapse_ws(override_item.get("prompt_source", "")) or str(source),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "state_segment_id": override_item.get("state_segment_id"),
                "state_label": override_item.get("state_label"),
                "motion_trend": override_item.get("motion_trend"),
                "deploy_segment_id": override_item.get("deploy_segment_id"),
                "match_source": override_item.get("match_source"),
                "prompt_strength": override_item.get("prompt_strength"),
                "base_prompt": override_item.get("base_prompt"),
                "local_prompt": override_item.get("local_prompt"),
                "resolved_prompt": override_item.get("resolved_prompt"),
            }
        )
    return resolved


def build_execution_plan(runtime_prompt_plan):
    # type: (Dict[str, object]) -> Dict[str, object]
    plan = dict(runtime_prompt_plan or {})
    execution_segments = []
    for item in list(plan.get("segments") or []):
        execution_segments.append(
            {
                "seg": int(item.get("seg", 0) or 0),
                "segment_id": int(item.get("segment_id", item.get("deploy_segment_id", 0)) or 0),
                "schedule_source": str(item.get("schedule_source", "") or plan.get("schedule_source", PROMPT_SOURCE_RUNTIME_PLAN)),
                "execution_backend": str(item.get("execution_backend", "") or plan.get("execution_backend", "")),
                "raw_start_idx": int(item.get("raw_start_idx", item.get("start_idx", 0)) or 0),
                "raw_end_idx": int(item.get("raw_end_idx", item.get("end_idx", 0)) or 0),
                "deploy_start_idx": int(item.get("deploy_start_idx", item.get("start_idx", 0)) or 0),
                "deploy_end_idx": int(item.get("deploy_end_idx", item.get("end_idx", 0)) or 0),
                "start_idx": int(item.get("start_idx", 0) or 0),
                "end_idx": int(item.get("end_idx", 0) or 0),
                "raw_gap": int(item.get("raw_gap", 0) or 0),
                "deploy_gap": int(item.get("deploy_gap", 0) or 0),
                "num_frames": int(item.get("num_frames", 0) or 0),
                "boundary_shift": int(item.get("boundary_shift", 0) or 0),
                "gap_error": int(item.get("gap_error", 0) or 0),
            }
        )
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schedule_source": str(plan.get("schedule_source", "") or PROMPT_SOURCE_RUNTIME_PLAN),
        "execution_backend": str(plan.get("execution_backend", "") or ""),
        "segments": execution_segments,
    }


def write_execution_plan(infer_dir, execution_plan):
    # type: (Path, Dict[str, object]) -> Path
    infer_root = Path(infer_dir).resolve()
    infer_root.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="execution_plan_", suffix=".json", dir=str(infer_root))
    Path(tmp_path).unlink()
    write_json_atomic(Path(tmp_path), execution_plan, indent=2)
    return Path(tmp_path).resolve()


def build_prompt_resolution(runtime_prompt_plan, execution_segments, exp_dir):
    # type: (Dict[str, object], List[Dict[str, object]], Path) -> Dict[str, object]
    exp_root = Path(exp_dir).resolve()
    plan_segments = {}
    for item in list(runtime_prompt_plan.get("segments") or []):
        plan_segments[int(item.get("deploy_segment_id", item.get("segment_id", 0)) or 0)] = dict(item)

    resolution_items = []
    for idx, raw_exec_segment in enumerate(list(execution_segments or [])):
        exec_segment = _as_dict(raw_exec_segment)
        seg = _safe_int(exec_segment.get("seg", idx), idx)
        deploy_segment_id = _safe_int(exec_segment.get("segment_id", seg), seg)
        plan_item = _as_dict(plan_segments.get(int(deploy_segment_id), {}))
        if not plan_item:
            raise ValueError("runtime prompt plan missing deploy segment {}".format(int(deploy_segment_id)))
        if int(plan_item.get("start_idx", 0) or 0) != int(exec_segment.get("start_idx", 0) or 0):
            raise ValueError("runtime prompt plan start mismatch for deploy segment {}".format(int(deploy_segment_id)))
        if int(plan_item.get("end_idx", 0) or 0) != int(exec_segment.get("end_idx", 0) or 0):
            raise ValueError("runtime prompt plan end mismatch for deploy segment {}".format(int(deploy_segment_id)))

        resolution_items.append(
            {
                "seg": int(seg),
                "deploy_segment_id": int(deploy_segment_id),
                "start_frame": int(plan_item.get("start_idx", 0) or 0),
                "end_frame": int(plan_item.get("end_idx", 0) or 0),
                "state_segment_id": _safe_int(plan_item.get("state_segment_id"), None),
                "state_label": plan_item.get("state_label"),
                "motion_trend": plan_item.get("motion_trend"),
                "match_source": plan_item.get("match_source"),
                "prompt_source": str(plan_item.get("prompt_source", "") or PROMPT_SOURCE_RUNTIME_PLAN),
                "base_prompt": str(plan_item.get("base_prompt", runtime_prompt_plan.get("prompt", "")) or ""),
                "local_prompt": str(plan_item.get("local_prompt", "") or ""),
                "resolved_prompt": str(plan_item.get("resolved_prompt", "") or runtime_prompt_plan.get("prompt", "")),
                "negative_prompt": str(plan_item.get("negative_prompt", "") or runtime_prompt_plan.get("negative_prompt", "")),
                "prompt_strength": float(plan_item.get("prompt_strength", 0.5) or 0.5),
                "prompt_preview": _prompt_preview(plan_item.get("resolved_prompt", "")),
            }
        )

    prompt_source_counts = _count_by_key(resolution_items, "prompt_source")
    state_motion_trend_counts = _count_by_key([item for item in resolution_items if item.get("motion_trend")], "motion_trend")
    state_label_counts = _count_by_key([item for item in resolution_items if item.get("state_label")], "state_label")
    raw_path = runtime_prompt_plan.get("_path")

    debug_payload = {
        "version": int(runtime_prompt_plan.get("version", 1) or 1),
        "schema": str(runtime_prompt_plan.get("schema", "") or "runtime_prompt_plan.v1"),
        "state_prompt_enabled": bool(len(resolution_items) > 0),
        "state_prompt_segment_count": int(len(list(runtime_prompt_plan.get("segments") or []))),
        "matched_execution_segment_count": int(len(resolution_items)),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "state_label_counts": dict(state_label_counts),
        "prompt_file_used": str(raw_path) if raw_path is not None else "",
        "source_files": {
            "runtime_prompt_plan": (
                _relative_path(exp_root, raw_path) if raw_path is not None else ""
            ),
        },
        "warnings": [],
        "segments": [
            {
                "seg": int(item["seg"]),
                "deploy_segment_id": int(item["deploy_segment_id"]),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
                "motion_trend": item.get("motion_trend"),
                "prompt_source": str(item["prompt_source"]),
                "prompt_preview": str(item["prompt_preview"]),
                "prompt_strength": float(item.get("prompt_strength", 0.5) or 0.5),
            }
            for item in resolution_items
        ],
    }

    return {
        "prompt_file_path": raw_path,
        "runtime_prompt_plan_path": raw_path,
        "state_prompt_enabled": bool(len(resolution_items) > 0),
        "state_prompt_segment_count": int(len(list(runtime_prompt_plan.get("segments") or []))),
        "matched_execution_segment_count": int(len(resolution_items)),
        "prompt_file_version": int(runtime_prompt_plan.get("version", 1) or 1),
        "prompt_file_source": str(runtime_prompt_plan.get("source", "") or PROMPT_SOURCE_RUNTIME_PLAN),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "state_label_counts": dict(state_label_counts),
        "prompt_resolution": dict(debug_payload),
        "segment_resolutions": list(resolution_items),
        "warnings": [],
    }


def merge_prompt_resolution_into_runs_plan(plan_obj, segment_resolutions):
    # type: (dict, list) -> dict
    plan = dict(plan_obj or {})
    plan_segments = list(plan.get("segments", []) or [])
    resolution_map = {}
    for raw_item in list(segment_resolutions or []):
        if not isinstance(raw_item, dict):
            continue
        seg = raw_item.get("seg")
        if seg is None:
            continue
        resolution_map[int(seg)] = dict(raw_item)

    for idx, raw_segment in enumerate(plan_segments):
        if not isinstance(raw_segment, dict):
            continue
        seg_item = raw_segment
        seg_key = seg_item.get("seg", idx)
        resolved = resolution_map.get(int(seg_key), {})
        if not resolved:
            continue
        seg_item["prompt_source"] = str(resolved.get("prompt_source", seg_item.get("prompt_source", "")) or "")
        seg_item["deploy_segment_id"] = int(resolved.get("deploy_segment_id", seg_item.get("segment_id", idx)) or 0)
        seg_item["state_segment_id"] = resolved.get("state_segment_id")
        seg_item["state_label"] = resolved.get("state_label")
        seg_item["motion_trend"] = resolved.get("motion_trend")
        seg_item["base_prompt"] = resolved.get("base_prompt")
        seg_item["local_prompt"] = resolved.get("local_prompt")
        seg_item["resolved_prompt"] = resolved.get("resolved_prompt")
        seg_item["negative_prompt"] = resolved.get("negative_prompt")
        seg_item["prompt_strength"] = resolved.get("prompt_strength")
        seg_item["prompt_preview"] = resolved.get("prompt_preview")
    plan["segments"] = plan_segments
    return plan
