from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_ws(text):
    # type: (object) -> str
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


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


def _load_json_object(path_obj):
    # type: (Path) -> Tuple[Dict[str, object], str]
    path = Path(path_obj).resolve()
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, "json_error: {}".format(exc)
    if not isinstance(payload, dict):
        return {}, "not_object"
    return payload, ""


def _relative_path(base_dir, target_path):
    # type: (Path, Path) -> str
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


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


def build_prompt_resolution_for_infer(prompt_dir, infer_dir, execution_segments):
    # type: (Path, Path, List[Dict[str, object]]) -> Dict[str, object]
    prompt_dir = Path(prompt_dir).resolve()
    infer_dir = Path(infer_dir).resolve()
    exp_dir = prompt_dir.parent.resolve()

    base_prompt_path = (prompt_dir / "base_prompt.json").resolve()
    state_manifest_path = (prompt_dir / "state_prompt_manifest.json").resolve()
    runtime_plan_path = (prompt_dir / "runtime_prompt_plan.json").resolve()

    runtime_payload, runtime_err = _load_json_object(runtime_plan_path)
    if runtime_err:
        raise ValueError("invalid runtime_prompt_plan.json: {} ({})".format(runtime_plan_path, runtime_err))

    base_payload, base_err = _load_json_object(base_prompt_path)
    if base_err:
        raise ValueError("invalid base_prompt.json: {} ({})".format(base_prompt_path, base_err))

    state_payload, state_err = _load_json_object(state_manifest_path)
    if state_err:
        raise ValueError("invalid state_prompt_manifest.json: {} ({})".format(state_manifest_path, state_err))

    plan_segments = {}
    for idx, raw_item in enumerate(list(runtime_payload.get("segments") or [])):
        item = _as_dict(raw_item)
        deploy_segment_id = _safe_int(item.get("deploy_segment_id", item.get("segment_id", idx)), idx)
        if deploy_segment_id is None:
            continue
        plan_segments[int(deploy_segment_id)] = item

    resolution_items = []
    warnings = []  # type: List[str]
    matched_execution_segment_count = 0
    base_prompt_text = _collapse_ws(runtime_payload.get("base_prompt", "")) or _collapse_ws(base_payload.get("base_prompt", ""))
    base_negative_prompt = _collapse_ws(runtime_payload.get("negative_prompt", "")) or _collapse_ws(
        base_payload.get("negative_prompt", "")
    )

    if not base_prompt_text:
        raise ValueError("runtime_prompt_plan.json must contain base_prompt")

    for idx, raw_exec_segment in enumerate(list(execution_segments or [])):
        exec_segment = _as_dict(raw_exec_segment)
        seg = _safe_int(exec_segment.get("seg", idx), idx)
        deploy_segment_id = _safe_int(exec_segment.get("segment_id", seg), seg)
        if seg is None or deploy_segment_id is None:
            raise ValueError("execution segment missing seg/segment_id at index {}".format(idx))
        plan_item = _as_dict(plan_segments.get(int(deploy_segment_id), {}))
        if not plan_item:
            raise ValueError("runtime prompt plan missing deploy segment {}".format(int(deploy_segment_id)))

        plan_start = _safe_int(plan_item.get("start_frame"), None)
        plan_end = _safe_int(plan_item.get("end_frame"), None)
        exec_start = _safe_int(exec_segment.get("start_idx"), None)
        exec_end = _safe_int(exec_segment.get("end_idx"), None)
        if plan_start is not None and exec_start is not None and int(plan_start) != int(exec_start):
            raise ValueError(
                "runtime prompt plan start mismatch for deploy segment {}: plan={} exec={}".format(
                    int(deploy_segment_id),
                    int(plan_start),
                    int(exec_start),
                )
            )
        if plan_end is not None and exec_end is not None and int(plan_end) != int(exec_end):
            raise ValueError(
                "runtime prompt plan end mismatch for deploy segment {}: plan={} exec={}".format(
                    int(deploy_segment_id),
                    int(plan_end),
                    int(exec_end),
                )
            )

        matched_execution_segment_count += 1
        resolved_prompt = _collapse_ws(plan_item.get("resolved_prompt", ""))
        local_prompt = _collapse_ws(plan_item.get("local_prompt", ""))
        negative_prompt = _collapse_ws(plan_item.get("negative_prompt", "")) or str(base_negative_prompt)
        if not resolved_prompt:
            warnings.append("deploy segment {} missing resolved_prompt; fallback to base_prompt".format(int(deploy_segment_id)))
            resolved_prompt = str(base_prompt_text)
        resolution_items.append(
            {
                "seg": int(seg),
                "deploy_segment_id": int(deploy_segment_id),
                "start_frame": plan_start,
                "end_frame": plan_end,
                "state_segment_id": _safe_int(plan_item.get("state_segment_id"), None),
                "state_label": _collapse_ws(plan_item.get("state_label", "")) or None,
                "motion_trend": _collapse_ws(plan_item.get("motion_trend", "")) or None,
                "match_source": _collapse_ws(plan_item.get("match_source", "")) or None,
                "prompt_source": _collapse_ws(plan_item.get("prompt_source", "")) or "runtime_prompt_plan",
                "base_prompt": _collapse_ws(plan_item.get("base_prompt", "")) or str(base_prompt_text),
                "local_prompt": str(local_prompt),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": str(negative_prompt),
                "prompt_strength": float(plan_item.get("prompt_strength", 0.5) or 0.5),
                "prompt_preview": _prompt_preview(resolved_prompt),
            }
        )

    prompt_source_counts = _count_by_key(resolution_items, "prompt_source")
    state_motion_trend_counts = _count_by_key(
        [item for item in resolution_items if item.get("motion_trend")], "motion_trend"
    )
    state_label_counts = _count_by_key(
        [item for item in resolution_items if item.get("state_label")], "state_label"
    )

    debug_payload = {
        "version": int(runtime_payload.get("version", 1) or 1),
        "schema": str(runtime_payload.get("schema", "") or "runtime_prompt_plan.v1"),
        "state_prompt_enabled": bool(len(resolution_items) > 0),
        "state_prompt_segment_count": int(len(list(state_payload.get("state_segments") or []))),
        "matched_execution_segment_count": int(matched_execution_segment_count),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "state_label_counts": dict(state_label_counts),
        "prompt_file_used": str(runtime_plan_path),
        "source_files": {
            "base_prompt": _relative_path(exp_dir, base_prompt_path),
            "state_prompt_manifest": _relative_path(exp_dir, state_manifest_path),
            "runtime_prompt_plan": _relative_path(exp_dir, runtime_plan_path),
        },
        "warnings": list(warnings),
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
        "prompt_file_path": runtime_plan_path,
        "base_prompt_path": base_prompt_path,
        "state_prompt_manifest_path": state_manifest_path,
        "runtime_prompt_plan_path": runtime_plan_path,
        "state_prompt_enabled": bool(len(resolution_items) > 0),
        "state_prompt_segment_count": int(len(list(state_payload.get("state_segments") or []))),
        "matched_execution_segment_count": int(matched_execution_segment_count),
        "prompt_file_version": int(runtime_payload.get("version", 1) or 1),
        "prompt_file_source": str(runtime_payload.get("source", "") or "runtime_prompt_plan_v1"),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "state_label_counts": dict(state_label_counts),
        "prompt_resolution": dict(debug_payload),
        "segment_resolutions": list(resolution_items),
        "warnings": list(warnings),
    }
