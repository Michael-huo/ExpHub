from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _common import write_json_atomic


_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_ws(text):
    # type: (object) -> str
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


def _safe_int(value, default=None):
    # type: (object, Optional[int]) -> Optional[int]
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


def _join_prompt_parts(base_prompt, local_prompt):
    # type: (str, str) -> str
    base = _collapse_ws(base_prompt)
    local = _collapse_ws(local_prompt)
    if not base:
        return local
    if not local:
        return base
    return _collapse_ws("{} {}".format(base, local))


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


def _build_state_segments(state_payload):
    # type: (Dict[str, object]) -> Dict[int, Dict[str, object]]
    out = {}
    for idx, raw_item in enumerate(list(state_payload.get("state_segments") or [])):
        item = _as_dict(raw_item)
        state_segment_id = _safe_int(item.get("state_segment_id"), idx)
        if state_segment_id is None:
            continue
        out[int(state_segment_id)] = item
    return out


def _build_deploy_segments(deploy_payload):
    # type: (Dict[str, object]) -> Dict[int, Dict[str, object]]
    out = {}
    for idx, raw_item in enumerate(list(deploy_payload.get("deploy_segments") or [])):
        item = _as_dict(raw_item)
        deploy_segment_id = _safe_int(item.get("deploy_segment_id"), idx)
        if deploy_segment_id is None:
            continue
        out[int(deploy_segment_id)] = item
    return out


def build_prompt_resolution_for_infer(prompt_dir, infer_dir, execution_segments):
    # type: (Path, Path, List[Dict[str, object]]) -> Dict[str, object]
    prompt_dir = Path(prompt_dir).resolve()
    infer_dir = Path(infer_dir).resolve()
    exp_dir = prompt_dir.parent.resolve()

    final_prompt_path = (prompt_dir / "final_prompt.json").resolve()
    state_manifest_path = (prompt_dir / "state_prompt_manifest.json").resolve()
    deploy_map_path = (prompt_dir / "deploy_to_state_prompt_map.json").resolve()
    derived_manifest_path = (infer_dir / "prompt_manifest_resolved.json").resolve()

    final_payload, final_err = _load_json_object(final_prompt_path)
    if final_err:
        raise ValueError("invalid final_prompt.json: {} ({})".format(final_prompt_path, final_err))

    global_prompt = _collapse_ws(final_payload.get("prompt", "")) or _collapse_ws(final_payload.get("base_prompt", ""))
    negative_prompt = _collapse_ws(final_payload.get("negative_prompt", "")) or _collapse_ws(
        final_payload.get("base_neg_prompt", "")
    )
    if not global_prompt:
        raise ValueError("final_prompt.json must contain prompt: {}".format(final_prompt_path))

    original_prompt_source = _collapse_ws(final_payload.get("source", "")) or "prompt_profile_v1"
    state_prompt_files_detected = bool(state_manifest_path.is_file() and deploy_map_path.is_file())
    state_prompt_manifest = {}
    deploy_to_state_map = {}
    warnings = []  # type: List[str]

    if state_prompt_files_detected:
        state_prompt_manifest, state_err = _load_json_object(state_manifest_path)
        deploy_to_state_map, deploy_err = _load_json_object(deploy_map_path)
        if state_err:
            warnings.append("state prompt manifest invalid: {}".format(state_err))
        if deploy_err:
            warnings.append("deploy-to-state prompt map invalid: {}".format(deploy_err))
    else:
        if state_manifest_path.is_file() != deploy_map_path.is_file():
            warnings.append(
                "state prompt files incomplete: manifest_exists={} deploy_map_exists={}".format(
                    bool(state_manifest_path.is_file()),
                    bool(deploy_map_path.is_file()),
                )
            )

    state_segments = _build_state_segments(state_prompt_manifest) if not warnings and state_prompt_manifest else {}
    deploy_segments = _build_deploy_segments(deploy_to_state_map) if not warnings and deploy_to_state_map else {}
    state_prompt_resolution_active = bool(state_segments and deploy_segments)

    resolution_items = []  # type: List[Dict[str, object]]
    mapped_execution_segment_count = 0

    for idx, raw_exec_segment in enumerate(list(execution_segments or [])):
        exec_segment = _as_dict(raw_exec_segment)
        seg = _safe_int(exec_segment.get("seg"), idx)
        deploy_segment_id = _safe_int(exec_segment.get("segment_id"), seg if seg is not None else idx)
        if seg is None:
            seg = idx
        if deploy_segment_id is None:
            deploy_segment_id = seg

        resolved_prompt = str(global_prompt)
        prompt_source = "global_only"
        state_segment_id = None  # type: Optional[int]
        state_label = ""
        motion_trend = ""
        local_prompt = ""
        match_source = ""

        if state_prompt_resolution_active:
            mapping_row = _as_dict(deploy_segments.get(int(deploy_segment_id), {}))
            state_segment_id = _safe_int(mapping_row.get("state_segment_id"))
            match_source = _collapse_ws(mapping_row.get("match_source", ""))
            state_row = _as_dict(state_segments.get(int(state_segment_id), {})) if state_segment_id is not None else {}
            if state_row:
                mapped_execution_segment_count += 1
                state_label = _collapse_ws(state_row.get("state_label", ""))
                motion_trend = _collapse_ws(state_row.get("motion_trend", ""))
                local_prompt = _collapse_ws(state_row.get("prompt_text", ""))
                if local_prompt:
                    resolved_prompt = _join_prompt_parts(global_prompt, local_prompt)
                    prompt_source = "global_plus_state_local"
                else:
                    prompt_source = "global_only_fallback"
            elif mapping_row:
                prompt_source = "global_only_fallback"

        resolution_items.append(
            {
                "seg": int(seg),
                "deploy_segment_id": int(deploy_segment_id),
                "state_segment_id": int(state_segment_id) if state_segment_id is not None else None,
                "state_label": str(state_label) if state_label else None,
                "motion_trend": str(motion_trend) if motion_trend else None,
                "match_source": str(match_source) if match_source else None,
                "prompt_source": str(prompt_source),
                "prompt": str(resolved_prompt),
                "negative_prompt": str(negative_prompt),
                "local_prompt": str(local_prompt) if local_prompt else None,
                "prompt_preview": _prompt_preview(resolved_prompt),
            }
        )

    prompt_source_counts = _count_by_key(resolution_items, "prompt_source")
    state_motion_trend_counts = _count_by_key(
        [item for item in resolution_items if item.get("motion_trend")], "motion_trend"
    )
    state_prompt_enabled = bool(prompt_source_counts.get("global_plus_state_local", 0) > 0)
    prompt_manifest_mode = "global_plus_state_local" if state_prompt_enabled else "global_only"

    if prompt_manifest_mode == "global_plus_state_local":
        manifest_source = "infer_prompt_manifest_v1_global_plus_state_local"
    elif state_prompt_files_detected:
        manifest_source = "infer_prompt_manifest_v1_global_fallback"
    else:
        manifest_source = "infer_prompt_manifest_v1_global_only"

    manifest_payload = {
        "version": 1,
        "schema": "infer_prompt_manifest_v1",
        "prompt_manifest_mode": str(prompt_manifest_mode),
        "prompt": str(global_prompt),
        "negative_prompt": str(negative_prompt),
        "profile": dict(final_payload.get("profile", {}) or {}),
        "source": str(manifest_source),
        "base_prompt_source": str(original_prompt_source),
        "source_files": {
            "final_prompt": _relative_path(exp_dir, final_prompt_path),
            "state_prompt_manifest": _relative_path(exp_dir, state_manifest_path) if state_manifest_path.is_file() else None,
            "deploy_to_state_prompt_map": _relative_path(exp_dir, deploy_map_path) if deploy_map_path.is_file() else None,
        },
        "state_prompt_enabled": bool(state_prompt_enabled),
        "state_prompt_files_detected": bool(state_prompt_files_detected),
        "state_prompt_segment_count": int(len(state_segments)),
        "mapped_execution_segment_count": int(mapped_execution_segment_count),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "segment_overrides": [
            {
                "seg": int(item["seg"]),
                "segment_id": int(item["deploy_segment_id"]),
                "prompt": str(item["prompt"]),
                "prompt_source": str(item["prompt_source"]),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
                "motion_trend": item.get("motion_trend"),
                "match_source": item.get("match_source"),
            }
            for item in resolution_items
        ],
    }
    write_json_atomic(derived_manifest_path, manifest_payload, indent=2)

    debug_payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_manifest_mode": str(prompt_manifest_mode),
        "state_prompt_enabled": bool(state_prompt_enabled),
        "state_prompt_files_detected": bool(state_prompt_files_detected),
        "state_prompt_segment_count": int(len(state_segments)),
        "mapped_execution_segment_count": int(mapped_execution_segment_count),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "prompt_file_used": str(derived_manifest_path),
        "base_prompt_source": str(original_prompt_source),
        "source_files": manifest_payload.get("source_files", {}),
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
            }
            for item in resolution_items
        ],
    }
    return {
        "prompt_file_path": derived_manifest_path,
        "base_prompt_path": final_prompt_path,
        "state_prompt_manifest_path": state_manifest_path,
        "deploy_to_state_prompt_map_path": deploy_map_path,
        "prompt_manifest_mode": str(prompt_manifest_mode),
        "state_prompt_enabled": bool(state_prompt_enabled),
        "state_prompt_files_detected": bool(state_prompt_files_detected),
        "state_prompt_segment_count": int(len(state_segments)),
        "mapped_execution_segment_count": int(mapped_execution_segment_count),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_motion_trend_counts": dict(state_motion_trend_counts),
        "prompt_resolution": dict(debug_payload),
        "segment_resolutions": list(debug_payload.get("segments", [])),
        "warnings": list(warnings),
    }
