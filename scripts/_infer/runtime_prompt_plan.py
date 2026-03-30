from __future__ import annotations

import json
import re
from typing import Dict, List


PROMPT_SOURCE_RUNTIME_PLAN = "runtime_prompt_plan"

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


def _normalize_segment_overrides(payload, default_prompt, default_negative_prompt, source):
    # type: (dict, str, str, str) -> List[dict]
    override_rows = list(payload.get("segments") or [])

    normalized = []
    for idx, raw_item in enumerate(override_rows):
        item = _as_dict(raw_item)
        seg = _safe_int(item.get("seg", item.get("deploy_segment_id", item.get("segment_id", idx))), idx)
        item_prompt = (
            _collapse_ws(item.get("resolved_prompt", ""))
            or _collapse_ws(item.get("prompt", ""))
            or _collapse_ws(item.get("final_prompt", ""))
            or str(default_prompt)
        )
        item_negative_prompt = (
            _collapse_ws(item.get("negative_prompt", ""))
            or _collapse_ws(item.get("final_neg_prompt", ""))
            or str(default_negative_prompt)
        )
        normalized.append(
            {
                "seg": int(seg),
                "final_prompt": str(item_prompt),
                "final_neg_prompt": str(item_negative_prompt),
                "prompt_source": _collapse_ws(item.get("prompt_source", "")) or str(source),
                "num_inference_steps": item.get("num_inference_steps"),
                "guidance_scale": item.get("guidance_scale"),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
                "motion_trend": item.get("motion_trend"),
                "deploy_segment_id": item.get("deploy_segment_id", item.get("segment_id")),
                "match_source": item.get("match_source"),
                "prompt_strength": item.get("prompt_strength"),
                "base_prompt": item.get("base_prompt"),
                "local_prompt": item.get("local_prompt"),
                "resolved_prompt": item.get("resolved_prompt"),
            }
        )
    return normalized


def load_runtime_prompt_plan_for_infer(path, default_prompt="", default_negative_prompt=""):
    # type: (str, str, str) -> dict
    with open(path, "r", encoding="utf-8") as fobj:
        payload = json.load(fobj)
    if not isinstance(payload, dict):
        raise ValueError("runtime prompt plan must be a JSON object")

    base_prompt = _collapse_ws(payload.get("base_prompt", "")) or _collapse_ws(payload.get("prompt", ""))
    negative_prompt = _collapse_ws(payload.get("negative_prompt", ""))
    source = str(payload.get("source", "") or "").strip() or PROMPT_SOURCE_RUNTIME_PLAN

    if not base_prompt:
        base_prompt = _collapse_ws(default_prompt)
    if not negative_prompt:
        negative_prompt = _collapse_ws(default_negative_prompt)
    if not base_prompt:
        raise ValueError("runtime prompt plan must contain base_prompt")

    return {
        "version": int(payload.get("version", 1) or 1),
        "schema": str(payload.get("schema", "") or "runtime_prompt_plan.v1"),
        "prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "source": str(source),
        "segment_overrides": _normalize_segment_overrides(payload, base_prompt, negative_prompt, source),
        "_raw": payload,
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
        resolved.append(
            {
                "seg": int(seg),
                "final_prompt": _collapse_ws(override_item.get("final_prompt", "")) or str(prompt),
                "final_neg_prompt": _collapse_ws(override_item.get("final_neg_prompt", "")) or str(negative_prompt),
                "prompt_source": _collapse_ws(override_item.get("prompt_source", "")) or str(source),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "state_segment_id": override_item.get("state_segment_id"),
                "state_label": override_item.get("state_label"),
                "motion_trend": override_item.get("motion_trend"),
                "deploy_segment_id": override_item.get("deploy_segment_id"),
                "match_source": override_item.get("match_source"),
                "prompt_strength": override_item.get("prompt_strength"),
            }
        )
    return resolved
