from __future__ import annotations

import json
import re
from typing import Dict, List


PROMPT_SOURCE_PROFILE_V1 = "prompt_profile_v1"
PROMPT_SOURCE_LEGACY_FILE = "legacy_prompt_file"

_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_ws(text):
    # type: (object) -> str
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


def _safe_int(value, default=None):
    # type: (object, int) -> int
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _normalize_segment_overrides(payload, prompt, negative_prompt, source):
    # type: (dict, str, str, str) -> List[dict]
    override_rows = list(payload.get("segment_overrides") or [])
    if not override_rows:
        override_rows = list(payload.get("segments") or [])

    normalized = []
    for idx, raw_item in enumerate(override_rows):
        item = _as_dict(raw_item)
        seg = _safe_int(item.get("seg", item.get("segment_id", idx)), idx)
        item_prompt = (
            _collapse_ws(item.get("prompt", ""))
            or _collapse_ws(item.get("final_prompt", ""))
            or _collapse_ws(item.get("final_prompt_preview", ""))
            or str(prompt)
        )
        item_negative_prompt = (
            _collapse_ws(item.get("negative_prompt", ""))
            or _collapse_ws(item.get("final_neg_prompt", ""))
            or _collapse_ws(item.get("final_neg_prompt_preview", ""))
            or str(negative_prompt)
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
            }
        )
    return normalized


def load_prompt_manifest_for_infer(path, default_prompt="", default_negative_prompt=""):
    # type: (str, str, str) -> dict
    with open(path, "r", encoding="utf-8") as fobj:
        payload = json.load(fobj)
    if not isinstance(payload, dict):
        raise ValueError("prompt file must be a JSON object")

    prompt = _collapse_ws(payload.get("prompt", ""))
    negative_prompt = _collapse_ws(payload.get("negative_prompt", ""))
    source = str(payload.get("source", "") or "").strip() or PROMPT_SOURCE_PROFILE_V1

    if not prompt:
        prompt = _collapse_ws(payload.get("base_prompt", "")) or _collapse_ws(default_prompt)
        if prompt:
            source = PROMPT_SOURCE_LEGACY_FILE
    if not negative_prompt:
        negative_prompt = _collapse_ws(payload.get("base_neg_prompt", "")) or _collapse_ws(default_negative_prompt)
        if negative_prompt and source == PROMPT_SOURCE_PROFILE_V1 and not payload.get("prompt", ""):
            source = PROMPT_SOURCE_LEGACY_FILE

    if not prompt:
        raise ValueError("prompt file must contain prompt")

    return {
        "version": int(payload.get("version", 1) or 1),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "profile": dict(payload.get("profile", {}) or {}),
        "source": source,
        "segment_overrides": _normalize_segment_overrides(payload, prompt, negative_prompt, source),
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
    source = str(manifest_info.get("source", "") or PROMPT_SOURCE_PROFILE_V1)
    override_map = {}
    for raw_item in list(manifest_info.get("segment_overrides") or []):
        item = _as_dict(raw_item)
        seg = _safe_int(item.get("seg"), -1)
        if seg < 0:
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
            }
        )
    return resolved
