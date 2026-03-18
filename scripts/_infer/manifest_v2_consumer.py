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

    resolved = []
    for seg in range(count):
        resolved.append(
            {
                "seg": int(seg),
                "final_prompt": str(prompt),
                "final_neg_prompt": str(negative_prompt),
                "prompt_source": str(source),
                "num_inference_steps": int(default_num_inference_steps),
                "guidance_scale": float(default_guidance_scale),
            }
        )
    return resolved
