from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from _prompt.manifest_v2 import MANIFEST_SCHEMA, MANIFEST_VERSION, compile_legacy_prompts
from _schedule import extract_execution_segments_from_manifest


POLICY_SOURCE_MANIFEST_V2 = "manifest_v2_structured"
POLICY_SOURCE_LEGACY = "legacy"
POLICY_SOURCE_FALLBACK = "fallback"

_WHITESPACE_RE = re.compile(r"\s+")
_NEGATIVE_SPLIT_RE = re.compile(r"[\n,;]+")


def _collapse_ws(text):
    # type: (object) -> str
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


def _strip_terminal(text):
    # type: (object) -> str
    out = _collapse_ws(text)
    while out.endswith(".") or out.endswith(",") or out.endswith(";") or out.endswith(":"):
        out = out[:-1].strip()
    return out


def _normalize_key(text):
    # type: (object) -> str
    lowered = _strip_terminal(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _combine_prompt_blocks(blocks):
    # type: (List[object]) -> str
    out = []
    seen = set()
    for block in list(blocks or []):
        text = str(block or "").strip()
        if not text:
            continue
        key = _normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return "\n".join(out).strip()


def _split_negative_items(text):
    # type: (object) -> List[str]
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []
    out = []
    seen = set()
    for item in _NEGATIVE_SPLIT_RE.split(raw):
        cleaned = _strip_terminal(item)
        if not cleaned:
            continue
        key = _normalize_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _combine_negative_blocks(blocks):
    # type: (List[object]) -> str
    out = []
    seen = set()
    for block in list(blocks or []):
        for item in _split_negative_items(block):
            key = _normalize_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
    return ", ".join(out).strip()


def _normalize_level(value, default):
    # type: (object, str) -> str
    text = _strip_terminal(value).lower()
    if not text:
        return str(default)
    if text in ("low", "medium", "high"):
        return text
    if "high" in text or "strong" in text:
        return "high"
    if "medium" in text or "moderate" in text or "mid" in text:
        return "medium"
    if "low" in text or "weak" in text:
        return "low"
    return str(default)


def _risk_negative_boost(risk_level):
    # type: (str) -> List[str]
    level = _normalize_level(risk_level, "low")
    if level == "high":
        return [
            "temporal instability",
            "viewpoint drift",
            "geometry collapse",
            "abrupt motion spikes",
        ]
    if level == "medium":
        return [
            "temporal instability",
            "viewpoint drift",
        ]
    return []


def _steps_from_motion(default_steps, motion_intensity):
    # type: (int, str) -> int
    base = max(1, int(default_steps))
    level = _normalize_level(motion_intensity, "medium")
    if level == "low":
        return max(32, int(base - 10))
    if level == "high":
        return int(base + 10)
    return int(base)


def _guidance_from_geometry(default_guidance, geometry_priority):
    # type: (float, str) -> float
    base = float(default_guidance)
    level = _normalize_level(geometry_priority, "medium")
    if level == "low":
        return round(max(1.0, base - 0.5), 3)
    if level == "high":
        return round(base + 0.5, 3)
    return round(base, 3)


def is_manifest_v2(manifest_obj):
    # type: (object) -> bool
    if not isinstance(manifest_obj, dict):
        return False
    if int(manifest_obj.get("version", 0) or 0) != int(MANIFEST_VERSION):
        return False
    return str(manifest_obj.get("schema", "") or "").strip() == str(MANIFEST_SCHEMA)


def _segment_map(manifest_obj):
    # type: (dict) -> Dict[int, dict]
    out = {}
    raw_segments = manifest_obj.get("segments", [])
    if not isinstance(raw_segments, list):
        return out
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            seg = int(item.get("seg"))
        except Exception:
            continue
        out[seg] = item
    return out


def _segment_has_structured_fields(segment):
    # type: (dict) -> bool
    if not isinstance(segment, dict):
        return False
    intent_card = segment.get("intent_card", {})
    control_hints = segment.get("control_hints", {})
    if isinstance(intent_card, dict):
        if _strip_terminal(intent_card.get("scene_anchor", "")):
            return True
        if _strip_terminal(intent_card.get("motion_intent", "")):
            return True
        if list(intent_card.get("geometry_constraints", []) or []):
            return True
        if list(intent_card.get("appearance_constraints", []) or []):
            return True
        if list(intent_card.get("suppressions", []) or []):
            return True
    if isinstance(control_hints, dict):
        if _strip_terminal(control_hints.get("motion_intensity", "")):
            return True
        if _strip_terminal(control_hints.get("geometry_priority", "")):
            return True
        if _strip_terminal(control_hints.get("risk_level", "")):
            return True
    return False


def load_prompt_manifest_for_infer(path, default_prompt="", default_negative_prompt=""):
    # type: (str, str, str) -> dict
    with open(path, "r", encoding="utf-8") as fobj:
        manifest = json.load(fobj)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    version = int(manifest.get("version", 1) or 1)
    if version not in (1, 2):
        raise ValueError("unsupported manifest version: {}".format(version))

    base_prompt = str(manifest.get("base_prompt", "") or "").strip()
    if not base_prompt:
        base_prompt = str(default_prompt or "").strip()
    if not base_prompt:
        raise ValueError("manifest.base_prompt is required and cannot be empty")

    base_neg_prompt = str(manifest.get("base_neg_prompt", "") or "").strip()
    if not base_neg_prompt:
        base_neg_prompt = str(default_negative_prompt or "").strip()

    return {
        "version": int(version),
        "schema": str(manifest.get("schema", "") or "").strip(),
        "base_prompt": base_prompt,
        "base_neg_prompt": base_neg_prompt,
        "global_invariants": dict(manifest.get("global_invariants", {}) or {}),
        "segments": _segment_map(manifest),
        "execution_segments": extract_execution_segments_from_manifest(manifest),
        "consumer_mode": POLICY_SOURCE_MANIFEST_V2 if is_manifest_v2(manifest) else POLICY_SOURCE_LEGACY,
        "_raw": manifest,
    }


def _resolve_legacy_segment(base_prompt, base_neg_prompt, segment):
    # type: (str, str, dict) -> dict
    delta_prompt = str(segment.get("delta_prompt", "") or "").strip()
    delta_neg_prompt = str(segment.get("delta_neg_prompt", "") or "").strip()
    legacy = segment.get("legacy", {})
    if isinstance(legacy, dict):
        if not delta_prompt:
            delta_prompt = str(legacy.get("delta_prompt", "") or "").strip()
        if not delta_neg_prompt:
            delta_neg_prompt = str(legacy.get("delta_neg_prompt", delta_neg_prompt) or "").strip()

    compiled = segment.get("compiled", {})
    final_prompt = _combine_prompt_blocks([base_prompt, delta_prompt])
    if not final_prompt and isinstance(compiled, dict):
        final_prompt = str(compiled.get("final_prompt_preview", "") or "").strip()
    if not final_prompt:
        final_prompt = str(base_prompt or "").strip()

    final_neg_prompt = _combine_negative_blocks([base_neg_prompt, delta_neg_prompt])
    if not final_neg_prompt and isinstance(compiled, dict):
        final_neg_prompt = str(compiled.get("final_neg_prompt_preview", "") or "").strip()
    has_legacy_signal = bool(
        delta_prompt
        or delta_neg_prompt
        or (isinstance(compiled, dict) and (compiled.get("final_prompt_preview") or compiled.get("final_neg_prompt_preview")))
    )
    source_mode = POLICY_SOURCE_LEGACY if has_legacy_signal else POLICY_SOURCE_FALLBACK
    return {
        "delta_prompt": delta_prompt,
        "delta_neg_prompt": delta_neg_prompt,
        "final_prompt": final_prompt,
        "final_neg_prompt": final_neg_prompt,
        "prompt_source": source_mode,
        "policy_source": source_mode,
        "control_hints": {
            "motion_intensity": "medium",
            "geometry_priority": "medium",
            "risk_level": "low",
        },
    }


def _resolve_structured_segment(base_prompt, base_neg_prompt, global_invariants, segment, default_num_inference_steps, default_guidance_scale):
    # type: (str, str, dict, dict, int, float) -> dict
    intent_card = dict(segment.get("intent_card", {}) or {})
    control_hints = dict(segment.get("control_hints", {}) or {})
    motion_intensity = _normalize_level(control_hints.get("motion_intensity", "medium"), "medium")
    geometry_priority = _normalize_level(control_hints.get("geometry_priority", "medium"), "medium")
    risk_level = _normalize_level(control_hints.get("risk_level", "low"), "low")
    normalized_hints = {
        "motion_intensity": motion_intensity,
        "geometry_priority": geometry_priority,
        "risk_level": risk_level,
    }

    compiled = compile_legacy_prompts(
        base_prompt=base_prompt,
        base_neg_prompt=base_neg_prompt,
        global_invariants=global_invariants,
        intent_card=intent_card,
        control_hints=normalized_hints,
    )
    delta_prompt = str(compiled.get("delta_prompt", "") or "").strip()
    delta_neg_prompt = str(compiled.get("delta_neg_prompt", "") or "").strip()
    final_prompt = _combine_prompt_blocks([base_prompt, delta_prompt])
    final_neg_prompt = _combine_negative_blocks([base_neg_prompt, delta_neg_prompt, _risk_negative_boost(risk_level)])

    return {
        "delta_prompt": delta_prompt,
        "delta_neg_prompt": delta_neg_prompt,
        "final_prompt": final_prompt,
        "final_neg_prompt": final_neg_prompt,
        "prompt_source": POLICY_SOURCE_MANIFEST_V2,
        "policy_source": POLICY_SOURCE_MANIFEST_V2,
        "control_hints": normalized_hints,
        "num_inference_steps": _steps_from_motion(default_num_inference_steps, motion_intensity),
        "guidance_scale": _guidance_from_geometry(default_guidance_scale, geometry_priority),
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
    base_prompt = str(manifest_info.get("base_prompt", "") or default_prompt or "").strip()
    base_neg_prompt = str(manifest_info.get("base_neg_prompt", "") or default_negative_prompt or "").strip()
    is_v2 = is_manifest_v2(manifest_info.get("_raw", {}))
    global_invariants = dict(manifest_info.get("global_invariants", {}) or {})
    seg_map = dict(manifest_info.get("segments", {}) or {})
    resolved = []

    for seg in range(count):
        segment = seg_map.get(seg, {})
        prompt_source = POLICY_SOURCE_FALLBACK
        policy_source = POLICY_SOURCE_FALLBACK
        delta_prompt = ""
        delta_neg_prompt = ""
        final_prompt = str(base_prompt or default_prompt or "").strip()
        final_neg_prompt = str(base_neg_prompt or default_negative_prompt or "").strip()
        control_hints = {
            "motion_intensity": "medium",
            "geometry_priority": "medium",
            "risk_level": "low",
        }
        num_inference_steps = int(default_num_inference_steps)
        guidance_scale = float(default_guidance_scale)

        if is_v2 and _segment_has_structured_fields(segment):
            structured = _resolve_structured_segment(
                base_prompt=base_prompt,
                base_neg_prompt=base_neg_prompt,
                global_invariants=global_invariants,
                segment=segment,
                default_num_inference_steps=int(default_num_inference_steps),
                default_guidance_scale=float(default_guidance_scale),
            )
            prompt_source = str(structured["prompt_source"])
            policy_source = str(structured["policy_source"])
            delta_prompt = str(structured.get("delta_prompt", "") or "").strip()
            delta_neg_prompt = str(structured.get("delta_neg_prompt", "") or "").strip()
            final_prompt = str(structured.get("final_prompt", "") or "").strip() or final_prompt
            final_neg_prompt = str(structured.get("final_neg_prompt", "") or "").strip()
            control_hints = dict(structured.get("control_hints", {}) or control_hints)
            num_inference_steps = int(structured.get("num_inference_steps", num_inference_steps))
            guidance_scale = float(structured.get("guidance_scale", guidance_scale))
        elif isinstance(segment, dict) and segment:
            legacy = _resolve_legacy_segment(base_prompt, base_neg_prompt, segment)
            prompt_source = str(legacy["prompt_source"])
            policy_source = str(legacy["policy_source"])
            delta_prompt = str(legacy.get("delta_prompt", "") or "").strip()
            delta_neg_prompt = str(legacy.get("delta_neg_prompt", "") or "").strip()
            final_prompt = str(legacy.get("final_prompt", "") or "").strip() or final_prompt
            final_neg_prompt = str(legacy.get("final_neg_prompt", "") or "").strip()
            control_hints = dict(legacy.get("control_hints", {}) or control_hints)

        final_prompt = final_prompt or str(default_prompt or "").strip()
        final_neg_prompt = final_neg_prompt or str(default_negative_prompt or "").strip()
        resolved.append(
            {
                "seg": int(seg),
                "base_prompt": base_prompt,
                "base_neg_prompt": base_neg_prompt,
                "delta_prompt": delta_prompt,
                "delta_neg_prompt": delta_neg_prompt,
                "final_prompt": final_prompt,
                "final_neg_prompt": final_neg_prompt,
                "prompt_source": prompt_source,
                "policy_source": policy_source,
                "control_hints": control_hints,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "has_delta": bool(delta_prompt or delta_neg_prompt),
            }
        )
    return resolved
