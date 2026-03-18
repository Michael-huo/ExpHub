from __future__ import annotations

import re
from typing import Dict, List

from .templates import (
    BASE_NEGATIVE_ITEMS,
    BEST_NEGATIVE_PROMPT,
    BEST_PROMPT,
    DYNAMIC_NEGATIVE_ITEMS,
    FORBIDDEN_POSITIVE_TOKENS,
    REPETITION_NEGATIVE_ITEMS,
    SCENE_PHRASE_MAP,
    SIDE_STRUCTURE_PRIORITY,
    SIDE_STRUCTURE_PHRASE_MAP,
    SURFACE_PHRASE_MAP,
)


def _join_items(items):
    # type: (List[str]) -> str
    values = [str(item) for item in list(items or []) if str(item).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return "{} and {}".format(values[0], values[1])
    return "{}, {}, and {}".format(values[0], values[1], values[2])


def _dedupe_keep_order(items):
    # type: (List[str]) -> List[str]
    out = []  # type: List[str]
    seen = set()
    for item in list(items or []):
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(item).strip())
    return out


def _canonical_side_structures(profile):
    # type: (Dict[str, object]) -> List[str]
    raw_items = [str(item) for item in list(profile.get("side_structures", []) or [])]
    ordered = []  # type: List[str]
    seen = set()
    for key in SIDE_STRUCTURE_PRIORITY:
        if key in raw_items and key not in seen:
            seen.add(key)
            ordered.append(str(key))
    return ordered


def is_best_prompt_case(profile):
    # type: (Dict[str, object]) -> bool
    side_structures = set(_canonical_side_structures(profile))
    return (
        str(profile.get("scene_type", "") or "") == "park_walkway"
        and str(profile.get("surface_type", "") or "") == "pavement"
        and "grass" in side_structures
        and "trees" in side_structures
    )


def build_texture_phrase(profile):
    # type: (Dict[str, object]) -> str
    surface_type = str(profile.get("surface_type", "unknown") or "unknown")
    items = [SURFACE_PHRASE_MAP.get(surface_type, SURFACE_PHRASE_MAP["unknown"])]
    for value in _canonical_side_structures(profile)[:2]:
        phrase = SIDE_STRUCTURE_PHRASE_MAP.get(str(value), "")
        if phrase:
            items.append(phrase)
    deduped = _dedupe_keep_order(items)[:3]
    return _join_items(deduped)


def build_prompt(profile):
    # type: (Dict[str, object]) -> str
    if is_best_prompt_case(profile):
        return str(BEST_PROMPT)
    scene_type = str(profile.get("scene_type", "unknown") or "unknown")
    scene_phrase = SCENE_PHRASE_MAP.get(scene_type, SCENE_PHRASE_MAP["unknown"])
    texture_phrase = build_texture_phrase(profile)
    sentences = [
        "First-person camera moving forward along an {}.".format(scene_phrase),
        "Photorealistic.",
        "Stable exposure and white balance.",
        "Consistent perspective and geometry, level horizon.",
        "Sharp, stable textures on {}.".format(texture_phrase),
        "No flicker, no warping, no artifacts.",
    ]
    prompt = " ".join(sentences).strip()
    lowered = prompt.lower().replace("first-person", "fp_camera")
    for token in FORBIDDEN_POSITIVE_TOKENS:
        pattern = r"\b{}\b".format(re.escape(str(token).lower()))
        if re.search(pattern, lowered) is not None:
            raise ValueError("forbidden token leaked into prompt: {}".format(token))
    return prompt


def build_negative_prompt(profile):
    # type: (Dict[str, object]) -> str
    dynamic_risk = str(profile.get("dynamic_risk", "low") or "low")
    repetition_risk = str(profile.get("repetition_risk", "medium") or "medium")
    items = list(BASE_NEGATIVE_ITEMS)
    items.extend(list(DYNAMIC_NEGATIVE_ITEMS.get(dynamic_risk, [])))
    items.extend(list(REPETITION_NEGATIVE_ITEMS.get(repetition_risk, [])))
    deduped = _dedupe_keep_order(items)
    if deduped == list(BASE_NEGATIVE_ITEMS):
        return str(BEST_NEGATIVE_PROMPT)
    return ", ".join(deduped).strip()


def build_final_prompt_payload(profile):
    # type: (Dict[str, object]) -> Dict[str, object]
    normalized_profile = dict(profile or {})
    normalized_profile["side_structures"] = _canonical_side_structures(normalized_profile)
    rules_hit = []  # type: List[str]
    if is_best_prompt_case(normalized_profile):
        prompt = str(BEST_PROMPT)
        negative_prompt = str(BEST_NEGATIVE_PROMPT)
        source = "prompt_profile_v1_canonical_best"
        rules_hit.append("canonical_best")
        rules_hit.append("best_negative_prompt_kept")
    elif str(normalized_profile.get("profile_confidence", "low") or "low") != "high":
        prompt = str(BEST_PROMPT)
        negative_prompt = str(BEST_NEGATIVE_PROMPT)
        source = "prompt_profile_v1_fallback_best"
        rules_hit.append("fallback_best_due_to_non_high_confidence")
    else:
        prompt = build_prompt(normalized_profile)
        negative_prompt = build_negative_prompt(normalized_profile)
        source = "prompt_profile_v1"
        rules_hit.append("high_confidence_slot_replace")
        if negative_prompt == str(BEST_NEGATIVE_PROMPT):
            rules_hit.append("best_negative_prompt_kept")
        else:
            rules_hit.append("negative_prompt_append_only")
    return {
        "version": 1,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "profile": normalized_profile,
        "source": source,
        "rules_hit": rules_hit,
    }
