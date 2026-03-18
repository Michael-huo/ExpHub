from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .templates import SIDE_STRUCTURE_PRIORITY

PROMPT_PROFILE_VERSION = 1

SCENE_TYPES = [
    "park_walkway",
    "campus_path",
    "orchard_row",
    "field_path",
    "road_edge",
    "corridor",
    "indoor_walkway",
    "unknown",
]  # type: List[str]
SURFACE_TYPES = ["pavement", "concrete", "brick", "dirt", "mixed", "unknown"]  # type: List[str]
SIDE_STRUCTURES = [
    "grass",
    "trees",
    "shrubs",
    "walls",
    "columns",
    "fence",
    "buildings",
    "crops",
    "soil_edges",
    "hedges",
]  # type: List[str]
LIGHTING_TYPES = ["daylight", "overcast", "indoor_even", "unknown"]  # type: List[str]
DYNAMIC_RISKS = ["low", "people", "vehicles", "animals", "mixed"]  # type: List[str]
REPETITION_RISKS = ["low", "medium", "high"]  # type: List[str]
PROFILE_CONFIDENCES = ["low", "medium", "high"]  # type: List[str]

CONFIDENCE_WEIGHTS = {"low": 1, "medium": 2, "high": 3}  # type: Dict[str, int]

_NON_ALNUM_RE = re.compile(r"[^a-z0-9_]+")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def default_prompt_profile():
    # type: () -> Dict[str, object]
    return {
        "version": int(PROMPT_PROFILE_VERSION),
        "scene_type": "unknown",
        "surface_type": "unknown",
        "side_structures": [],
        "lighting_type": "unknown",
        "dynamic_risk": "low",
        "repetition_risk": "medium",
        "profile_confidence": "low",
    }


def build_profile_instruction():
    # type: () -> str
    return (
        "Classify this frame into a closed-set PromptProfile for first-person walkway video generation.\n"
        "Return JSON only with keys: scene_type, surface_type, side_structures, lighting_type, dynamic_risk, repetition_risk, profile_confidence.\n"
        "scene_type must be one of: park_walkway, campus_path, orchard_row, field_path, road_edge, corridor, indoor_walkway, unknown.\n"
        "surface_type must be one of: pavement, concrete, brick, dirt, mixed, unknown.\n"
        "side_structures must be an array with up to 2 items chosen only from: grass, trees, shrubs, walls, columns, fence, buildings, crops, soil_edges, hedges.\n"
        "lighting_type must be one of: daylight, overcast, indoor_even, unknown.\n"
        "dynamic_risk must be one of: low, people, vehicles, animals, mixed.\n"
        "repetition_risk must be one of: low, medium, high.\n"
        "profile_confidence must be one of: low, medium, high.\n"
        "Do not output descriptions, explanations, colors, local decorations, or any free text outside the JSON object.\n"
        "If uncertain, use unknown and low."
    )


def _normalize_key(text):
    # type: (object) -> str
    value = str(text or "").strip().lower()
    value = value.replace("soil edges", "soil_edges")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = _NON_ALNUM_RE.sub("_", value).strip("_")
    return value


def _coerce_choice(value, allowed, default):
    # type: (object, List[str], str) -> str
    key = _normalize_key(value)
    if key in allowed:
        return key
    return str(default)


def _coerce_side_structures(value):
    # type: (object) -> List[str]
    items = []  # type: List[str]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        text = str(value or "").strip()
        if not text:
            raw_items = []
        else:
            raw_items = re.split(r"[\n,;/]+", text)
    seen = set()
    for item in raw_items:
        key = _normalize_key(item)
        if key not in SIDE_STRUCTURES or key in seen:
            continue
        seen.add(key)
        items.append(key)
    return _sort_side_structures(items)[:2]


def _sort_side_structures(items):
    # type: (List[str]) -> List[str]
    seen = set()
    out = []  # type: List[str]
    for key in SIDE_STRUCTURE_PRIORITY:
        if key in list(items or []) and key not in seen:
            seen.add(key)
            out.append(str(key))
    return out


def normalize_prompt_profile(payload):
    # type: (object) -> Dict[str, object]
    base = default_prompt_profile()
    data = payload if isinstance(payload, dict) else {}
    base["scene_type"] = _coerce_choice(data.get("scene_type", ""), SCENE_TYPES, "unknown")
    base["surface_type"] = _coerce_choice(data.get("surface_type", ""), SURFACE_TYPES, "unknown")
    base["side_structures"] = _sort_side_structures(_coerce_side_structures(data.get("side_structures", [])))
    base["lighting_type"] = _coerce_choice(data.get("lighting_type", ""), LIGHTING_TYPES, "unknown")
    base["dynamic_risk"] = _coerce_choice(data.get("dynamic_risk", ""), DYNAMIC_RISKS, "low")
    base["repetition_risk"] = _coerce_choice(data.get("repetition_risk", ""), REPETITION_RISKS, "medium")
    base["profile_confidence"] = _coerce_choice(data.get("profile_confidence", ""), PROFILE_CONFIDENCES, "low")
    return base


def _extract_json_dict(text):
    # type: (str) -> Optional[Dict[str, object]]
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    match = _JSON_BLOCK_RE.search(raw)
    if match is not None:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    return None


def _scan_choice(raw_text, allowed, default):
    # type: (str, List[str], str) -> str
    text = " " + _normalize_key(raw_text).replace("_", " ") + " "
    for item in allowed:
        token = " " + str(item).replace("_", " ") + " "
        if token in text:
            return str(item)
    return str(default)


def _scan_side_structures(raw_text):
    # type: (str) -> List[str]
    text = " " + _normalize_key(raw_text).replace("_", " ") + " "
    found = []  # type: List[str]
    for item in SIDE_STRUCTURES:
        token = " " + str(item).replace("_", " ") + " "
        if token in text:
            found.append(str(item))
        if len(found) >= 2:
            break
    return found


def parse_profile_response(raw_text):
    # type: (object) -> Dict[str, object]
    raw = str(raw_text or "").strip()
    parsed = _extract_json_dict(raw)
    if parsed is not None:
        return normalize_prompt_profile(parsed)
    return normalize_prompt_profile(
        {
            "scene_type": _scan_choice(raw, SCENE_TYPES, "unknown"),
            "surface_type": _scan_choice(raw, SURFACE_TYPES, "unknown"),
            "side_structures": _scan_side_structures(raw),
            "lighting_type": _scan_choice(raw, LIGHTING_TYPES, "unknown"),
            "dynamic_risk": _scan_choice(raw, DYNAMIC_RISKS, "low"),
            "repetition_risk": _scan_choice(raw, REPETITION_RISKS, "medium"),
            "profile_confidence": _scan_choice(raw, PROFILE_CONFIDENCES, "low"),
        }
    )


def _vote_scalar(candidates, field_name, allowed, default, ignore_default=False):
    # type: (List[Dict[str, object]], str, List[str], str, bool) -> str
    scores = {}  # type: Dict[str, int]
    first_seen = {}  # type: Dict[str, int]
    for idx, candidate in enumerate(candidates):
        value = str(candidate.get(field_name, default) or default)
        if value not in allowed:
            value = str(default)
        if ignore_default and value == default:
            continue
        if value not in first_seen:
            first_seen[value] = int(idx)
        weight = int(CONFIDENCE_WEIGHTS.get(str(candidate.get("profile_confidence", "low")), 1))
        scores[value] = int(scores.get(value, 0)) + weight
    if not scores:
        return str(default)
    best_value = str(default)
    best_rank = None
    for value in allowed:
        score = int(scores.get(value, 0))
        rank = (
            score,
            0 if value == default else 1,
            -int(first_seen.get(value, 999999)),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_value = str(value)
    return best_value


def _vote_side_structures(candidates):
    # type: (List[Dict[str, object]]) -> List[str]
    scores = {}  # type: Dict[str, int]
    first_seen = {}  # type: Dict[str, int]
    for idx, candidate in enumerate(candidates):
        weight = int(CONFIDENCE_WEIGHTS.get(str(candidate.get("profile_confidence", "low")), 1))
        for item in list(candidate.get("side_structures", []) or []):
            value = str(item)
            if value not in SIDE_STRUCTURES:
                continue
            if value not in first_seen:
                first_seen[value] = int(idx)
            scores[value] = int(scores.get(value, 0)) + weight
    ranked = sorted(
        list(scores.keys()),
        key=lambda key: (
            -int(scores.get(key, 0)),
            SIDE_STRUCTURE_PRIORITY.index(key),
            int(first_seen.get(key, 999999)),
        ),
    )
    return _sort_side_structures([str(item) for item in ranked[:2]])


def _count_values(candidates, field_name, default):
    # type: (List[Dict[str, object]], str, str) -> Dict[str, int]
    counts = {}  # type: Dict[str, int]
    for candidate in candidates:
        value = str(candidate.get(field_name, default) or default)
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _count_side_presence(candidates):
    # type: (List[Dict[str, object]]) -> Dict[str, int]
    counts = {}  # type: Dict[str, int]
    for candidate in candidates:
        seen = set()
        for value in list(candidate.get("side_structures", []) or []):
            key = str(value)
            if key not in SIDE_STRUCTURES or key in seen:
                continue
            seen.add(key)
            counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _has_obvious_conflict(counts, default, total):
    # type: (Dict[str, int], str, int) -> bool
    active = []
    for key, value in counts.items():
        if str(key) == str(default):
            continue
        if int(value) <= 0:
            continue
        active.append((str(key), int(value)))
    if len(active) <= 1:
        return False
    active.sort(key=lambda item: (-int(item[1]), item[0]))
    if int(active[1][1]) >= max(2, (int(total) + 2) // 3):
        return True
    return False


def aggregate_prompt_profiles(candidates):
    # type: (List[Dict[str, object]]) -> Dict[str, object]
    cleaned = [normalize_prompt_profile(item) for item in list(candidates or []) if isinstance(item, dict)]
    if not cleaned:
        return default_prompt_profile()

    total = int(len(cleaned))
    majority = int(total // 2) + 1
    profile = default_prompt_profile()
    profile["scene_type"] = _vote_scalar(cleaned, "scene_type", SCENE_TYPES, "unknown", ignore_default=True)
    profile["surface_type"] = _vote_scalar(cleaned, "surface_type", SURFACE_TYPES, "unknown", ignore_default=True)
    profile["side_structures"] = _vote_side_structures(cleaned)
    profile["lighting_type"] = _vote_scalar(cleaned, "lighting_type", LIGHTING_TYPES, "unknown", ignore_default=True)
    profile["dynamic_risk"] = _vote_scalar(cleaned, "dynamic_risk", DYNAMIC_RISKS, "low", ignore_default=False)
    profile["repetition_risk"] = _vote_scalar(cleaned, "repetition_risk", REPETITION_RISKS, "medium", ignore_default=False)

    scene_counts = _count_values(cleaned, "scene_type", "unknown")
    surface_counts = _count_values(cleaned, "surface_type", "unknown")
    side_counts = _count_side_presence(cleaned)
    confidence_counts = _count_values(cleaned, "profile_confidence", "low")

    scene_majority = int(scene_counts.get(str(profile["scene_type"]), 0)) >= majority and str(profile["scene_type"]) != "unknown"
    surface_majority = int(surface_counts.get(str(profile["surface_type"]), 0)) >= majority and str(profile["surface_type"]) != "unknown"
    top_side_hits = 0
    if list(profile.get("side_structures", []) or []):
        top_side_hits = int(max([side_counts.get(item, 0) for item in list(profile.get("side_structures", []) or [])]))
    side_consistent = (not list(profile.get("side_structures", []) or [])) or top_side_hits >= majority
    low_majority = int(confidence_counts.get("low", 0)) >= majority
    scene_conflict = _has_obvious_conflict(scene_counts, "unknown", total)
    surface_conflict = _has_obvious_conflict(surface_counts, "unknown", total)
    side_conflict = len([key for key, value in side_counts.items() if int(value) >= max(2, majority - 1)]) > 2 and not side_consistent
    matching_core_frames = 0
    for item in cleaned:
        if str(item.get("scene_type", "")) == str(profile["scene_type"]) and str(item.get("surface_type", "")) == str(profile["surface_type"]):
            matching_core_frames += 1

    if scene_conflict or surface_conflict or side_conflict:
        profile["profile_confidence"] = "low"
    elif scene_majority and surface_majority and side_consistent and matching_core_frames >= majority and not low_majority:
        profile["profile_confidence"] = "high"
    elif scene_majority or surface_majority or matching_core_frames >= majority:
        profile["profile_confidence"] = "medium"
    else:
        profile["profile_confidence"] = "low"
    return profile
