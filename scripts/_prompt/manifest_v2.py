from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple


MANIFEST_VERSION = 2
MANIFEST_SCHEMA = "prompt_manifest_v2"

_KV_SPLIT_RE = re.compile(r"[;\n]+")
_CLAUSE_SPLIT_RE = re.compile(r"[.;\n]+")
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_KEY_RE = re.compile(
    r"\b(scene_anchor|motion_intent|geometry_constraints|appearance_constraints|suppressions|control_hints|motion_intensity|geometry_priority|risk_level)\b",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")

DEFAULT_SCENE_ANCHOR = "Stable egocentric scene"
DEFAULT_MOTION_INTENSITY = "low"
DEFAULT_GEOMETRY_PRIORITY = "high"
DEFAULT_RISK_LEVEL = "low"
DEFAULT_MOTION_PHRASE = "steady forward egocentric motion along the path"
DEFAULT_GEOMETRY_PHRASES = ["continuous geometry", "consistent perspective"]

_SCENE_KEYWORDS = [
    "walkway",
    "sidewalk",
    "path",
    "road",
    "street",
    "corridor",
    "hallway",
    "room",
    "park",
    "plaza",
    "bridge",
    "stairs",
    "indoor",
    "outdoor",
    "building",
    "trail",
]
_MOTION_KEYWORDS = [
    "forward",
    "walking",
    "walk",
    "moving",
    "move",
    "turning",
    "turn",
    "approaching",
    "approach",
    "entering",
    "exiting",
    "driving",
    "ascending",
    "descending",
]
_GEOMETRY_KEYWORDS = [
    "geometry",
    "perspective",
    "horizon",
    "path",
    "corridor",
    "road",
    "lane",
    "door",
    "wall",
    "boundary",
    "straight",
    "curve",
    "alignment",
]

_STOP_TOKENS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "b",
}

_OBJECT_MOTION_HINTS = [
    "person",
    "people",
    "pedestrian",
    "man",
    "woman",
    "someone",
    "somebody",
    "dog",
    "car",
    "vehicle",
    "cyclist",
    "biker",
]


def clean_generation_text(text):
    # type: (str) -> str
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    raw = _FENCE_RE.sub(lambda match: match.group(1).strip(), raw)
    raw = raw.strip()
    if raw.lower().startswith("json\n"):
        raw = raw[5:].strip()
    return raw


def _collapse_ws(text):
    # type: (str) -> str
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).strip()


def _strip_terminal(text):
    # type: (str) -> str
    text = _collapse_ws(clean_generation_text(text))
    while text.endswith(".") or text.endswith(",") or text.endswith(";") or text.endswith(":"):
        text = text[:-1].strip()
    return text


def _ensure_sentence(text):
    # type: (str) -> str
    text = _strip_terminal(text)
    if not text:
        return ""
    if len(text) >= 2:
        text = text[0].upper() + text[1:]
    else:
        text = text.upper()
    return text + "."


def _contains_json_artifact(text):
    # type: (str) -> bool
    raw = str(text or "")
    if not raw:
        return False
    if "{" in raw or "}" in raw or "[" in raw or "]" in raw:
        return True
    if _JSON_KEY_RE.search(raw) and ":" in raw:
        return True
    return False


def _sanitize_fragment(text):
    # type: (str) -> str
    raw = clean_generation_text(text)
    if not raw:
        return ""
    raw = raw.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    raw = raw.replace('"', " ").replace("`", " ").replace(":", " ").replace("=", " ")
    raw = raw.replace("_", " ")
    raw = _JSON_KEY_RE.sub(" ", raw)
    raw = _collapse_ws(raw)
    raw = raw.strip(" ,;.-")
    trailing_fillers = [" with", " and", " of", " to", " for", " into", " on", " in", " at", " by"]
    changed = True
    while changed and raw:
        changed = False
        lowered = raw.lower()
        for suffix in trailing_fillers:
            if lowered.endswith(suffix):
                raw = raw[: -len(suffix)].rstrip(" ,;.-")
                changed = True
                break
    return raw


def _is_meaningful_fragment(text):
    # type: (str) -> bool
    raw = _sanitize_fragment(text)
    if not raw:
        return False
    if _contains_json_artifact(raw):
        return False
    raw_lower = raw.lower()
    if raw_lower in _STOP_TOKENS:
        return False
    if len(raw_lower) == 1:
        return False
    if not re.search(r"[a-zA-Z0-9]", raw):
        return False
    words = [word for word in re.split(r"\s+", raw_lower) if word]
    if len(words) == 1 and words[0] in _STOP_TOKENS:
        return False
    return True


def _post_clean_list(items):
    # type: (List[str]) -> List[str]
    out = []
    seen = set()
    for item in items:
        text = _sanitize_fragment(item)
        if not _is_meaningful_fragment(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _default_ego_motion(scene_anchor, geometry_constraints):
    # type: (str, List[str]) -> str
    context = " ".join([str(scene_anchor or "")] + list(geometry_constraints or [])).lower()
    if "corridor" in context or "hallway" in context:
        return "steady forward egocentric motion through the corridor"
    if "stairs" in context or "stair" in context:
        return "steady forward egocentric motion along the stairs"
    return DEFAULT_MOTION_PHRASE


def _normalize_motion_intent(text, scene_anchor, geometry_constraints):
    # type: (str, str, List[str]) -> str
    raw = _sanitize_fragment(text)
    if not raw:
        return _default_ego_motion(scene_anchor, geometry_constraints)

    lowered = raw.lower()
    if "ego" in lowered or "camera" in lowered or "viewer" in lowered or "first-person" in lowered:
        return raw

    if any(token in lowered for token in _OBJECT_MOTION_HINTS):
        return _default_ego_motion(scene_anchor, geometry_constraints)

    motion_verbs = ["walk", "walking", "move", "moving", "forward", "turn", "turning", "approach", "approaching"]
    if any(token in lowered for token in motion_verbs):
        return _default_ego_motion(scene_anchor, geometry_constraints)

    return raw


def _split_sentences(text):
    # type: (str) -> List[str]
    raw = clean_generation_text(text)
    if not raw:
        return []
    items = [item.strip() for item in _CLAUSE_SPLIT_RE.split(raw) if item.strip()]
    out = []
    seen = set()
    for item in items:
        norm = _sanitize_fragment(item)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _split_phrase_items(text):
    # type: (str) -> List[str]
    raw = clean_generation_text(text)
    if not raw:
        return []
    if "," in raw and raw.count(",") >= raw.count("."):
        parts = [item.strip() for item in raw.split(",")]
    else:
        parts = _split_sentences(raw)
    out = []
    seen = set()
    for item in parts:
        norm = _sanitize_fragment(item)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _normalize_text_list(value):
    # type: (object) -> List[str]
    out = []  # type: List[str]
    seen = set()
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        items = _split_phrase_items(value)
    else:
        items = []
    for item in items:
        text = _sanitize_fragment(item if isinstance(item, str) else str(item))
        if not _is_meaningful_fragment(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _extract_first_json_candidate(text):
    # type: (str) -> Tuple[str, bool]
    raw = clean_generation_text(text)
    start = raw.find("{")
    if start < 0:
        return "", False

    stack = []  # type: List[str]
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    return raw[start : idx + 1], True
            else:
                break

    return raw[start:].strip(), False


def _trim_dangling_json_tail(text):
    # type: (str) -> str
    out = str(text or "").rstrip()
    patterns = [
        r',\s*$',
        r':\s*$',
        r'"[^"]*"\s*:\s*$',
        r'"[^"]*"\s*:\s*\[\s*$',
        r'"[^"]*"\s*:\s*\{\s*$',
        r',\s*"[^"]*"\s*:\s*$',
        r',\s*"[^"]*"\s*:\s*\[\s*$',
        r',\s*"[^"]*"\s*:\s*\{\s*$',
    ]
    changed = True
    while changed and out:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", out)
            if updated != out:
                out = updated.rstrip()
                changed = True
    out = re.sub(r",\s*([\]}])", r"\1", out)
    return out.rstrip()


def _normalize_json_quotes(text):
    # type: (str) -> str
    out = str(text or "")
    out = out.replace("：", ":")
    out = re.sub(r"'([^']+)'\s*:", r'"\1":', out)
    out = re.sub(r":\s*'([^']*)'", r': "\1"', out)
    return out


def _repair_json_candidate(candidate):
    # type: (str) -> str
    out = clean_generation_text(candidate)
    if not out:
        return ""
    out = _normalize_json_quotes(out)

    stack = []  # type: List[str]
    rebuilt = []  # type: List[str]
    in_string = False
    escape = False
    for ch in out:
        rebuilt.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]") and stack and ch == stack[-1]:
            stack.pop()

    repaired = "".join(rebuilt).rstrip()
    if in_string:
        repaired += '"'
    repaired = _trim_dangling_json_tail(repaired)
    repaired += "".join(reversed(stack))
    repaired = _trim_dangling_json_tail(repaired)
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
    return repaired.strip()


def _try_json_object(candidate):
    # type: (str) -> Optional[dict]
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _try_parse_json(text):
    # type: (str) -> Tuple[Optional[dict], str]
    candidate, complete = _extract_first_json_candidate(text)
    if not candidate:
        return None, ""

    obj = _try_json_object(candidate)
    if obj is not None:
        return obj, "json"

    repaired = _repair_json_candidate(candidate)
    if repaired:
        obj = _try_json_object(repaired)
        if obj is not None:
            return obj, "json_repaired"

    if complete:
        normalized = _normalize_json_quotes(candidate)
        if normalized != candidate:
            obj = _try_json_object(normalized)
            if obj is not None:
                return obj, "json_repaired"
    return None, ""


def _parse_kv_text(text):
    # type: (str) -> Optional[dict]
    raw = clean_generation_text(text)
    if not raw or "{" in raw or "[" in raw:
        return None

    payload = {}  # type: Dict[str, str]
    for item in _KV_SPLIT_RE.split(raw):
        line = item.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key_norm = key.strip().lower().replace(" ", "_")
        value_norm = _sanitize_fragment(value.strip())
        if not key_norm or not value_norm:
            continue
        payload[key_norm] = value_norm

    if not payload:
        return None

    intent_card = {
        "scene_anchor": payload.get("scene_anchor", payload.get("scene", "")),
        "motion_intent": payload.get("motion_intent", payload.get("motion", "")),
        "geometry_constraints": _normalize_text_list(
            payload.get("geometry_constraints", payload.get("geometry", payload.get("structures", "")))
        ),
        "appearance_constraints": _normalize_text_list(
            payload.get(
                "appearance_constraints",
                ", ".join([payload.get("objects", ""), payload.get("lighting", ""), payload.get("appearance", "")]).strip(
                    ", "
                ),
            )
        ),
        "suppressions": _normalize_text_list(payload.get("suppressions", payload.get("negative", ""))),
    }
    control_hints = {
        "motion_intensity": payload.get("motion_intensity", ""),
        "geometry_priority": payload.get("geometry_priority", ""),
        "risk_level": payload.get("risk_level", ""),
    }
    return {"intent_card": intent_card, "control_hints": control_hints}


def _extract_named_value(raw_text, key):
    # type: (str, str) -> str
    raw = clean_generation_text(raw_text)
    patterns = [
        r'"%s"\s*:\s*"([^"\n\r\]\}]*)' % re.escape(key),
        r"%s\s*[:=]\s*([^,\n;\]\}]*)" % re.escape(key),
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if not match:
            continue
        value = _sanitize_fragment(match.group(1))
        if value and not _contains_json_artifact(value):
            return value
    return ""


def _extract_keyword_clause(raw_text, keywords):
    # type: (str, List[str]) -> str
    for clause in _split_sentences(raw_text):
        lowered = clause.lower()
        for keyword in keywords:
            if keyword in lowered:
                return clause
    return ""


def _build_safe_fallback_intent(raw_text):
    # type: (str) -> dict
    scene_anchor = _extract_named_value(raw_text, "scene_anchor") or _extract_keyword_clause(raw_text, _SCENE_KEYWORDS)
    motion_intent = _extract_named_value(raw_text, "motion_intent") or _extract_keyword_clause(raw_text, _MOTION_KEYWORDS)
    geometry_hint = _extract_named_value(raw_text, "geometry_constraints") or _extract_keyword_clause(raw_text, _GEOMETRY_KEYWORDS)
    appearance_hint = _extract_named_value(raw_text, "appearance_constraints")

    geometry_constraints = _post_clean_list(_normalize_text_list([geometry_hint] if geometry_hint else []))
    if not geometry_constraints:
        geometry_constraints = list(DEFAULT_GEOMETRY_PHRASES)

    appearance_constraints = _post_clean_list(_normalize_text_list([appearance_hint] if appearance_hint else []))
    motion_intent = _normalize_motion_intent(motion_intent, scene_anchor, geometry_constraints)

    return {
        "scene_anchor": scene_anchor or DEFAULT_SCENE_ANCHOR,
        "motion_intent": motion_intent,
        "geometry_constraints": geometry_constraints,
        "appearance_constraints": appearance_constraints,
        "suppressions": [],
    }


def _normalize_priority(value, allowed, default):
    # type: (object, List[str], str) -> str
    text = _strip_terminal(value if isinstance(value, str) else str(value))
    if not text:
        return default
    text = text.lower()
    if text in allowed:
        return text
    if "high" in text or "strong" in text:
        return "high"
    if "mid" in text or "medium" in text or "moderate" in text:
        return "medium"
    return "low"


def _infer_motion_intensity(text):
    # type: (str) -> str
    lowered = clean_generation_text(text).lower()
    if not lowered:
        return DEFAULT_MOTION_INTENSITY
    if "fast" in lowered or "rapid" in lowered or "sharp" in lowered or "aggressive" in lowered:
        return "high"
    if "turn" in lowered or "curve" in lowered or "approach" in lowered or "move" in lowered or "walk" in lowered:
        return "medium"
    return DEFAULT_MOTION_INTENSITY


def _infer_risk_level(intent_card, parse_mode):
    # type: (dict, str) -> str
    suppressions = _normalize_text_list(intent_card.get("suppressions", []))
    motion_intent = str(intent_card.get("motion_intent", "") or "")
    if parse_mode.startswith("fallback"):
        return "medium"
    if len(suppressions) >= 4:
        return "medium"
    if _infer_motion_intensity(motion_intent) == "high":
        return "high"
    return DEFAULT_RISK_LEVEL


def _normalize_intent_card(payload):
    # type: (Optional[dict]) -> dict
    container = payload if isinstance(payload, dict) else {}
    source = container.get("intent_card", container)
    if not isinstance(source, dict):
        source = {}
    scene_anchor = _sanitize_fragment(source.get("scene_anchor", ""))
    geometry_constraints = _post_clean_list(_normalize_text_list(source.get("geometry_constraints", [])))
    appearance_constraints = _post_clean_list(_normalize_text_list(source.get("appearance_constraints", [])))
    suppressions = _post_clean_list(_normalize_text_list(source.get("suppressions", [])))
    return {
        "scene_anchor": scene_anchor,
        "motion_intent": _normalize_motion_intent(source.get("motion_intent", ""), scene_anchor, geometry_constraints),
        "geometry_constraints": geometry_constraints,
        "appearance_constraints": appearance_constraints,
        "suppressions": suppressions,
    }


def _structured_field_count(intent_card):
    # type: (dict) -> int
    count = 0
    if _sanitize_fragment(intent_card.get("scene_anchor", "")):
        count += 1
    if _sanitize_fragment(intent_card.get("motion_intent", "")):
        count += 1
    if _normalize_text_list(intent_card.get("geometry_constraints", [])):
        count += 1
    return count


def _normalize_control_hints(payload, intent_card, raw_text, parse_mode):
    # type: (Optional[dict], dict, str, str) -> dict
    source = payload if isinstance(payload, dict) else {}
    control_source = source.get("control_hints", source)
    if not isinstance(control_source, dict):
        control_source = {}
    motion_text = " ".join([str(intent_card.get("motion_intent", "") or ""), clean_generation_text(raw_text)]).strip()
    return {
        "motion_intensity": _normalize_priority(
            control_source.get("motion_intensity", _infer_motion_intensity(motion_text)),
            ["low", "medium", "high"],
            DEFAULT_MOTION_INTENSITY,
        ),
        "geometry_priority": _normalize_priority(
            control_source.get("geometry_priority", DEFAULT_GEOMETRY_PRIORITY),
            ["low", "medium", "high"],
            DEFAULT_GEOMETRY_PRIORITY,
        ),
        "risk_level": _normalize_priority(
            control_source.get("risk_level", _infer_risk_level(intent_card, parse_mode)),
            ["low", "medium", "high"],
            DEFAULT_RISK_LEVEL,
        ),
    }


def _default_negative_suppressions(base_neg_prompt, risk_level):
    # type: (str, str) -> List[str]
    base = str(base_neg_prompt or "").lower()
    out = [
        "ghosting" if "ghosting" in base else "ghosting",
        "double edges" if "double edges" in base else "double edges",
        "unstable geometry" if "inconsistent geometry" in base or "geometry" in base else "unstable geometry",
        "warped perspective" if "wrong perspective" in base or "perspective" in base else "warped perspective",
    ]
    if risk_level in ("medium", "high"):
        out.append("flickering" if "flickering" in base else "flickering")
    if risk_level == "high":
        out.append("heavy motion blur" if "motion blur" in base else "heavy motion blur")
    return out


def parse_intent_response(raw_text):
    # type: (str) -> dict
    cleaned = clean_generation_text(raw_text)
    payload = None  # type: Optional[dict]
    parse_mode = "raw_text_fallback"

    payload, json_mode = _try_parse_json(cleaned)
    if payload is not None:
        parse_mode = json_mode
    else:
        payload = _parse_kv_text(cleaned)
        if payload is not None:
            parse_mode = "kv_pairs"

    parsed_intent = _normalize_intent_card(payload)
    structured_ok = _structured_field_count(parsed_intent) >= 2

    if structured_ok:
        intent_card = parsed_intent
    else:
        intent_card = _build_safe_fallback_intent(cleaned)
        if payload is None:
            parse_mode = "raw_text_fallback"
        else:
            parse_mode = "fallback_from_{}".format(parse_mode)

    control_hints = _normalize_control_hints(payload, intent_card, cleaned, parse_mode)
    return {
        "raw_text": cleaned,
        "parse_mode": parse_mode,
        "structured_ok": structured_ok,
        "intent_card": intent_card,
        "control_hints": control_hints,
    }


def build_intent_instruction(base_instruction="", backend_name="", strict_json=False):
    # type: (str, str, bool) -> str
    prefix = clean_generation_text(base_instruction)
    if not prefix:
        prefix = "You will be given multiple frames sampled from a short video segment."
    backend = str(backend_name or "").strip().lower()
    lines = [
        prefix,
        "Extract a compact semantic intent card for video generation instead of writing a final natural-language prompt.",
        "Return JSON only with keys: scene_anchor, motion_intent, geometry_constraints, appearance_constraints, suppressions, control_hints.",
        "scene_anchor: one short sentence for stable scene identity and layout.",
        "motion_intent: one short sentence for observed ego-motion or local dynamic intent when relevant.",
        "geometry_constraints: array of short phrases for geometry, perspective, path shape, horizon, boundaries, rigid structures.",
        "appearance_constraints: array of short phrases for lighting, materials, weather, key objects, and texture cues.",
        "suppressions: array of short phrases for artifacts or content to avoid in this segment.",
        "control_hints: object with motion_intensity, geometry_priority, risk_level using only low, medium, or high.",
        "Keep every field conservative, concrete, and short.",
    ]
    if backend == "smolvlm2":
        lines.append("Prefer 3-6 short items total across the arrays and avoid verbose explanations.")
    if strict_json:
        lines.append("Do not add markdown, commentary, or any text outside the JSON object.")
    else:
        lines.append("If exact JSON is hard, still keep the response close to the requested keys and values.")
    return "\n".join(lines)


def build_global_invariants(base_prompt, base_neg_prompt):
    # type: (str, str) -> dict
    positive_constraints = [item for item in _split_sentences(base_prompt) if item]
    negative_constraints = [item for item in _split_phrase_items(base_neg_prompt) if item]
    return {
        "positive_constraints": positive_constraints,
        "negative_constraints": negative_constraints,
        "notes": [
            "Keep sequence-level scene identity stable unless the segment intent explicitly requires a local change.",
            "Prefer stable geometry and coherent appearance over speculative detail.",
        ],
    }


def _dedupe_sentences(items, blocked_items):
    # type: (List[str], List[str]) -> List[str]
    out = []
    seen = set()
    blocked = set([_sanitize_fragment(item).lower() for item in blocked_items if _sanitize_fragment(item)])
    for item in items:
        text = _ensure_sentence(_sanitize_fragment(item))
        key = _sanitize_fragment(text).lower()
        if not key or key in seen or key in blocked:
            continue
        seen.add(key)
        out.append(text)
    return out


def compile_legacy_prompts(base_prompt, base_neg_prompt, global_invariants, intent_card, control_hints=None):
    # type: (str, str, dict, dict, Optional[dict]) -> dict
    global_invariants = global_invariants if isinstance(global_invariants, dict) else {}
    intent_card = intent_card if isinstance(intent_card, dict) else {}
    control_hints = control_hints if isinstance(control_hints, dict) else {}

    scene_anchor = _sanitize_fragment(intent_card.get("scene_anchor", "")) or DEFAULT_SCENE_ANCHOR
    geometry_constraints = _post_clean_list(_normalize_text_list(intent_card.get("geometry_constraints", [])))
    appearance_constraints = _post_clean_list(_normalize_text_list(intent_card.get("appearance_constraints", [])))
    suppressions = _post_clean_list(_normalize_text_list(intent_card.get("suppressions", [])))
    motion_intent = _normalize_motion_intent(intent_card.get("motion_intent", ""), scene_anchor, geometry_constraints)

    if not geometry_constraints:
        geometry_constraints = list(DEFAULT_GEOMETRY_PHRASES)

    prompt_parts = [scene_anchor, motion_intent]
    prompt_parts.append("Preserve {}.".format(", ".join(geometry_constraints)))
    if appearance_constraints:
        prompt_parts.append("Keep {}.".format(", ".join(appearance_constraints)))
    if control_hints.get("geometry_priority", "") == "high" and "stable geometry" not in ",".join(geometry_constraints).lower():
        prompt_parts.append("Maintain stable geometry and perspective.")

    blocked_prompt_items = list(global_invariants.get("positive_constraints", []))
    blocked_prompt_items.append(str(base_prompt or ""))
    delta_prompt_parts = _dedupe_sentences(prompt_parts, blocked_prompt_items)
    delta_prompt = " ".join(delta_prompt_parts).strip()

    risk_level = _normalize_priority(control_hints.get("risk_level", DEFAULT_RISK_LEVEL), ["low", "medium", "high"], DEFAULT_RISK_LEVEL)
    blocked_neg = _normalize_text_list(global_invariants.get("negative_constraints", []))
    blocked_neg_set = set([item.lower() for item in blocked_neg])

    delta_neg_items = []  # type: List[str]
    seen_neg = set()
    for item in suppressions:
        key = item.lower()
        if key in seen_neg or key in blocked_neg_set:
            continue
        seen_neg.add(key)
        delta_neg_items.append(item)

    if not delta_neg_items:
        for item in _default_negative_suppressions(base_neg_prompt, risk_level):
            key = item.lower()
            if key in seen_neg:
                continue
            seen_neg.add(key)
            delta_neg_items.append(item)

    delta_neg_prompt = ", ".join(delta_neg_items).strip()

    final_prompt_preview = str(base_prompt or "").strip()
    if delta_prompt:
        final_prompt_preview = (final_prompt_preview + "\n" + delta_prompt).strip()

    final_neg_prompt_preview = str(base_neg_prompt or "").strip()
    if delta_neg_prompt:
        if final_neg_prompt_preview:
            final_neg_prompt_preview = (final_neg_prompt_preview + "\n" + delta_neg_prompt).strip()
        else:
            final_neg_prompt_preview = delta_neg_prompt

    return {
        "delta_prompt": delta_prompt,
        "delta_neg_prompt": delta_neg_prompt,
        "final_prompt_preview": final_prompt_preview,
        "final_neg_prompt_preview": final_neg_prompt_preview,
    }


def build_manifest_v2(base_prompt, base_neg_prompt, sequence_meta, global_invariants, compiler, segments):
    # type: (str, str, dict, dict, dict, List[dict]) -> dict
    return {
        "version": MANIFEST_VERSION,
        "schema": MANIFEST_SCHEMA,
        "base_prompt": str(base_prompt or "").strip(),
        "base_neg_prompt": str(base_neg_prompt or "").strip(),
        "sequence_meta": dict(sequence_meta or {}),
        "global_invariants": dict(global_invariants or {}),
        "compiler": dict(compiler or {}),
        "segments": list(segments or []),
    }
