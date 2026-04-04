from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from exphub.pipeline.prompt.backends import create_backend


SCENE_PROMPT_INSTRUCTION = (
    "Encode this single representative frame from a first-person video segment as one compact scene prompt. "
    "Prefer short scene phrases for environment, layout, path or corridor, surface, obstacles or pedestrians, "
    "and forward visibility. Avoid full sentences, storytelling, image-quality notes, emotions, motion description, "
    "uncertainty, and background filler. Keep it under 18 words. Output only the scene prompt."
)

_SCENE_PROMPT_PREFIXES = [
    "scene prompt:",
    "prompt:",
    "output:",
    "the image shows",
    "this image shows",
    "this frame shows",
    "a first-person view of",
    "first-person view of",
    "a first person view of",
    "first person view of",
    "a view of",
    "view of",
]
_SCENE_PROMPT_WORD_LIMIT = 18
_SCENE_PROMPT_MAX_PHRASES = 4

_SCENE_PROMPT_LEADIN_PATTERNS = [
    r"^(?:this|the)\s+(?:image|frame|scene|view)\s+(?:shows|depicts|features)\s+",
    r"^(?:image|frame|scene)\s+(?:shows|depicts|features)\s+",
    r"^(?:there is|there are)\s+",
    r"^(?:showing|featuring)\s+",
]
_SCENE_PROMPT_DROP_PATTERNS = [
    r"\bin the background\b",
    r"\bin background\b",
    r"\bin the distance\b",
    r"\bin distance\b",
    r"\bon it\b",
    r"\bcan be seen\b",
    r"\bcan be visible\b",
    r"\bis visible\b",
    r"\bare visible\b",
    r"\bappears to be\b",
    r"\bseems to be\b",
]
_SCENE_PROMPT_FILLER_WORDS = {
    "clearly",
    "fairly",
    "mainly",
    "mostly",
    "overall",
    "rather",
    "somewhat",
    "very",
}
_SCENE_PROMPT_WEAK_PHRASES = {
    "image",
    "frame",
    "scene",
    "view",
}
_SCENE_PROMPT_MODIFIER_ONLY = {
    "bright",
    "dim",
    "long",
    "narrow",
    "open",
    "straight",
    "wide",
}
_SCENE_PROMPT_REPLACEMENTS = [
    (r"\bfirst[\s-]?person(?:\s+view)?\b", ""),
    (r"\bwalkway\b", "path"),
    (r"\bpathway\b", "path"),
    (r"\bhallway\b", "corridor"),
    (r"\bindoors\b", "indoor"),
    (r"\binside\b", "indoor"),
    (r"\boutdoors\b", "outdoor"),
    (r"\boutside\b", "outdoor"),
    (r"\bwell[\s-]?lit\b", "bright"),
    (r"\bbrightly lit\b", "bright"),
    (r"\bdimly lit\b", "dim"),
    (r"\blow[- ]light\b", "dim"),
    (r"\bperson walking\b", "pedestrian"),
    (r"\bpeople walking\b", "pedestrians"),
    (r"\bperson\b", "pedestrian"),
    (r"\bpeople\b", "pedestrians"),
    (r"\bwalks? down\b", "on"),
    (r"\bwalking down\b", "on"),
    (r"\bwalking along\b", "on"),
    (r"\bwalking on\b", "on"),
    (r"\bstraight ahead\b", "forward"),
    (r"\bview ahead\b", "forward visibility"),
    (r"\bvisibility ahead\b", "forward visibility"),
]
_SCENE_PROMPT_CONNECTOR_REPLACEMENTS = [
    (r"\s+with\s+", ", "),
    (r"\s+featuring\s+", ", "),
    (r"\s+showing\s+", ", "),
    (r"\s+including\s+", ", "),
    (r"\s+lined with\s+", ", "),
    (r"\s+and\s+", ", "),
]


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def _relative_to_exp(exp_dir, target_path):
    # type: (Path, Path) -> str
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _safe_int(value, default=0):
    # type: (object, int) -> int
    try:
        return int(value)
    except Exception:
        return int(default)


def _word_list(text):
    # type: (object) -> List[str]
    return [part for part in _collapse_ws(text).split(" ") if part]


def _drop_filler_words(text):
    # type: (str) -> str
    kept = []
    for token in _word_list(text):
        if token in _SCENE_PROMPT_FILLER_WORDS:
            continue
        kept.append(token)
    return " ".join(kept).strip()


def _normalize_scene_phrase(text):
    # type: (str) -> str
    phrase = _collapse_ws(text).strip(" \t\r\n,;:.")
    if not phrase:
        return ""
    phrase = re.sub(r"\b(?:a|an|the)\b\s*", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = _drop_filler_words(phrase)
    phrase = re.sub(r"\b(?:photo|picture|image|frame)\b", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase.strip(" \t\r\n,;:.")


def _dedupe_scene_phrases(phrases):
    # type: (List[str]) -> List[str]
    unique = []
    seen = set()
    for raw_phrase in list(phrases or []):
        phrase = _normalize_scene_phrase(raw_phrase)
        if not phrase:
            continue
        if phrase in _SCENE_PROMPT_WEAK_PHRASES:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        unique.append(phrase)
        if len(unique) >= int(_SCENE_PROMPT_MAX_PHRASES):
            break
    return unique


def _merge_scene_phrase_parts(parts):
    # type: (List[str]) -> List[str]
    merged = []
    modifier_prefix = []
    for raw_part in list(parts or []):
        phrase = _normalize_scene_phrase(raw_part)
        if not phrase:
            continue
        if phrase in _SCENE_PROMPT_MODIFIER_ONLY:
            modifier_prefix.append(phrase)
            continue
        if modifier_prefix:
            merged.append("{} {}".format(" ".join(modifier_prefix), phrase).strip())
            modifier_prefix = []
            continue
        merged.append(phrase)
    if modifier_prefix:
        merged.append(" ".join(modifier_prefix).strip())
    return merged


def _trim_phrase_words(phrases, word_limit):
    # type: (List[str], int) -> str
    remaining = max(1, int(word_limit))
    out = []
    for phrase in list(phrases or []):
        words = _word_list(phrase)
        if not words:
            continue
        if remaining <= 0:
            break
        if len(words) > remaining:
            phrase = " ".join(words[:remaining]).rstrip(" ,;:.")
            words = _word_list(phrase)
        if not words:
            continue
        out.append(" ".join(words))
        remaining -= len(words)
    return ", ".join(out).strip(" \t\r\n,;:.")


def normalize_scene_prompt(text):
    # type: (object) -> Tuple[str, Dict[str, object]]
    raw_prompt = _collapse_ws(text).strip(" \t\r\n\"'`")
    prompt = str(raw_prompt)
    prompt_lower = prompt.lower()
    for prefix in _SCENE_PROMPT_PREFIXES:
        if prompt_lower.startswith(prefix):
            prompt = prompt[len(prefix) :].strip(" \t\r\n:,-")
            prompt_lower = prompt.lower()
            break

    prompt = prompt.lower()
    for pattern in _SCENE_PROMPT_LEADIN_PATTERNS:
        prompt = re.sub(pattern, "", prompt)

    prompt = re.sub(r"\s*;\s*", ", ", prompt)
    prompt = re.sub(r"\s*:\s*", " ", prompt)
    prompt = re.sub(r"\s*[/|]\s*", ", ", prompt)
    prompt = re.sub(r"\s*[-]\s*", " ", prompt)
    for pattern in _SCENE_PROMPT_DROP_PATTERNS:
        prompt = re.sub(pattern, "", prompt)
    for pattern, replacement in _SCENE_PROMPT_REPLACEMENTS:
        prompt = re.sub(pattern, replacement, prompt)
    for pattern, replacement in _SCENE_PROMPT_CONNECTOR_REPLACEMENTS:
        prompt = re.sub(pattern, replacement, prompt)

    prompt = re.sub(r"\b(?:maybe|perhaps|possibly|probably|likely|unclear|unknown)\b", "", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"\s*,\s*", ", ", prompt)
    prompt = re.sub(r"(?:,\s*){2,}", ", ", prompt)
    prompt = prompt.strip(" \t\r\n,;:.")

    phrases = _dedupe_scene_phrases(_merge_scene_phrase_parts([part for part in prompt.split(",") if part.strip()]))
    normalized = _trim_phrase_words(phrases, _SCENE_PROMPT_WORD_LIMIT)
    if not normalized:
        normalized = _trim_phrase_words(_dedupe_scene_phrases([prompt]), _SCENE_PROMPT_WORD_LIMIT)
    normalized = _collapse_ws(normalized).strip(" \t\r\n,;:.")

    return normalized, {
        "raw_word_count": int(len(_word_list(raw_prompt))),
        "normalized_word_count": int(len(_word_list(normalized))),
        "changed": bool(_collapse_ws(raw_prompt).lower() != str(normalized).lower()),
        "phrase_count": int(len(list(phrases))),
    }


def _resolve_frame_path(frames_dir, start_frame, end_frame, preferred_frame):
    # type: (Path, int, int, int) -> Path
    frames_root = Path(frames_dir).resolve()
    start_idx = int(start_frame)
    end_idx = int(end_frame)
    preferred_idx = max(int(start_idx), min(int(end_idx), int(preferred_frame)))

    search_order = [int(preferred_idx)]
    max_offset = int(max(abs(preferred_idx - start_idx), abs(end_idx - preferred_idx)))
    for offset in range(1, max_offset + 1):
        left_idx = int(preferred_idx - offset)
        right_idx = int(preferred_idx + offset)
        if left_idx >= int(start_idx):
            search_order.append(int(left_idx))
        if right_idx <= int(end_idx):
            search_order.append(int(right_idx))

    for frame_idx in search_order:
        image_path = frames_root / "{:06d}.png".format(int(frame_idx))
        if image_path.is_file():
            return image_path
    raise RuntimeError(
        "cannot resolve representative frame under {} for state range [{}:{}]".format(
            frames_root,
            int(start_idx),
            int(end_idx),
        )
    )


def _pick_primary_frame(state_row, keyframe_indices, frames_dir):
    # type: (Dict[str, object], List[int], Path) -> Dict[str, object]
    start_frame = int(state_row.get("start_frame", 0) or 0)
    end_frame = int(state_row.get("end_frame", 0) or 0)
    midpoint_frame = int(start_frame + max(0, int(end_frame - start_frame)) // 2)
    candidate_keyframes = [int(idx) for idx in list(keyframe_indices or []) if int(start_frame) <= int(idx) <= int(end_frame)]

    selection_source = "state_segment_midpoint_fallback"
    selected_frame = int(midpoint_frame)
    if candidate_keyframes:
        selected_frame = min(candidate_keyframes, key=lambda idx: (abs(int(idx) - int(midpoint_frame)), int(idx)))
        selection_source = "segment_keyframe_nearest_midpoint"

    image_path = _resolve_frame_path(
        frames_dir=frames_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        preferred_frame=selected_frame,
    )
    frame_idx = int(image_path.stem)
    return {
        "frame_idx": int(frame_idx),
        "image_path": image_path,
        "selection_source": str(selection_source),
        "midpoint_frame": int(midpoint_frame),
        "candidate_keyframe_count": int(len(candidate_keyframes)),
    }


def build_state_scene_encoding(segment_inputs, frames_dir, prompt_model_ref=""):
    # type: (Dict[str, object], Path, str) -> Dict[str, object]
    manifest = _as_dict(segment_inputs.get("segment_manifest"))
    state_payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(state_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("state_segments payload has no segments for scene encoding")

    keyframe_indices = sorted(set(int(v) for v in list(_as_dict(manifest.get("keyframes")).get("indices") or [])))
    exp_dir = Path(segment_inputs.get("exp_dir")).resolve()
    frames_root = Path(frames_dir).resolve()

    backend = create_backend(model_ref=str(prompt_model_ref or ""))
    backend.load()
    backend_meta = dict(backend.meta() or {})
    backend_name = str(backend_meta.get("backend", "smolvlm2") or "smolvlm2")

    encoded_segments = []
    for idx, raw_item in enumerate(state_rows):
        state_row = _as_dict(raw_item)
        primary_frame = _pick_primary_frame(
            state_row=state_row,
            keyframe_indices=keyframe_indices,
            frames_dir=frames_root,
        )
        raw_prompt = backend.generate([str(primary_frame["image_path"])], SCENE_PROMPT_INSTRUCTION)
        scene_prompt, normalization_meta = normalize_scene_prompt(raw_prompt)
        if not scene_prompt:
            raise RuntimeError(
                "scene encoding returned empty prompt for state segment {}".format(
                    _safe_int(state_row.get("segment_id"), idx)
                )
            )
        encoded_segments.append(
            {
                "state_segment_id": _safe_int(state_row.get("segment_id"), idx),
                "source_segment_id": _safe_int(state_row.get("segment_id"), idx),
                "state_label": str(state_row.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "start_frame": int(state_row.get("start_frame", 0) or 0),
                "end_frame": int(state_row.get("end_frame", 0) or 0),
                "scene_prompt": str(scene_prompt),
                "raw_scene_prompt": _collapse_ws(raw_prompt),
                "raw_scene_prompt_word_count": int(normalization_meta.get("raw_word_count", 0) or 0),
                "scene_prompt_word_count": int(normalization_meta.get("normalized_word_count", 0) or 0),
                "scene_prompt_normalization": dict(normalization_meta),
                "scene_prompt_source": "state_v2t_primary_frame",
                "scene_encoding_backend": str(backend_name),
                "representative_frame": {
                    "frame_idx": int(primary_frame["frame_idx"]),
                    "image_path": _relative_to_exp(exp_dir, primary_frame["image_path"]),
                    "selection_source": str(primary_frame["selection_source"]),
                    "midpoint_frame": int(primary_frame["midpoint_frame"]),
                    "candidate_keyframe_count": int(primary_frame["candidate_keyframe_count"]),
                },
            }
        )

    return {
        "version": 1,
        "source": "state_level_scene_encoding_v2",
        "scene_prompt_mode": "state_v2t_primary_frame",
        "backend": str(backend_name),
        "backend_meta": dict(backend_meta),
        "scene_prompt_style": "compact_canonical_phrase_v1",
        "state_segments": encoded_segments,
    }
