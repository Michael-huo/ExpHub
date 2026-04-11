from __future__ import annotations

import re
from datetime import datetime


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_GROUP_BREAK_THRESHOLD = 0.45


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _tokenize_scene_prompt(text):
    return set(_TOKEN_RE.findall(_collapse_ws(text).lower()))


def _jaccard_distance(left_tokens, right_tokens):
    if not left_tokens and not right_tokens:
        return 0.0
    union = set(left_tokens) | set(right_tokens)
    if not union:
        return 0.0
    intersection = set(left_tokens) & set(right_tokens)
    return float(1.0 - (float(len(intersection)) / float(len(union))))


def build_semantic_shift_payload(segment_manifest, prompt_manifest, group_break_threshold=_DEFAULT_GROUP_BREAK_THRESHOLD):
    manifest = _as_dict(segment_manifest)
    prompt_payload = _as_dict(prompt_manifest)
    state_payload = _as_dict(manifest.get("state_segments"))
    state_rows = list(state_payload.get("segments") or [])
    prompt_rows = list(prompt_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("segment manifest missing state_segments.segments for semantic shift")
    if not prompt_rows:
        raise RuntimeError("prompt manifest missing segments for semantic shift")

    prompt_map = {}
    for idx, raw_item in enumerate(prompt_rows):
        item = _as_dict(raw_item)
        prompt_map[_safe_int(item.get("state_segment_id"), idx)] = item

    scene_group_idx = 0
    prev_tokens = set()
    prev_prompt = ""
    prev_segment_id = None
    segments = []
    for idx, raw_item in enumerate(state_rows):
        row = _as_dict(raw_item)
        segment_id = _safe_int(row.get("segment_id"), idx)
        prompt_item = _as_dict(prompt_map.get(segment_id))
        scene_prompt = _collapse_ws(prompt_item.get("scene_prompt", ""))
        if not scene_prompt:
            raise RuntimeError("missing scene prompt for state segment {}".format(segment_id))
        scene_tokens = _tokenize_scene_prompt(scene_prompt)
        semantic_shift_score = _jaccard_distance(prev_tokens, scene_tokens) if idx > 0 else 0.0
        is_scene_break = bool(idx > 0 and semantic_shift_score >= float(group_break_threshold))
        if is_scene_break:
            scene_group_idx += 1
        scene_label = "scene_group_{:03d}".format(int(scene_group_idx))
        segments.append(
            {
                "segment_id": int(segment_id),
                "start_frame": _safe_int(row.get("start_frame"), 0),
                "end_frame": _safe_int(row.get("end_frame"), 0),
                "duration_frames": _safe_int(row.get("duration_frames"), 0),
                "scene_prompt": str(scene_prompt),
                "scene_label": str(scene_label),
                "scene_group_id": int(scene_group_idx),
                "semantic_shift": float(semantic_shift_score),
                "is_scene_break": bool(is_scene_break),
                "previous_segment_id": int(prev_segment_id) if prev_segment_id is not None else None,
                "previous_scene_prompt": str(prev_prompt),
                "source": "prompt_manifest.segments[].scene_prompt",
            }
        )
        prev_tokens = scene_tokens
        prev_prompt = scene_prompt
        prev_segment_id = int(segment_id)

    return {
        "version": 1,
        "schema": "semantic_shift.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "state_analysis",
        "source": "encode.state_analysis.semantic_shift",
        "signal_name": "semantic_change_signal",
        "signal_method": "scene_prompt_jaccard_distance_between_adjacent_state_segments",
        "group_break_threshold": float(group_break_threshold),
        "segments": segments,
        "summary": {
            "segment_count": int(len(segments)),
            "sequence_start_idx": int(segments[0]["start_frame"]) if segments else 0,
            "sequence_end_idx": int(segments[-1]["end_frame"]) if segments else 0,
            "scene_group_count": int(segments[-1]["scene_group_id"] + 1) if segments else 0,
            "scene_break_count": int(len([item for item in segments if item.get("is_scene_break")])),
            "max_semantic_shift": float(max([item.get("semantic_shift", 0.0) for item in segments]) if segments else 0.0),
        },
    }
