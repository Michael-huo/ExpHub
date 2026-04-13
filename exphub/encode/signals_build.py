from __future__ import annotations

import re
from datetime import datetime

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_GROUP_BREAK_THRESHOLD = 0.45
_DYNAMIC_THRESHOLD = 0.66
_MIXED_THRESHOLD = 0.33
_HIGH_RISK_THRESHOLD = 0.66
_MEDIUM_RISK_THRESHOLD = 0.33
_MOTION_WEIGHT = 0.7
_SEMANTIC_WEIGHT = 0.3


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _motion_label_for_score(score):
    score = float(score)
    if score >= _DYNAMIC_THRESHOLD:
        return "dynamic"
    if score >= _MIXED_THRESHOLD:
        return "mixed"
    return "steady"


def _risk_level_for_score(score):
    score = float(score)
    if score >= _HIGH_RISK_THRESHOLD:
        return "high"
    if score >= _MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _tokenize_scene_prompt(text):
    return set(_TOKEN_RE.findall(_collapse_ws(text).lower()))


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _jaccard_distance(left_tokens, right_tokens):
    if not left_tokens and not right_tokens:
        return 0.0
    union = set(left_tokens) | set(right_tokens)
    if not union:
        return 0.0
    intersection = set(left_tokens) & set(right_tokens)
    return float(1.0 - (float(len(intersection)) / float(len(union))))


def build_motion_score_payload(segment_manifest):
    manifest = _as_dict(segment_manifest)
    state_payload = _as_dict(manifest.get("state_segments"))
    state_rows = list(state_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("segment manifest missing state_segments.segments for motion score")

    segments = []
    for idx, raw_item in enumerate(state_rows):
        row = _as_dict(raw_item)
        state_score_mean = _safe_float(row.get("state_score_mean"))
        state_score_peak = _safe_float(row.get("state_score_peak"))
        motion_score = _clamp01(0.65 * state_score_mean + 0.35 * state_score_peak)
        segments.append(
            {
                "segment_id": _safe_int(row.get("segment_id"), idx),
                "state_label": str(row.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "start_frame": _safe_int(row.get("start_frame"), 0),
                "end_frame": _safe_int(row.get("end_frame"), 0),
                "duration_frames": _safe_int(row.get("duration_frames"), 0),
                "motion_score": float(motion_score),
                "motion_label": _motion_label_for_score(motion_score),
                "state_score_mean": float(state_score_mean),
                "state_score_peak": float(state_score_peak),
                "source": "segment_manifest.state_segments",
            }
        )

    motion_values = [float(item.get("motion_score", 0.0) or 0.0) for item in segments]
    return {
        "version": 1,
        "schema": "motion_score.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "state_analysis",
        "source": "encode.state_analysis.motion_score",
        "signal_name": "motion_state_signal",
        "signal_method": "0.65 * state_score_mean + 0.35 * state_score_peak",
        "label_thresholds": {
            "steady_lt": float(_MIXED_THRESHOLD),
            "mixed_lt": float(_DYNAMIC_THRESHOLD),
        },
        "segments": segments,
        "summary": {
            "segment_count": int(len(segments)),
            "sequence_start_idx": int(segments[0]["start_frame"]),
            "sequence_end_idx": int(segments[-1]["end_frame"]),
            "mean_motion_score": float(sum(motion_values) / float(len(motion_values))) if motion_values else 0.0,
            "dynamic_segment_count": int(len([item for item in segments if item.get("motion_label") == "dynamic"])),
        },
    }


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


def build_generation_risk_payload(motion_score_payload, semantic_shift_payload):
    motion_payload = _as_dict(motion_score_payload)
    semantic_payload = _as_dict(semantic_shift_payload)
    motion_rows = list(motion_payload.get("segments") or [])
    semantic_rows = list(semantic_payload.get("segments") or [])
    if not motion_rows or not semantic_rows:
        raise RuntimeError("generation risk requires non-empty motion score and semantic shift segments")
    if len(motion_rows) != len(semantic_rows):
        raise RuntimeError("motion score and semantic shift segment counts must match")

    for idx, (motion_raw, semantic_raw) in enumerate(zip(motion_rows, semantic_rows)):
        motion_row = _as_dict(motion_raw)
        semantic_row = _as_dict(semantic_raw)
        motion_segment_id = _safe_int(motion_row.get("segment_id"), idx)
        semantic_segment_id = _safe_int(semantic_row.get("segment_id"), idx)
        if motion_segment_id != semantic_segment_id:
            raise RuntimeError(
                "generation risk requires matched segment ids: motion={} semantic={}".format(
                    int(motion_segment_id),
                    int(semantic_segment_id),
                )
            )
        motion_range = (
            _safe_int(motion_row.get("start_frame"), 0),
            _safe_int(motion_row.get("end_frame"), 0),
        )
        semantic_range = (
            _safe_int(semantic_row.get("start_frame"), 0),
            _safe_int(semantic_row.get("end_frame"), 0),
        )
        if motion_range != semantic_range:
            raise RuntimeError(
                "generation risk requires matched segment ranges: motion={} semantic={}".format(
                    motion_range,
                    semantic_range,
                )
            )

    segments = []
    for idx, (motion_raw, semantic_raw) in enumerate(zip(motion_rows, semantic_rows)):
        motion_row = _as_dict(motion_raw)
        semantic_row = _as_dict(semantic_raw)
        motion_score = _safe_float(motion_row.get("motion_score"))
        semantic_shift = _safe_float(semantic_row.get("semantic_shift"))
        combined_score = _clamp01((_MOTION_WEIGHT * motion_score) + (_SEMANTIC_WEIGHT * semantic_shift))
        segments.append(
            {
                "segment_id": _safe_int(motion_row.get("segment_id"), idx),
                "start_frame": _safe_int(motion_row.get("start_frame"), 0),
                "end_frame": _safe_int(motion_row.get("end_frame"), 0),
                "duration_frames": _safe_int(motion_row.get("duration_frames"), 0),
                "scene_label": str(semantic_row.get("scene_label", "scene_group_000") or "scene_group_000"),
                "motion_score": float(motion_score),
                "semantic_shift": float(semantic_shift),
                "generation_risk": float(combined_score),
                "risk_level": _risk_level_for_score(combined_score),
                "combination": {
                    "motion_weight": float(_MOTION_WEIGHT),
                    "semantic_weight": float(_SEMANTIC_WEIGHT),
                    "formula": "{} * motion_score + {} * semantic_shift".format(_MOTION_WEIGHT, _SEMANTIC_WEIGHT),
                },
                "source": "motion_score + semantic_shift",
            }
        )

    return {
        "version": 1,
        "schema": "generation_risk.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "state_analysis",
        "source": "encode.state_analysis.risk_score",
        "signal_name": "generation_risk_signal",
        "signal_method": "weighted_combination_of_motion_score_and_semantic_shift",
        "combination_policy": {
            "motion_weight": float(_MOTION_WEIGHT),
            "semantic_weight": float(_SEMANTIC_WEIGHT),
            "high_risk_threshold": float(_HIGH_RISK_THRESHOLD),
            "medium_risk_threshold": float(_MEDIUM_RISK_THRESHOLD),
        },
        "segments": segments,
        "summary": {
            "segment_count": int(len(segments)),
            "sequence_start_idx": int(segments[0]["start_frame"]) if segments else 0,
            "sequence_end_idx": int(segments[-1]["end_frame"]) if segments else 0,
            "high_risk_segment_count": int(len([item for item in segments if item.get("risk_level") == "high"])),
            "mean_generation_risk": float(sum([item.get("generation_risk", 0.0) for item in segments]) / float(len(segments))),
        },
    }


__all__ = [
    "build_generation_risk_payload",
    "build_motion_score_payload",
    "build_semantic_shift_payload",
]
