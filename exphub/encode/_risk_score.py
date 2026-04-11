from __future__ import annotations

from datetime import datetime


_HIGH_RISK_THRESHOLD = 0.66
_MEDIUM_RISK_THRESHOLD = 0.33
_MOTION_WEIGHT = 0.7
_SEMANTIC_WEIGHT = 0.3


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _risk_level_for_score(score):
    score = float(score)
    if score >= _HIGH_RISK_THRESHOLD:
        return "high"
    if score >= _MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _assert_matching_segment_rows(motion_rows, semantic_rows):
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


def build_generation_risk_payload(motion_score_payload, semantic_shift_payload):
    motion_payload = _as_dict(motion_score_payload)
    semantic_payload = _as_dict(semantic_shift_payload)
    motion_rows = list(motion_payload.get("segments") or [])
    semantic_rows = list(semantic_payload.get("segments") or [])
    if not motion_rows or not semantic_rows:
        raise RuntimeError("generation risk requires non-empty motion score and semantic shift segments")
    if len(motion_rows) != len(semantic_rows):
        raise RuntimeError("motion score and semantic shift segment counts must match")
    _assert_matching_segment_rows(motion_rows, semantic_rows)

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
