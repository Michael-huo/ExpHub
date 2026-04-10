from __future__ import annotations

from datetime import datetime


_SEMANTIC_PEAK_THRESHOLD = 0.45
_MOTION_JUMP_THRESHOLD = 0.20
_RISK_JUMP_THRESHOLD = 0.18
_BOUNDARY_STRENGTH_FLOOR = 0.25


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


def build_candidate_boundaries_payload(motion_score_payload, semantic_shift_payload, generation_risk_payload):
    motion_rows = list(_as_dict(motion_score_payload).get("segments") or [])
    semantic_rows = list(_as_dict(semantic_shift_payload).get("segments") or [])
    risk_rows = list(_as_dict(generation_risk_payload).get("segments") or [])
    if not motion_rows or not semantic_rows or not risk_rows:
        raise RuntimeError("candidate boundary extraction requires motion, semantic, and risk segments")
    if not (len(motion_rows) == len(semantic_rows) == len(risk_rows)):
        raise RuntimeError("candidate boundary extraction requires aligned segment counts")

    boundaries = []
    for idx in range(1, len(risk_rows)):
        prev_motion = _as_dict(motion_rows[idx - 1])
        curr_motion = _as_dict(motion_rows[idx])
        prev_semantic = _as_dict(semantic_rows[idx - 1])
        curr_semantic = _as_dict(semantic_rows[idx])
        prev_risk = _as_dict(risk_rows[idx - 1])
        curr_risk = _as_dict(risk_rows[idx])

        frame_idx = _safe_int(curr_risk.get("start_frame"), 0)
        motion_jump = abs(_safe_float(curr_motion.get("motion_score")) - _safe_float(prev_motion.get("motion_score")))
        semantic_peak = _safe_float(curr_semantic.get("semantic_shift"))
        risk_jump = abs(_safe_float(curr_risk.get("generation_risk")) - _safe_float(prev_risk.get("generation_risk")))
        scene_change = bool(curr_semantic.get("scene_label") != prev_semantic.get("scene_label"))
        risk_level_change = bool(curr_risk.get("risk_level") != prev_risk.get("risk_level"))
        motion_label_change = bool(curr_motion.get("motion_label") != prev_motion.get("motion_label"))
        stability_change = 1.0 if (scene_change or risk_level_change or motion_label_change) else 0.0
        boundary_strength = min(
            1.0,
            (0.35 * semantic_peak) + (0.30 * risk_jump) + (0.20 * motion_jump) + (0.15 * stability_change),
        )

        reasons = []
        if semantic_peak >= _SEMANTIC_PEAK_THRESHOLD:
            reasons.append("semantic_peak")
        if motion_jump >= _MOTION_JUMP_THRESHOLD:
            reasons.append("motion_jump")
        if risk_jump >= _RISK_JUMP_THRESHOLD:
            reasons.append("risk_jump")
        if scene_change:
            reasons.append("scene_group_change")
        if risk_level_change:
            reasons.append("risk_level_change")
        if motion_label_change:
            reasons.append("motion_label_change")

        if not reasons and boundary_strength < _BOUNDARY_STRENGTH_FLOOR:
            continue

        boundaries.append(
            {
                "candidate_id": int(len(boundaries)),
                "frame_idx": int(frame_idx),
                "previous_segment_id": _safe_int(prev_risk.get("segment_id"), idx - 1),
                "next_segment_id": _safe_int(curr_risk.get("segment_id"), idx),
                "strength": float(boundary_strength),
                "reasons": reasons,
                "source_scores": {
                    "motion_jump": float(motion_jump),
                    "semantic_peak": float(semantic_peak),
                    "risk_jump": float(risk_jump),
                    "stability_change": float(stability_change),
                },
            }
        )

    return {
        "version": 1,
        "schema": "candidate_boundaries.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "candidate_boundaries",
        "source": "encode.candidate_boundaries.extract",
        "extractor": "signal_transition_peaks_v1",
        "criteria": {
            "semantic_peak_threshold": float(_SEMANTIC_PEAK_THRESHOLD),
            "motion_jump_threshold": float(_MOTION_JUMP_THRESHOLD),
            "risk_jump_threshold": float(_RISK_JUMP_THRESHOLD),
            "boundary_strength_floor": float(_BOUNDARY_STRENGTH_FLOOR),
        },
        "boundaries": boundaries,
        "summary": {
            "candidate_count": int(len(boundaries)),
            "sequence_start_idx": _safe_int(_as_dict(risk_rows[0]).get("start_frame"), 0),
            "sequence_end_idx": _safe_int(_as_dict(risk_rows[-1]).get("end_frame"), 0),
            "max_strength": float(max([item.get("strength", 0.0) for item in boundaries]) if boundaries else 0.0),
        },
    }
