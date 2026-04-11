from __future__ import annotations

from datetime import datetime

from ._risk_score import build_generation_risk_payload
from ._semantic_shift import build_semantic_shift_payload


_DYNAMIC_THRESHOLD = 0.66
_MIXED_THRESHOLD = 0.33


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


__all__ = [
    "build_generation_risk_payload",
    "build_motion_score_payload",
    "build_semantic_shift_payload",
]
