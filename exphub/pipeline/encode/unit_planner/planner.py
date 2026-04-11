from __future__ import annotations

from datetime import datetime

from .constraints import (
    DEFAULT_MIN_EXECUTABLE_FRAMES,
    enforce_contiguous_shared_anchors,
    is_valid_for_decode,
    is_valid_for_export,
    unit_duration_frames,
)
from .costs import boundary_choice_cost, span_policy_for_risk_level


_HIGH_RISK_THRESHOLD = 0.66
_MEDIUM_RISK_THRESHOLD = 0.33


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


def _mean(values):
    values = [float(item) for item in list(values or [])]
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _risk_level_for_score(score):
    score = float(score)
    if score >= _HIGH_RISK_THRESHOLD:
        return "high"
    if score >= _MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _motion_label_for_score(score):
    score = float(score)
    if score >= 0.66:
        return "dynamic"
    if score >= 0.33:
        return "mixed"
    return "steady"


def _segments_covering_range(rows, start_idx, end_idx):
    selected = []
    for raw_item in list(rows or []):
        item = _as_dict(raw_item)
        seg_start = _safe_int(item.get("start_frame"), 0)
        seg_end = _safe_int(item.get("end_frame"), 0)
        if seg_end < int(start_idx) or seg_start > int(end_idx):
            continue
        selected.append(item)
    return selected


def _choose_next_anchor(current_start, sequence_end_idx, internal_candidates, candidate_strength_map, risk_level):
    policy = span_policy_for_risk_level(risk_level)
    min_span = max(
        int(policy.get("min", DEFAULT_MIN_EXECUTABLE_FRAMES) or DEFAULT_MIN_EXECUTABLE_FRAMES),
        int(DEFAULT_MIN_EXECUTABLE_FRAMES),
    )
    max_span = int(policy.get("max", 48) or 48)
    remaining = int(sequence_end_idx) - int(current_start) + 1
    if remaining <= max_span and remaining >= min_span:
        return int(sequence_end_idx)

    lower_bound = int(current_start) + int(min_span)
    upper_bound = min(int(sequence_end_idx), int(current_start) + int(max_span))
    viable = [idx for idx in list(internal_candidates or []) if lower_bound <= int(idx) <= upper_bound]
    if viable:
        viable.sort(
            key=lambda idx: boundary_choice_cost(
                start_idx=current_start,
                candidate_idx=idx,
                policy=policy,
                candidate_strength=candidate_strength_map.get(int(idx), 0.0),
            )
        )
        proposed = int(viable[0])
    else:
        proposed = min(int(sequence_end_idx), int(current_start) + int(policy.get("target", 32) or 32))

    tail_frames = int(sequence_end_idx) - int(proposed) + 1
    if proposed < int(sequence_end_idx) and tail_frames < int(min_span):
        return int(sequence_end_idx)
    if proposed <= int(current_start):
        return min(int(sequence_end_idx), int(current_start) + int(min_span))
    return int(proposed)


def build_generation_units_payload(
    motion_score_payload,
    semantic_shift_payload,
    generation_risk_payload,
    candidate_boundaries_payload,
    sequence_start_idx,
    sequence_end_idx,
):
    motion_rows = list(_as_dict(motion_score_payload).get("segments") or [])
    semantic_rows = list(_as_dict(semantic_shift_payload).get("segments") or [])
    risk_rows = list(_as_dict(generation_risk_payload).get("segments") or [])
    boundary_rows = list(_as_dict(candidate_boundaries_payload).get("boundaries") or [])
    if not motion_rows or not semantic_rows or not risk_rows:
        raise RuntimeError("generation unit planner requires motion, semantic, and risk segments")
    if not (len(motion_rows) == len(semantic_rows) == len(risk_rows)):
        raise RuntimeError("generation unit planner requires matched signal segment counts")
    if int(sequence_end_idx) < int(sequence_start_idx):
        raise RuntimeError("generation unit planner received inverted sequence range")

    sequence_start_idx = int(sequence_start_idx)
    sequence_end_idx = int(sequence_end_idx)
    candidate_strength_map = {}
    internal_candidates = []
    for raw_item in boundary_rows:
        item = _as_dict(raw_item)
        frame_idx = _safe_int(item.get("frame_idx"), -1)
        if frame_idx <= sequence_start_idx or frame_idx >= sequence_end_idx:
            continue
        candidate_strength_map[frame_idx] = max(
            _safe_float(item.get("strength"), 0.0),
            _safe_float(candidate_strength_map.get(frame_idx), 0.0),
        )
        internal_candidates.append(frame_idx)
    internal_candidates = sorted(set(internal_candidates))

    units = []
    current_start = int(sequence_start_idx)
    span_idx = -1
    prev_span_signature = None
    unit_id = 0
    while current_start < sequence_end_idx:
        active_risk = _segments_covering_range(risk_rows, current_start, current_start)
        active_risk_row = active_risk[0] if active_risk else _as_dict(risk_rows[-1])
        next_anchor = _choose_next_anchor(
            current_start=current_start,
            sequence_end_idx=sequence_end_idx,
            internal_candidates=internal_candidates,
            candidate_strength_map=candidate_strength_map,
            risk_level=str(active_risk_row.get("risk_level", "medium") or "medium"),
        )
        covered_motion = _segments_covering_range(motion_rows, current_start, next_anchor)
        covered_semantic = _segments_covering_range(semantic_rows, current_start, next_anchor)
        covered_risk = _segments_covering_range(risk_rows, current_start, next_anchor)
        if not covered_motion or not covered_semantic or not covered_risk:
            raise RuntimeError(
                "generation unit planner found an empty unit coverage: start={} end={}".format(current_start, next_anchor)
            )

        mean_motion_score = _mean([item.get("motion_score", 0.0) for item in covered_motion])
        mean_risk_score = _mean([item.get("generation_risk", 0.0) for item in covered_risk])
        scene_group_id = min([_safe_int(item.get("scene_group_id"), 0) for item in covered_semantic]) if covered_semantic else 0
        scene_label = "scene_group_{:03d}".format(int(scene_group_id))
        motion_label = _motion_label_for_score(mean_motion_score)
        risk_level = _risk_level_for_score(mean_risk_score)

        span_signature = (scene_label, motion_label)
        if span_signature != prev_span_signature:
            span_idx += 1
            prev_span_signature = span_signature
        span_id = "span_{:03d}".format(int(span_idx))
        prompt_ref = {
            "artifact_path": "prompt/prompt_spans.json",
            "span_id": str(span_id),
        }

        unit = {
            "unit_id": "unit_{:03d}".format(int(unit_id)),
            "anchor_start_idx": int(current_start),
            "anchor_end_idx": int(next_anchor),
            "duration_frames": int(unit_duration_frames(current_start, next_anchor)),
            "motion_label": str(motion_label),
            "scene_label": str(scene_label),
            "risk_level": str(risk_level),
            "prompt_ref": prompt_ref,
            "is_valid_for_decode": bool(is_valid_for_decode(current_start, next_anchor)),
            "is_valid_for_export": bool(is_valid_for_export(current_start, next_anchor)),
            "source_segment_ids": sorted(set([_safe_int(item.get("segment_id"), 0) for item in covered_risk])),
            "generation_risk_mean": float(mean_risk_score),
            "motion_score_mean": float(mean_motion_score),
        }
        units.append(unit)
        unit_id += 1
        if next_anchor >= sequence_end_idx:
            break
        current_start = int(next_anchor)

    enforce_contiguous_shared_anchors(units, sequence_start_idx, sequence_end_idx)
    return {
        "version": 1,
        "schema": "generation_units.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "unit_planner",
        "source": "encode.unit_planner.planner",
        "planner_policy": {
            "risk_span_policy": {
                "low": dict(span_policy_for_risk_level("low")),
                "medium": dict(span_policy_for_risk_level("medium")),
                "high": dict(span_policy_for_risk_level("high")),
            },
            "shared_anchor_rule": "prev.anchor_end_idx == next.anchor_start_idx",
            "minimum_executable_frames": int(DEFAULT_MIN_EXECUTABLE_FRAMES),
        },
        "sequence_range": {
            "start_idx": int(sequence_start_idx),
            "end_idx": int(sequence_end_idx),
        },
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "decode_valid_unit_count": int(len([item for item in units if item.get("is_valid_for_decode")])),
            "export_valid_unit_count": int(len([item for item in units if item.get("is_valid_for_export")])),
            "multi_frame_unit_count": int(len([item for item in units if int(item.get("duration_frames", 0) or 0) > 1])),
            "shared_anchor_count": int(max(0, len(units) - 1)),
        },
        "artifact_paths": {
            "motion_score": "segment/motion_score.json",
            "semantic_shift": "segment/semantic_shift.json",
            "generation_risk": "segment/generation_risk.json",
            "candidate_boundaries": "segment/candidate_boundaries.json",
            "generation_units": "segment/generation_units.json",
            "prompt_spans": "prompt/prompt_spans.json",
        },
    }
