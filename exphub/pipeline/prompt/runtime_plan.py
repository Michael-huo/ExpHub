from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from exphub.pipeline.prompt.scene_encoding import normalize_scene_prompt


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


def _join_prompt_parts(*parts):
    # type: (*str) -> str
    values = [_collapse_ws(part) for part in list(parts or [])]
    values = [item for item in values if item]
    return _collapse_ws(" ".join(values))


def _join_negative_prompt(base_negative_prompt, negative_delta):
    # type: (str, str) -> str
    base = _collapse_ws(base_negative_prompt)
    delta = _collapse_ws(negative_delta)
    if not base:
        return delta
    if not delta:
        return base
    return "{}, {}".format(base, delta)


def _overlap_len(start_a, end_a, start_b, end_b):
    # type: (int, int, int, int) -> int
    left = max(int(start_a), int(start_b))
    right = min(int(end_a), int(end_b))
    if right < left:
        return 0
    return int(right - left + 1)


def _segment_midpoint(start_frame, end_frame):
    # type: (int, int) -> float
    return 0.5 * float(int(start_frame) + int(end_frame))


def _as_state_control(state_row):
    # type: (Dict[str, object]) -> Dict[str, object]
    control = _as_dict(state_row.get("state_control"))
    return {
        "prompt_strength": float(control.get("prompt_strength", 0.5) or 0.5),
        "negative_prompt_delta": _collapse_ws(control.get("negative_prompt_delta", "")),
        "motion_trend": str(control.get("motion_trend", "uncertain_interval") or "uncertain_interval"),
        "continuity_emphasis": str(control.get("continuity_emphasis", "balanced") or "balanced"),
    }


def _humanize_token(value):
    # type: (object) -> str
    return _collapse_ws(str(value or "").replace("_", " "))


def _state_control_clause(state_control):
    # type: (Dict[str, object]) -> str
    parts = []
    continuity_emphasis = _humanize_token(state_control.get("continuity_emphasis", ""))
    motion_trend = _humanize_token(state_control.get("motion_trend", ""))
    if continuity_emphasis:
        parts.append("continuity emphasis {}".format(continuity_emphasis))
    if motion_trend:
        parts.append("motion trend {}".format(motion_trend))
    return ", ".join([part for part in parts if part]).strip()


def _labeled_prompt_clause(label, text):
    # type: (str, str) -> str
    body = _collapse_ws(text).rstrip(" ,;:.")
    if not body:
        return ""
    return "{}: {}.".format(str(label), body)


def _build_scene_prompt_map(state_scene_encoding):
    # type: (Dict[str, object]) -> Dict[int, Dict[str, object]]
    scene_segments = list(_as_dict(state_scene_encoding).get("state_segments") or [])
    if not scene_segments:
        raise RuntimeError("state_scene_encoding has no encoded state segments")
    out = {}
    for raw_item in scene_segments:
        item = _as_dict(raw_item)
        state_segment_id = _safe_int(item.get("state_segment_id"), -1)
        if state_segment_id < 0:
            raise RuntimeError("state_scene_encoding contains invalid state_segment_id")
        scene_prompt, _ = normalize_scene_prompt(item.get("scene_prompt", ""))
        scene_prompt = _collapse_ws(scene_prompt)
        if not scene_prompt:
            raise RuntimeError("state_scene_encoding contains empty scene_prompt for state_segment_id={}".format(state_segment_id))
        normalized_item = dict(item)
        normalized_item["scene_prompt"] = scene_prompt
        out[int(state_segment_id)] = normalized_item
    return out


def _pick_state_segment(raw_start, raw_end, state_segments):
    # type: (int, int, List[Dict[str, object]]) -> Tuple[Dict[str, object], int, str]
    best_segment = None
    best_overlap = -1
    for state_segment in list(state_segments or []):
        overlap = _overlap_len(
            raw_start,
            raw_end,
            int(state_segment.get("start_frame", 0) or 0),
            int(state_segment.get("end_frame", 0) or 0),
        )
        if overlap > best_overlap:
            best_overlap = int(overlap)
            best_segment = state_segment
        elif overlap == best_overlap and best_segment is not None:
            current_id = int(state_segment.get("state_segment_id", 0) or 0)
            best_id = int(best_segment.get("state_segment_id", 0) or 0)
            if current_id < best_id:
                best_segment = state_segment

    if best_segment is not None and best_overlap > 0:
        return dict(best_segment), int(best_overlap), "max_overlap"

    nearest_segment = None
    nearest_rank = None
    target_mid = _segment_midpoint(raw_start, raw_end)
    for state_segment in list(state_segments or []):
        state_mid = _segment_midpoint(
            int(state_segment.get("start_frame", 0) or 0),
            int(state_segment.get("end_frame", 0) or 0),
        )
        rank = (
            abs(float(state_mid) - float(target_mid)),
            int(state_segment.get("state_segment_id", 0) or 0),
        )
        if nearest_rank is None or rank < nearest_rank:
            nearest_rank = rank
            nearest_segment = state_segment

    if nearest_segment is None:
        raise RuntimeError("cannot match deploy segment to state segment")
    return dict(nearest_segment), 0, "nearest_state_segment"


def build_runtime_prompt_plan(segment_inputs, state_prompt_manifest, state_scene_encoding, base_prompt_payload, prompt_dir):
    # type: (Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object], Path) -> Dict[str, object]
    deploy_schedule = _as_dict(segment_inputs.get("deploy_schedule_payload"))
    deploy_segments = list(deploy_schedule.get("segments") or [])
    if not deploy_segments:
        raise RuntimeError("deploy_schedule payload has no segments")

    state_segments = list(_as_dict(state_prompt_manifest).get("state_segments") or [])
    if not state_segments:
        raise RuntimeError("state_prompt_manifest has no state_segments")
    scene_prompt_map = _build_scene_prompt_map(state_scene_encoding)

    exp_dir = Path(segment_inputs.get("exp_dir")).resolve()
    prompt_root = Path(prompt_dir).resolve()
    base_prompt_path = prompt_root / "base_prompt.json"
    state_manifest_path = prompt_root / "state_prompt_manifest.json"

    base_prompt_text = _collapse_ws(_as_dict(base_prompt_payload).get("base_prompt", ""))
    base_negative_prompt = _collapse_ws(_as_dict(base_prompt_payload).get("negative_prompt", ""))

    runtime_segments = []
    schedule_backend = str(deploy_schedule.get("backend", "") or "")
    for idx, raw_item in enumerate(deploy_segments):
        item = _as_dict(raw_item)
        deploy_segment_id = int(item.get("segment_id", idx) or idx)
        deploy_start_idx = int(item.get("deploy_start_idx", item.get("start_idx", 0)) or 0)
        deploy_end_idx = int(item.get("deploy_end_idx", item.get("end_idx", 0)) or 0)
        raw_start_idx = int(item.get("raw_start_idx", deploy_start_idx) or deploy_start_idx)
        raw_end_idx = int(item.get("raw_end_idx", deploy_end_idx) or deploy_end_idx)
        state_row, overlap_frames, match_source = _pick_state_segment(raw_start_idx, raw_end_idx, state_segments)
        state_control = _as_state_control(state_row)
        state_control_clause = _state_control_clause(state_control)
        negative_prompt_delta = str(state_control.get("negative_prompt_delta", "") or "")
        state_segment_id = int(state_row.get("state_segment_id", 0) or 0)
        scene_info = _as_dict(scene_prompt_map.get(state_segment_id))
        if not scene_info:
            raise RuntimeError("missing state scene encoding for state_segment_id={}".format(state_segment_id))
        representative_frame = _as_dict(scene_info.get("representative_frame"))
        scene_prompt, _ = normalize_scene_prompt(scene_info.get("scene_prompt", ""))
        scene_prompt = _collapse_ws(scene_prompt)
        resolved_prompt = _join_prompt_parts(
            base_prompt_text,
            _labeled_prompt_clause("Scene", scene_prompt),
            _labeled_prompt_clause("Control", state_control_clause),
        )
        runtime_segments.append(
            {
                "seg": int(deploy_segment_id),
                "segment_id": int(deploy_segment_id),
                "deploy_segment_id": int(deploy_segment_id),
                "schedule_source": "runtime_prompt_plan",
                "execution_backend": schedule_backend,
                "start_frame": int(deploy_start_idx),
                "end_frame": int(deploy_end_idx),
                "start_idx": int(deploy_start_idx),
                "end_idx": int(deploy_end_idx),
                "raw_start_frame": int(raw_start_idx),
                "raw_end_frame": int(raw_end_idx),
                "raw_start_idx": int(raw_start_idx),
                "raw_end_idx": int(raw_end_idx),
                "deploy_start_idx": int(deploy_start_idx),
                "deploy_end_idx": int(deploy_end_idx),
                "raw_gap": int(item.get("raw_gap", raw_end_idx - raw_start_idx) or (raw_end_idx - raw_start_idx)),
                "deploy_gap": int(
                    item.get("deploy_gap", deploy_end_idx - deploy_start_idx) or (deploy_end_idx - deploy_start_idx)
                ),
                "num_frames": int(item.get("num_frames", deploy_end_idx - deploy_start_idx + 1) or (deploy_end_idx - deploy_start_idx + 1)),
                "boundary_shift": int(item.get("boundary_shift", 0) or 0),
                "gap_error": int(item.get("gap_error", 0) or 0),
                "state_segment_id": int(state_segment_id),
                "state_label": str(state_row.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "state_control": dict(state_control),
                "scene_prompt": scene_prompt,
                "scene_prompt_source": str(scene_info.get("scene_prompt_source", "state_v2t_primary_frame") or "state_v2t_primary_frame"),
                "scene_encoding_backend": str(scene_info.get("scene_encoding_backend", "") or ""),
                "scene_prompt_frame_idx": int(representative_frame.get("frame_idx", raw_start_idx) or raw_start_idx),
                "scene_prompt_image": str(representative_frame.get("image_path", "") or ""),
                "scene_prompt_selection_source": str(
                    representative_frame.get("selection_source", "") or "segment_keyframe_nearest_midpoint"
                ),
                "scene_prompt_candidate_keyframe_count": int(
                    representative_frame.get("candidate_keyframe_count", 0) or 0
                ),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": _join_negative_prompt(base_negative_prompt, negative_prompt_delta),
                "negative_prompt_delta": negative_prompt_delta,
                "prompt_strength": float(state_control.get("prompt_strength", 0.5) or 0.5),
                "motion_trend": str(state_control.get("motion_trend", "uncertain_interval") or "uncertain_interval"),
                "match_source": str(match_source),
                "overlap_frames": int(overlap_frames),
                "prompt_source": "runtime_prompt_plan",
            }
        )

    return {
        "version": 2,
        "schema": "runtime_prompt_plan.v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "runtime_prompt_plan_formal",
        "schedule_source": "segment_manifest.deploy_schedule",
        "execution_backend": schedule_backend,
        "base_prompt": base_prompt_text,
        "negative_prompt": base_negative_prompt,
        "state_control_mode": "minimal_state_control",
        "scene_prompt_mode": str(_as_dict(state_scene_encoding).get("scene_prompt_mode", "") or "state_v2t_primary_frame"),
        "scene_encoding_backend": str(_as_dict(state_scene_encoding).get("backend", "") or ""),
        "scene_prompt_style": str(_as_dict(state_scene_encoding).get("scene_prompt_style", "") or "compact_canonical_phrase_v1"),
        "deploy_segment_count": int(len(runtime_segments)),
        "source_files": {
            "base_prompt": _relative_to_exp(exp_dir, base_prompt_path),
            "state_prompt_manifest": _relative_to_exp(exp_dir, state_manifest_path),
            "segment_manifest": str((segment_inputs.get("source_files") or {}).get("segment_manifest", "") or ""),
            "state_segments": str((segment_inputs.get("source_files") or {}).get("state_segments", "") or ""),
            "deploy_schedule": str((segment_inputs.get("source_files") or {}).get("deploy_schedule", "") or ""),
        },
        "segments": runtime_segments,
    }
