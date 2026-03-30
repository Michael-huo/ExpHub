from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STATE_PROMPT_PRESETS = {
    "low_state": {
        "prompt_text": (
            "Within this interval, keep motion transitions gentle and preserve stable first-person continuity, "
            "local geometry, and temporal consistency."
        ),
        "negative_prompt_delta": "",
        "prompt_strength": 0.35,
        "motion_trend": "stable_interval",
    },
    "high_state": {
        "prompt_text": (
            "Within this interval, treat the motion as high-risk and preserve first-person continuity, "
            "local geometry, exposure stability, and temporal coherence under stronger viewpoint change."
        ),
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "prompt_strength": 0.75,
        "motion_trend": "risk_interval",
    },
    "unknown": {
        "prompt_text": (
            "Within this interval, preserve first-person continuity, stable geometry, and temporal coherence."
        ),
        "negative_prompt_delta": "",
        "prompt_strength": 0.50,
        "motion_trend": "unknown_interval",
    },
}  # type: Dict[str, Dict[str, object]]


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def _load_json_if_object(path_obj):
    # type: (Path) -> Dict[str, object]
    path = Path(path_obj).resolve()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _relative_path(base_dir, target_path):
    # type: (Optional[Path], Optional[Path]) -> str
    if target_path is None:
        return ""
    target = Path(target_path).resolve()
    if base_dir is not None:
        try:
            return str(target.relative_to(base_dir.resolve()))
        except Exception:
            pass
    return str(target)


def _resolve_segment_dir(exp_dir, frames_dir, segment_dir):
    # type: (Optional[Path], Path, Optional[Path]) -> Optional[Path]
    if segment_dir is not None:
        candidate = Path(segment_dir).resolve()
        if candidate.is_dir():
            return candidate
    if exp_dir is not None:
        candidate = (Path(exp_dir).resolve() / "segment").resolve()
        if candidate.is_dir():
            return candidate
    frames_root = Path(frames_dir).resolve()
    if frames_root.name == "frames" and frames_root.parent.is_dir():
        return frames_root.parent.resolve()
    return None


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


def _preset_for_state_label(state_label):
    # type: (str) -> Dict[str, object]
    return dict(STATE_PROMPT_PRESETS.get(str(state_label or "unknown"), STATE_PROMPT_PRESETS["unknown"]))


def _coerce_state_segments(payload):
    # type: (Dict[str, object]) -> List[Dict[str, object]]
    rows = []
    for idx, raw_item in enumerate(list(payload.get("segments") or [])):
        item = _as_dict(raw_item)
        try:
            start_frame = int(item.get("start_frame", 0) or 0)
            end_frame = int(item.get("end_frame", 0) or 0)
        except Exception:
            continue
        if end_frame < start_frame:
            continue
        state_label = str(item.get("state_label", "unknown") or "unknown")
        preset = _preset_for_state_label(state_label)
        rows.append(
            {
                "state_segment_id": int(item.get("segment_id", idx) or idx),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "state_label": str(state_label),
                "prompt_text": str(preset.get("prompt_text", "") or ""),
                "negative_prompt_delta": str(preset.get("negative_prompt_delta", "") or ""),
                "prompt_strength": float(preset.get("prompt_strength", 0.5) or 0.5),
                "motion_trend": str(preset.get("motion_trend", "unknown_interval") or "unknown_interval"),
                "source_segment_id": int(item.get("segment_id", idx) or idx),
            }
        )
    return rows


def _fallback_state_segments(frames_count):
    # type: (int) -> List[Dict[str, object]]
    end_frame = max(0, int(frames_count) - 1)
    preset = _preset_for_state_label("unknown")
    return [
        {
            "state_segment_id": 0,
            "start_frame": 0,
            "end_frame": int(end_frame),
            "state_label": "unknown",
            "prompt_text": str(preset.get("prompt_text", "") or ""),
            "negative_prompt_delta": str(preset.get("negative_prompt_delta", "") or ""),
            "prompt_strength": float(preset.get("prompt_strength", 0.5) or 0.5),
            "motion_trend": str(preset.get("motion_trend", "unknown_interval") or "unknown_interval"),
            "source_segment_id": None,
        }
    ]


def _coerce_deploy_segments(payload):
    # type: (Dict[str, object]) -> List[Dict[str, object]]
    rows = []
    for idx, raw_item in enumerate(list(payload.get("segments") or [])):
        item = _as_dict(raw_item)
        try:
            deploy_start = int(item.get("deploy_start_idx", item.get("start_idx", 0)) or 0)
            deploy_end = int(item.get("deploy_end_idx", item.get("end_idx", 0)) or 0)
            raw_start = int(item.get("raw_start_idx", deploy_start) or deploy_start)
            raw_end = int(item.get("raw_end_idx", deploy_end) or deploy_end)
        except Exception:
            continue
        if deploy_end < deploy_start or raw_end < raw_start:
            continue
        rows.append(
            {
                "deploy_segment_id": int(item.get("segment_id", idx) or idx),
                "start_frame": int(deploy_start),
                "end_frame": int(deploy_end),
                "raw_start_frame": int(raw_start),
                "raw_end_frame": int(raw_end),
            }
        )
    return rows


def _pick_state_segment(raw_start, raw_end, state_segments):
    # type: (int, int, List[Dict[str, object]]) -> Tuple[int, int, str]
    if not state_segments:
        return -1, 0, "no_state_segments"

    best_segment = None
    best_overlap = -1
    for state_segment in list(state_segments):
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
            if int(state_segment.get("state_segment_id", 0) or 0) < int(best_segment.get("state_segment_id", 0) or 0):
                best_segment = state_segment

    if best_segment is not None and best_overlap > 0:
        return int(best_segment.get("state_segment_id", 0) or 0), int(best_overlap), "max_overlap"

    target_mid = _segment_midpoint(raw_start, raw_end)
    nearest_segment = None
    nearest_rank = None
    for state_segment in list(state_segments):
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
        return -1, 0, "no_state_segments"
    return int(nearest_segment.get("state_segment_id", 0) or 0), 0, "nearest_state_segment"


def _join_prompt_parts(base_prompt, local_prompt):
    # type: (str, str) -> str
    base = _collapse_ws(base_prompt)
    local = _collapse_ws(local_prompt)
    if not base:
        return local
    if not local:
        return base
    return _collapse_ws("{} {}".format(base, local))


def _join_negative_prompt(base_negative_prompt, negative_delta):
    # type: (str, str) -> str
    base = _collapse_ws(base_negative_prompt)
    delta = _collapse_ws(negative_delta)
    if not delta:
        return base
    if not base:
        return delta
    return "{}, {}".format(base, delta)


def build_state_prompt_artifacts(
    frames_dir,
    frames_count,
    prompt_dir,
    base_prompt_payload,
    exp_dir=None,
    segment_dir=None,
):
    # type: (Path, int, Path, Dict[str, object], Optional[Path], Optional[Path]) -> Dict[str, object]
    prompt_dir = Path(prompt_dir).resolve()
    frames_dir = Path(frames_dir).resolve()
    exp_dir_path = Path(exp_dir).resolve() if exp_dir is not None else None
    segment_dir_path = _resolve_segment_dir(exp_dir_path, frames_dir, segment_dir)
    base_dir = exp_dir_path
    if base_dir is None and prompt_dir.parent.is_dir():
        base_dir = prompt_dir.parent.resolve()

    state_segments_path = None  # type: Optional[Path]
    deploy_schedule_path = None  # type: Optional[Path]
    state_segments_payload = {}  # type: Dict[str, object]
    deploy_schedule_payload = {}  # type: Dict[str, object]
    if segment_dir_path is not None:
        state_segments_path = (segment_dir_path / "state_segmentation" / "state_segments.json").resolve()
        deploy_schedule_path = (segment_dir_path / "deploy_schedule.json").resolve()
        state_segments_payload = _load_json_if_object(state_segments_path)
        deploy_schedule_payload = _load_json_if_object(deploy_schedule_path)

    has_state_segments = bool(state_segments_payload)
    state_segments = _coerce_state_segments(state_segments_payload)
    if not state_segments:
        state_segments = _fallback_state_segments(frames_count)

    base_prompt_text = _collapse_ws((base_prompt_payload or {}).get("base_prompt", ""))
    base_negative_prompt = _collapse_ws((base_prompt_payload or {}).get("negative_prompt", ""))
    base_prompt_path = prompt_dir / "base_prompt.json"

    state_manifest = {
        "version": 1,
        "schema": "state_prompt_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_type": "interval_prompt_manifest",
        "base_prompt_path": _relative_path(base_dir, base_prompt_path),
        "state_segment_count": int(len(state_segments)),
        "source_files": {
            "state_segments": _relative_path(base_dir, state_segments_path) if state_segments_path is not None else "",
            "deploy_schedule": _relative_path(base_dir, deploy_schedule_path) if deploy_schedule_path is not None else "",
            "has_state_segments": bool(has_state_segments),
        },
        "state_segments": list(state_segments),
    }

    deploy_segments = _coerce_deploy_segments(deploy_schedule_payload)
    runtime_segments = []
    for deploy_segment in list(deploy_segments):
        raw_start = int(deploy_segment.get("raw_start_frame", deploy_segment.get("start_frame", 0)) or 0)
        raw_end = int(deploy_segment.get("raw_end_frame", deploy_segment.get("end_frame", 0)) or 0)
        state_segment_id, overlap_frames, match_source = _pick_state_segment(raw_start, raw_end, state_segments)
        state_row = {}
        if state_segment_id >= 0:
            for item in list(state_segments):
                current_state_segment_id = item.get("state_segment_id", -1)
                if int(current_state_segment_id if current_state_segment_id is not None else -1) == int(state_segment_id):
                    state_row = dict(item)
                    break

        state_label = str(state_row.get("state_label", "unknown") or "unknown")
        local_prompt = _collapse_ws(state_row.get("prompt_text", ""))
        negative_prompt_delta = _collapse_ws(state_row.get("negative_prompt_delta", ""))
        prompt_strength = float(state_row.get("prompt_strength", 0.5) or 0.5)
        resolved_prompt = _join_prompt_parts(base_prompt_text, local_prompt)
        negative_prompt = _join_negative_prompt(base_negative_prompt, negative_prompt_delta)
        runtime_segments.append(
            {
                "seg": int(deploy_segment.get("deploy_segment_id", 0) or 0),
                "segment_id": int(deploy_segment.get("deploy_segment_id", 0) or 0),
                "deploy_segment_id": int(deploy_segment.get("deploy_segment_id", 0) or 0),
                "start_frame": int(deploy_segment.get("start_frame", 0) or 0),
                "end_frame": int(deploy_segment.get("end_frame", 0) or 0),
                "raw_start_frame": int(raw_start),
                "raw_end_frame": int(raw_end),
                "state_segment_id": int(state_segment_id) if state_segment_id >= 0 else None,
                "state_label": str(state_label),
                "base_prompt": str(base_prompt_text),
                "local_prompt": str(local_prompt),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": str(negative_prompt),
                "negative_prompt_delta": str(negative_prompt_delta),
                "prompt_strength": float(prompt_strength),
                "motion_trend": str(state_row.get("motion_trend", "unknown_interval") or "unknown_interval"),
                "match_source": str(match_source),
                "overlap_frames": int(overlap_frames),
                "prompt_source": "runtime_prompt_plan",
            }
        )

    runtime_plan = {
        "version": 1,
        "schema": "runtime_prompt_plan.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "runtime_prompt_plan_v1",
        "base_prompt": str(base_prompt_text),
        "negative_prompt": str(base_negative_prompt),
        "state_prompt_manifest_path": _relative_path(base_dir, prompt_dir / "state_prompt_manifest.json"),
        "deploy_segment_count": int(len(runtime_segments)),
        "source_files": {
            "base_prompt": _relative_path(base_dir, base_prompt_path),
            "state_prompt_manifest": _relative_path(base_dir, prompt_dir / "state_prompt_manifest.json"),
            "state_segments": _relative_path(base_dir, state_segments_path) if state_segments_path is not None else "",
            "deploy_schedule": _relative_path(base_dir, deploy_schedule_path) if deploy_schedule_path is not None else "",
        },
        "segments": list(runtime_segments),
    }

    return {
        "state_prompt_manifest": state_manifest,
        "runtime_prompt_plan": runtime_plan,
        "summary": {
            "has_state_segments": bool(has_state_segments),
            "state_segment_count": int(len(state_segments)),
            "deploy_segment_count": int(len(runtime_segments)),
            "segment_dir": _relative_path(base_dir, segment_dir_path) if segment_dir_path is not None else "",
            "source_files": {
                "state_segments": _relative_path(base_dir, state_segments_path) if state_segments_path is not None else "",
                "deploy_schedule": _relative_path(base_dir, deploy_schedule_path) if deploy_schedule_path is not None else "",
            },
        },
    }
