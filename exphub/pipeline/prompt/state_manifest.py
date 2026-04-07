from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

from exphub.common.io import ensure_file, read_json_dict


DEFAULT_STATE_CONTROL = {
    "negative_prompt_delta": "",
    "continuity_emphasis": "balanced",
}  # type: Dict[str, object]

STATE_CONTROL_BY_LABEL = {
    "low_state": {
        "negative_prompt_delta": "",
        "continuity_emphasis": "steady",
    },
    "high_state": {
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "continuity_emphasis": "reinforced",
    },
}  # type: Dict[str, Dict[str, object]]


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _relative_to_exp(exp_dir, target_path):
    # type: (Path, Path) -> str
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _state_control_for_label(state_label):
    # type: (str) -> Dict[str, object]
    name = str(state_label or "").strip()
    control = dict(DEFAULT_STATE_CONTROL)
    control.update(_as_dict(STATE_CONTROL_BY_LABEL.get(name)))
    return control


def load_segment_prompt_inputs(segment_manifest_path):
    # type: (Path) -> Dict[str, object]
    manifest_path = ensure_file(segment_manifest_path, "segment manifest")
    manifest = read_json_dict(manifest_path)
    if not manifest:
        raise RuntimeError("invalid segment manifest: {}".format(manifest_path))

    exp_dir = manifest_path.parent.parent.resolve()
    state_segments_payload = _as_dict(manifest.get("state_segments"))
    deploy_schedule_payload = _as_dict(manifest.get("deploy_schedule"))

    if not state_segments_payload:
        raise RuntimeError("segment manifest missing embedded state_segments payload: {}".format(manifest_path))
    if not deploy_schedule_payload:
        raise RuntimeError("segment manifest missing deploy_schedule payload: {}".format(manifest_path))

    frames_meta = _as_dict(manifest.get("frames"))
    state_segments_source = "{}#state_segments".format(_relative_to_exp(exp_dir, manifest_path))
    deploy_schedule_source = "{}#deploy_schedule".format(_relative_to_exp(exp_dir, manifest_path))
    return {
        "segment_manifest_path": manifest_path,
        "segment_manifest": manifest,
        "state_segments_path": manifest_path,
        "state_segments_payload": state_segments_payload,
        "deploy_schedule_path": manifest_path,
        "deploy_schedule_payload": deploy_schedule_payload,
        "frame_count": int(frames_meta.get("frame_count", 0) or 0),
        "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
        "exp_dir": exp_dir,
        "source_files": {
            "segment_manifest": _relative_to_exp(exp_dir, manifest_path),
            "state_segments": state_segments_source,
            "deploy_schedule": deploy_schedule_source,
        },
    }


def build_state_prompt_manifest(segment_inputs):
    # type: (Dict[str, object]) -> Dict[str, object]
    payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("state_segments payload has no segments")

    segments = []
    for idx, raw_item in enumerate(state_rows):
        item = _as_dict(raw_item)
        start_frame = int(item.get("start_frame", 0) or 0)
        end_frame = int(item.get("end_frame", 0) or 0)
        if end_frame < start_frame:
            raise RuntimeError("invalid state segment range at index {}".format(idx))
        state_label = str(item.get("state_label", "state_unlabeled") or "state_unlabeled")
        control = _state_control_for_label(state_label)
        state_segment_id = int(item.get("segment_id", idx) or idx)
        segments.append(
            {
                "state_segment_id": int(state_segment_id),
                "scene_segment_id": int(state_segment_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "state_label": state_label,
                "state_control": {
                    "negative_prompt_delta": str(control.get("negative_prompt_delta", "") or ""),
                    "continuity_emphasis": str(control.get("continuity_emphasis", "balanced") or "balanced"),
                },
                "source_segment_id": int(state_segment_id),
            }
        )

    return {
        "version": 3,
        "schema": "state_prompt_manifest.v3",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_type": "state_control_manifest",
        "state_segment_count": int(len(segments)),
        "source_files": dict(segment_inputs.get("source_files") or {}),
        "state_segments": segments,
        "summary": {
            "state_segment_count": int(len(segments)),
            "frame_count": int(segment_inputs.get("frame_count", 0) or 0),
            "frame_count_used": int(segment_inputs.get("frame_count_used", 0) or 0),
            "state_control_mode": "minimal_state_control",
            "scene_binding_key": "state_segment_id",
        },
    }
