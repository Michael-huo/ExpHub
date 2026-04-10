from __future__ import annotations

from datetime import datetime


DEFAULT_MOTION_PROMPT = {
    "motion_prompt": "steady egomotion, smooth viewpoint progression.",
    "negative_prompt_delta": "",
    "continuity_emphasis": "balanced",
}

MOTION_PROMPT_BY_STATE = {
    "low_state": {
        "motion_prompt": "steady egomotion, smooth viewpoint progression.",
        "negative_prompt_delta": "",
        "continuity_emphasis": "steady",
    },
    "high_state": {
        "motion_prompt": "elevated motion change, preserve transition continuity and camera stability.",
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "continuity_emphasis": "reinforced",
    },
}


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def build_motion_prompt_payload(segment_inputs):
    state_payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(state_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("segment manifest has no state segments for motion prompts")

    segments = []
    for idx, raw_item in enumerate(state_rows):
        item = _as_dict(raw_item)
        state_label = str(item.get("state_label", "state_unlabeled") or "state_unlabeled")
        motion_cfg = dict(DEFAULT_MOTION_PROMPT)
        motion_cfg.update(_as_dict(MOTION_PROMPT_BY_STATE.get(state_label)))
        state_segment_id = int(item.get("segment_id", idx) or idx)
        segments.append(
            {
                "state_segment_id": int(state_segment_id),
                "state_label": str(state_label),
                "start_frame": int(item.get("start_frame", 0) or 0),
                "end_frame": int(item.get("end_frame", 0) or 0),
                "motion_prompt": str(motion_cfg.get("motion_prompt", "") or ""),
                "negative_prompt_delta": str(motion_cfg.get("negative_prompt_delta", "") or ""),
                "continuity_emphasis": str(motion_cfg.get("continuity_emphasis", "balanced") or "balanced"),
                "motion_prompt_source": "scene_split.state_segments",
            }
        )

    return {
        "version": 1,
        "schema": "motion_prompt_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "encode.text_gen.motion_prompt",
        "motion_prompt_mode": "state_label_mapping_v1",
        "segments": segments,
        "summary": {
            "state_segment_count": int(len(segments)),
        },
    }
