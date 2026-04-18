from __future__ import annotations

from pathlib import Path

from exphub.common.io import write_json_atomic


BASE_PROMPT = (
    "Maintain first-person viewpoint continuity across the full sequence. "
    "Preserve stable scene geometry, perspective, and camera alignment. "
    "Keep exposure and white balance stable over time. "
    "Preserve temporal coherence without flicker or drifting structure."
)

NEGATIVE_PROMPT = (
    "flickering, warping, ghosting, geometry drift, inconsistent perspective, exposure instability, "
    "white balance shifts, rolling shutter wobble, texture swimming, motion tearing, double edges, heavy blur, low quality"
)

MOTION_PROMPTS = {
    "stop": "near-static camera pose, preserve still-scene stability and fine geometry.",
    "forward": "smooth forward egomotion, preserve clear depth progression and stable perspective.",
    "left_turn": "smooth left turn, keep rotation continuous and geometry aligned.",
    "right_turn": "smooth right turn, keep rotation continuous and geometry aligned.",
    "mixed": "mixed egomotion, keep transitions coherent and camera movement readable.",
}


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _collapse_ws(value):
    return " ".join(str(value or "").strip().split()).strip()


def _semantic_prompt(unit, semantic_anchors):
    span = _as_dict(unit.get("anchor_span"))
    start_reason = str(span.get("start_reason", "") or "")
    end_reason = str(span.get("end_reason", "") or "")
    motion_label = str(unit.get("motion_label", "mixed") or "mixed")
    duration_sec = float(unit.get("duration_sec", 0.0) or 0.0)
    if end_reason == "semantic_gain" or start_reason == "semantic_gain":
        return _collapse_ws(
            "scene content changes near the unit boundary; preserve object layout, road geometry, and viewpoint continuity during {}.".format(
                motion_label.replace("_", " ")
            )
        )
    if end_reason == "duration_fallback" or start_reason == "duration_fallback":
        return _collapse_ws(
            "scene content remains broadly continuous over {:.1f} seconds; maintain stable surroundings and consistent details.".format(
                duration_sec
            )
        )
    if motion_label in ("left_turn", "right_turn"):
        return "scene content is continuous; preserve building edges, trees, and ground plane alignment through the turn."
    if motion_label == "forward":
        return "scene content is continuous; preserve depth cues, path geometry, and stable foreground-background layout."
    if motion_label == "stop":
        return "scene content is nearly static; preserve fine texture, lighting, and camera pose stability."
    return "scene content is continuous; preserve stable surroundings and viewpoint context."


def build_prompts(generation_units, motion_segments, semantic_anchors, frames_dir=None, prompt_model_dir="", out_path=None):
    units = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        motion_label = str(unit.get("motion_label", "mixed") or "mixed")
        motion_prompt = MOTION_PROMPTS.get(motion_label, MOTION_PROMPTS["mixed"])
        semantic_prompt = _semantic_prompt(unit, semantic_anchors)
        assembled = _collapse_ws(
            "{} Motion: {} Semantic: {}".format(
                BASE_PROMPT,
                motion_prompt,
                semantic_prompt,
            )
        )
        units.append(
            {
                "unit_id": str(unit.get("unit_id", "") or ""),
                "seg_id": str(unit.get("seg_id", "") or ""),
                "base_prompt": str(BASE_PROMPT),
                "motion_prompt": str(motion_prompt),
                "semantic_prompt": str(semantic_prompt),
                "assembled_prompt": str(assembled),
                "negative_prompt": str(NEGATIVE_PROMPT),
                "motion_label": str(motion_label),
                "prompt_mode": "base+motion+semantic",
            }
        )
    payload = {
        "version": 1,
        "source": "encode.synthetic_prompt.v1",
        "prompt_mode": "base+motion+semantic",
        "base_prompt": str(BASE_PROMPT),
        "negative_prompt": str(NEGATIVE_PROMPT),
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "prompt_model_dir": str(prompt_model_dir or ""),
            "semantic_slot_backend": "rule_template_v1",
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
