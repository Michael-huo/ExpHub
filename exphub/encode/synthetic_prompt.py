from __future__ import annotations

from exphub.common.io import write_json_atomic


PROMPT_STRATEGY = "base_motion_fixed_prompt_v1"

PROMPT_BASE = (
    "first-person viewpoint continuity, stable scene geometry, consistent perspective and camera alignment, "
    "stable exposure and white balance, temporal coherence without flicker"
)

PROMPT_NEGATIVE = (
    "blurry, low detail, low quality, distorted geometry, warped structure, flicker, temporal inconsistency, "
    "drifting objects, duplicated objects, broken perspective, unstable camera, sudden viewpoint change, "
    "ghosting, artifacts, oversmoothing"
)

MOTION_PROMPTS = {
    "stop": "near-static camera pose, stable still-scene geometry, preserved fine structure",
    "forward": "smooth forward egomotion, clear depth progression, stable perspective",
    "left_turn": "smooth left turn, continuous camera rotation, geometry-aligned motion",
    "right_turn": "smooth right turn, continuous camera rotation, geometry-aligned motion",
    "mixed": "coherent mixed egomotion, readable camera movement, stable transition",
}

PROMPT_STABILITY = "stable foreground-background layout"


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _collapse_ws(value):
    return " ".join(str(value or "").strip().split()).strip()


def _phrases(*values):
    out = []
    seen = set()
    for value in values:
        for part in str(value or "").replace(".", ",").split(","):
            text = _collapse_ws(part).strip(" ,:;-")
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _join_prompt(*values):
    return ", ".join(_phrases(*values))


def _visual_anchor_count(semantic_anchors):
    count = 0
    for raw_motion_state in list(_as_dict(semantic_anchors).get("motion_states") or []):
        count += len(list(_as_dict(raw_motion_state).get("semantic_states") or []))
    return int(count)


def build_prompts(
    generation_units,
    motion_segments,
    semantic_anchors,
    frames_dir=None,
    out_path=None,
):
    del motion_segments, frames_dir
    visual_anchor_count = _visual_anchor_count(semantic_anchors)

    units = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        motion_label = str(unit.get("motion_label", "mixed") or "mixed")
        semantic_state_id = str(unit.get("semantic_state_id", "") or "")
        if not semantic_state_id:
            raise RuntimeError("generation unit {} missing semantic_state_id".format(unit_id))
        prompt_motion = MOTION_PROMPTS.get(motion_label, MOTION_PROMPTS["mixed"])
        prompt_positive = _join_prompt(PROMPT_BASE, prompt_motion, PROMPT_STABILITY)
        for forbidden in tuple("{}{}".format(label, ":") for label in ("Motion", "Semantic", "Base")):
            if forbidden in prompt_positive:
                raise RuntimeError("prompt_positive for {} contains a forbidden label prefix".format(unit_id))
        units.append(
            {
                "unit_id": str(unit_id),
                "start_idx": int(unit.get("start_idx")),
                "end_idx": int(unit.get("end_idx")),
                "motion_state_id": str(unit.get("motion_state_id", "") or ""),
                "motion_label": str(motion_label),
                "semantic_state_id": str(semantic_state_id),
                "prompt_negative": str(PROMPT_NEGATIVE),
                "prompt_base": str(PROMPT_BASE),
                "prompt_motion": str(prompt_motion),
                "prompt_stability": str(PROMPT_STABILITY),
                "prompt_positive": str(prompt_positive),
                "assembled_prompt": str(prompt_positive),
            }
        )

    payload = {
        "schema": "prompts.v3",
        "prompt_strategy": PROMPT_STRATEGY,
        "anchor_backend": {
            "name": "image_embedding",
            "source": "semantic_anchors.motion_states.semantic_states",
        },
        "prompt_negative": str(PROMPT_NEGATIVE),
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "visual_anchor_count": int(visual_anchor_count),
            "prompt_positive_source": "prompt_base + prompt_motion + prompt_stability",
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
