from __future__ import annotations

from exphub.common.io import write_json_atomic


DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
PROMPT_STRATEGY = "four_part_text_image_semantic_state_v1"

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


def _semantic_state_map(semantic_anchors):
    out = {}
    for raw_motion_state in list(_as_dict(semantic_anchors).get("motion_states") or []):
        for raw_state in list(_as_dict(raw_motion_state).get("semantic_states") or []):
            state = _as_dict(raw_state)
            state_id = str(state.get("semantic_state_id", "") or "")
            if state_id:
                out[state_id] = state
    if not out:
        raise RuntimeError("semantic_anchors.json v2 must contain semantic_states")
    return out


def build_prompts(
    generation_units,
    motion_segments,
    semantic_anchors,
    frames_dir=None,
    prompt_python="",
    prompt_backend="blip2",
    prompt_blip2_model=DEFAULT_BLIP2_MODEL,
    out_path=None,
    exphub_root=None,
):
    del motion_segments, frames_dir, prompt_python, prompt_backend, prompt_blip2_model, exphub_root
    state_by_id = _semantic_state_map(semantic_anchors)
    backend_meta = _as_dict(semantic_anchors).get("backend_meta") or {}

    units = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        motion_label = str(unit.get("motion_label", "mixed") or "mixed")
        semantic_state_id = str(unit.get("semantic_state_id", "") or "")
        if not semantic_state_id:
            raise RuntimeError("generation unit {} missing semantic_state_id".format(unit_id))
        state = state_by_id.get(semantic_state_id)
        if not state:
            raise RuntimeError("generation unit {} references missing semantic_state_id {}".format(unit_id, semantic_state_id))
        prompt_semantic = _collapse_ws(state.get("prompt_semantic"))
        if not prompt_semantic:
            raise RuntimeError("semantic state {} missing prompt_semantic".format(semantic_state_id))
        if prompt_semantic.lower().startswith("semantic" + ":"):
            raise RuntimeError("semantic state {} prompt_semantic must not use a label prefix".format(semantic_state_id))
        prompt_motion = MOTION_PROMPTS.get(motion_label, MOTION_PROMPTS["mixed"])
        prompt_positive = _join_prompt(PROMPT_BASE, prompt_motion, prompt_semantic, PROMPT_STABILITY)
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
                "prompt_semantic": str(prompt_semantic),
                "prompt_positive": str(prompt_positive),
            }
        )

    payload = {
        "schema": "prompts.v2",
        "prompt_strategy": PROMPT_STRATEGY,
        "semantic_backend": {
            "name": "blip2",
            "model": str(backend_meta.get("caption_model", DEFAULT_BLIP2_MODEL) or DEFAULT_BLIP2_MODEL),
            "source": "semantic_anchors.motion_states.semantic_states",
        },
        "prompt_negative": str(PROMPT_NEGATIVE),
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "semantic_state_count": int(len(state_by_id)),
            "prompt_positive_source": "prompt_base + prompt_motion + prompt_semantic",
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
