from __future__ import annotations


INVARIANT_BASE_POSITIVE_LINES = [
    "Maintain first-person viewpoint continuity across the full sequence.",
    "Preserve stable scene geometry, perspective, and camera alignment.",
    "Keep exposure and white balance stable over time.",
    "Preserve temporal coherence without flicker or drifting structure.",
]

INVARIANT_BASE_NEGATIVE_ITEMS = [
    "flickering",
    "warping",
    "ghosting",
    "geometry drift",
    "inconsistent perspective",
    "exposure instability",
    "white balance shifts",
    "rolling shutter wobble",
    "texture swimming",
    "motion tearing",
    "double edges",
    "heavy blur",
    "low quality",
]


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def get_base_prompt():
    return _collapse_ws(" ".join([str(item).strip() for item in INVARIANT_BASE_POSITIVE_LINES if str(item).strip()]))


def get_negative_prompt():
    return ", ".join([str(item).strip() for item in INVARIANT_BASE_NEGATIVE_ITEMS if str(item).strip()]).strip()


def build_base_prompt_payload():
    return {
        "version": 1,
        "schema": "base_prompt.v1",
        "base_prompt": get_base_prompt(),
        "negative_prompt": get_negative_prompt(),
        "source": "encode.text_gen.base_prompt",
        "geometry_constraints_included": True,
    }
