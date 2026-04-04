from __future__ import annotations

from typing import Dict, List


INVARIANT_BASE_POSITIVE_LINES = [
    "Maintain first-person viewpoint continuity across the full sequence.",
    "Preserve stable scene geometry, perspective, and camera alignment.",
    "Keep exposure and white balance stable over time.",
    "Preserve temporal coherence without flicker or drifting structure.",
]  # type: List[str]

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
]  # type: List[str]


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def get_invariant_base_prompt():
    # type: () -> str
    return _collapse_ws(" ".join([str(item).strip() for item in INVARIANT_BASE_POSITIVE_LINES if str(item).strip()]))


def get_invariant_negative_prompt():
    # type: () -> str
    return ", ".join([str(item).strip() for item in INVARIANT_BASE_NEGATIVE_ITEMS if str(item).strip()]).strip()


def build_base_prompt_payload():
    # type: () -> Dict[str, object]
    return {
        "version": 2,
        "schema": "base_prompt.v2",
        "base_prompt": get_invariant_base_prompt(),
        "negative_prompt": get_invariant_negative_prompt(),
        "source": "prompt_invariant_base_v2",
        "rules_hit": [
            "invariant_base_only",
            "clip_profile_removed_from_base_prompt",
            "scene_prompt_excluded_from_base_prompt",
            "state_control_separated_from_scene_prompt",
        ],
    }
