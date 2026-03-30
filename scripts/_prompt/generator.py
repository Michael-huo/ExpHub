from __future__ import annotations

from typing import Dict, List


BASE_PROMPT_SENTENCES = [
    "Maintain first-person viewpoint continuity across the full sequence.",
    "Preserve consistent scene geometry, perspective, and camera alignment.",
    "Keep exposure and white balance stable over time.",
    "Avoid flicker, warping, ghosting, geometry drift, and temporal instability.",
]  # type: List[str]

BASE_NEGATIVE_ITEMS = [
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


def build_base_prompt(profile):
    # type: (Dict[str, object]) -> str
    return " ".join([str(item).strip() for item in BASE_PROMPT_SENTENCES if str(item).strip()]).strip()


def build_negative_prompt(profile):
    # type: (Dict[str, object]) -> str
    return ", ".join([str(item).strip() for item in BASE_NEGATIVE_ITEMS if str(item).strip()]).strip()


def build_base_prompt_payload(profile):
    # type: (Dict[str, object]) -> Dict[str, object]
    normalized_profile = dict(profile or {})
    return {
        "version": 1,
        "schema": "base_prompt.v1",
        "base_prompt": build_base_prompt(normalized_profile),
        "negative_prompt": build_negative_prompt(normalized_profile),
        "profile": normalized_profile,
        "source": "prompt_runtime_base_v1",
        "rules_hit": [
            "global_invariants_only",
            "scene_specific_positive_terms_removed",
            "interval_prompts_drive_runtime_specialization",
        ],
    }
