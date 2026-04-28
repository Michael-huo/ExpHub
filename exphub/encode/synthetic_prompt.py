from __future__ import annotations

from pathlib import Path

from exphub.common.io import read_json_dict, write_json_atomic


PROMPT_PROFILE = "base_motion_prompt"
REQUIRED_MOTION_LABELS = ("stop", "forward", "left_turn", "right_turn", "mixed")


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


def _join_prompt(*values, joiner=", "):
    return str(joiner).join(_phrases(*values))


def _default_prompt_manifest_path():
    return Path(__file__).resolve().parents[2] / "config" / "prompt_manifest.json"


def _non_empty_string(value, label):
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("{} must be a non-empty string".format(label))
    return text


def _load_prompt_manifest(prompt_manifest_path=None):
    path = Path(prompt_manifest_path).resolve() if prompt_manifest_path is not None else _default_prompt_manifest_path()
    manifest = read_json_dict(path)
    if not manifest:
        raise RuntimeError("invalid prompt manifest: {}".format(path))
    if str(manifest.get("schema", "") or "") != "prompt_manifest":
        raise RuntimeError("prompt manifest schema must be prompt_manifest: {}".format(path))
    profile = _non_empty_string(manifest.get("prompt_profile"), "prompt_manifest.prompt_profile")
    if profile != PROMPT_PROFILE:
        raise RuntimeError("prompt manifest profile must be {}: {}".format(PROMPT_PROFILE, path))

    positive = _as_dict(manifest.get("positive"))
    negative = _as_dict(manifest.get("negative"))
    composition = _as_dict(manifest.get("composition"))
    base_prompt = _non_empty_string(positive.get("base_prompt"), "positive.base_prompt")
    negative_prompt = _non_empty_string(negative.get("negative_prompt"), "negative.negative_prompt")
    motion_prompts = _as_dict(positive.get("motion_prompts"))
    missing = [label for label in REQUIRED_MOTION_LABELS if not str(motion_prompts.get(label, "") or "").strip()]
    if missing:
        raise RuntimeError("prompt manifest missing motion prompts: {}".format(", ".join(missing)))
    joiner = composition.get("joiner", ", ")
    if not isinstance(joiner, str):
        raise RuntimeError("composition.joiner must be a string")
    return {
        "path": path,
        "schema": "prompt_manifest",
        "profile": profile,
        "base_prompt": base_prompt,
        "negative_prompt": negative_prompt,
        "motion_prompts": {
            label: _non_empty_string(motion_prompts.get(label), "motion_prompts.{}".format(label))
            for label in REQUIRED_MOTION_LABELS
        },
        "joiner": joiner,
    }


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
    prompt_manifest_path=None,
):
    del motion_segments, frames_dir
    manifest = _load_prompt_manifest(prompt_manifest_path)
    visual_anchor_count = _visual_anchor_count(semantic_anchors)

    units = []
    motion_prompt_fallback_count = 0
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        motion_label = str(unit.get("motion_label", "mixed") or "mixed")
        semantic_state_id = str(unit.get("semantic_state_id", "") or "")
        if not semantic_state_id:
            raise RuntimeError("generation unit {} missing semantic_state_id".format(unit_id))
        if motion_label not in manifest["motion_prompts"]:
            motion_prompt_fallback_count += 1
        prompt_motion = manifest["motion_prompts"].get(motion_label, manifest["motion_prompts"]["mixed"])
        prompt_positive = _join_prompt(manifest["base_prompt"], prompt_motion, joiner=manifest["joiner"])
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
                "prompt_base": str(manifest["base_prompt"]),
                "prompt_motion": str(prompt_motion),
                "prompt_positive": str(prompt_positive),
                "assembled_prompt": str(prompt_positive),
                "prompt_negative": str(manifest["negative_prompt"]),
            }
        )

    payload = {
        "schema": "prompts",
        "prompt_profile": str(manifest["profile"]),
        "prompt_manifest": {
            "path": str(manifest["path"]),
            "schema": str(manifest["schema"]),
            "profile": str(manifest["profile"]),
        },
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "visual_anchor_count": int(visual_anchor_count),
            "prompt_positive_source": "base_prompt + motion_prompt",
            "motion_prompt_fallback_count": int(motion_prompt_fallback_count),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
