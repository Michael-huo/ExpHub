from __future__ import annotations

from datetime import datetime

from exphub.pipeline.encode.text_gen.motion_prompt import resolve_motion_prompt_from_planner_label


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _join_prompt_parts(*parts):
    values = [_collapse_ws(item) for item in list(parts or [])]
    values = [item for item in values if item]
    return _collapse_ws(" ".join(values))


def _join_negative_prompt(base_negative_prompt, negative_delta):
    base = _collapse_ws(base_negative_prompt)
    delta = _collapse_ws(negative_delta)
    if not base:
        return delta
    if not delta:
        return base
    return "{}, {}".format(base, delta)


def _labeled_prompt_clause(label, text):
    body = _collapse_ws(text).rstrip(" ,;:.")
    if not body:
        return ""
    return "{}: {}.".format(str(label), body)


def _overlap_frames(start_a, end_a, start_b, end_b):
    left = max(int(start_a), int(start_b))
    right = min(int(end_a), int(end_b))
    return max(0, int(right) - int(left) + 1)


def _segment_prompt_map(prompt_manifest):
    mapping = {}
    for idx, raw_item in enumerate(list(_as_dict(prompt_manifest).get("segments") or [])):
        item = _as_dict(raw_item)
        mapping[_safe_int(item.get("state_segment_id"), idx)] = item
    return mapping


def _resolve_scene_prompt(span_units, prompt_segment_map):
    weight_by_prompt = {}
    for unit in list(span_units or []):
        unit_start = _safe_int(unit.get("anchor_start_idx"), 0)
        unit_end = _safe_int(unit.get("anchor_end_idx"), 0)
        for segment_id in list(unit.get("source_segment_ids") or []):
            prompt_item = _as_dict(prompt_segment_map.get(_safe_int(segment_id), {}))
            if not prompt_item:
                continue
            scene_prompt = _collapse_ws(prompt_item.get("scene_prompt", ""))
            if not scene_prompt:
                continue
            overlap = _overlap_frames(
                unit_start,
                unit_end,
                _safe_int(prompt_item.get("start_frame"), 0),
                _safe_int(prompt_item.get("end_frame"), 0),
            )
            if overlap <= 0:
                overlap = max(
                    1,
                    _safe_int(prompt_item.get("end_frame"), 0) - _safe_int(prompt_item.get("start_frame"), 0) + 1,
                )
            weight_by_prompt[scene_prompt] = int(weight_by_prompt.get(scene_prompt, 0) + overlap)
    if not weight_by_prompt:
        return ""
    return max(weight_by_prompt.items(), key=lambda item: (int(item[1]), item[0]))[0]


def build_prompt_spans_payload(prompt_manifest, generation_units_payload):
    prompt_payload = _as_dict(prompt_manifest)
    units_payload = _as_dict(generation_units_payload)
    units = list(units_payload.get("units") or [])
    if not units:
        raise RuntimeError("prompt spans require generation units")

    base_prompt = str(prompt_payload.get("base_prompt", "") or "")
    negative_prompt = str(prompt_payload.get("negative_prompt", "") or "")
    prompt_segment_map = _segment_prompt_map(prompt_payload)

    grouped = {}
    for raw_unit in units:
        unit = _as_dict(raw_unit)
        prompt_ref = _as_dict(unit.get("prompt_ref"))
        span_id = str(prompt_ref.get("span_id", "") or "")
        if not span_id:
            raise RuntimeError("generation unit missing prompt_ref.span_id")
        grouped.setdefault(span_id, []).append(unit)

    ordered_span_ids = sorted(grouped.keys())
    spans = []
    for span_id in ordered_span_ids:
        span_units = list(grouped.get(span_id) or [])
        motion_label = str(span_units[0].get("motion_label", "steady") or "steady")
        scene_label = str(span_units[0].get("scene_label", "scene_group_000") or "scene_group_000")
        motion_prompt_payload = resolve_motion_prompt_from_planner_label(motion_label)
        scene_prompt = _resolve_scene_prompt(span_units, prompt_segment_map)
        resolved_prompt = _join_prompt_parts(
            base_prompt,
            _labeled_prompt_clause("Scene", scene_prompt),
            _labeled_prompt_clause("Motion", motion_prompt_payload.get("motion_prompt", "")),
        )
        spans.append(
            {
                "span_id": str(span_id),
                "scene_label": str(scene_label),
                "motion_label": str(motion_label),
                "anchor_start_idx": int(min([_safe_int(unit.get("anchor_start_idx"), 0) for unit in span_units])),
                "anchor_end_idx": int(max([_safe_int(unit.get("anchor_end_idx"), 0) for unit in span_units])),
                "unit_ids": [str(unit.get("unit_id", "") or "") for unit in span_units],
                "source_segment_ids": sorted(
                    set([int(segment_id) for unit in span_units for segment_id in list(unit.get("source_segment_ids") or [])])
                ),
                "shared_unit_count": int(len(span_units)),
                "base_prompt": str(base_prompt),
                "scene_prompt": str(scene_prompt),
                "scene_prompt_source": "prompt_manifest.scene_prompt_majority_overlap",
                "motion_prompt": str(motion_prompt_payload.get("motion_prompt", "") or ""),
                "motion_prompt_source": "planner.motion_label_mapping",
                "negative_prompt_delta": str(motion_prompt_payload.get("negative_prompt_delta", "") or ""),
                "continuity_emphasis": str(motion_prompt_payload.get("continuity_emphasis", "balanced") or "balanced"),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": _join_negative_prompt(negative_prompt, motion_prompt_payload.get("negative_prompt_delta", "")),
            }
        )

    return {
        "version": 1,
        "schema": "prompt_spans.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "text_gen",
        "source": "encode.text_gen.prompt_spans",
        "prompt_structure": "base_scene_motion",
        "grouping_policy": "contiguous_generation_units_share_prompt_when_prompt_ref.span_id_matches",
        "base_prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "spans": spans,
        "summary": {
            "span_count": int(len(spans)),
            "shared_prompt_unit_count": int(sum([int(item.get("shared_unit_count", 0) or 0) for item in spans])),
            "multi_unit_span_count": int(len([item for item in spans if int(item.get("shared_unit_count", 0) or 0) > 1])),
        },
        "artifact_paths": {
            "prompt_manifest": "prompt/prompt_manifest.json",
            "generation_units": "segment/generation_units.json",
            "prompt_spans": "prompt/prompt_spans.json",
        },
    }
