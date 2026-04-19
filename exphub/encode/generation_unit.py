from __future__ import annotations

from pathlib import Path

from exphub.common.io import write_json_atomic


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _time_at(prepare_result, idx):
    values = list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])
    if idx < 0 or idx >= len(values):
        raise RuntimeError("prepare_result frame_index_map missing abs time for frame {}".format(int(idx)))
    return float(values[int(idx)])


def _anchor_map(semantic_anchors):
    out = {}
    for raw_group in list(_as_dict(semantic_anchors).get("segments") or []):
        group = _as_dict(raw_group)
        out[str(group.get("seg_id", "") or "")] = [_as_dict(item) for item in list(group.get("anchor_items") or [])]
    return out


def _motion_segment_map(motion_segments):
    return {str(item.get("seg_id", "") or ""): _as_dict(item) for item in list(_as_dict(motion_segments).get("segments") or [])}


def _validate_legal(value, legal_set, label):
    if int(value) not in legal_set:
        raise RuntimeError("{} must be a legal position, got {}".format(label, int(value)))


def build_generation_units(prepare_result, motion_segments, semantic_anchors, out_path=None):
    prepare = _as_dict(prepare_result)
    legal_grid = _as_dict(prepare.get("legal_grid"))
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    legal_set = set(legal_positions)
    allowed_deltas = set(int(item) for item in list(legal_grid.get("allowed_delta_indices") or []))
    allowed_num_frames = set(int(item) for item in list(legal_grid.get("allowed_num_frames") or []))
    if not legal_positions or not allowed_deltas or not allowed_num_frames:
        raise RuntimeError("prepare_result legal_grid missing legal unit constraints")

    segment_by_id = _motion_segment_map(motion_segments)
    anchors_by_seg = _anchor_map(semantic_anchors)
    units = []
    previous_end = None
    for seg_index, raw_segment in enumerate(list(_as_dict(motion_segments).get("segments") or [])):
        segment = _as_dict(raw_segment)
        seg_id = str(segment.get("seg_id", "") or "")
        seg_start = int(segment.get("start_idx"))
        seg_end = int(segment.get("end_idx"))
        _validate_legal(seg_start, legal_set, "motion segment start")
        _validate_legal(seg_end, legal_set, "motion segment end")
        if previous_end is not None and int(seg_start) != int(previous_end):
            raise RuntimeError(
                "motion segments must join with shared endpoints: prev_end={} current_start={}".format(
                    int(previous_end),
                    int(seg_start),
                )
            )

        anchor_items = list(anchors_by_seg.get(seg_id) or [])
        if not anchor_items:
            raise RuntimeError("semantic anchors missing for motion segment {}".format(seg_id))
        anchor_items.sort(key=lambda item: int(item.get("frame_idx", -1)))
        anchor_indices = [int(item.get("frame_idx")) for item in anchor_items]
        if anchor_indices[0] != seg_start or anchor_indices[-1] != seg_end:
            raise RuntimeError(
                "semantic anchors must cover motion segment {}: anchors={} segment={}-{}".format(
                    seg_id,
                    anchor_indices,
                    seg_start,
                    seg_end,
                )
            )

        for idx in anchor_indices:
            _validate_legal(idx, legal_set, "semantic anchor")
            if idx < seg_start or idx > seg_end:
                raise RuntimeError("semantic anchor crosses motion segment {}: {}".format(seg_id, idx))

        for local_idx, (start_idx, end_idx) in enumerate(zip(anchor_indices[:-1], anchor_indices[1:])):
            delta = int(end_idx) - int(start_idx)
            length = int(delta + 1)
            if delta not in allowed_deltas or length not in allowed_num_frames:
                raise RuntimeError(
                    "generation unit span is not allowed: seg_id={} start={} end={} delta={} length={}".format(
                        seg_id,
                        start_idx,
                        end_idx,
                        delta,
                        length,
                    )
                )
            if start_idx < seg_start or end_idx > seg_end:
                raise RuntimeError("generation unit crosses motion segment {}".format(seg_id))
            unit_id = "unit_{:04d}".format(int(len(units)))
            units.append(
                {
                    "unit_id": str(unit_id),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "anchor_start_idx": int(start_idx),
                    "anchor_end_idx": int(end_idx),
                    "length": int(length),
                    "duration_frames": int(length),
                    "duration_sec": float(delta) / float(max(int(prepare.get("target_fps", legal_grid.get("fps", 1)) or 1), 1)),
                    "start_abs_time_sec": _time_at(prepare, int(start_idx)),
                    "end_abs_time_sec": _time_at(prepare, int(end_idx)),
                    "motion_label": str(segment.get("motion_label", "") or "mixed"),
                    "seg_id": str(seg_id),
                    "motion_segment_index": int(seg_index),
                    "anchor_span": {
                        "start_anchor": int(start_idx),
                        "end_anchor": int(end_idx),
                        "start_reason": str(anchor_items[local_idx].get("reason", "") or ""),
                        "end_reason": str(anchor_items[local_idx + 1].get("reason", "") or ""),
                    },
                    "prompt_ref": {
                        "artifact_path": "encode/prompts.json",
                        "unit_id": str(unit_id),
                    },
                    "scene_label": "motion_segment_{:04d}".format(int(seg_index)),
                    "is_valid_for_decode": True,
                    "source_segment_ids": [int(seg_index)],
                }
            )
        previous_end = int(seg_end)

    if not units:
        raise RuntimeError("generation unit builder produced zero units")
    if units[0]["start_idx"] != legal_positions[0] or units[-1]["end_idx"] != legal_positions[-1]:
        raise RuntimeError("generation units must cover the full prepared legal interval")
    for idx in range(1, len(units)):
        if int(units[idx - 1]["end_idx"]) != int(units[idx]["start_idx"]):
            raise RuntimeError(
                "generation units must use shared endpoints: prev_end={} current_start={}".format(
                    units[idx - 1]["end_idx"],
                    units[idx]["start_idx"],
                )
            )
        if str(units[idx - 1]["seg_id"]) != str(units[idx]["seg_id"]):
            prev_seg = segment_by_id[str(units[idx - 1]["seg_id"])]
            if int(units[idx - 1]["end_idx"]) != int(prev_seg.get("end_idx")):
                raise RuntimeError("generation unit segment transition does not land on motion segment boundary")

    payload = {
        "version": 1,
        "source": "encode.generation_unit.v1",
        "sequence_range": {
            "start_idx": int(units[0]["start_idx"]),
            "end_idx": int(units[-1]["end_idx"]),
        },
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "decode_valid_unit_count": int(len([item for item in units if item.get("is_valid_for_decode")])),
            "motion_segment_count": int(len(list(_as_dict(motion_segments).get("segments") or []))),
            "shared_anchor_count": int(max(0, len(units) - 1)),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
