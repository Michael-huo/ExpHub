from __future__ import annotations

from exphub.common.io import write_json_atomic


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _time_at(prepare_result, idx):
    values = list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])
    if idx < 0 or idx >= len(values):
        raise RuntimeError("prepare_result frame_index_map missing abs time for frame {}".format(int(idx)))
    return float(values[int(idx)])


def _motion_segment_map(motion_segments):
    return {str(item.get("seg_id", "") or ""): _as_dict(item) for item in list(_as_dict(motion_segments).get("segments") or [])}


def _states_by_segment(semantic_anchors):
    out = {}
    for raw_group in list(_as_dict(semantic_anchors).get("segments") or []):
        group = _as_dict(raw_group)
        seg_id = str(group.get("seg_id", "") or "")
        states = [_as_dict(item) for item in list(group.get("semantic_states") or [])]
        states.sort(key=lambda item: int(item.get("start_idx", item.get("anchor_idx", -1))))
        out[seg_id] = states
    return out


def _anchors_by_segment(semantic_anchors):
    out = {}
    for raw_group in list(_as_dict(semantic_anchors).get("segments") or []):
        group = _as_dict(raw_group)
        seg_id = str(group.get("seg_id", "") or "")
        anchors = [_as_dict(item) for item in list(group.get("anchors") or group.get("anchor_items") or [])]
        out[seg_id] = {int(item.get("frame_idx", -1)): item for item in anchors if item.get("frame_idx") is not None}
    return out


def _validate_legal(value, legal_set, label):
    if int(value) not in legal_set:
        raise RuntimeError("{} must be a legal position, got {}".format(label, int(value)))


def _legal_between(legal_positions, start_idx, end_idx):
    return [int(idx) for idx in legal_positions if int(start_idx) <= int(idx) <= int(end_idx)]


def _can_cover(start_idx, end_idx, legal_positions, allowed_deltas):
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if start_idx == end_idx:
        return True
    legal = [int(idx) for idx in legal_positions if int(start_idx) <= int(idx) <= int(end_idx)]
    reachable = {start_idx}
    for idx in legal:
        if idx not in reachable:
            continue
        for nxt in legal:
            if nxt <= idx:
                continue
            if int(nxt) - int(idx) in allowed_deltas:
                reachable.add(int(nxt))
    return int(end_idx) in reachable


def _nearest_legal(value, legal_positions, seg_start, seg_end):
    viable = [int(idx) for idx in legal_positions if int(seg_start) < int(idx) < int(seg_end)]
    if not viable:
        return None
    value = int(value)
    return min(viable, key=lambda idx: (abs(int(idx) - value), int(idx)))


def _semantic_cut_candidates(states, legal_positions, legal_set, allowed_deltas, seg_start, seg_end, diagnostics):
    grid_step = min(
        [int(b) - int(a) for a, b in zip(legal_positions[:-1], legal_positions[1:]) if int(b) > int(a)] or [1]
    )
    cuts = set()
    for state in states:
        start_idx = int(state.get("start_idx", state.get("anchor_idx", seg_start)))
        if start_idx <= int(seg_start) or start_idx >= int(seg_end):
            continue
        chosen = None
        if start_idx in legal_set:
            chosen = int(start_idx)
        else:
            snapped = _nearest_legal(start_idx, legal_positions, seg_start, seg_end)
            if snapped is not None and abs(int(snapped) - int(start_idx)) <= max(1, int(grid_step) // 2):
                chosen = int(snapped)
                diagnostics.append(
                    {
                        "type": "semantic_boundary_snapped",
                        "semantic_state_id": str(state.get("semantic_state_id", "") or ""),
                        "requested_idx": int(start_idx),
                        "cut_idx": int(chosen),
                    }
                )
        if chosen is None:
            diagnostics.append(
                {
                    "type": "semantic_boundary_ignored_not_legal",
                    "semantic_state_id": str(state.get("semantic_state_id", "") or ""),
                    "requested_idx": int(start_idx),
                }
            )
            continue
        if not _can_cover(seg_start, chosen, legal_positions, allowed_deltas) or not _can_cover(
            chosen,
            seg_end,
            legal_positions,
            allowed_deltas,
        ):
            diagnostics.append(
                {
                    "type": "semantic_boundary_ignored_illegal_span",
                    "semantic_state_id": str(state.get("semantic_state_id", "") or ""),
                    "requested_idx": int(start_idx),
                    "cut_idx": int(chosen),
                }
            )
            continue
        cuts.add(int(chosen))
    return cuts


def _build_cut_points(seg_start, seg_end, legal_positions, allowed_deltas, semantic_cuts):
    cuts = [int(seg_start)]
    current = int(seg_start)
    while current < int(seg_end):
        allowed_next = [
            int(idx)
            for idx in legal_positions
            if int(current) < int(idx) <= int(seg_end) and int(idx) - int(current) in allowed_deltas
        ]
        allowed_next = [idx for idx in allowed_next if idx == int(seg_end) or _can_cover(idx, seg_end, legal_positions, allowed_deltas)]
        if not allowed_next:
            raise RuntimeError("cannot build legal generation unit span from {} to {}".format(current, seg_end))
        semantic_next = [idx for idx in allowed_next if int(idx) in semantic_cuts]
        chosen = min(semantic_next) if semantic_next else max(allowed_next)
        cuts.append(int(chosen))
        current = int(chosen)
    return cuts


def _overlap_len(a_start, a_end, b_start, b_end):
    left = max(int(a_start), int(b_start))
    right = min(int(a_end), int(b_end))
    if right < left:
        return 0
    return int(right - left + 1)


def _assign_state(unit_start, unit_end, states):
    if not states:
        raise RuntimeError("generation unit cannot be assigned without semantic states")
    scored = []
    unit_len = int(unit_end) - int(unit_start) + 1
    for state in states:
        overlap = _overlap_len(unit_start, unit_end, state.get("start_idx"), state.get("end_idx"))
        scored.append(
            (
                int(overlap),
                -abs(int(state.get("anchor_idx", state.get("start_idx"))) - int(unit_start)),
                str(state.get("semantic_state_id", "") or ""),
                state,
            )
        )
    scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
    best = scored[0][3]
    best_overlap = int(scored[0][0])
    return best, float(best_overlap) / float(max(1, unit_len))


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
    states_by_seg = _states_by_segment(semantic_anchors)
    anchors_by_seg = _anchors_by_segment(semantic_anchors)
    units = []
    diagnostics = []
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

        segment_legal = _legal_between(legal_positions, seg_start, seg_end)
        states = list(states_by_seg.get(seg_id) or [])
        if not states:
            raise RuntimeError("semantic states missing for motion segment {}".format(seg_id))
        if int(states[0].get("start_idx", states[0].get("anchor_idx", -1))) != int(seg_start):
            raise RuntimeError("first semantic state must start at motion segment start: {}".format(seg_id))

        semantic_cuts = _semantic_cut_candidates(states, segment_legal, legal_set, allowed_deltas, seg_start, seg_end, diagnostics)
        cut_points = _build_cut_points(seg_start, seg_end, segment_legal, allowed_deltas, semantic_cuts)
        anchor_lookup = dict(anchors_by_seg.get(seg_id) or {})
        for start_idx, end_idx in zip(cut_points[:-1], cut_points[1:]):
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
            state, overlap_ratio = _assign_state(start_idx, end_idx, states)
            semantic_state_id = str(state.get("semantic_state_id", "") or "")
            if overlap_ratio < 1.0:
                diagnostics.append(
                    {
                        "type": "unit_crosses_semantic_state_boundary",
                        "unit_start_idx": int(start_idx),
                        "unit_end_idx": int(end_idx),
                        "assigned_semantic_state_id": str(semantic_state_id),
                        "overlap_ratio": float(overlap_ratio),
                    }
                )
            unit_id = "unit_{:04d}".format(int(len(units)))
            start_anchor = _as_dict(anchor_lookup.get(int(start_idx)))
            end_anchor = _as_dict(anchor_lookup.get(int(end_idx)))
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
                    "semantic_state_id": str(semantic_state_id),
                    "semantic_anchor_idx": int(state.get("anchor_idx", state.get("start_idx"))),
                    "semantic_state_anchor_idx": int(state.get("anchor_idx", state.get("start_idx"))),
                    "semantic_state_overlap_ratio": float(overlap_ratio),
                    "anchor_span": {
                        "start_anchor": int(start_idx),
                        "end_anchor": int(end_idx),
                        "start_reason": str(start_anchor.get("reason", "legal_grid") or "legal_grid"),
                        "end_reason": str(end_anchor.get("reason", "legal_grid") or "legal_grid"),
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
        "version": 2,
        "source": "encode.generation_unit.v2",
        "sequence_range": {
            "start_idx": int(units[0]["start_idx"]),
            "end_idx": int(units[-1]["end_idx"]),
        },
        "units": units,
        "diagnostics": diagnostics,
        "summary": {
            "unit_count": int(len(units)),
            "decode_valid_unit_count": int(len([item for item in units if item.get("is_valid_for_decode")])),
            "motion_segment_count": int(len(list(_as_dict(motion_segments).get("segments") or []))),
            "semantic_state_count": int(sum(len(items) for items in states_by_seg.values())),
            "shared_anchor_count": int(max(0, len(units) - 1)),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
