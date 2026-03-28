#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _common import log_info

from _segment.signal_extraction import (
    extract_signal_timeseries_from_frames,
)
from _segment.state_segmentation import (
    STATE_HIGH,
    STATE_LOW,
    compute_state_segments,
    save_state_segmentation_plots,
    write_state_segmentation_outputs,
)


STATE_POLICY_NAME = "state"
DEFAULT_HIGH_GAP = 24
DEFAULT_TRANSITION_GAP = 24
DEFAULT_PRE_TRANSITION_FRAMES = 24
DEFAULT_POST_TRANSITION_FRAMES = 12
DEFAULT_MIN_ANCHOR_SPACING = 12
DEFAULT_MIN_SEGMENT_FRAMES = 16
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_WEIGHTS = {
    "motion_velocity": 0.75,
    "semantic_velocity": 0.25,
}
DEFAULT_ENTER_TH = 0.65
DEFAULT_EXIT_TH = 0.45
DEFAULT_MIN_HIGH_LEN = 24
DEFAULT_MIN_LOW_LEN = 24
DEFAULT_GLITCH_MERGE_LEN = 12

ZONE_LOW = "low"
ZONE_TRANSITION = "transition"
ZONE_HIGH = "high"

_ZONE_PRIORITY = {
    ZONE_LOW: 0,
    ZONE_TRANSITION: 1,
    ZONE_HIGH: 2,
}

def _build_state_items(build_item, final_indices, density_rows, safe_base_indices):
    safe_base_set = set(int(idx) for idx in list(safe_base_indices or []))
    inserted_count = 0
    high_anchor_count = 0
    transition_anchor_count = 0
    low_anchor_count = 0
    items = []

    for frame_idx in list(final_indices or []):
        density_row = density_rows[int(frame_idx)] if density_rows else {}
        schedule_zone = str(density_row.get("schedule_zone", ZONE_LOW))
        state_label = str(density_row.get("state_label", STATE_LOW))
        if schedule_zone == ZONE_HIGH:
            high_anchor_count += 1
        elif schedule_zone == ZONE_TRANSITION:
            transition_anchor_count += 1
        else:
            low_anchor_count += 1

        if frame_idx in safe_base_set and schedule_zone == ZONE_LOW and state_label == STATE_LOW:
            items.append(
                build_item(
                    frame_idx,
                    source_type="uniform",
                    source_role="uniform",
                    legacy_meta={
                        "candidate_role": "uniform",
                        "promotion_source": "uniform",
                    },
                )
            )
            continue

        source_role = "{}_zone".format(str(schedule_zone))
        is_inserted = bool(frame_idx not in safe_base_set)
        if is_inserted:
            inserted_count += 1
        items.append(
            build_item(
                frame_idx,
                source_type=STATE_POLICY_NAME,
                source_role=source_role,
                legacy_meta={
                    "candidate_role": source_role,
                    "is_inserted": is_inserted,
                    "promotion_source": "state_schedule",
                    "promotion_reason": "global_density_scan_anchor",
                    "window_id": int(frame_idx),
                },
            )
        )

    return items, {
        "inserted_count": int(inserted_count),
        "high_anchor_count": int(high_anchor_count),
        "transition_anchor_count": int(transition_anchor_count),
        "low_anchor_count": int(low_anchor_count),
    }


def _segment_ranges(segments):
    rows = []
    for segment in list(segments or []):
        segment_id = int(segment.get("segment_id", 0) or 0)
        rows.append(
            {
                "segment_id": int(segment_id),
                "state_label": str(segment.get("state_label", STATE_LOW)),
                "start_frame": int(segment.get("start_frame", 0) or 0),
                "end_frame": int(segment.get("end_frame", 0) or 0),
                "duration_frames": int(segment.get("duration_frames", 0) or 0),
            }
        )
    return rows


def _zone_priority(zone_name):
    return int(_ZONE_PRIORITY.get(str(zone_name), 0))


def _target_gap_for_zone(zone_name, safe_gap, transition_gap, high_gap):
    zone_name = str(zone_name)
    if zone_name == ZONE_HIGH:
        return int(high_gap)
    if zone_name == ZONE_TRANSITION:
        return int(transition_gap)
    return int(safe_gap)


def _build_density_schedule_rows(
    frame_rows,
    segments,
    safe_gap,
    transition_gap,
    high_gap,
    pre_transition_frames,
    post_transition_frames,
):
    rows = []
    row_map = {}
    for row in list(frame_rows or []):
        state_label = str(row.get("state_label", STATE_LOW))
        schedule_zone = ZONE_HIGH if state_label == STATE_HIGH else ZONE_LOW
        item = {
            "frame_idx": int(row.get("frame_idx", 0) or 0),
            "state_label": state_label,
            "state_score": float(row.get("state_score", 0.0) or 0.0),
            "schedule_zone": str(schedule_zone),
            "zone_priority": int(_zone_priority(schedule_zone)),
        }
        rows.append(item)
        row_map[int(item["frame_idx"])] = item

    if not rows:
        return []

    frame_idx_last = int(rows[-1]["frame_idx"])
    for segment in list(segments or []):
        if str(segment.get("state_label", STATE_LOW)) != STATE_HIGH:
            continue
        start_frame = int(segment.get("start_frame", 0) or 0)
        end_frame = int(segment.get("end_frame", 0) or 0)
        transition_start = max(0, int(start_frame - int(pre_transition_frames)))
        transition_end = min(frame_idx_last, int(end_frame + int(post_transition_frames)))

        for frame_idx in range(transition_start, start_frame):
            item = row_map.get(int(frame_idx))
            if item is None:
                continue
            if _zone_priority(ZONE_TRANSITION) > int(item.get("zone_priority", 0) or 0):
                item["schedule_zone"] = ZONE_TRANSITION
                item["zone_priority"] = int(_zone_priority(ZONE_TRANSITION))

        for frame_idx in range(end_frame + 1, transition_end + 1):
            item = row_map.get(int(frame_idx))
            if item is None:
                continue
            if _zone_priority(ZONE_TRANSITION) > int(item.get("zone_priority", 0) or 0):
                item["schedule_zone"] = ZONE_TRANSITION
                item["zone_priority"] = int(_zone_priority(ZONE_TRANSITION))

    for item in rows:
        item["target_gap"] = int(
            _target_gap_for_zone(
                item.get("schedule_zone", ZONE_LOW),
                safe_gap=safe_gap,
                transition_gap=transition_gap,
                high_gap=high_gap,
            )
        )
    return rows


def _density_schedule_runs(density_rows):
    runs = []
    if not density_rows:
        return runs

    start_pos = 0
    current_zone = str(density_rows[0].get("schedule_zone", ZONE_LOW))
    current_gap = int(density_rows[0].get("target_gap", 0) or 0)
    for pos in range(1, len(density_rows)):
        zone_name = str(density_rows[pos].get("schedule_zone", ZONE_LOW))
        gap_value = int(density_rows[pos].get("target_gap", 0) or 0)
        if zone_name == current_zone and gap_value == current_gap:
            continue
        runs.append(
            {
                "run_id": int(len(runs)),
                "schedule_zone": str(current_zone),
                "target_gap": int(current_gap),
                "start_frame": int(density_rows[start_pos].get("frame_idx", 0) or 0),
                "end_frame": int(density_rows[pos - 1].get("frame_idx", 0) or 0),
                "duration_frames": int(pos - start_pos),
            }
        )
        start_pos = pos
        current_zone = str(zone_name)
        current_gap = int(gap_value)
    runs.append(
        {
            "run_id": int(len(runs)),
            "schedule_zone": str(current_zone),
            "target_gap": int(current_gap),
            "start_frame": int(density_rows[start_pos].get("frame_idx", 0) or 0),
            "end_frame": int(density_rows[-1].get("frame_idx", 0) or 0),
            "duration_frames": int(len(density_rows) - start_pos),
        }
    )
    return runs


def _scan_density_schedule_anchors(density_rows):
    if not density_rows:
        return []

    anchors = [int(density_rows[0].get("frame_idx", 0) or 0)]
    current_frame = int(anchors[0])
    last_frame = int(density_rows[-1].get("frame_idx", 0) or 0)

    while current_frame < last_frame:
        current_gap = int(density_rows[current_frame].get("target_gap", 1) or 1)
        next_frame = min(last_frame, int(current_frame + max(1, current_gap)))
        if next_frame <= current_frame:
            next_frame = min(last_frame, int(current_frame + 1))
        anchors.append(int(next_frame))
        current_frame = int(next_frame)

    if anchors[-1] != int(last_frame):
        anchors.append(int(last_frame))
    return sorted(set(int(value) for value in anchors))


def _segment_zone_profile(start_frame, end_frame, density_rows):
    counts = {
        ZONE_LOW: 0,
        ZONE_TRANSITION: 0,
        ZONE_HIGH: 0,
    }
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        return {
            "primary_zone": ZONE_LOW,
            "counts": counts,
        }

    for frame_idx in range(start_frame, end_frame + 1):
        zone_name = str(density_rows[frame_idx].get("schedule_zone", ZONE_LOW))
        counts[zone_name] = int(counts.get(zone_name, 0)) + 1

    primary_zone = ZONE_LOW
    best_key = None
    for zone_name in (ZONE_LOW, ZONE_TRANSITION, ZONE_HIGH):
        zone_rank_key = (
            int(counts.get(zone_name, 0)),
            int(_zone_priority(zone_name)),
        )
        if best_key is None or zone_rank_key > best_key:
            best_key = zone_rank_key
            primary_zone = str(zone_name)
    return {
        "primary_zone": str(primary_zone),
        "counts": counts,
    }


def _anchor_spacing_meta(left_frame, right_frame, density_rows):
    left_zone = str(density_rows[int(left_frame)].get("schedule_zone", ZONE_LOW))
    right_zone = str(density_rows[int(right_frame)].get("schedule_zone", ZONE_LOW))
    return {
        "left_frame": int(left_frame),
        "right_frame": int(right_frame),
        "left_zone": str(left_zone),
        "right_zone": str(right_zone),
        "left_priority": int(_zone_priority(left_zone)),
        "right_priority": int(_zone_priority(right_zone)),
        "gap": int(int(right_frame) - int(left_frame)),
    }


def _enforce_min_anchor_spacing(anchors, density_rows, min_anchor_spacing):
    anchors = list(sorted(set(int(value) for value in list(anchors or []))))
    merge_rules = []
    frame_last = int(density_rows[-1].get("frame_idx", 0) or 0) if density_rows else 0

    while len(anchors) >= 2:
        target_idx = None
        for idx in range(len(anchors) - 1):
            if int(anchors[idx + 1] - anchors[idx]) < int(min_anchor_spacing):
                target_idx = int(idx)
                break
        if target_idx is None:
            break

        left_frame = int(anchors[target_idx])
        right_frame = int(anchors[target_idx + 1])
        record = _anchor_spacing_meta(left_frame, right_frame, density_rows)

        if left_frame <= 0:
            remove_pos = int(target_idx + 1)
            keep_frame = int(left_frame)
            remove_reason = "preserve_first_anchor"
        elif right_frame >= frame_last:
            remove_pos = int(target_idx)
            keep_frame = int(right_frame)
            remove_reason = "preserve_last_anchor"
        elif int(record["left_priority"]) > int(record["right_priority"]):
            remove_pos = int(target_idx + 1)
            keep_frame = int(left_frame)
            remove_reason = "keep_higher_density_left"
        else:
            remove_pos = int(target_idx)
            keep_frame = int(right_frame)
            remove_reason = "keep_higher_or_equal_density_right"

        removed_frame = int(anchors.pop(remove_pos))
        record["kept_frame"] = int(keep_frame)
        record["removed_frame"] = int(removed_frame)
        record["resolution"] = str(remove_reason)
        merge_rules.append(record)

    return anchors, {
        "minimum_spacing": int(min_anchor_spacing),
        "merge_count": int(len(merge_rules)),
        "tie_break_rule": "keep higher density zone anchor; when equal priority keep later anchor; always preserve first/last anchor",
        "applied_rules": merge_rules,
    }


def _anchor_segment_records(anchors, density_rows):
    records = []
    for idx in range(len(anchors) - 1):
        start_frame = int(anchors[idx])
        end_frame = int(anchors[idx + 1])
        zone_profile = _segment_zone_profile(start_frame, end_frame, density_rows)
        records.append(
            {
                "segment_idx": int(idx),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "duration_frames": int(end_frame - start_frame),
                "primary_zone": str(zone_profile["primary_zone"]),
                "zone_counts": dict(zone_profile["counts"]),
            }
        )
    return records


def _merge_short_segments(anchors, density_rows, min_segment_frames):
    anchors = list(sorted(set(int(value) for value in list(anchors or []))))
    merge_rules = []
    while len(anchors) > 2:
        segment_records = _anchor_segment_records(anchors, density_rows)
        short_record = None
        for record in segment_records:
            if int(record["duration_frames"]) < int(min_segment_frames):
                short_record = record
                break
        if short_record is None:
            break

        seg_idx = int(short_record["segment_idx"])
        if seg_idx <= 0:
            remove_pos = int(seg_idx + 1)
            merge_side = "next"
            remove_reason = "edge_merge_into_next_only_neighbor"
        elif seg_idx >= len(segment_records) - 1:
            remove_pos = int(seg_idx)
            merge_side = "previous"
            remove_reason = "edge_merge_into_previous_only_neighbor"
        else:
            prev_record = segment_records[seg_idx - 1]
            next_record = segment_records[seg_idx + 1]
            prev_zone = str(prev_record["primary_zone"])
            curr_zone = str(short_record["primary_zone"])
            next_zone = str(next_record["primary_zone"])
            prev_len = int(prev_record["duration_frames"])
            next_len = int(next_record["duration_frames"])

            if prev_zone == curr_zone and next_zone != curr_zone:
                remove_pos = int(seg_idx)
                merge_side = "previous"
                remove_reason = "merge_into_same_zone_previous"
            elif next_zone == curr_zone and prev_zone != curr_zone:
                remove_pos = int(seg_idx + 1)
                merge_side = "next"
                remove_reason = "merge_into_same_zone_next"
            elif prev_zone == curr_zone and next_zone == curr_zone:
                if prev_len >= next_len:
                    remove_pos = int(seg_idx)
                    merge_side = "previous"
                    remove_reason = "both_neighbors_same_zone_keep_previous_on_longer_or_tie"
                else:
                    remove_pos = int(seg_idx + 1)
                    merge_side = "next"
                    remove_reason = "both_neighbors_same_zone_keep_next_on_longer"
            elif prev_len >= next_len:
                remove_pos = int(seg_idx)
                merge_side = "previous"
                remove_reason = "merge_into_longer_previous_or_tie"
            else:
                remove_pos = int(seg_idx + 1)
                merge_side = "next"
                remove_reason = "merge_into_longer_next"

        removed_anchor = int(anchors.pop(remove_pos))
        merge_rules.append(
            {
                "short_segment_idx": int(seg_idx),
                "short_segment_start": int(short_record["start_frame"]),
                "short_segment_end": int(short_record["end_frame"]),
                "short_segment_frames": int(short_record["duration_frames"]),
                "short_segment_zone": str(short_record["primary_zone"]),
                "removed_anchor": int(removed_anchor),
                "merge_side": str(merge_side),
                "resolution": str(remove_reason),
            }
        )

    final_segment_records = _anchor_segment_records(anchors, density_rows)
    min_final_segment_frames = 0
    if final_segment_records:
        min_final_segment_frames = min(int(item["duration_frames"]) for item in final_segment_records)
    return anchors, {
        "minimum_segment_frames": int(min_segment_frames),
        "merge_count": int(len(merge_rules)),
        "tie_break_rule": "prefer same-zone neighbor; otherwise merge into longer neighbor; if tie merge into previous",
        "applied_rules": merge_rules,
        "min_final_segment_frames": int(min_final_segment_frames),
    }


def _schedule_rows_with_anchor_counts(schedule_runs, final_indices):
    rows = []
    for run in list(schedule_runs or []):
        start_frame = int(run.get("start_frame", 0) or 0)
        end_frame = int(run.get("end_frame", 0) or 0)
        anchor_count = 0
        for frame_idx in list(final_indices or []):
            if start_frame <= int(frame_idx) <= end_frame:
                anchor_count += 1
        rows.append(
            {
                "run_id": int(run.get("run_id", 0) or 0),
                "schedule_zone": str(run.get("schedule_zone", ZONE_LOW)),
                "frame_range": [int(start_frame), int(end_frame)],
                "target_gap": int(run.get("target_gap", 0) or 0),
                "duration_frames": int(run.get("duration_frames", 0) or 0),
                "anchor_count": int(anchor_count),
            }
        )
    return rows


def _transition_band_count(schedule_runs):
    count = 0
    for run in list(schedule_runs or []):
        if str(run.get("schedule_zone", ZONE_LOW)) == ZONE_TRANSITION:
            count += 1
    return int(count)


def _min_gap(final_indices):
    if len(final_indices) < 2:
        return 0
    return min(int(final_indices[idx + 1] - final_indices[idx]) for idx in range(len(final_indices) - 1))


def build_policy_plan(context):
    safe_gap = max(1, int(context["kf_gap"]))
    transition_gap = int(DEFAULT_TRANSITION_GAP)
    high_gap = int(DEFAULT_HIGH_GAP)
    pre_transition_frames = int(DEFAULT_PRE_TRANSITION_FRAMES)
    post_transition_frames = int(DEFAULT_POST_TRANSITION_FRAMES)
    min_anchor_spacing = int(DEFAULT_MIN_ANCHOR_SPACING)
    min_segment_frames = int(DEFAULT_MIN_SEGMENT_FRAMES)
    frame_count_used = int(context["frame_count_used"])
    used_last_idx = int(context["used_last_idx"])
    safe_base_indices = list(context["uniform_base_indices"])

    if frame_count_used <= 0 or not safe_base_indices:
        return {
            "frame_count_used": int(frame_count_used),
            "tail_drop": int(context["tail_drop"]),
            "uniform_base_indices": list(safe_base_indices),
            "keyframe_indices": [],
            "keyframe_items": [],
            "summary": {
                "policy_name": STATE_POLICY_NAME,
                "num_uniform_base": 0,
                "num_final_keyframes": 0,
                "extra_kf_ratio": 0.0,
            },
            "policy_meta": {
                "policy_name": STATE_POLICY_NAME,
                "safe_gap": int(safe_gap),
                "transition_gap": int(transition_gap),
                "high_gap": int(high_gap),
                "pre_transition_frames": int(pre_transition_frames),
                "post_transition_frames": int(post_transition_frames),
                "min_anchor_spacing": int(min_anchor_spacing),
                "min_segment_frames": int(min_segment_frames),
                "state_segment_count": 0,
                "high_state_count": 0,
                "low_state_count": 0,
                "transition_band_count": 0,
                "final_keyframe_count": 0,
                "min_final_gap": 0,
                "min_final_segment_frames": 0,
                "short_segment_merge_count": 0,
                "state_frame_ranges": [],
                "density_schedule_summary": [],
            },
        }

    exp_dir = context["root_dir"].parent
    segment_dir = context["root_dir"]
    state_output_dir = segment_dir / "state_segmentation"
    used_frame_paths = list(context["frame_paths"][:frame_count_used])
    used_timestamps = list(context["timestamps"][:frame_count_used])

    log_info(
        "state policy formal inputs start: frames={} used={} signals=motion_velocity,semantic_velocity".format(
            int(context["frame_count_total"]),
            int(frame_count_used),
        )
    )
    signal_payload = extract_signal_timeseries_from_frames(
        frame_paths=used_frame_paths,
        timestamps=used_timestamps,
        exp_dir=exp_dir,
        segment_dir=segment_dir,
        keyframes_meta={"policy_name": STATE_POLICY_NAME},
        output_dir=context["policy_cache_dir"] / "formal_state_inputs",
        cache_dir=context["policy_cache_dir"],
    )

    state_result = compute_state_segments(
        rows=signal_payload["rows"],
        exp_dir=exp_dir,
        input_csv=None,
        output_dir=state_output_dir,
        smoothing_window=DEFAULT_SMOOTHING_WINDOW,
        enter_th=DEFAULT_ENTER_TH,
        exit_th=DEFAULT_EXIT_TH,
        min_high_len=DEFAULT_MIN_HIGH_LEN,
        min_low_len=DEFAULT_MIN_LOW_LEN,
        glitch_merge_len=DEFAULT_GLITCH_MERGE_LEN,
        weights=dict(DEFAULT_WEIGHTS),
    )

    density_rows = _build_density_schedule_rows(
        frame_rows=state_result["frame_rows"],
        segments=state_result["segments"],
        safe_gap=safe_gap,
        transition_gap=transition_gap,
        high_gap=high_gap,
        pre_transition_frames=pre_transition_frames,
        post_transition_frames=post_transition_frames,
    )
    schedule_runs = _density_schedule_runs(density_rows)
    scanned_indices = _scan_density_schedule_anchors(density_rows)
    scanned_indices, spacing_meta = _enforce_min_anchor_spacing(
        scanned_indices,
        density_rows=density_rows,
        min_anchor_spacing=min_anchor_spacing,
    )
    final_indices, short_segment_meta = _merge_short_segments(
        scanned_indices,
        density_rows=density_rows,
        min_segment_frames=min_segment_frames,
    )
    final_indices = [int(idx) for idx in final_indices if 0 <= int(idx) <= int(used_last_idx)]
    density_schedule_summary = _schedule_rows_with_anchor_counts(schedule_runs, final_indices)
    transition_band_count = _transition_band_count(schedule_runs)
    min_final_gap = int(_min_gap(final_indices))
    min_final_segment_frames = int(short_segment_meta.get("min_final_segment_frames", 0) or 0)

    keyframe_items, allocation_meta = _build_state_items(
        context["build_item"],
        final_indices,
        density_rows,
        safe_base_indices,
    )

    state_frame_ranges = _segment_ranges(state_result["segments"])
    high_state_count = len([row for row in state_frame_ranges if str(row.get("state_label")) == STATE_HIGH])
    low_state_count = len([row for row in state_frame_ranges if str(row.get("state_label")) == STATE_LOW])
    summary = {
        "policy_name": STATE_POLICY_NAME,
        "num_uniform_base": int(len(safe_base_indices)),
        "num_final_keyframes": int(len(final_indices)),
        "inserted_count": int(allocation_meta["inserted_count"]),
        "high_state_anchor_count": int(allocation_meta["high_anchor_count"]),
        "transition_anchor_count": int(allocation_meta["transition_anchor_count"]),
        "low_state_anchor_count": int(allocation_meta["low_anchor_count"]),
        "state_segment_count": int(len(state_frame_ranges)),
        "high_state_count": int(high_state_count),
        "low_state_count": int(low_state_count),
        "transition_band_count": int(transition_band_count),
        "safe_gap": int(safe_gap),
        "transition_gap": int(transition_gap),
        "high_gap": int(high_gap),
        "pre_transition_frames": int(pre_transition_frames),
        "post_transition_frames": int(post_transition_frames),
        "min_anchor_spacing": int(min_anchor_spacing),
        "min_segment_frames": int(min_segment_frames),
        "min_final_gap": int(min_final_gap),
        "min_final_segment_frames": int(min_final_segment_frames),
        "short_segment_merge_count": int(short_segment_meta.get("merge_count", 0) or 0),
        "extra_kf_ratio": float(
            max(0.0, float(len(final_indices) - len(safe_base_indices))) / float(len(safe_base_indices))
        ) if safe_base_indices else 0.0,
    }

    policy_meta = {
        "policy_name": STATE_POLICY_NAME,
        "safe_gap": int(safe_gap),
        "transition_gap": int(transition_gap),
        "high_gap": int(high_gap),
        "pre_transition_frames": int(pre_transition_frames),
        "post_transition_frames": int(post_transition_frames),
        "min_anchor_spacing": int(min_anchor_spacing),
        "min_segment_frames": int(min_segment_frames),
        "state_segment_count": int(len(state_frame_ranges)),
        "high_state_count": int(high_state_count),
        "low_state_count": int(low_state_count),
        "transition_band_count": int(transition_band_count),
        "final_keyframe_count": int(len(final_indices)),
        "min_final_gap": int(min_final_gap),
        "min_final_segment_frames": int(min_final_segment_frames),
        "short_segment_merge_count": int(short_segment_meta.get("merge_count", 0) or 0),
        "state_frame_ranges": state_frame_ranges,
        "density_schedule_summary": density_schedule_summary,
        "scheduling_rules": {
            "scan_strategy": "global_left_to_right_density_scan",
            "spacing_rule": str(spacing_meta.get("tie_break_rule", "")),
            "short_segment_merge_rule": str(short_segment_meta.get("tie_break_rule", "")),
        },
        "spacing_merge_meta": spacing_meta,
        "short_segment_merge_meta": short_segment_meta,
    }

    write_state_segmentation_outputs(state_result)
    save_state_segmentation_plots(
        output_dir=state_result["output_dir"],
        frame_rows=state_result["frame_rows"],
        segments=state_result["segments"],
        enter_th=DEFAULT_ENTER_TH,
        exit_th=DEFAULT_EXIT_TH,
        density_rows=density_rows,
        final_indices=final_indices,
        uniform_indices=safe_base_indices,
        signal_rows=signal_payload["rows"],
    )

    log_info(
        "state policy selected: safe_base={} segments={} transition_bands={} final={} min_gap={} short_merge={}".format(
            int(len(safe_base_indices)),
            int(len(state_frame_ranges)),
            int(transition_band_count),
            int(len(final_indices)),
            int(min_final_gap),
            int(short_segment_meta.get("merge_count", 0) or 0),
        )
    )

    return {
        "frame_count_used": int(frame_count_used),
        "tail_drop": int(context["tail_drop"]),
        "uniform_base_indices": list(safe_base_indices),
        "keyframe_indices": list(final_indices),
        "keyframe_items": keyframe_items,
        "summary": summary,
        "policy_meta": policy_meta,
    }
