from __future__ import annotations


DEFAULT_MIN_DECODE_FRAMES = 10
DEFAULT_MIN_EXPORT_FRAMES = 12
DEFAULT_MIN_EXECUTABLE_FRAMES = max(DEFAULT_MIN_DECODE_FRAMES, DEFAULT_MIN_EXPORT_FRAMES)


def unit_duration_frames(start_idx, end_idx):
    return int(max(0, int(end_idx) - int(start_idx) + 1))


def is_valid_for_decode(start_idx, end_idx):
    return bool(unit_duration_frames(start_idx, end_idx) >= int(DEFAULT_MIN_DECODE_FRAMES))


def is_valid_for_export(start_idx, end_idx):
    return bool(unit_duration_frames(start_idx, end_idx) >= int(DEFAULT_MIN_EXPORT_FRAMES))


def enforce_contiguous_shared_anchors(units, sequence_start_idx, sequence_end_idx):
    items = list(units or [])
    if not items:
        raise RuntimeError("generation unit planner produced zero units")
    if int(items[0].get("anchor_start_idx", -1)) != int(sequence_start_idx):
        raise RuntimeError("generation units must start at the sequence start")
    if int(items[-1].get("anchor_end_idx", -1)) != int(sequence_end_idx):
        raise RuntimeError("generation units must end at the sequence end")
    for idx in range(1, len(items)):
        prev_end = int(items[idx - 1].get("anchor_end_idx", -1))
        current_start = int(items[idx].get("anchor_start_idx", -2))
        if prev_end != current_start:
            raise RuntimeError(
                "generation units must use shared anchors: prev_end={} current_start={}".format(prev_end, current_start)
            )
    for item in items:
        start_idx = int(item.get("anchor_start_idx", -1))
        end_idx = int(item.get("anchor_end_idx", -1))
        if end_idx < start_idx:
            raise RuntimeError("generation unit has invalid anchor range: start={} end={}".format(start_idx, end_idx))
