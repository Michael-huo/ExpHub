from __future__ import annotations

from bisect import bisect_left


def build_legal_grid(
    num_frames,
    fps,
    grid_seconds=1,
    allowed_unit_seconds=(1, 2, 3, 4, 5),
    tail_policy="drop",
):
    fps_i = int(fps)
    grid_seconds_i = int(grid_seconds)
    if num_frames < 0:
        raise ValueError("num_frames must be >= 0, got {}".format(num_frames))
    if fps_i <= 0:
        raise ValueError("fps must be > 0, got {}".format(fps))
    if grid_seconds_i <= 0:
        raise ValueError("grid_seconds must be > 0, got {}".format(grid_seconds))

    allowed_seconds = [int(item) for item in allowed_unit_seconds]
    if any(item <= 0 for item in allowed_seconds):
        raise ValueError("allowed_unit_seconds must contain positive integers")

    grid_step = int(fps_i * grid_seconds_i)
    allowed_delta_indices = [int(fps_i * seconds) for seconds in allowed_seconds]
    legal_positions = list(range(0, int(num_frames), int(grid_step)))

    return {
        "fps": int(fps_i),
        "grid_seconds": int(grid_seconds_i),
        "grid_step": int(grid_step),
        "allowed_unit_seconds": allowed_seconds,
        "allowed_delta_indices": allowed_delta_indices,
        "allowed_num_frames": [int(delta) + 1 for delta in allowed_delta_indices],
        "legal_positions": legal_positions,
        "tail_policy": str(tail_policy),
    }


def is_legal_position(idx, legal_positions):
    return int(idx) in set(int(item) for item in legal_positions)


def snap_to_nearest_legal_position(idx, legal_positions, prefer="nearest"):
    positions = sorted(int(item) for item in legal_positions)
    if not positions:
        raise ValueError("legal_positions is empty")

    value = int(idx)
    pos = bisect_left(positions, value)
    lower = positions[pos - 1] if pos > 0 else None
    higher = positions[pos] if pos < len(positions) else None
    prefer_value = str(prefer or "nearest").strip().lower()

    if prefer_value == "lower":
        return int(lower if lower is not None else positions[0])
    if prefer_value == "higher":
        return int(higher if higher is not None else positions[-1])
    if prefer_value != "nearest":
        raise ValueError("unsupported snap prefer value: {}".format(prefer))

    if lower is None:
        return int(higher)
    if higher is None:
        return int(lower)
    if abs(value - lower) <= abs(higher - value):
        return int(lower)
    return int(higher)


def is_legal_delta(delta_idx, allowed_delta_indices):
    return int(delta_idx) in set(int(item) for item in allowed_delta_indices)
