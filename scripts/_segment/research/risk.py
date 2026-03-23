#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .kinematics import minmax_normalize, moving_average


RISK_BUNDLE_VERSION = "unified_risk_signals_v1"
DEFAULT_RISK_WEIGHTS = {
    "turn_proxy": 0.45,
    "motion_proxy": 0.35,
    "semantic_proxy": 0.2,
}
DEFAULT_RISK_THRESHOLDS = {
    "medium": 0.4,
    "high": 0.65,
    "window": 0.65,
}
DEFAULT_TURN_PROXY_COMPONENTS = {
    "feature_motion": 0.4,
    "appearance_delta": 0.25,
    "brightness_jump": 0.15,
    "motion_acceleration_smooth": 0.2,
}
DEFAULT_RAW_MERGE_GAP_FRAMES = 2
DEFAULT_EXPANDED_MERGE_GAP_FRAMES = 4
DEFAULT_HARDEST_WINDOW_SELECTION = (
    "peak_score desc -> integrated_score desc -> width_expanded desc -> expanded_start_frame asc"
)


@dataclass
class RiskFrameRow(object):
    frame_idx: int
    timestamp: float
    turn_proxy_raw: float
    turn_proxy: float
    motion_proxy_raw: float
    motion_proxy: float
    semantic_proxy_raw: float
    semantic_proxy: float
    risk_score_raw: float
    risk_score: float
    risk_level: str
    is_uniform_base_kf: bool
    is_final_kf: bool
    local_gap_left: int
    local_gap_right: int
    is_in_raw_risk_window: bool = False
    raw_risk_window_id: Optional[int] = None
    is_in_expanded_risk_window: bool = False
    expanded_risk_window_id: Optional[int] = None


@dataclass
class RiskWindowRegion(object):
    window_id: int
    raw_start_frame: int
    raw_end_frame: int
    expanded_start_frame: int
    expanded_end_frame: int
    raw_start_time: float
    raw_end_time: float
    expanded_start_time: float
    expanded_end_time: float
    peak_frame_idx: int
    peak_timestamp: float
    peak_score: float
    width_raw: int
    width_expanded: int
    integrated_score: float
    window_rank: int


@dataclass
class WindowCoverageSummary(object):
    window_id: int
    window_rank: int
    raw_start_frame: int
    raw_end_frame: int
    expanded_start_frame: int
    expanded_end_frame: int
    raw_start_time: float
    raw_end_time: float
    expanded_start_time: float
    expanded_end_time: float
    peak_frame_idx: int
    peak_timestamp: float
    peak_score: float
    width_raw: int
    width_expanded: int
    integrated_score: float
    uniform_base_count: int
    final_count: int
    prev_uniform_kf_idx: Optional[int]
    next_uniform_kf_idx: Optional[int]
    prev_final_kf_idx: Optional[int]
    next_final_kf_idx: Optional[int]
    prev_uniform_kf_time: Optional[float]
    next_uniform_kf_time: Optional[float]
    prev_final_kf_time: Optional[float]
    next_final_kf_time: Optional[float]
    uniform_span_across_window: Optional[int]
    final_span_across_window: Optional[int]
    uniform_left_distance_to_window: Optional[int]
    uniform_right_distance_to_window: Optional[int]
    final_left_distance_to_window: Optional[int]
    final_right_distance_to_window: Optional[int]


@dataclass
class RiskMetadataConfig(object):
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    smooth_window: int
    expansion_radius_frames: int
    hardest_window_selection: str
    version: str
    raw_merge_gap_frames: int
    expanded_merge_gap_frames: int
    integrated_score_definition: str
    turn_proxy_components: Dict[str, float]
    motion_proxy_source: str
    semantic_proxy_source: str
    local_gap_definition: str
    window_distance_definition: str


@dataclass
class RiskBundle(object):
    frame_rows: List[RiskFrameRow]
    raw_windows: List[RiskWindowRegion]
    expanded_windows: List[RiskWindowRegion]
    window_coverages: List[WindowCoverageSummary]
    hardest_window: Optional[WindowCoverageSummary]
    metadata: RiskMetadataConfig


def _frame_timestamp(row):
    value = row.get("timestamp", row.get("ts_sec", 0.0))
    return float(value or 0.0)


def _series(rows, key):
    values = []
    for row in rows:
        values.append(float(row.get(key, 0.0) or 0.0))
    return values


def _safe_window(window_size):
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    return int(window_size)


def _nearest_left_right(frame_idx, sorted_indices, max_frame_idx):
    if not sorted_indices:
        return int(frame_idx), int(max_frame_idx - frame_idx)
    pos_exact = bisect_left(sorted_indices, int(frame_idx))
    if pos_exact < len(sorted_indices) and int(sorted_indices[pos_exact]) == int(frame_idx):
        return 0, 0
    pos = bisect_right(sorted_indices, int(frame_idx))
    prev_idx = sorted_indices[pos - 1] if pos > 0 else None
    next_idx = sorted_indices[pos] if pos < len(sorted_indices) else None
    if prev_idx is None:
        left_gap = int(frame_idx)
    else:
        left_gap = int(frame_idx) - int(prev_idx)
    if next_idx is None:
        right_gap = int(max_frame_idx - frame_idx)
    else:
        right_gap = int(next_idx) - int(frame_idx)
    return int(max(left_gap, 0)), int(max(right_gap, 0))


def _risk_level(score, thresholds):
    score = float(score)
    if score >= float(thresholds.get("high", 0.65) or 0.65):
        return "high"
    if score >= float(thresholds.get("medium", 0.4) or 0.4):
        return "medium"
    return "low"


def _default_expansion_radius(uniform_keyframe_indices, kf_gap):
    if kf_gap is not None:
        try:
            kf_gap_value = int(kf_gap)
        except Exception:
            kf_gap_value = 0
        if kf_gap_value > 0:
            return max(1, int(kf_gap_value // 2))
    uniform_keyframe_indices = sorted(int(idx) for idx in uniform_keyframe_indices or [])
    if len(uniform_keyframe_indices) >= 2:
        gaps = []
        for idx in range(1, len(uniform_keyframe_indices)):
            gaps.append(int(uniform_keyframe_indices[idx]) - int(uniform_keyframe_indices[idx - 1]))
        if gaps:
            gaps.sort()
            return max(1, int(gaps[len(gaps) // 2] // 2))
    return 12


def _turn_proxy_raw(rows, component_weights):
    component_weights = dict(component_weights or {})
    feature_motion = minmax_normalize(_series(rows, "feature_motion"))
    appearance_delta = minmax_normalize(_series(rows, "appearance_delta"))
    brightness_jump = minmax_normalize(_series(rows, "brightness_jump"))
    motion_acc = minmax_normalize(_series(rows, "motion_acceleration_smooth"))

    values = []
    for idx in range(len(rows)):
        score = 0.0
        score += float(component_weights.get("feature_motion", 0.0) or 0.0) * float(feature_motion[idx])
        score += float(component_weights.get("appearance_delta", 0.0) or 0.0) * float(appearance_delta[idx])
        score += float(component_weights.get("brightness_jump", 0.0) or 0.0) * float(brightness_jump[idx])
        score += float(component_weights.get("motion_acceleration_smooth", 0.0) or 0.0) * float(motion_acc[idx])
        values.append(float(score))
    return values


def _windowed(values, smooth_window):
    smoothed, actual_window = moving_average(values, smooth_window)
    return [float(v) for v in smoothed], int(actual_window)


def _assign_window_membership(frame_rows, raw_windows, expanded_windows):
    for window in raw_windows:
        for idx in range(int(window.raw_start_frame), int(window.raw_end_frame) + 1):
            frame_rows[idx].is_in_raw_risk_window = True
            frame_rows[idx].raw_risk_window_id = int(window.window_id)
    for window in expanded_windows:
        for idx in range(int(window.expanded_start_frame), int(window.expanded_end_frame) + 1):
            frame_rows[idx].is_in_expanded_risk_window = True
            frame_rows[idx].expanded_risk_window_id = int(window.window_id)


def _window_sort_key(window):
    return (
        -float(window.peak_score),
        -float(window.integrated_score),
        -int(window.width_expanded),
        int(window.expanded_start_frame),
    )


def _rank_windows(windows):
    ranked = sorted(list(windows), key=_window_sort_key)
    for rank, window in enumerate(ranked, start=1):
        window.window_rank = int(rank)
    ranked.sort(key=lambda item: int(item.window_id))
    return ranked


def _build_window_region(frame_rows, start_idx, end_idx, expanded_start, expanded_end, window_id):
    peak_idx = int(start_idx)
    peak_score = float(frame_rows[start_idx].risk_score)
    for idx in range(int(start_idx), int(end_idx) + 1):
        score = float(frame_rows[idx].risk_score)
        if score > peak_score:
            peak_idx = int(idx)
            peak_score = float(score)

    integrated_score = 0.0
    for idx in range(int(expanded_start), int(expanded_end) + 1):
        integrated_score += float(frame_rows[idx].risk_score)

    return RiskWindowRegion(
        window_id=int(window_id),
        raw_start_frame=int(start_idx),
        raw_end_frame=int(end_idx),
        expanded_start_frame=int(expanded_start),
        expanded_end_frame=int(expanded_end),
        raw_start_time=float(frame_rows[start_idx].timestamp),
        raw_end_time=float(frame_rows[end_idx].timestamp),
        expanded_start_time=float(frame_rows[expanded_start].timestamp),
        expanded_end_time=float(frame_rows[expanded_end].timestamp),
        peak_frame_idx=int(peak_idx),
        peak_timestamp=float(frame_rows[peak_idx].timestamp),
        peak_score=float(peak_score),
        width_raw=int(end_idx - start_idx + 1),
        width_expanded=int(expanded_end - expanded_start + 1),
        integrated_score=float(integrated_score),
        window_rank=0,
    )


def _extract_raw_windows(frame_rows, thresholds, raw_merge_gap_frames):
    risk_indices = []
    window_threshold = float(thresholds.get("window", thresholds.get("high", 0.65)) or 0.65)
    for row in frame_rows:
        if float(row.risk_score) >= window_threshold:
            risk_indices.append(int(row.frame_idx))

    if not risk_indices:
        return []

    groups = []
    cur_start = risk_indices[0]
    cur_end = risk_indices[0]
    max_selected_gap = int(raw_merge_gap_frames) + 1
    for idx in risk_indices[1:]:
        if int(idx) - int(cur_end) <= max_selected_gap:
            cur_end = int(idx)
            continue
        groups.append((int(cur_start), int(cur_end)))
        cur_start = int(idx)
        cur_end = int(idx)
    groups.append((int(cur_start), int(cur_end)))

    windows = []
    for window_id, (start_idx, end_idx) in enumerate(groups):
        windows.append(
            _build_window_region(
                frame_rows=frame_rows,
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                expanded_start=int(start_idx),
                expanded_end=int(end_idx),
                window_id=int(window_id),
            )
        )
    return _rank_windows(windows)


def _expand_and_merge_windows(frame_rows, raw_windows, expansion_radius_frames, expanded_merge_gap_frames):
    if not raw_windows:
        return []

    last_frame_idx = int(len(frame_rows) - 1)
    ranges = []
    for raw_window in raw_windows:
        start_idx = max(0, int(raw_window.raw_start_frame) - int(expansion_radius_frames))
        end_idx = min(last_frame_idx, int(raw_window.raw_end_frame) + int(expansion_radius_frames))
        ranges.append(
            {
                "raw_start_frame": int(raw_window.raw_start_frame),
                "raw_end_frame": int(raw_window.raw_end_frame),
                "expanded_start_frame": int(start_idx),
                "expanded_end_frame": int(end_idx),
            }
        )

    merged = []
    current = dict(ranges[0])
    for item in ranges[1:]:
        if int(item["expanded_start_frame"]) <= int(current["expanded_end_frame"]) + int(expanded_merge_gap_frames) + 1:
            current["raw_start_frame"] = min(int(current["raw_start_frame"]), int(item["raw_start_frame"]))
            current["raw_end_frame"] = max(int(current["raw_end_frame"]), int(item["raw_end_frame"]))
            current["expanded_start_frame"] = min(int(current["expanded_start_frame"]), int(item["expanded_start_frame"]))
            current["expanded_end_frame"] = max(int(current["expanded_end_frame"]), int(item["expanded_end_frame"]))
            continue
        merged.append(dict(current))
        current = dict(item)
    merged.append(dict(current))

    windows = []
    for window_id, item in enumerate(merged):
        windows.append(
            _build_window_region(
                frame_rows=frame_rows,
                start_idx=int(item["raw_start_frame"]),
                end_idx=int(item["raw_end_frame"]),
                expanded_start=int(item["expanded_start_frame"]),
                expanded_end=int(item["expanded_end_frame"]),
                window_id=int(window_id),
            )
        )
    return _rank_windows(windows)


def _count_inside(indices, start_idx, end_idx):
    count = 0
    for idx in indices:
        if int(start_idx) <= int(idx) <= int(end_idx):
            count += 1
    return int(count)


def _prev_next_window_boundary(indices, start_idx, end_idx):
    if not indices:
        return None, None
    prev_pos = bisect_right(indices, int(start_idx))
    next_pos = bisect_left(indices, int(end_idx))
    prev_idx = indices[prev_pos - 1] if prev_pos > 0 else None
    next_idx = indices[next_pos] if next_pos < len(indices) else None
    return prev_idx, next_idx


def _span(prev_idx, next_idx):
    if prev_idx is None or next_idx is None:
        return None
    return int(next_idx) - int(prev_idx)


def _distance_left(start_idx, prev_idx):
    if prev_idx is None:
        return None
    return int(start_idx) - int(prev_idx)


def _distance_right(end_idx, next_idx):
    if next_idx is None:
        return None
    return int(next_idx) - int(end_idx)


def _timestamp_lookup(frame_rows, frame_idx):
    if frame_idx is None:
        return None
    return float(frame_rows[int(frame_idx)].timestamp)


def _build_window_coverages(frame_rows, expanded_windows, uniform_keyframe_indices, final_keyframe_indices):
    uniform_keyframe_indices = sorted(int(idx) for idx in uniform_keyframe_indices or [])
    final_keyframe_indices = sorted(int(idx) for idx in final_keyframe_indices or [])
    summaries = []

    for window in expanded_windows:
        start_idx = int(window.expanded_start_frame)
        end_idx = int(window.expanded_end_frame)
        prev_uniform, next_uniform = _prev_next_window_boundary(uniform_keyframe_indices, start_idx, end_idx)
        prev_final, next_final = _prev_next_window_boundary(final_keyframe_indices, start_idx, end_idx)
        summaries.append(
            WindowCoverageSummary(
                window_id=int(window.window_id),
                window_rank=int(window.window_rank),
                raw_start_frame=int(window.raw_start_frame),
                raw_end_frame=int(window.raw_end_frame),
                expanded_start_frame=int(window.expanded_start_frame),
                expanded_end_frame=int(window.expanded_end_frame),
                raw_start_time=float(window.raw_start_time),
                raw_end_time=float(window.raw_end_time),
                expanded_start_time=float(window.expanded_start_time),
                expanded_end_time=float(window.expanded_end_time),
                peak_frame_idx=int(window.peak_frame_idx),
                peak_timestamp=float(window.peak_timestamp),
                peak_score=float(window.peak_score),
                width_raw=int(window.width_raw),
                width_expanded=int(window.width_expanded),
                integrated_score=float(window.integrated_score),
                uniform_base_count=_count_inside(uniform_keyframe_indices, start_idx, end_idx),
                final_count=_count_inside(final_keyframe_indices, start_idx, end_idx),
                prev_uniform_kf_idx=None if prev_uniform is None else int(prev_uniform),
                next_uniform_kf_idx=None if next_uniform is None else int(next_uniform),
                prev_final_kf_idx=None if prev_final is None else int(prev_final),
                next_final_kf_idx=None if next_final is None else int(next_final),
                prev_uniform_kf_time=_timestamp_lookup(frame_rows, prev_uniform),
                next_uniform_kf_time=_timestamp_lookup(frame_rows, next_uniform),
                prev_final_kf_time=_timestamp_lookup(frame_rows, prev_final),
                next_final_kf_time=_timestamp_lookup(frame_rows, next_final),
                uniform_span_across_window=_span(prev_uniform, next_uniform),
                final_span_across_window=_span(prev_final, next_final),
                uniform_left_distance_to_window=_distance_left(start_idx, prev_uniform),
                uniform_right_distance_to_window=_distance_right(end_idx, next_uniform),
                final_left_distance_to_window=_distance_left(start_idx, prev_final),
                final_right_distance_to_window=_distance_right(end_idx, next_final),
            )
        )
    return summaries


def _top_risk_peaks(frame_rows, top_k):
    peaks = []
    for idx, row in enumerate(frame_rows):
        score = float(row.risk_score)
        left_score = float(frame_rows[idx - 1].risk_score) if idx > 0 else score
        right_score = float(frame_rows[idx + 1].risk_score) if idx + 1 < len(frame_rows) else score
        if score + 1e-12 < left_score or score + 1e-12 < right_score:
            continue
        peaks.append(
            {
                "frame_idx": int(row.frame_idx),
                "timestamp": float(row.timestamp),
                "risk_score": float(row.risk_score),
                "risk_level": str(row.risk_level),
            }
        )
    peaks.sort(key=lambda item: (-float(item["risk_score"]), int(item["frame_idx"])))
    if len(peaks) >= int(top_k):
        return peaks[: int(top_k)]
    fallback = []
    for row in frame_rows:
        fallback.append(
            {
                "frame_idx": int(row.frame_idx),
                "timestamp": float(row.timestamp),
                "risk_score": float(row.risk_score),
                "risk_level": str(row.risk_level),
            }
        )
    fallback.sort(key=lambda item: (-float(item["risk_score"]), int(item["frame_idx"])))
    return fallback[: int(top_k)]


def _worst_window(window_coverages):
    if not window_coverages:
        return None
    ordered = sorted(
        list(window_coverages),
        key=lambda item: (
            -float(item.integrated_score),
            -float(item.peak_score),
            -int(item.width_expanded),
            int(item.expanded_start_frame),
        ),
    )
    return ordered[0]


def compute_risk_bundle(
    rows,
    uniform_keyframe_indices,
    final_keyframe_indices,
    smooth_window=5,
    kf_gap=None,
    expansion_radius_frames=None,
    weights=None,
    thresholds=None,
    raw_merge_gap_frames=DEFAULT_RAW_MERGE_GAP_FRAMES,
    expanded_merge_gap_frames=DEFAULT_EXPANDED_MERGE_GAP_FRAMES,
    turn_proxy_components=None,
    version=RISK_BUNDLE_VERSION,
):
    rows = list(rows or [])
    uniform_keyframe_indices = sorted(int(idx) for idx in uniform_keyframe_indices or [])
    final_keyframe_indices = sorted(int(idx) for idx in final_keyframe_indices or [])
    weights = dict(DEFAULT_RISK_WEIGHTS, **dict(weights or {}))
    thresholds = dict(DEFAULT_RISK_THRESHOLDS, **dict(thresholds or {}))
    turn_proxy_components = dict(DEFAULT_TURN_PROXY_COMPONENTS, **dict(turn_proxy_components or {}))
    smooth_window = _safe_window(smooth_window)
    if expansion_radius_frames is None:
        expansion_radius_frames = _default_expansion_radius(uniform_keyframe_indices, kf_gap)
    expansion_radius_frames = max(0, int(expansion_radius_frames))

    if not rows:
        metadata = RiskMetadataConfig(
            weights=dict((str(k), float(v)) for k, v in weights.items()),
            thresholds=dict((str(k), float(v)) for k, v in thresholds.items()),
            smooth_window=int(smooth_window),
            expansion_radius_frames=int(expansion_radius_frames),
            hardest_window_selection=str(DEFAULT_HARDEST_WINDOW_SELECTION),
            version=str(version),
            raw_merge_gap_frames=int(raw_merge_gap_frames),
            expanded_merge_gap_frames=int(expanded_merge_gap_frames),
            integrated_score_definition="sum(risk_score over expanded window)",
            turn_proxy_components=dict((str(k), float(v)) for k, v in turn_proxy_components.items()),
            motion_proxy_source="motion_density",
            semantic_proxy_source="semantic_density",
            local_gap_definition="per-frame distance from frame_idx to previous/next final keyframe",
            window_distance_definition="distance from expanded window boundary to previous/next anchor outside or on the boundary",
        )
        return RiskBundle(
            frame_rows=[],
            raw_windows=[],
            expanded_windows=[],
            window_coverages=[],
            hardest_window=None,
            metadata=metadata,
        )

    turn_raw = _turn_proxy_raw(rows, turn_proxy_components)
    turn_proxy, _turn_window = _windowed(turn_raw, smooth_window)
    motion_raw = _series(rows, "motion_density")
    motion_base = minmax_normalize(motion_raw)
    motion_proxy, _motion_window = _windowed(motion_base, smooth_window)
    semantic_raw = _series(rows, "semantic_density")
    semantic_base = minmax_normalize(semantic_raw)
    semantic_proxy, _semantic_window = _windowed(semantic_base, smooth_window)

    risk_score_raw = []
    for idx in range(len(rows)):
        risk_score_raw.append(
            float(weights.get("turn_proxy", 0.0) or 0.0) * float(turn_raw[idx])
            + float(weights.get("motion_proxy", 0.0) or 0.0) * float(motion_base[idx])
            + float(weights.get("semantic_proxy", 0.0) or 0.0) * float(semantic_base[idx])
        )
    risk_score, _risk_window = _windowed(risk_score_raw, smooth_window)

    uniform_set = set(uniform_keyframe_indices)
    final_set = set(final_keyframe_indices)
    max_frame_idx = max(0, len(rows) - 1)
    frame_rows = []
    for idx, row in enumerate(rows):
        left_gap, right_gap = _nearest_left_right(idx, final_keyframe_indices, max_frame_idx)
        frame_rows.append(
            RiskFrameRow(
                frame_idx=int(idx),
                timestamp=_frame_timestamp(row),
                turn_proxy_raw=float(turn_raw[idx]),
                turn_proxy=float(turn_proxy[idx]),
                motion_proxy_raw=float(motion_raw[idx]),
                motion_proxy=float(motion_proxy[idx]),
                semantic_proxy_raw=float(semantic_raw[idx]),
                semantic_proxy=float(semantic_proxy[idx]),
                risk_score_raw=float(risk_score_raw[idx]),
                risk_score=float(risk_score[idx]),
                risk_level=_risk_level(risk_score[idx], thresholds),
                is_uniform_base_kf=bool(idx in uniform_set),
                is_final_kf=bool(idx in final_set),
                local_gap_left=int(left_gap),
                local_gap_right=int(right_gap),
            )
        )

    raw_windows = _extract_raw_windows(frame_rows, thresholds, raw_merge_gap_frames)
    expanded_windows = _expand_and_merge_windows(
        frame_rows=frame_rows,
        raw_windows=raw_windows,
        expansion_radius_frames=expansion_radius_frames,
        expanded_merge_gap_frames=expanded_merge_gap_frames,
    )
    window_coverages = _build_window_coverages(
        frame_rows=frame_rows,
        expanded_windows=expanded_windows,
        uniform_keyframe_indices=uniform_keyframe_indices,
        final_keyframe_indices=final_keyframe_indices,
    )
    hardest_window = None
    if window_coverages:
        hardest_window = sorted(
            list(window_coverages),
            key=lambda item: (
                -float(item.peak_score),
                -float(item.integrated_score),
                -int(item.width_expanded),
                int(item.expanded_start_frame),
            ),
        )[0]

    _assign_window_membership(frame_rows, raw_windows, expanded_windows)

    metadata = RiskMetadataConfig(
        weights=dict((str(k), float(v)) for k, v in weights.items()),
        thresholds=dict((str(k), float(v)) for k, v in thresholds.items()),
        smooth_window=int(smooth_window),
        expansion_radius_frames=int(expansion_radius_frames),
        hardest_window_selection=str(DEFAULT_HARDEST_WINDOW_SELECTION),
        version=str(version),
        raw_merge_gap_frames=int(raw_merge_gap_frames),
        expanded_merge_gap_frames=int(expanded_merge_gap_frames),
        integrated_score_definition="sum(risk_score over expanded window)",
        turn_proxy_components=dict((str(k), float(v)) for k, v in turn_proxy_components.items()),
        motion_proxy_source="motion_density",
        semantic_proxy_source="semantic_density",
        local_gap_definition="per-frame distance from frame_idx to previous/next final keyframe",
        window_distance_definition="distance from expanded window boundary to previous/next anchor outside or on the boundary",
    )
    return RiskBundle(
        frame_rows=frame_rows,
        raw_windows=raw_windows,
        expanded_windows=expanded_windows,
        window_coverages=window_coverages,
        hardest_window=hardest_window,
        metadata=metadata,
    )


def risk_bundle_to_dict(bundle):
    if isinstance(bundle, dict):
        return bundle
    return asdict(bundle)


def risk_frame_rows_to_dicts(bundle):
    payload = risk_bundle_to_dict(bundle)
    return list(payload.get("frame_rows", []) or [])


def risk_windows_to_dicts(bundle):
    payload = risk_bundle_to_dict(bundle)
    return list(payload.get("window_coverages", []) or [])


def build_risk_summary(bundle, top_k=5):
    payload = risk_bundle_to_dict(bundle)
    frame_rows = list(payload.get("frame_rows", []) or [])
    raw_windows = list(payload.get("raw_windows", []) or [])
    expanded_windows = list(payload.get("expanded_windows", []) or [])
    window_coverages = list(payload.get("window_coverages", []) or [])
    hardest_window = payload.get("hardest_window")
    worst_window = None
    if window_coverages:
        worst_window = risk_bundle_to_dict(_worst_window([_dict_to_window_coverage(item) for item in window_coverages]))

    top_risk_peaks = _top_risk_peaks([_dict_to_frame_row(item) for item in frame_rows], top_k)
    uniform_base_count_in_windows = 0
    final_count_in_windows = 0
    max_gap_in_windows = 0
    for item in window_coverages:
        uniform_base_count_in_windows += int(item.get("uniform_base_count", 0) or 0)
        final_count_in_windows += int(item.get("final_count", 0) or 0)
        for key in ("uniform_span_across_window", "final_span_across_window"):
            value = item.get(key)
            if value is None:
                continue
            max_gap_in_windows = max(int(max_gap_in_windows), int(value))

    return {
        "top_risk_peaks": top_risk_peaks,
        "raw_risk_windows": raw_windows,
        "risk_windows": expanded_windows,
        "window_coverage": window_coverages,
        "keyframe_coverage": {
            "uniform_base_count_in_windows": int(uniform_base_count_in_windows),
            "final_count_in_windows": int(final_count_in_windows),
            "max_gap_in_windows": int(max_gap_in_windows),
            "hardest_window": hardest_window,
            "worst_window": worst_window,
        },
        "metadata": dict(payload.get("metadata", {}) or {}),
    }


def _dict_to_frame_row(item):
    return RiskFrameRow(
        frame_idx=int(item.get("frame_idx", 0) or 0),
        timestamp=float(item.get("timestamp", 0.0) or 0.0),
        turn_proxy_raw=float(item.get("turn_proxy_raw", 0.0) or 0.0),
        turn_proxy=float(item.get("turn_proxy", 0.0) or 0.0),
        motion_proxy_raw=float(item.get("motion_proxy_raw", 0.0) or 0.0),
        motion_proxy=float(item.get("motion_proxy", 0.0) or 0.0),
        semantic_proxy_raw=float(item.get("semantic_proxy_raw", 0.0) or 0.0),
        semantic_proxy=float(item.get("semantic_proxy", 0.0) or 0.0),
        risk_score_raw=float(item.get("risk_score_raw", 0.0) or 0.0),
        risk_score=float(item.get("risk_score", 0.0) or 0.0),
        risk_level=str(item.get("risk_level", "low") or "low"),
        is_uniform_base_kf=bool(item.get("is_uniform_base_kf", False)),
        is_final_kf=bool(item.get("is_final_kf", False)),
        local_gap_left=int(item.get("local_gap_left", 0) or 0),
        local_gap_right=int(item.get("local_gap_right", 0) or 0),
        is_in_raw_risk_window=bool(item.get("is_in_raw_risk_window", False)),
        raw_risk_window_id=item.get("raw_risk_window_id"),
        is_in_expanded_risk_window=bool(item.get("is_in_expanded_risk_window", False)),
        expanded_risk_window_id=item.get("expanded_risk_window_id"),
    )


def _dict_to_window_coverage(item):
    return WindowCoverageSummary(
        window_id=int(item.get("window_id", 0) or 0),
        window_rank=int(item.get("window_rank", 0) or 0),
        raw_start_frame=int(item.get("raw_start_frame", 0) or 0),
        raw_end_frame=int(item.get("raw_end_frame", 0) or 0),
        expanded_start_frame=int(item.get("expanded_start_frame", 0) or 0),
        expanded_end_frame=int(item.get("expanded_end_frame", 0) or 0),
        raw_start_time=float(item.get("raw_start_time", 0.0) or 0.0),
        raw_end_time=float(item.get("raw_end_time", 0.0) or 0.0),
        expanded_start_time=float(item.get("expanded_start_time", 0.0) or 0.0),
        expanded_end_time=float(item.get("expanded_end_time", 0.0) or 0.0),
        peak_frame_idx=int(item.get("peak_frame_idx", 0) or 0),
        peak_timestamp=float(item.get("peak_timestamp", 0.0) or 0.0),
        peak_score=float(item.get("peak_score", 0.0) or 0.0),
        width_raw=int(item.get("width_raw", 0) or 0),
        width_expanded=int(item.get("width_expanded", 0) or 0),
        integrated_score=float(item.get("integrated_score", 0.0) or 0.0),
        uniform_base_count=int(item.get("uniform_base_count", 0) or 0),
        final_count=int(item.get("final_count", 0) or 0),
        prev_uniform_kf_idx=item.get("prev_uniform_kf_idx"),
        next_uniform_kf_idx=item.get("next_uniform_kf_idx"),
        prev_final_kf_idx=item.get("prev_final_kf_idx"),
        next_final_kf_idx=item.get("next_final_kf_idx"),
        prev_uniform_kf_time=item.get("prev_uniform_kf_time"),
        next_uniform_kf_time=item.get("next_uniform_kf_time"),
        prev_final_kf_time=item.get("prev_final_kf_time"),
        next_final_kf_time=item.get("next_final_kf_time"),
        uniform_span_across_window=item.get("uniform_span_across_window"),
        final_span_across_window=item.get("final_span_across_window"),
        uniform_left_distance_to_window=item.get("uniform_left_distance_to_window"),
        uniform_right_distance_to_window=item.get("uniform_right_distance_to_window"),
        final_left_distance_to_window=item.get("final_left_distance_to_window"),
        final_right_distance_to_window=item.get("final_right_distance_to_window"),
    )
