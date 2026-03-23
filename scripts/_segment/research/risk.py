#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..policies.uniform import compute_uniform_base
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
DEFAULT_PROPOSED_TEACHER_GAP = 24
DEFAULT_PROPOSED_SAFE_GAP = 60
PROPOSED_SCHEDULE_VERSION = "risk_window_v1"


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


def _resolve_schedule_range(frame_count=None, frame_idx_start=0, frame_idx_end=None):
    frame_idx_start = max(0, int(frame_idx_start or 0))
    if frame_idx_end is None:
        if frame_count is None:
            frame_idx_end = frame_idx_start - 1
        else:
            frame_idx_end = frame_idx_start + max(0, int(frame_count) - 1)
    frame_idx_end = int(frame_idx_end)
    if frame_idx_end < frame_idx_start:
        return frame_idx_start, frame_idx_start - 1, 0
    return frame_idx_start, frame_idx_end, int(frame_idx_end - frame_idx_start + 1)


def _uniform_anchor_range(frame_count=None, frame_idx_start=0, frame_idx_end=None, gap=1):
    start_idx, end_idx, total_count = _resolve_schedule_range(
        frame_count=frame_count,
        frame_idx_start=frame_idx_start,
        frame_idx_end=frame_idx_end,
    )
    if total_count <= 0:
        return {
            "indices": [],
            "used_last_idx": int(start_idx - 1),
            "used_count": 0,
            "tail_drop": 0,
        }
    base = compute_uniform_base(total_count, gap)
    return {
        "indices": [int(start_idx + idx) for idx in list(base.get("indices") or [])],
        "used_last_idx": int(start_idx + int(base.get("used_last_idx", -1) or -1)),
        "used_count": int(base.get("used_count", 0) or 0),
        "tail_drop": int(base.get("tail_drop", 0) or 0),
    }


def _clip_window_to_range(window, frame_idx_start, frame_idx_end):
    raw_start = max(int(frame_idx_start), int(window.get("raw_start_frame", 0) or 0))
    raw_end = min(int(frame_idx_end), int(window.get("raw_end_frame", 0) or 0))
    expanded_start = max(int(frame_idx_start), int(window.get("expanded_start_frame", raw_start) or raw_start))
    expanded_end = min(int(frame_idx_end), int(window.get("expanded_end_frame", raw_end) or raw_end))
    if expanded_end < expanded_start:
        return None
    if raw_end < raw_start:
        raw_start = int(expanded_start)
        raw_end = int(expanded_end)
    clipped = dict(window)
    clipped["raw_start_frame"] = int(raw_start)
    clipped["raw_end_frame"] = int(raw_end)
    clipped["expanded_start_frame"] = int(expanded_start)
    clipped["expanded_end_frame"] = int(expanded_end)
    clipped["width_raw"] = int(max(0, raw_end - raw_start + 1))
    clipped["width_expanded"] = int(max(0, expanded_end - expanded_start + 1))
    return clipped


def _window_time(frame_rows, frame_idx):
    if frame_idx is None:
        return None
    if not frame_rows:
        return None
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= len(frame_rows):
        return None
    row = frame_rows[frame_idx]
    if isinstance(row, dict):
        return float(row.get("timestamp", 0.0) or 0.0)
    return float(getattr(row, "timestamp", 0.0) or 0.0)


def _merge_schedule_windows(windows, merge_gap_frames):
    windows = sorted(
        [dict(item) for item in windows or []],
        key=lambda item: (
            int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0),
            int(item.get("expanded_end_frame", item.get("raw_end_frame", 0)) or 0),
        ),
    )
    if not windows:
        return []
    merge_gap_frames = max(0, int(merge_gap_frames or 0))
    merged = [dict(windows[0])]
    for item in windows[1:]:
        current = merged[-1]
        current_end = int(current.get("expanded_end_frame", current.get("raw_end_frame", 0)) or 0)
        next_start = int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0)
        if next_start <= current_end + merge_gap_frames + 1:
            current["raw_start_frame"] = min(
                int(current.get("raw_start_frame", next_start) or next_start),
                int(item.get("raw_start_frame", next_start) or next_start),
            )
            current["raw_end_frame"] = max(
                int(current.get("raw_end_frame", current_end) or current_end),
                int(item.get("raw_end_frame", current_end) or current_end),
            )
            current["expanded_start_frame"] = min(
                int(current.get("expanded_start_frame", next_start) or next_start),
                int(item.get("expanded_start_frame", next_start) or next_start),
            )
            current["expanded_end_frame"] = max(
                int(current.get("expanded_end_frame", current_end) or current_end),
                int(item.get("expanded_end_frame", current_end) or current_end),
            )
            current["peak_score"] = max(
                float(current.get("peak_score", 0.0) or 0.0),
                float(item.get("peak_score", 0.0) or 0.0),
            )
            current["integrated_score"] = float(current.get("integrated_score", 0.0) or 0.0) + float(
                item.get("integrated_score", 0.0) or 0.0
            )
            peak_frame_current = int(current.get("peak_frame_idx", current["expanded_start_frame"]) or current["expanded_start_frame"])
            peak_frame_item = int(item.get("peak_frame_idx", item["expanded_start_frame"]) or item["expanded_start_frame"])
            if float(item.get("peak_score", 0.0) or 0.0) >= float(current.get("peak_score", 0.0) or 0.0):
                current["peak_frame_idx"] = int(peak_frame_item)
                if "peak_timestamp" in item:
                    current["peak_timestamp"] = item.get("peak_timestamp")
            else:
                current["peak_frame_idx"] = int(peak_frame_current)
            continue
        merged.append(dict(item))
    return merged


def _window_dicts_for_schedule(bundle, frame_rows, use_expanded_windows=True):
    payload = risk_bundle_to_dict(bundle)
    key = "expanded_windows" if use_expanded_windows else "raw_windows"
    windows = list(payload.get(key, []) or [])
    prepared = []
    for item in windows:
        window = dict(item)
        if "window_rank" not in window:
            window["window_rank"] = 0
        if "raw_start_time" not in window:
            window["raw_start_time"] = _window_time(frame_rows, window.get("raw_start_frame"))
        if "raw_end_time" not in window:
            window["raw_end_time"] = _window_time(frame_rows, window.get("raw_end_frame"))
        if "expanded_start_time" not in window:
            window["expanded_start_time"] = _window_time(frame_rows, window.get("expanded_start_frame"))
        if "expanded_end_time" not in window:
            window["expanded_end_time"] = _window_time(frame_rows, window.get("expanded_end_frame"))
        if "peak_timestamp" not in window:
            window["peak_timestamp"] = _window_time(frame_rows, window.get("peak_frame_idx"))
        prepared.append(window)
    return prepared


def _nearest_teacher_anchor(frame_idx, teacher_anchors, prefer_right):
    if not teacher_anchors:
        return None
    pos = bisect_left(teacher_anchors, int(frame_idx))
    left_idx = teacher_anchors[pos - 1] if pos > 0 else None
    right_idx = teacher_anchors[pos] if pos < len(teacher_anchors) else None
    if left_idx is None:
        return None if right_idx is None else int(right_idx)
    if right_idx is None:
        return int(left_idx)
    left_dist = abs(int(frame_idx) - int(left_idx))
    right_dist = abs(int(right_idx) - int(frame_idx))
    if left_dist < right_dist:
        return int(left_idx)
    if right_dist < left_dist:
        return int(right_idx)
    if prefer_right:
        return int(right_idx)
    return int(left_idx)


def _safe_snapped_teacher_anchors(teacher_anchors, frame_count=None, frame_idx_start=0, frame_idx_end=None, safe_gap=1):
    safe_uniform = _uniform_anchor_range(
        frame_count=frame_count,
        frame_idx_start=frame_idx_start,
        frame_idx_end=frame_idx_end,
        gap=safe_gap,
    )
    snapped = []
    for frame_idx in list(safe_uniform.get("indices") or []):
        snapped_idx = _nearest_teacher_anchor(frame_idx, teacher_anchors, prefer_right=True)
        if snapped_idx is None:
            continue
        snapped.append(int(snapped_idx))
    return safe_uniform, sorted(set(snapped))


def _window_anchor_indices(anchor_indices, start_idx, end_idx):
    anchor_indices = sorted(int(idx) for idx in anchor_indices or [])
    if not anchor_indices:
        return []
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    pos_left = bisect_left(anchor_indices, start_idx)
    pos_right = bisect_right(anchor_indices, end_idx)
    indices = []
    if pos_left > 0:
        indices.append(int(anchor_indices[pos_left - 1]))
    indices.extend(int(idx) for idx in anchor_indices[pos_left:pos_right])
    if pos_right < len(anchor_indices):
        next_idx = int(anchor_indices[pos_right])
        if not indices or int(indices[-1]) != next_idx:
            indices.append(int(next_idx))
    return indices


def _max_anchor_gap(anchor_indices):
    if len(anchor_indices) < 2:
        return 0
    max_gap = 0
    for pos in range(1, len(anchor_indices)):
        max_gap = max(int(max_gap), int(anchor_indices[pos]) - int(anchor_indices[pos - 1]))
    return int(max_gap)


def _count_in_or_around(anchor_indices, start_idx, end_idx):
    return int(len(_window_anchor_indices(anchor_indices, start_idx, end_idx)))


def _distance_to_window(frame_idx, window):
    start_idx = int(window.get("expanded_start_frame", window.get("raw_start_frame", 0)) or 0)
    end_idx = int(window.get("expanded_end_frame", window.get("raw_end_frame", 0)) or 0)
    frame_idx = int(frame_idx)
    if start_idx <= frame_idx <= end_idx:
        return 0
    if frame_idx < start_idx:
        return int(start_idx - frame_idx)
    return int(frame_idx - end_idx)


def _nearest_window_rank(frame_idx, windows):
    if not windows:
        return None
    best_rank = None
    best_distance = None
    best_start = None
    for item in windows:
        distance = _distance_to_window(frame_idx, item)
        rank = int(item.get("window_rank", 0) or 0)
        start_idx = int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0)
        if best_distance is None or distance < best_distance or (
            distance == best_distance and (best_rank is None or rank < best_rank or (rank == best_rank and start_idx < best_start))
        ):
            best_distance = int(distance)
            best_rank = int(rank)
            best_start = int(start_idx)
    return best_rank


def _schedule_window_stats(frame_rows, risky_windows_used, teacher_dense_anchors, proposed_final_anchors, teacher_gap):
    if not risky_windows_used:
        return []
    window_objects = [_build_window_region(
        frame_rows=frame_rows,
        start_idx=int(item.get("raw_start_frame", item.get("expanded_start_frame", 0)) or 0),
        end_idx=int(item.get("raw_end_frame", item.get("expanded_end_frame", 0)) or 0),
        expanded_start=int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0),
        expanded_end=int(item.get("expanded_end_frame", item.get("raw_end_frame", 0)) or 0),
        window_id=int(item.get("window_id", 0) or 0),
    ) for item in risky_windows_used]
    for pos, item in enumerate(window_objects):
        item.window_rank = int(risky_windows_used[pos].get("window_rank", pos + 1) or (pos + 1))
    coverages = _build_window_coverages(
        frame_rows=frame_rows,
        expanded_windows=window_objects,
        uniform_keyframe_indices=teacher_dense_anchors,
        final_keyframe_indices=proposed_final_anchors,
    )
    stats = []
    for coverage in coverages:
        start_idx = int(coverage.expanded_start_frame)
        end_idx = int(coverage.expanded_end_frame)
        teacher_sequence = _window_anchor_indices(teacher_dense_anchors, start_idx, end_idx)
        proposed_sequence = _window_anchor_indices(proposed_final_anchors, start_idx, end_idx)
        teacher_gap_max = _max_anchor_gap(teacher_sequence)
        proposed_gap_max = _max_anchor_gap(proposed_sequence)
        stats.append(
            {
                "window_id": int(coverage.window_id),
                "window_rank": int(coverage.window_rank),
                "raw_start_frame": int(coverage.raw_start_frame),
                "raw_end_frame": int(coverage.raw_end_frame),
                "expanded_start_frame": int(coverage.expanded_start_frame),
                "expanded_end_frame": int(coverage.expanded_end_frame),
                "raw_start_time": float(coverage.raw_start_time),
                "raw_end_time": float(coverage.raw_end_time),
                "expanded_start_time": float(coverage.expanded_start_time),
                "expanded_end_time": float(coverage.expanded_end_time),
                "peak_frame_idx": int(coverage.peak_frame_idx),
                "peak_timestamp": float(coverage.peak_timestamp),
                "peak_score": float(coverage.peak_score),
                "integrated_score": float(coverage.integrated_score),
                "teacher_anchor_count_in_window": int(coverage.uniform_base_count),
                "proposed_anchor_count_in_window": int(coverage.final_count),
                "teacher_anchor_count_in_or_around_window": int(_count_in_or_around(teacher_dense_anchors, start_idx, end_idx)),
                "proposed_anchor_count_in_or_around_window": int(_count_in_or_around(proposed_final_anchors, start_idx, end_idx)),
                "teacher_span_across_window": coverage.uniform_span_across_window,
                "proposed_span_across_window": coverage.final_span_across_window,
                "teacher_left_distance_to_window": coverage.uniform_left_distance_to_window,
                "teacher_right_distance_to_window": coverage.uniform_right_distance_to_window,
                "proposed_left_distance_to_window": coverage.final_left_distance_to_window,
                "proposed_right_distance_to_window": coverage.final_right_distance_to_window,
                "teacher_max_gap_across_window": int(teacher_gap_max),
                "proposed_max_gap_across_window": int(proposed_gap_max),
                "protected": bool(proposed_gap_max <= int(teacher_gap) and bool(proposed_sequence)),
            }
        )
    return stats


def build_proposed_schedule_from_risk_bundle(
    risk_bundle,
    frame_count=None,
    frame_idx_start=0,
    frame_idx_end=None,
    teacher_gap=DEFAULT_PROPOSED_TEACHER_GAP,
    safe_gap=DEFAULT_PROPOSED_SAFE_GAP,
    use_expanded_windows=True,
    merge_nearby_windows=False,
    merge_gap_frames=None,
    edge_protection_teacher_hops=1,
):
    payload = risk_bundle_to_dict(risk_bundle)
    frame_rows = [_dict_to_frame_row(item) for item in list(payload.get("frame_rows", []) or [])]
    if frame_count is None and frame_idx_end is None:
        frame_count = len(frame_rows)
    frame_idx_start, frame_idx_end, total_count = _resolve_schedule_range(
        frame_count=frame_count,
        frame_idx_start=frame_idx_start,
        frame_idx_end=frame_idx_end,
    )
    if frame_rows:
        frame_idx_end = min(int(frame_idx_end), int(len(frame_rows) - 1))
        total_count = max(0, int(frame_idx_end - frame_idx_start + 1))

    teacher_gap = max(1, int(teacher_gap))
    safe_gap = max(teacher_gap, int(safe_gap))
    edge_protection_teacher_hops = max(0, int(edge_protection_teacher_hops))
    teacher_uniform = _uniform_anchor_range(
        frame_count=total_count,
        frame_idx_start=frame_idx_start,
        frame_idx_end=frame_idx_end,
        gap=teacher_gap,
    )
    teacher_dense_anchors = list(teacher_uniform.get("indices") or [])
    safe_uniform, safe_snapped_anchors = _safe_snapped_teacher_anchors(
        teacher_dense_anchors,
        frame_count=total_count,
        frame_idx_start=frame_idx_start,
        frame_idx_end=teacher_uniform.get("used_last_idx"),
        safe_gap=safe_gap,
    )

    schedule_frame_end = int(teacher_uniform.get("used_last_idx", frame_idx_start - 1) or (frame_idx_start - 1))
    risky_windows_used = []
    if teacher_dense_anchors:
        windows = _window_dicts_for_schedule(risk_bundle, frame_rows, use_expanded_windows=use_expanded_windows)
        for window in windows:
            clipped = _clip_window_to_range(window, frame_idx_start, schedule_frame_end)
            if clipped is None:
                continue
            risky_windows_used.append(clipped)
        if merge_nearby_windows and risky_windows_used:
            if merge_gap_frames is None:
                merge_gap_frames = max(0, teacher_gap // 2)
            risky_windows_used = _merge_schedule_windows(risky_windows_used, merge_gap_frames)
        risky_windows_used.sort(
            key=lambda item: (
                int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0),
                int(item.get("expanded_end_frame", item.get("raw_end_frame", 0)) or 0),
            )
        )
        for pos, item in enumerate(risky_windows_used, start=1):
            item["window_id"] = int(item.get("window_id", pos - 1) or (pos - 1))
            item["window_rank"] = int(item.get("window_rank", pos) or pos)
            item["width_raw"] = int(
                max(0, int(item.get("raw_end_frame", 0) or 0) - int(item.get("raw_start_frame", 0) or 0) + 1)
            )
            item["width_expanded"] = int(
                max(
                    0,
                    int(item.get("expanded_end_frame", 0) or 0) - int(item.get("expanded_start_frame", 0) or 0) + 1,
                )
            )
            item["raw_start_time"] = _window_time(frame_rows, item.get("raw_start_frame"))
            item["raw_end_time"] = _window_time(frame_rows, item.get("raw_end_frame"))
            item["expanded_start_time"] = _window_time(frame_rows, item.get("expanded_start_frame"))
            item["expanded_end_time"] = _window_time(frame_rows, item.get("expanded_end_frame"))
            item["peak_timestamp"] = _window_time(frame_rows, item.get("peak_frame_idx"))

    proposed_anchor_set = set(int(idx) for idx in safe_snapped_anchors)
    if teacher_dense_anchors:
        proposed_anchor_set.add(int(teacher_dense_anchors[0]))
    teacher_positions = sorted(int(idx) for idx in teacher_dense_anchors)
    for window in risky_windows_used:
        start_idx = int(window.get("expanded_start_frame", 0) or 0)
        end_idx = int(window.get("expanded_end_frame", 0) or 0)
        left_pos = bisect_left(teacher_positions, start_idx)
        right_pos = bisect_right(teacher_positions, end_idx)
        keep_start = max(0, int(left_pos - edge_protection_teacher_hops))
        keep_end = min(len(teacher_positions), int(right_pos + edge_protection_teacher_hops))
        for pos in range(keep_start, keep_end):
            proposed_anchor_set.add(int(teacher_positions[pos]))
    proposed_final_anchors = sorted(proposed_anchor_set)

    per_window_anchor_stats = _schedule_window_stats(
        frame_rows=frame_rows,
        risky_windows_used=risky_windows_used,
        teacher_dense_anchors=teacher_dense_anchors,
        proposed_final_anchors=proposed_final_anchors,
        teacher_gap=teacher_gap,
    )
    all_risky_windows_protected = bool(all(bool(item.get("protected", False)) for item in per_window_anchor_stats))
    max_gap_inside_risky_windows = 0
    worst_window = None
    for item in per_window_anchor_stats:
        gap_value = int(item.get("proposed_max_gap_across_window", 0) or 0)
        if gap_value > max_gap_inside_risky_windows:
            max_gap_inside_risky_windows = int(gap_value)
            worst_window = dict(item)

    return {
        "teacher_dense_anchors": list(teacher_dense_anchors),
        "proposed_final_anchors": list(proposed_final_anchors),
        "risky_windows_used": list(risky_windows_used),
        "per_window_anchor_stats": list(per_window_anchor_stats),
        "validation": {
            "all_risky_windows_protected": bool(all_risky_windows_protected),
            "max_gap_inside_risky_windows": int(max_gap_inside_risky_windows),
            "worst_window_gap": int(max_gap_inside_risky_windows),
            "worst_window": worst_window,
            "violating_window_count": int(
                len([item for item in per_window_anchor_stats if not bool(item.get("protected", False))])
            ),
        },
        "estimated_uniform60_anchor_count": int(len(list(safe_uniform.get("indices") or []))),
        "schedule_metadata": {
            "version": str(PROPOSED_SCHEDULE_VERSION),
            "strategy": "dense uniform teacher -> keep risky/edge teacher anchors -> sparse safe anchors snapped from gap60 baseline",
            "teacher_gap": int(teacher_gap),
            "safe_gap": int(safe_gap),
            "frame_idx_start": int(frame_idx_start),
            "frame_idx_end": int(schedule_frame_end),
            "source_frame_idx_end": int(frame_idx_end),
            "frame_count_total": int(total_count),
            "teacher_used_count": int(teacher_uniform.get("used_count", 0) or 0),
            "teacher_used_last_idx": int(schedule_frame_end),
            "teacher_tail_drop": int(teacher_uniform.get("tail_drop", 0) or 0),
            "use_expanded_windows": bool(use_expanded_windows),
            "merge_nearby_windows": bool(merge_nearby_windows),
            "merge_gap_frames": None if merge_gap_frames is None else int(merge_gap_frames),
            "edge_protection_teacher_hops": int(edge_protection_teacher_hops),
            "safe_uniform_anchor_count": int(len(list(safe_uniform.get("indices") or []))),
            "safe_snapped_teacher_anchor_count": int(len(safe_snapped_anchors)),
            "risky_window_count": int(len(risky_windows_used)),
            "bundle_version": str((payload.get("metadata", {}) or {}).get("version", "")),
            "tail_policy": "teacher anchors follow compute_uniform_base; tail beyond used_last_idx is excluded from proposed schedule",
        },
    }


def proposed_schedule_anchor_rows(schedule_payload, risk_bundle):
    payload = risk_bundle_to_dict(risk_bundle)
    rows = list(payload.get("frame_rows", []) or [])
    teacher_set = set(int(idx) for idx in list(schedule_payload.get("teacher_dense_anchors") or []))
    proposed_set = set(int(idx) for idx in list(schedule_payload.get("proposed_final_anchors") or []))
    windows = list(schedule_payload.get("risky_windows_used") or [])
    all_indices = sorted(teacher_set | proposed_set)
    output_rows = []
    for frame_idx in all_indices:
        timestamp = 0.0
        if 0 <= int(frame_idx) < len(rows):
            timestamp = float(rows[int(frame_idx)].get("timestamp", 0.0) or 0.0)
        in_risky_window = False
        for window in windows:
            start_idx = int(window.get("expanded_start_frame", 0) or 0)
            end_idx = int(window.get("expanded_end_frame", 0) or 0)
            if start_idx <= int(frame_idx) <= end_idx:
                in_risky_window = True
                break
        output_rows.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp": float(timestamp),
                "source": "proposed" if int(frame_idx) in proposed_set else "teacher",
                "is_teacher_anchor": bool(int(frame_idx) in teacher_set),
                "is_proposed_anchor": bool(int(frame_idx) in proposed_set),
                "in_risky_window": bool(in_risky_window),
                "nearest_window_rank": _nearest_window_rank(frame_idx, windows),
            }
        )
    return output_rows


def proposed_schedule_window_rows(schedule_payload):
    rows = []
    for item in list(schedule_payload.get("per_window_anchor_stats") or []):
        rows.append(
            {
                "window_rank": int(item.get("window_rank", 0) or 0),
                "expanded_start_frame": int(item.get("expanded_start_frame", 0) or 0),
                "expanded_end_frame": int(item.get("expanded_end_frame", 0) or 0),
                "peak_score": float(item.get("peak_score", 0.0) or 0.0),
                "integrated_score": float(item.get("integrated_score", 0.0) or 0.0),
                "teacher_anchor_count_in_or_around_window": int(
                    item.get("teacher_anchor_count_in_or_around_window", 0) or 0
                ),
                "proposed_anchor_count_in_or_around_window": int(
                    item.get("proposed_anchor_count_in_or_around_window", 0) or 0
                ),
                "teacher_span_across_window": item.get("teacher_span_across_window"),
                "proposed_span_across_window": item.get("proposed_span_across_window"),
                "proposed_left_distance_to_window": item.get("proposed_left_distance_to_window"),
                "proposed_right_distance_to_window": item.get("proposed_right_distance_to_window"),
                "protected": bool(item.get("protected", False)),
            }
        )
    return rows


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
