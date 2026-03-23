#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _common import log_info

from _segment.policies.signal_builders import load_risk_signal_bundle
from _segment.research import (
    build_formal_risk_policy_meta,
    build_proposed_schedule_from_risk_bundle,
    compute_risk_bundle,
)


RISK_POLICY_NAME = "risk"
DEFAULT_RISKY_GAP = 24
DEFAULT_SMOOTH_WINDOW = 5


def _risk_used_last_idx(frame_count_total):
    frame_count_total = int(frame_count_total)
    if frame_count_total <= 0:
        return -1
    return int(((frame_count_total - 1) // 4) * 4)


def _safe_base_indices(used_last_idx, safe_gap):
    used_last_idx = int(used_last_idx)
    safe_gap = max(1, int(safe_gap))
    if used_last_idx < 0:
        return []
    indices = list(range(0, used_last_idx + 1, safe_gap))
    if not indices:
        indices = [0]
    if indices[-1] != int(used_last_idx):
        indices.append(int(used_last_idx))
    return indices


def _safe_snap_map(safe_snapped_pairs):
    mapping = {}
    for item in list(safe_snapped_pairs or []):
        teacher_idx = int(item.get("teacher_frame_idx", 0) or 0)
        safe_idx = int(item.get("safe_frame_idx", 0) or 0)
        candidate_key = (
            0 if teacher_idx == safe_idx else 1,
            abs(teacher_idx - safe_idx),
            safe_idx,
        )
        current = mapping.get(teacher_idx)
        if current is None or candidate_key < current["key"]:
            mapping[teacher_idx] = {
                "safe_frame_idx": int(safe_idx),
                "key": candidate_key,
            }
    out = {}
    for teacher_idx, item in mapping.items():
        out[int(teacher_idx)] = int(item["safe_frame_idx"])
    return out


def _window_id_for_frame(frame_idx, windows):
    best_window_id = None
    best_distance = None
    best_rank = None
    for item in list(windows or []):
        start_idx = int(item.get("expanded_start_frame", item.get("raw_start_frame", 0)) or 0)
        end_idx = int(item.get("expanded_end_frame", item.get("raw_end_frame", 0)) or 0)
        if start_idx <= int(frame_idx) <= end_idx:
            return int(item.get("window_id", 0) or 0)
        if int(frame_idx) < start_idx:
            distance = int(start_idx - int(frame_idx))
        else:
            distance = int(int(frame_idx) - end_idx)
        window_rank = int(item.get("window_rank", 0) or 0)
        if best_distance is None or distance < best_distance or (
            distance == best_distance and (best_rank is None or window_rank < best_rank)
        ):
            best_distance = int(distance)
            best_rank = int(window_rank)
            best_window_id = int(item.get("window_id", 0) or 0)
    return best_window_id


def _build_risk_items(build_item, safe_base_indices, schedule_payload):
    safe_base_set = set(int(idx) for idx in safe_base_indices)
    safe_snap_map = _safe_snap_map(schedule_payload.get("safe_snapped_pairs"))
    risky_windows = list(schedule_payload.get("risky_windows_used") or [])
    items = []
    inserted_count = 0
    relocated_count = 0

    for frame_idx in list(schedule_payload.get("proposed_final_anchors") or []):
        frame_idx = int(frame_idx)
        if frame_idx in safe_base_set:
            items.append(
                build_item(
                    frame_idx,
                    source_type="uniform",
                    source_role="uniform",
                    candidate_role="uniform",
                    promotion_source="uniform",
                )
            )
            continue

        if frame_idx in safe_snap_map:
            base_idx = int(safe_snap_map[frame_idx])
            relocated = bool(frame_idx != base_idx)
            relocated_count += 1 if relocated else 0
            items.append(
                build_item(
                    frame_idx,
                    source_type=RISK_POLICY_NAME,
                    source_role="safe_snapped",
                    candidate_role="safe_snapped",
                    is_relocated=relocated,
                    replaced_uniform_index=int(base_idx) if relocated else None,
                    promotion_source="safe_gap",
                    promotion_reason="snap_to_teacher_anchor",
                    window_id=_window_id_for_frame(frame_idx, risky_windows),
                )
            )
            continue

        inserted_count += 1
        items.append(
            build_item(
                frame_idx,
                source_type=RISK_POLICY_NAME,
                source_role="risky_teacher",
                candidate_role="risky_teacher",
                is_inserted=True,
                promotion_source="risk_window",
                promotion_reason="protect_risky_window",
                window_id=_window_id_for_frame(frame_idx, risky_windows),
            )
        )
    return items, int(inserted_count), int(relocated_count)


def build_policy_plan(context):
    safe_gap = max(1, int(context["kf_gap"]))
    risky_gap = int(DEFAULT_RISKY_GAP)
    used_last_idx = _risk_used_last_idx(context["frame_count_total"])
    frame_count_used = int(max(0, used_last_idx + 1))
    safe_base_indices = _safe_base_indices(used_last_idx, safe_gap)

    if frame_count_used <= 0 or not safe_base_indices:
        return {
            "used_last_idx": int(used_last_idx),
            "frame_count_used": int(frame_count_used),
            "tail_drop": int(max(0, int(context["frame_count_total"]) - int(frame_count_used))),
            "uniform_base_indices": list(safe_base_indices),
            "keyframe_indices": [],
            "keyframe_items": [],
            "summary": {
                "policy_name": RISK_POLICY_NAME,
                "num_uniform_base": 0,
                "num_final_keyframes": 0,
                "extra_kf_ratio": 0.0,
                "risk_window_count": 0,
                "all_risky_windows_protected": True,
                "reduction_vs_teacher": 0.0,
            },
            "policy_meta": {
                "policy_name": RISK_POLICY_NAME,
                "safe_gap": int(safe_gap),
                "risky_gap": int(risky_gap),
                "teacher_gap": int(risky_gap),
                "risk_window_count": 0,
                "proposed_anchor_count": 0,
                "final_keyframe_count": 0,
                "all_risky_windows_protected": True,
                "worst_window_gap": 0,
                "reduction_vs_teacher": 0.0,
                "hardest_window_frame_range": None,
                "hardest_window_peak_score": 0.0,
                "window_protection": {
                    "protected_window_count": 0,
                    "violating_window_count": 0,
                    "all_risky_windows_protected": True,
                    "max_gap_inside_risky_windows": 0,
                    "worst_window_gap": 0,
                    "worst_window": None,
                    "per_window": [],
                },
            },
        }

    log_info(
        "risk observe start: frames={} used={} safe_gap={} risky_gap={}".format(
            int(context["frame_count_total"]),
            int(frame_count_used),
            int(safe_gap),
            int(risky_gap),
        )
    )
    signal_bundle = load_risk_signal_bundle(
        context,
        DEFAULT_SMOOTH_WINDOW,
        used_last_idx=used_last_idx,
        used_frame_count=frame_count_used,
    )
    risk_bundle = compute_risk_bundle(
        rows=signal_bundle["rows"],
        uniform_keyframe_indices=safe_base_indices,
        final_keyframe_indices=safe_base_indices,
        smooth_window=int(DEFAULT_SMOOTH_WINDOW),
        kf_gap=int(safe_gap),
    )
    schedule_payload = build_proposed_schedule_from_risk_bundle(
        risk_bundle=risk_bundle,
        frame_count=frame_count_used,
        frame_idx_start=0,
        frame_idx_end=used_last_idx,
        teacher_gap=risky_gap,
        safe_gap=safe_gap,
        use_expanded_windows=True,
        merge_nearby_windows=False,
        edge_protection_teacher_hops=1,
        preserve_range_boundaries=True,
    )
    risk_meta = build_formal_risk_policy_meta(
        risk_bundle=risk_bundle,
        schedule_payload=schedule_payload,
        safe_gap=safe_gap,
        risky_gap=risky_gap,
        policy_name=RISK_POLICY_NAME,
    )
    keyframe_items, inserted_count, relocated_count = _build_risk_items(
        context["build_item"],
        safe_base_indices,
        schedule_payload,
    )

    final_indices = list(schedule_payload.get("proposed_final_anchors") or [])
    teacher_count = int(len(list(schedule_payload.get("teacher_dense_anchors") or [])))
    summary = {
        "policy_name": RISK_POLICY_NAME,
        "uniform_count": int(len(safe_base_indices)),
        "num_uniform_base": int(len(safe_base_indices)),
        "final_keyframe_count": int(len(final_indices)),
        "num_final_keyframes": int(len(final_indices)),
        "inserted_count": int(inserted_count),
        "relocated_count": int(relocated_count),
        "risk_window_count": int(risk_meta.get("risk_window_count", 0) or 0),
        "all_risky_windows_protected": bool(risk_meta.get("all_risky_windows_protected", False)),
        "worst_window_gap": int(risk_meta.get("worst_window_gap", 0) or 0),
        "reduction_vs_teacher": float(risk_meta.get("reduction_vs_teacher", 0.0) or 0.0),
        "extra_kf_ratio": float(
            max(0.0, float(len(final_indices) - len(safe_base_indices))) / float(len(safe_base_indices))
        ) if safe_base_indices else 0.0,
        "teacher_anchor_count": int(teacher_count),
        "num_boundary_selected": 0,
        "num_support_selected": 0,
        "num_boundary_relocated": 0,
        "num_boundary_inserted": 0,
        "num_support_inserted": int(inserted_count),
        "num_promoted_support_inserted": int(inserted_count),
        "num_burst_windows_triggered": int(risk_meta.get("risk_window_count", 0) or 0),
    }

    risk_meta["signal_meta"] = {
        "frame_signals": dict(signal_bundle.get("meta", {}).get("frame_signals", {}) or {}),
        "semantic": dict(signal_bundle.get("meta", {}).get("semantic", {}) or {}),
        "motion": dict(signal_bundle.get("meta", {}).get("motion", {}) or {}),
    }
    risk_meta["safe_uniform_indices"] = list(safe_base_indices)
    risk_meta["teacher_dense_anchors"] = list(schedule_payload.get("teacher_dense_anchors") or [])
    risk_meta["safe_uniform_anchors"] = list(schedule_payload.get("safe_uniform_anchors") or [])
    risk_meta["safe_snapped_anchors"] = list(schedule_payload.get("safe_snapped_anchors") or [])
    risk_meta["safe_snapped_pairs"] = list(schedule_payload.get("safe_snapped_pairs") or [])
    risk_meta["risky_windows_used"] = list(schedule_payload.get("risky_windows_used") or [])

    log_info(
        "risk selected: safe_base={} teacher={} final={} windows={} reduction_vs_teacher={:.3f}".format(
            int(len(safe_base_indices)),
            int(teacher_count),
            int(len(final_indices)),
            int(risk_meta.get("risk_window_count", 0) or 0),
            float(risk_meta.get("reduction_vs_teacher", 0.0) or 0.0),
        )
    )

    return {
        "used_last_idx": int(used_last_idx),
        "frame_count_used": int(frame_count_used),
        "tail_drop": int(max(0, int(context["frame_count_total"]) - int(frame_count_used))),
        "uniform_base_indices": list(safe_base_indices),
        "keyframe_indices": list(final_indices),
        "keyframe_items": keyframe_items,
        "summary": summary,
        "policy_meta": risk_meta,
    }
