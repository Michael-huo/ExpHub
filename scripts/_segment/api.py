#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

from _common import list_frames_sorted, log_info, write_json_atomic

from .materialize import materialize_keyframes
from .policies import get_policy_builder
from .policies.uniform import compute_uniform_base


def _read_timestamps(path):
    values = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            values.append(float(text))
    return values


def _build_keyframe_item_factory(frame_paths, timestamps):
    def _make_item(
        frame_idx,
        source_type,
        source_role,
        rerank_score=None,
        semantic_relation="",
        is_inserted=False,
        is_relocated=False,
        replaced_uniform_index=None,
        candidate_role="",
    ):
        path = frame_paths[int(frame_idx)]
        return {
            "frame_idx": int(frame_idx),
            "file_name": path.name,
            "ts_sec": float(timestamps[int(frame_idx)]),
            "source_type": str(source_type),
            "source_role": str(source_role),
            "candidate_role": str(candidate_role or source_role),
            "rerank_score": float(rerank_score) if rerank_score is not None else None,
            "semantic_relation": str(semantic_relation or ""),
            "is_inserted": bool(is_inserted),
            "is_relocated": bool(is_relocated),
            "replaced_uniform_index": int(replaced_uniform_index) if replaced_uniform_index is not None else None,
        }

    return _make_item


def _normalize_plan(context, plan):
    raw_items = list(plan.get("keyframe_items") or [])
    item_map = {}
    for item in raw_items:
        frame_idx = int(item["frame_idx"])
        item_map[frame_idx] = dict(item)

    raw_indices = list(plan.get("keyframe_indices") or item_map.keys())
    indices = []
    seen = set()
    for value in raw_indices:
        frame_idx = int(value)
        if frame_idx in seen:
            continue
        if frame_idx < 0 or frame_idx > int(context["used_last_idx"]):
            continue
        seen.add(frame_idx)
        indices.append(frame_idx)
    indices.sort()

    build_item = context["build_item"]
    items = []
    for frame_idx in indices:
        item = item_map.get(frame_idx)
        if item is None:
            item = build_item(frame_idx, source_type="uniform", source_role="uniform", candidate_role="uniform")
        items.append(item)

    summary = dict(plan.get("summary") or {})
    summary["policy_name"] = str(context["policy_name"])
    summary["num_final_keyframes"] = int(len(indices))
    summary["num_uniform_base"] = int(len(context["uniform_base_indices"]))
    base_count = int(len(context["uniform_base_indices"]))
    if base_count > 0:
        summary["extra_kf_ratio"] = float(max(0, len(indices) - base_count) / float(base_count))
    else:
        summary["extra_kf_ratio"] = 0.0

    return {
        "policy_name": str(context["policy_name"]),
        "frame_count_total": int(context["frame_count_total"]),
        "frame_count_used": int(context["frame_count_used"]),
        "tail_drop": int(context["tail_drop"]),
        "uniform_base_indices": list(context["uniform_base_indices"]),
        "keyframe_indices": indices,
        "keyframe_items": items,
        "summary": summary,
        "policy_meta": dict(plan.get("policy_meta") or {}),
    }


def build_keyframe_plan(root_dir, frames_dir, timestamps_path, kf_gap, policy_name):
    root_dir = Path(root_dir).resolve()
    frames_dir = Path(frames_dir).resolve()
    timestamps_path = Path(timestamps_path).resolve()

    frame_paths = list_frames_sorted(frames_dir)
    timestamps = _read_timestamps(timestamps_path)
    if len(frame_paths) != len(timestamps):
        raise ValueError(
            "frame count and timestamps count mismatch: frames={} timestamps={}".format(
                len(frame_paths), len(timestamps)
            )
        )
    if not frame_paths:
        raise ValueError("no frames found under {}".format(frames_dir))

    uniform_base = compute_uniform_base(len(frame_paths), int(kf_gap))
    context = {
        "root_dir": root_dir,
        "frames_dir": frames_dir,
        "timestamps_path": timestamps_path,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "frame_count_total": int(len(frame_paths)),
        "kf_gap": int(kf_gap),
        "used_last_idx": int(uniform_base["used_last_idx"]),
        "frame_count_used": int(uniform_base["used_count"]),
        "tail_drop": int(uniform_base["tail_drop"]),
        "uniform_base_indices": list(uniform_base["indices"]),
        "policy_name": str(policy_name),
        "policy_cache_dir": (root_dir / ".segment_cache" / str(policy_name)).resolve(),
        "build_item": _build_keyframe_item_factory(frame_paths, timestamps),
    }
    os.makedirs(str(context["policy_cache_dir"]), exist_ok=True)

    builder = get_policy_builder(policy_name)
    log_info("segment policy start: name={} uniform_base={}".format(policy_name, len(context["uniform_base_indices"])))
    plan = builder(context)
    return _normalize_plan(context, plan)


def materialize_keyframe_plan(root_dir, frames_dir, timestamps_path, kf_gap, keyframes_mode, policy_name):
    root_dir = Path(root_dir).resolve()
    frames_dir = Path(frames_dir).resolve()
    keyframes_dir = root_dir / "keyframes"
    plan = build_keyframe_plan(
        root_dir=root_dir,
        frames_dir=frames_dir,
        timestamps_path=timestamps_path,
        kf_gap=kf_gap,
        policy_name=policy_name,
    )

    actual_mode, bytes_sum = materialize_keyframes(
        frames_dir=frames_dir,
        keyframes_dir=keyframes_dir,
        keyframe_indices=plan["keyframe_indices"],
        mode_requested=keyframes_mode,
    )

    keyframes_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kf_gap": int(kf_gap),
        "mode_requested": str(keyframes_mode),
        "mode_actual": str(actual_mode),
        "frame_count_total": int(plan["frame_count_total"]),
        "frame_count_used": int(plan["frame_count_used"]),
        "tail_drop": int(plan["tail_drop"]),
        "keyframe_count": int(len(plan["keyframe_indices"])),
        "keyframe_indices": list(plan["keyframe_indices"]),
        "keyframe_bytes_sum": int(bytes_sum),
        "policy_name": str(plan["policy_name"]),
        "uniform_base_indices": list(plan.get("uniform_base_indices", [])),
        "keyframes": list(plan["keyframe_items"]),
        "summary": dict(plan["summary"]),
        "policy_meta": dict(plan.get("policy_meta") or {}),
        "note": "Keyframes remain backward compatible with the legacy uniform layout fields. semantic_guarded_v1 uses a uniform skeleton plus boundary/support adjustments.",
    }

    write_json_atomic(keyframes_dir / "keyframes_meta.json", keyframes_meta, indent=2)
    return keyframes_meta
