from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from exphub.common.io import list_frames_sorted

def _read_timestamps(path):
    values = []
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            values.append(float(text))
    return values


def _compute_uniform_base(frame_count_total, kf_gap):
    frame_count_total = int(frame_count_total)
    kf_gap = max(1, int(kf_gap))
    used_last_idx = ((frame_count_total - 1) // kf_gap) * kf_gap
    used_count = int(used_last_idx + 1)
    tail_drop = int(frame_count_total - used_count)
    indices = list(range(0, used_count, kf_gap))
    if not indices:
        indices = [0]
    return {
        "indices": indices,
        "used_last_idx": int(used_last_idx),
        "used_count": int(used_count),
        "tail_drop": int(tail_drop),
    }


def _build_keyframe_item_factory(frame_paths, timestamps):
    def _legacy_item_fields(source_role, legacy_meta):
        legacy_meta = dict(legacy_meta or {})
        return {
            "candidate_role": str(legacy_meta.get("candidate_role") or source_role),
            "is_inserted": bool(legacy_meta.get("is_inserted", False)),
            "is_relocated": bool(legacy_meta.get("is_relocated", False)),
            "replaced_uniform_index": (
                int(legacy_meta["replaced_uniform_index"])
                if legacy_meta.get("replaced_uniform_index") is not None
                else None
            ),
            "promotion_source": str(legacy_meta.get("promotion_source") or ""),
            "promotion_reason": str(legacy_meta.get("promotion_reason") or ""),
            "window_id": int(legacy_meta["window_id"]) if legacy_meta.get("window_id") is not None else None,
        }

    def _make_item(frame_idx, source_type, source_role, rerank_score=None, legacy_meta=None):
        path = frame_paths[int(frame_idx)]
        item = {
            "frame_idx": int(frame_idx),
            "file_name": path.name,
            "ts_sec": float(timestamps[int(frame_idx)]),
            "source_type": str(source_type),
            "source_role": str(source_role),
            "rerank_score": float(rerank_score) if rerank_score is not None else None,
        }
        item.update(_legacy_item_fields(source_role, legacy_meta))
        return item

    return _make_item


def _normalize_plan(context, plan):
    frame_count_total = int(context["frame_count_total"])
    plan_used_last_idx = plan.get("used_last_idx")
    plan_frame_count_used = plan.get("frame_count_used")
    if plan_used_last_idx is None and plan_frame_count_used is None:
        used_last_idx = int(context["used_last_idx"])
        frame_count_used = int(context["frame_count_used"])
    else:
        if plan_used_last_idx is None:
            frame_count_used = max(0, int(plan_frame_count_used))
            used_last_idx = int(frame_count_used - 1) if frame_count_used > 0 else -1
        else:
            used_last_idx = int(plan_used_last_idx)
            frame_count_used = (
                int(plan_frame_count_used)
                if plan_frame_count_used is not None
                else int(used_last_idx + 1)
            )
        used_last_idx = min(max(used_last_idx, -1), max(-1, frame_count_total - 1))
        frame_count_used = min(max(frame_count_used, 0), frame_count_total)

    tail_drop = int(plan.get("tail_drop", max(0, frame_count_total - frame_count_used)))
    uniform_base_indices = list(plan.get("uniform_base_indices") or context["uniform_base_indices"])
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
        if frame_idx < 0 or frame_idx > int(used_last_idx):
            continue
        seen.add(frame_idx)
        indices.append(frame_idx)
    indices.sort()

    build_item = context["build_item"]
    items = []
    for frame_idx in indices:
        item = item_map.get(frame_idx)
        if item is None:
            item = build_item(
                frame_idx,
                source_type="state",
                source_role="state",
                legacy_meta={
                    "candidate_role": "state",
                    "promotion_source": "state",
                },
            )
        items.append(item)

    summary = dict(plan.get("summary") or {})
    summary["policy_name"] = "state"
    summary["num_final_keyframes"] = int(len(indices))
    summary["num_uniform_base"] = int(len(uniform_base_indices))
    base_count = int(len(uniform_base_indices))
    summary["extra_kf_ratio"] = (
        float(max(0, len(indices) - base_count) / float(base_count))
        if base_count > 0
        else 0.0
    )

    return {
        "policy_name": "state",
        "frame_count_total": int(frame_count_total),
        "frame_count_used": int(frame_count_used),
        "tail_drop": int(tail_drop),
        "uniform_base_indices": list(uniform_base_indices),
        "keyframe_indices": indices,
        "keyframe_items": items,
        "summary": summary,
        "policy_meta": dict(plan.get("policy_meta") or {}),
    }


def run_state_mainline(segment_dir, frames_dir, timestamps_path, kf_gap):
    from ._state_policy import build_policy_plan

    segment_dir_path = Path(segment_dir).resolve()
    frames_dir_path = Path(frames_dir).resolve()
    timestamps_path = Path(timestamps_path).resolve()

    frame_paths = list_frames_sorted(frames_dir_path)
    timestamps = _read_timestamps(timestamps_path)
    if len(frame_paths) != len(timestamps):
        raise RuntimeError(
            "frame count and timestamps count mismatch: frames={} timestamps={}".format(
                len(frame_paths),
                len(timestamps),
            )
        )
    if not frame_paths:
        raise RuntimeError("no frames found under {}".format(frames_dir_path))

    uniform_base = _compute_uniform_base(len(frame_paths), int(kf_gap))
    context = {
        "root_dir": segment_dir_path,
        "frames_dir": frames_dir_path,
        "timestamps_path": timestamps_path,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "frame_count_total": int(len(frame_paths)),
        "kf_gap": int(kf_gap),
        "used_last_idx": int(uniform_base["used_last_idx"]),
        "frame_count_used": int(uniform_base["used_count"]),
        "tail_drop": int(uniform_base["tail_drop"]),
        "uniform_base_indices": list(uniform_base["indices"]),
        "policy_name": "state",
        "build_item": _build_keyframe_item_factory(frame_paths, timestamps),
    }
    policy_result = build_policy_plan(context)
    plan = _normalize_plan(context, policy_result)

    return {
        "plan": plan,
        "state_overview_path": policy_result.get("state_overview_path"),
        "state_segments_payload": dict(policy_result.get("state_segments_payload") or {}),
        "state_report_payload": dict(policy_result.get("state_report_payload") or {}),
    }


def _json_ready(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mainline", action="store_true")
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--timestamps_path", required=True)
    parser.add_argument("--kf_gap", type=int, required=True)
    parser.add_argument("--out_path", required=True)
    return parser


def main(argv=None):
    from exphub.common.io import write_json_atomic

    args = _build_arg_parser().parse_args(argv)
    if not args.run_mainline:
        raise SystemExit("state detector helper requires --run-mainline")
    result = run_state_mainline(
        segment_dir=args.segment_dir,
        frames_dir=args.frames_dir,
        timestamps_path=args.timestamps_path,
        kf_gap=int(args.kf_gap),
    )
    write_json_atomic(args.out_path, _json_ready(result), indent=2)
    return result


if __name__ == "__main__":
    main()
