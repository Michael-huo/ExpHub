from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from exphub.common.io import write_json_atomic


TRANSITION_SOURCE = "encode.export_router.transition_compat.v1"
DERIVED_FROM = [
    "motion_segments.json",
    "semantic_anchors.json",
    "generation_units.json",
    "prompts.json",
]


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _relative(exp_dir, target):
    root = Path(exp_dir).resolve()
    path = Path(target).resolve()
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _time_range(prepare_result):
    return dict(_as_dict(prepare_result.get("time_range")))


def _frame_times(prepare_result):
    frame_index_map = _as_dict(prepare_result.get("frame_index_map"))
    return list(frame_index_map.get("prepared_to_rel_time_sec") or frame_index_map.get("prepared_to_time_sec") or [])


def _frame_abs_times(prepare_result):
    return list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])


def _calib_values(prepare_result):
    intrinsics = _as_dict(prepare_result.get("normalized_intrinsics"))
    values = [
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    ]
    values.extend([float(item) for item in list(intrinsics.get("dist") or [])])
    return values


def _png_bytes_for_indices(frames_dir, indices):
    total = 0
    for idx in list(indices or []):
        path = Path(frames_dir).resolve() / "{:06d}.png".format(int(idx))
        try:
            total += int(path.stat().st_size)
        except Exception:
            pass
    return int(total)


def _motion_label_counts(motion_segments):
    counts = {}
    for item in list(_as_dict(motion_segments).get("segments") or []):
        label = str(_as_dict(item).get("motion_label", "") or "unknown")
        counts[label] = int(counts.get(label, 0)) + 1
    return counts


def _build_legacy_manifest(exp_dir, prepare_result, frames_dir, motion_segments, generation_units):
    exp_root = Path(exp_dir).resolve()
    prepare = _as_dict(prepare_result)
    legal_grid = _as_dict(prepare.get("legal_grid"))
    normalized_resolution = _as_dict(prepare.get("normalized_resolution"))
    times = _frame_times(prepare)
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    state_segments = []
    for idx, raw_segment in enumerate(list(_as_dict(motion_segments).get("segments") or [])):
        item = _as_dict(raw_segment)
        start_idx = int(item.get("start_idx", 0) or 0)
        end_idx = int(item.get("end_idx", start_idx) or start_idx)
        state_segments.append(
            {
                "segment_id": int(idx),
                "seg_id": str(item.get("seg_id", "") or "seg_{:04d}".format(int(idx))),
                "state_label": "motion_{}".format(str(item.get("motion_label", "mixed") or "mixed")),
                "motion_label": str(item.get("motion_label", "mixed") or "mixed"),
                "risk_level": "",
                "start_frame": int(start_idx),
                "end_frame": int(end_idx),
                "start_time": float(times[start_idx]) if start_idx < len(times) else 0.0,
                "end_time": float(times[end_idx]) if end_idx < len(times) else 0.0,
                "start_abs_time_sec": item.get("start_abs_time_sec"),
                "end_abs_time_sec": item.get("end_abs_time_sec"),
                "duration_frames": int(end_idx - start_idx + 1),
                "duration_sec": float((end_idx - start_idx) / float(max(int(prepare.get("target_fps", 1) or 1), 1))),
            }
        )
    frame_count = int(prepare.get("num_frames", 0) or 0)
    return {
        "version": 1,
        "schema": "segment_manifest.v1",
        "stage": "encode",
        "substage": "transition_compat",
        "transition_only": True,
        "source": TRANSITION_SOURCE,
        "derived_from": list(DERIVED_FROM),
        "policy": "encode_pass1_transition_compat",
        "inputs": {
            "fps": float(prepare.get("target_fps", 0) or 0),
            "duration": float(_time_range(prepare).get("dur_sec", 0.0) or 0.0),
            "start_sec": float(_time_range(prepare).get("start_sec", 0.0) or 0.0),
            "width": int(normalized_resolution.get("width", 0) or 0),
            "height": int(normalized_resolution.get("height", 0) or 0),
            "prepare_result": "prepare/prepare_result.json",
            "frames_dir": "prepare/frames",
        },
        "artifacts": {
            "prepare_result": "prepare/prepare_result.json",
            "motion_segments": "encode/motion_segments.json",
            "semantic_anchors": "encode/semantic_anchors.json",
            "generation_units": "encode/generation_units.json",
            "prompts": "encode/prompts.json",
        },
        "frames": {
            "frame_count": int(frame_count),
            "frame_count_used": int(frame_count),
            "tail_drop": 0,
        },
        "keyframes": {
            "count": int(len(legal_positions)),
            "indices": list(legal_positions),
            "bytes_sum": _png_bytes_for_indices(frames_dir, legal_positions),
        },
        "state_segments": {
            "version": 1,
            "source": TRANSITION_SOURCE,
            "transition_only": True,
            "segments": state_segments,
        },
        "state_report": {
            "version": 1,
            "source": TRANSITION_SOURCE,
            "transition_only": True,
            "summary": {
                "state_segment_count": int(len(state_segments)),
                "note": "motion segments translated for decode compatibility only",
            },
        },
        "camera": {
            "timestamps": [float(item) for item in times],
            "abs_timestamps": [float(item) for item in _frame_abs_times(prepare)],
            "calib": _calib_values(prepare),
        },
        "extraction": {
            "source": "prepare_result",
            "frame_count": int(frame_count),
            "timestamps_count": int(len(times)),
        },
        "quality_diagnostics": {
            "source": TRANSITION_SOURCE,
            "transition_only": True,
            "warnings": [],
        },
        "summary": {
            "frame_count": int(frame_count),
            "frame_count_used": int(frame_count),
            "keyframe_count": int(len(legal_positions)),
            "state_segment_count": int(len(state_segments)),
            "generation_unit_count": int(len(list(_as_dict(generation_units).get("units") or []))),
        },
        "timings_sec": {
            "prepare_reuse": 0.0,
            "total": 0.0,
        },
        "prepare_result": dict(prepare),
    }


def _build_encode_plan(prepare_result, generation_units):
    units = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        item = dict(_as_dict(raw_unit))
        item["transition_only"] = True
        item["source"] = TRANSITION_SOURCE
        units.append(item)
    frame_count = int(_as_dict(prepare_result).get("num_frames", 0) or 0)
    return {
        "schema": "encode_plan.v1",
        "stage": "encode",
        "transition_only": True,
        "source": TRANSITION_SOURCE,
        "derived_from": list(DERIVED_FROM),
        "planner": "generation_units",
        "prompt_strategy": "prompt_spans",
        "video": {
            "frame_count": int(frame_count),
            "frame_count_used": int(frame_count),
            "fps": _as_dict(prepare_result).get("target_fps"),
        },
        "sequence_range": dict(_as_dict(generation_units).get("sequence_range") or {}),
        "units": units,
        "summary": dict(_as_dict(generation_units).get("summary") or {}),
    }


def _build_prompt_spans(prompts, generation_units):
    prompt_by_unit = {str(item.get("unit_id", "") or ""): _as_dict(item) for item in list(_as_dict(prompts).get("units") or [])}
    spans = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        prompt_ref = _as_dict(unit.get("prompt_ref"))
        span_id = str(prompt_ref.get("span_id", "") or "")
        prompt = prompt_by_unit.get(str(unit.get("unit_id", "") or ""), {})
        spans.append(
            {
                "span_id": str(span_id),
                "transition_only": True,
                "source": TRANSITION_SOURCE,
                "scene_label": str(unit.get("scene_label", "") or ""),
                "motion_label": str(unit.get("motion_label", "") or ""),
                "anchor_start_idx": int(unit.get("start_idx", 0) or 0),
                "anchor_end_idx": int(unit.get("end_idx", 0) or 0),
                "unit_ids": [str(unit.get("unit_id", "") or "")],
                "source_segment_ids": list(unit.get("source_segment_ids") or []),
                "shared_unit_count": 1,
                "base_prompt": str(prompt.get("base_prompt", _as_dict(prompts).get("base_prompt", "")) or ""),
                "scene_prompt": str(prompt.get("semantic_prompt", "") or ""),
                "scene_prompt_source": "encode.synthetic_prompt.semantic_slot",
                "motion_prompt": str(prompt.get("motion_prompt", "") or ""),
                "motion_prompt_source": "encode.synthetic_prompt.motion_label_mapping",
                "negative_prompt_delta": "",
                "continuity_emphasis": "balanced",
                "resolved_prompt": str(prompt.get("assembled_prompt", "") or ""),
                "negative_prompt": str(prompt.get("negative_prompt", _as_dict(prompts).get("negative_prompt", "")) or ""),
            }
        )
    return {
        "version": 1,
        "schema": "prompt_spans.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "transition_only": True,
        "source": TRANSITION_SOURCE,
        "derived_from": list(DERIVED_FROM),
        "prompt_structure": "base_motion_semantic",
        "base_prompt": str(_as_dict(prompts).get("base_prompt", "") or ""),
        "negative_prompt": str(_as_dict(prompts).get("negative_prompt", "") or ""),
        "spans": spans,
        "summary": {
            "span_count": int(len(spans)),
            "shared_prompt_unit_count": int(len(spans)),
            "multi_unit_span_count": 0,
        },
    }


def _json_bytes(obj):
    return len(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def _build_encode_report(exp_dir, motion_segments, semantic_anchors, generation_units, prompts, prompt_spans, elapsed_sec):
    return {
        "schema": "encode_report.v1",
        "stage": "encode",
        "status": "success",
        "transition_only": True,
        "source": TRANSITION_SOURCE,
        "derived_from": list(DERIVED_FROM),
        "timings_sec": {
            "encode_total": float(elapsed_sec),
        },
        "counts": {
            "motion_segments": int(len(list(_as_dict(motion_segments).get("segments") or []))),
            "semantic_anchor_groups": int(len(list(_as_dict(semantic_anchors).get("segments") or []))),
            "units": int(len(list(_as_dict(generation_units).get("units") or []))),
            "prompt_spans": int(len(list(prompt_spans.get("spans") or []))),
        },
        "config": {
            "planner": "generation_units",
            "prompt_strategy": "prompt_spans",
            "note": "transition-only report for legacy eval readers",
        },
        "artifacts": {
            "motion_segments": "encode/motion_segments.json",
            "semantic_anchors": "encode/semantic_anchors.json",
            "generation_units": "encode/generation_units.json",
            "prompts": "encode/prompts.json",
            "legacy_segment_manifest": "encode/legacy_segment_manifest.json",
            "encode_plan": "encode/encode_plan.json",
            "prompt_spans": "encode/prompt_spans.json",
            "encode_report": "encode/encode_report.json",
        },
        "outputs": {
            "prompt_bytes": int(_json_bytes(prompt_spans)),
            "bytes_sum": int(_json_bytes(prompt_spans)),
        },
    }


def _build_encode_result(motion_segments, semantic_anchors, generation_units, prompts):
    units = list(_as_dict(generation_units).get("units") or [])
    return {
        "version": 1,
        "source": "encode.result.v1",
        "num_motion_segments": int(len(list(_as_dict(motion_segments).get("segments") or []))),
        "num_semantic_anchor_groups": int(len(list(_as_dict(semantic_anchors).get("segments") or []))),
        "num_generation_units": int(len(units)),
        "motion_labels": _motion_label_counts(motion_segments),
        "unit_lengths": [int(item.get("length", item.get("duration_frames", 0)) or 0) for item in units],
        "prompt_mode": "base+motion+semantic",
        "legacy_manifest_path": "encode/legacy_segment_manifest.json",
        "transition_artifacts": [
            "encode/segment_manifest.json",
            "encode/encode_plan.json",
            "encode/prompt_spans.json",
            "encode/encode_report.json",
        ],
    }


def write_encode_overview(output_path, motion_segments, semantic_anchors, generation_units):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion_colors = {
        "stop": "#7f8c8d",
        "forward": "#2ca02c",
        "left_turn": "#1f77b4",
        "right_turn": "#d62728",
        "mixed": "#9467bd",
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    ax_motion, ax_semantic, ax_units = axes

    max_idx = 0
    for segment in list(_as_dict(motion_segments).get("segments") or []):
        start = int(segment.get("start_idx", 0) or 0)
        end = int(segment.get("end_idx", start) or start)
        label = str(segment.get("motion_label", "mixed") or "mixed")
        max_idx = max(max_idx, end)
        ax_motion.axvspan(start, end, color=motion_colors.get(label, "#8c564b"), alpha=0.35)
        ax_motion.text((start + end) / 2.0, 0.5, label, ha="center", va="center", fontsize=9)
    ax_motion.set_yticks([])
    ax_motion.set_ylabel("Motion")
    ax_motion.set_title("Motion segmentation")

    gap_rows = list(_as_dict(semantic_anchors).get("gap_rows") or [])
    if gap_rows:
        xs = [int(item.get("frame_idx", 0) or 0) for item in gap_rows]
        ys = [float(item.get("score", 0.0) or 0.0) for item in gap_rows]
        ax_semantic.plot(xs, ys, color="#444444", linewidth=1.2, label="semantic gap")
    reason_colors = {
        "segment_boundary": "#333333",
        "semantic_gain": "#e377c2",
        "duration_fallback": "#ff7f0e",
    }
    seen_reasons = set()
    for group in list(_as_dict(semantic_anchors).get("segments") or []):
        for item in list(_as_dict(group).get("anchor_items") or []):
            idx = int(item.get("frame_idx", 0) or 0)
            reason = str(item.get("reason", "") or "segment_boundary")
            score = float(item.get("score", 0.0) or 0.0)
            max_idx = max(max_idx, idx)
            label = reason if reason not in seen_reasons else None
            seen_reasons.add(reason)
            ax_semantic.axvline(idx, color=reason_colors.get(reason, "#17becf"), alpha=0.45, linewidth=1)
            ax_semantic.scatter([idx], [score], color=reason_colors.get(reason, "#17becf"), s=25, label=label)
    ax_semantic.set_ylabel("Anchor score")
    ax_semantic.set_title("Semantic anchors")
    if seen_reasons:
        ax_semantic.legend(loc="upper right", fontsize=8)

    for unit_idx, unit in enumerate(list(_as_dict(generation_units).get("units") or [])):
        start = int(unit.get("start_idx", 0) or 0)
        end = int(unit.get("end_idx", start) or start)
        label = "{} {}-{} len={}".format(str(unit.get("motion_label", "") or ""), start, end, int(unit.get("length", 0) or 0))
        max_idx = max(max_idx, end)
        color = "#17becf" if unit_idx % 2 == 0 else "#bcbd22"
        ax_units.axvspan(start, end, color=color, alpha=0.35)
        ax_units.text((start + end) / 2.0, 0.5, label, ha="center", va="center", fontsize=8)
    ax_units.set_yticks([])
    ax_units.set_ylabel("Units")
    ax_units.set_title("Generation units")
    ax_units.set_xlabel("frame_idx")
    ax_units.set_xlim(0, max(1, int(max_idx)))
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=140)
    plt.close(fig)


def export_encode_outputs(
    runtime,
    prepare_result,
    motion_segments,
    semantic_anchors,
    generation_units,
    prompts,
    elapsed_sec=0.0,
):
    exp_dir = runtime.paths.exp_dir
    encode_dir = runtime.paths.encode_dir
    encode_dir.mkdir(parents=True, exist_ok=True)

    legacy_manifest = _build_legacy_manifest(
        exp_dir=exp_dir,
        prepare_result=prepare_result,
        frames_dir=runtime.paths.prepare_frames_dir,
        motion_segments=motion_segments,
        generation_units=generation_units,
    )
    encode_plan = _build_encode_plan(prepare_result, generation_units)
    prompt_spans = _build_prompt_spans(prompts, generation_units)
    encode_report = _build_encode_report(exp_dir, motion_segments, semantic_anchors, generation_units, prompts, prompt_spans, elapsed_sec)
    encode_result = _build_encode_result(motion_segments, semantic_anchors, generation_units, prompts)

    write_json_atomic(encode_dir / "motion_segments.json", motion_segments, indent=2)
    write_json_atomic(encode_dir / "semantic_anchors.json", semantic_anchors, indent=2)
    write_json_atomic(encode_dir / "generation_units.json", generation_units, indent=2)
    write_json_atomic(encode_dir / "prompts.json", prompts, indent=2)
    write_json_atomic(encode_dir / "encode_result.json", encode_result, indent=2)
    write_json_atomic(runtime.paths.encode_legacy_manifest_path, legacy_manifest, indent=2)
    write_json_atomic(runtime.paths.segment_manifest_path, legacy_manifest, indent=2)
    write_json_atomic(runtime.paths.encode_plan_path, encode_plan, indent=2)
    write_json_atomic(runtime.paths.prompt_spans_path, prompt_spans, indent=2)
    write_json_atomic(runtime.paths.encode_report_path, encode_report, indent=2)
    write_encode_overview(encode_dir / "encode_overview.png", motion_segments, semantic_anchors, generation_units)
    shutil.copy2(str(encode_dir / "encode_overview.png"), str(runtime.paths.encode_segmentation_overview_path))
    return encode_dir / "encode_result.json"
