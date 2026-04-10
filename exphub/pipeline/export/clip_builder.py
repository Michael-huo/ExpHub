from __future__ import annotations

import json
import subprocess
from pathlib import Path

from exphub.common.types import canon_num_str, dot_to_p, sanitize_token


DEFAULT_EXPORT_FPS = 24
DEFAULT_EXPORT_NUM_FRAMES = 73
DEFAULT_EXPORT_WIDTH = 832
DEFAULT_EXPORT_HEIGHT = 480
TRAINING_BASE_CAPTION = "first-person viewpoint, stable geometry, temporal consistency"


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip(" ,;:.")


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _unique_texts(values, limit):
    out = []
    seen = set()
    for raw_value in list(values or []):
        text = _collapse_ws(raw_value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= int(limit):
            break
    return out


def build_export_profile(
    target_fps=DEFAULT_EXPORT_FPS,
    target_num_frames=DEFAULT_EXPORT_NUM_FRAMES,
    target_width=DEFAULT_EXPORT_WIDTH,
    target_height=DEFAULT_EXPORT_HEIGHT,
    harvest_sec=None,
    stride_sec=None,
):
    target_fps = int(target_fps)
    target_num_frames = int(target_num_frames)
    target_width = int(target_width)
    target_height = int(target_height)
    if target_fps <= 0:
        raise RuntimeError("export target_fps must be > 0")
    if target_num_frames <= 1:
        raise RuntimeError("export target_num_frames must be > 1")
    if target_width <= 0 or target_height <= 0:
        raise RuntimeError("export target resolution must be > 0")

    target_duration_sec = float(target_num_frames - 1) / float(target_fps)
    if harvest_sec is None or float(harvest_sec) <= 0.0:
        harvest_sec = float(target_duration_sec)
    if stride_sec is None or float(stride_sec) <= 0.0:
        stride_sec = float(target_duration_sec)
    step_frames = max(1, int(round(float(stride_sec) * float(target_fps))))

    return {
        "target_fps": int(target_fps),
        "target_num_frames": int(target_num_frames),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "target_duration_sec": float(target_duration_sec),
        "harvest_sec": float(harvest_sec),
        "stride_sec": float(stride_sec),
        "step_frames": int(step_frames),
    }


def training_spec(profile=None):
    profile_obj = dict(profile or build_export_profile())
    return {
        "target_fps": int(profile_obj.get("target_fps", DEFAULT_EXPORT_FPS)),
        "target_num_frames": int(profile_obj.get("target_num_frames", DEFAULT_EXPORT_NUM_FRAMES)),
        "target_width": int(profile_obj.get("target_width", DEFAULT_EXPORT_WIDTH)),
        "target_height": int(profile_obj.get("target_height", DEFAULT_EXPORT_HEIGHT)),
        "target_duration_sec": float(profile_obj.get("target_duration_sec", 0.0) or 0.0),
        "harvest_sec": float(profile_obj.get("harvest_sec", 0.0) or 0.0),
        "stride_sec": float(profile_obj.get("stride_sec", 0.0) or 0.0),
        "step_frames": int(profile_obj.get("step_frames", 1) or 1),
    }


def _load_manifest(path, artifact_name):
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("{} must be a JSON object: {}".format(artifact_name, Path(path).resolve()))
    return payload


def _load_aligned_segment_plan(segment_manifest, exp_dir):
    manifest = _as_dict(segment_manifest)
    exp_root = Path(exp_dir).resolve()
    artifact_path = str(
        _as_dict(manifest.get("aligned_segment_plan")).get("path", "")
        or _as_dict(manifest.get("artifacts")).get("aligned_segment_plan", "")
        or ""
    ).strip()
    if not artifact_path:
        raise RuntimeError("segment manifest missing aligned_segment_plan artifact path")
    plan_path = (exp_root / artifact_path).resolve()
    payload = _load_manifest(plan_path, "aligned segment plan")
    segments = list(payload.get("segments") or [])
    if not segments:
        raise RuntimeError("aligned segment plan contains zero segments: {}".format(plan_path))
    return payload, plan_path


def _load_generation_units_payload(exp_dir):
    path = (Path(exp_dir).resolve() / "segment" / "generation_units.json").resolve()
    payload = _load_manifest(path, "generation units")
    if not list(payload.get("units") or []):
        raise RuntimeError("generation units contains zero units: {}".format(path))
    return payload, path


def _load_prompt_spans_payload(exp_dir):
    path = (Path(exp_dir).resolve() / "prompt" / "prompt_spans.json").resolve()
    payload = _load_manifest(path, "prompt spans")
    if not list(payload.get("spans") or []):
        raise RuntimeError("prompt spans contains zero spans: {}".format(path))
    return payload, path


def _prompt_segment_map(prompt_manifest):
    mapping = {}
    for raw_item in list(_as_dict(prompt_manifest).get("segments") or []):
        item = _as_dict(raw_item)
        state_segment_id = _safe_int(item.get("state_segment_id"), -1)
        if state_segment_id < 0:
            continue
        mapping[int(state_segment_id)] = item
    return mapping


def _prompt_signature(prompt_item):
    item = _as_dict(prompt_item)
    return (
        _collapse_ws(item.get("scene_prompt", "")).lower(),
        _collapse_ws(item.get("motion_prompt", "")).lower(),
    )


def _is_contiguous_neighbor(left_item, right_item):
    left = _as_dict(left_item)
    right = _as_dict(right_item)
    left_end = _safe_int(left.get("aligned_end_idx"), -1)
    right_start = _safe_int(right.get("aligned_start_idx"), -1)
    return int(right_start) == int(left_end)


def _build_training_caption_from_segments(prompt_segments):
    scene_phrases = _unique_texts([_as_dict(item).get("scene_prompt") for item in prompt_segments], limit=2)
    motion_phrases = _unique_texts([_as_dict(item).get("motion_prompt") for item in prompt_segments], limit=2)
    parts = [TRAINING_BASE_CAPTION]
    parts.extend(scene_phrases)
    parts.extend(motion_phrases)
    return ", ".join([item for item in parts if item]).strip()


def _choose_centered_window(start_idx, end_idx, target_num_frames, anchor_center=None):
    start = int(start_idx)
    end = int(end_idx)
    target = int(target_num_frames)
    available = int(end - start + 1)
    if target <= 0 or available < target:
        raise RuntimeError("cannot derive centered window: {}..{} target={}".format(start, end, target))
    if anchor_center is None:
        anchor_center = 0.5 * float(start + end)
    clip_start = int(round(float(anchor_center) - 0.5 * float(target - 1)))
    clip_start = max(int(start), min(int(clip_start), int(end - target + 1)))
    clip_end = int(clip_start + target - 1)
    return int(clip_start), int(clip_end)


def _trace_segments(rows, clip_start_idx, clip_end_idx):
    traces = []
    prev_end = None
    for row in list(rows or []):
        seg = _as_dict(row)
        aligned_start = _safe_int(seg.get("aligned_start_idx"), 0)
        aligned_end = _safe_int(seg.get("aligned_end_idx"), -1)
        unique_start = int(aligned_start)
        if prev_end is not None and int(aligned_start) == int(prev_end):
            unique_start = int(aligned_start) + 1
        overlap_start = max(int(clip_start_idx), int(unique_start))
        overlap_end = min(int(clip_end_idx), int(aligned_end))
        if overlap_end < overlap_start:
            prev_end = int(aligned_end)
            continue
        traces.append(
            {
                "segment_id": int(_safe_int(seg.get("segment_id"), 0)),
                "state_label": str(seg.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "raw_start_idx": int(_safe_int(seg.get("raw_start_idx"), aligned_start)),
                "raw_end_idx": int(_safe_int(seg.get("raw_end_idx"), aligned_end)),
                "aligned_start_idx": int(aligned_start),
                "aligned_end_idx": int(aligned_end),
                "aligned_num_frames": int(_safe_int(seg.get("aligned_num_frames"), aligned_end - aligned_start + 1)),
                "shared_boundary_with_previous": bool(prev_end is not None and int(aligned_start) == int(prev_end)),
                "used_start_idx": int(overlap_start),
                "used_end_idx": int(overlap_end),
                "used_num_frames": int(overlap_end - overlap_start + 1),
                "left_shift": int(_safe_int(seg.get("left_shift"), 0)),
                "right_shift": int(_safe_int(seg.get("right_shift"), 0)),
                "align_reason": str(seg.get("align_reason", "") or ""),
            }
        )
        prev_end = int(aligned_end)
    return traces


def _trace_generation_units(rows, clip_start_idx, clip_end_idx):
    traces = []
    prev_end = None
    for raw_item in list(rows or []):
        unit = _as_dict(raw_item)
        anchor_start = _safe_int(unit.get("anchor_start_idx"), 0)
        anchor_end = _safe_int(unit.get("anchor_end_idx"), -1)
        unique_start = int(anchor_start)
        if prev_end is not None and int(anchor_start) == int(prev_end):
            unique_start = int(anchor_start) + 1
        overlap_start = max(int(clip_start_idx), int(unique_start))
        overlap_end = min(int(clip_end_idx), int(anchor_end))
        if overlap_end < overlap_start:
            prev_end = int(anchor_end)
            continue
        traces.append(
            {
                "unit_id": str(unit.get("unit_id", "") or ""),
                "anchor_start_idx": int(anchor_start),
                "anchor_end_idx": int(anchor_end),
                "duration_frames": int(_safe_int(unit.get("duration_frames"), anchor_end - anchor_start + 1)),
                "motion_label": str(unit.get("motion_label", "steady") or "steady"),
                "scene_label": str(unit.get("scene_label", "scene_group_000") or "scene_group_000"),
                "risk_level": str(unit.get("risk_level", "low") or "low"),
                "shared_anchor_with_previous": bool(prev_end is not None and int(anchor_start) == int(prev_end)),
                "used_start_idx": int(overlap_start),
                "used_end_idx": int(overlap_end),
                "used_num_frames": int(overlap_end - overlap_start + 1),
            }
        )
        prev_end = int(anchor_end)
    return traces


def _window_start_indices(span_start_idx, span_end_idx, target_num_frames, step_frames):
    start_idx = int(span_start_idx)
    end_idx = int(span_end_idx)
    target_frames = int(target_num_frames)
    step_frames = max(1, int(step_frames))
    max_clip_start = int(end_idx - target_frames + 1)
    if max_clip_start < int(start_idx):
        return []
    starts = []
    clip_start = int(start_idx)
    while clip_start <= max_clip_start:
        starts.append(int(clip_start))
        clip_start += int(step_frames)
    if not starts:
        starts.append(int(start_idx))
    return starts


def _select_aligned_training_candidates(segment_manifest, prompt_manifest, exp_dir, profile):
    segment_obj = _as_dict(segment_manifest)
    prompt_obj = _as_dict(prompt_manifest)
    spec = training_spec(profile)

    frame_count = int(_as_dict(segment_obj.get("frames")).get("frame_count_used", 0) or _as_dict(segment_obj.get("frames")).get("frame_count", 0) or 0)
    keyframe_count = int(_as_dict(segment_obj.get("keyframes")).get("count", 0) or 0)
    prompt_segments = list(prompt_obj.get("segments") or [])
    if frame_count < int(spec["target_num_frames"]):
        return {
            "candidates": [],
            "rejections": [{"reason": "frame_count_too_short:{}<{}".format(frame_count, int(spec["target_num_frames"]))}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "clip_count": 0,
            },
        }
    if keyframe_count < 2:
        return {
            "candidates": [],
            "rejections": [{"reason": "insufficient_keyframes:{}".format(keyframe_count)}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "clip_count": 0,
            },
        }
    if str(prompt_obj.get("prompt_structure", "") or "") != "base_scene_motion":
        return {
            "candidates": [],
            "rejections": [{"reason": "unsupported_prompt_structure:{}".format(str(prompt_obj.get("prompt_structure", "") or "<missing>"))}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "clip_count": 0,
            },
        }
    if not prompt_segments:
        return {
            "candidates": [],
            "rejections": [{"reason": "missing_prompt_segments"}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "clip_count": 0,
            },
        }

    aligned_plan, aligned_plan_path = _load_aligned_segment_plan(segment_obj, exp_dir)
    prompt_by_state = _prompt_segment_map(prompt_obj)
    aligned_rows = []
    for raw_item in list(aligned_plan.get("segments") or []):
        seg = _as_dict(raw_item)
        if not bool(seg.get("is_valid_for_export", False)):
            continue
        segment_id = _safe_int(seg.get("segment_id"), -1)
        prompt_item = _as_dict(prompt_by_state.get(segment_id))
        if not prompt_item:
            continue
        seg["scene_prompt"] = str(prompt_item.get("scene_prompt", "") or "")
        seg["motion_prompt"] = str(prompt_item.get("motion_prompt", "") or "")
        seg["resolved_prompt"] = str(prompt_item.get("resolved_prompt", "") or "")
        seg["negative_prompt"] = str(prompt_item.get("negative_prompt", "") or "")
        aligned_rows.append(seg)
    if not aligned_rows:
        return {
            "candidates": [],
            "rejections": [{"reason": "aligned_segments_missing_prompt_support"}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "clip_count": 0,
            },
        }

    target_frames = int(spec["target_num_frames"])
    chosen_rows = None
    clip_start_idx = None
    clip_end_idx = None
    selection_reason = ""

    for seg in aligned_rows:
        aligned_num_frames = _safe_int(seg.get("aligned_num_frames"), 0)
        if aligned_num_frames < target_frames:
            continue
        clip_start_idx, clip_end_idx = _choose_centered_window(
            _safe_int(seg.get("aligned_start_idx"), 0),
            _safe_int(seg.get("aligned_end_idx"), -1),
            target_frames,
        )
        chosen_rows = [seg]
        selection_reason = "single_aligned_segment"
        break

    if chosen_rows is None:
        idx = 0
        while idx < len(aligned_rows):
            current = aligned_rows[idx]
            group = [current]
            signature = _prompt_signature(current)
            j = idx + 1
            while (
                j < len(aligned_rows)
                and _prompt_signature(aligned_rows[j]) == signature
                and _is_contiguous_neighbor(aligned_rows[j - 1], aligned_rows[j])
            ):
                group.append(aligned_rows[j])
                j += 1
            group_total = int(_safe_int(group[-1].get("aligned_end_idx"), -1) - _safe_int(group[0].get("aligned_start_idx"), 0) + 1)
            if group_total >= target_frames:
                anchor_center = 0.5 * float(
                    _safe_int(current.get("aligned_start_idx"), 0) + _safe_int(current.get("aligned_end_idx"), 0)
                )
                clip_start_idx, clip_end_idx = _choose_centered_window(
                    _safe_int(group[0].get("aligned_start_idx"), 0),
                    _safe_int(group[-1].get("aligned_end_idx"), -1),
                    target_frames,
                    anchor_center=anchor_center,
                )
                chosen_rows = group
                selection_reason = "same_scene_motion_neighborhood_extension"
                break
            idx = j

    if chosen_rows is None:
        return {
            "candidates": [],
            "rejections": [{"reason": "no_aligned_exportable_clip"}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "aligned_segment_count": int(_as_dict(aligned_plan.get("summary")).get("segment_count", 0) or 0),
                "clip_count": 0,
            },
        }

    trace_segments = _trace_segments(chosen_rows, clip_start_idx, clip_end_idx)
    if sum([int(item.get("used_num_frames", 0) or 0) for item in trace_segments]) != target_frames:
        return {
            "candidates": [],
            "rejections": [{"reason": "clip_trace_frame_mismatch"}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "aligned_segment_count": int(_as_dict(aligned_plan.get("summary")).get("segment_count", 0) or 0),
                "clip_count": 0,
            },
        }

    prompt_trace = []
    for seg in chosen_rows:
        prompt_trace.append(
            {
                "segment_id": int(_safe_int(seg.get("segment_id"), 0)),
                "scene_prompt": str(seg.get("scene_prompt", "") or ""),
                "motion_prompt": str(seg.get("motion_prompt", "") or ""),
                "state_label": str(seg.get("state_label", "state_unlabeled") or "state_unlabeled"),
            }
        )
    caption = _build_training_caption_from_segments(prompt_trace)
    if not caption:
        return {
            "candidates": [],
            "rejections": [{"reason": "empty_training_caption"}],
            "summary": {
                "frame_count": int(frame_count),
                "keyframe_count": int(keyframe_count),
                "prompt_segment_count": int(len(prompt_segments)),
                "aligned_segment_count": int(_as_dict(aligned_plan.get("summary")).get("segment_count", 0) or 0),
                "clip_count": 0,
            },
        }

    clip_manifest = {
        "version": 2,
        "schema": "train_clip_manifest.v2",
        "source": "export.clip_builder",
        "export_source": "aligned",
        "prompt_structure": "base_scene_motion",
        "clip_id": "aligned_clip_000",
        "train_start_idx": int(clip_start_idx),
        "train_end_idx": int(clip_end_idx),
        "target_num_frames": int(target_frames),
        "target_fps": int(spec["target_fps"]),
        "resolved_prompt": str(caption),
        "selection_reason": str(selection_reason),
        "source_files": {
            "aligned_segment_plan": str(Path(aligned_plan_path).resolve()),
        },
        "trace": {
            "segment_ids": [int(item.get("segment_id", 0) or 0) for item in trace_segments],
            "segments": trace_segments,
            "prompt_trace": prompt_trace,
        },
    }

    candidate = {
        "clip_id": "aligned_clip_000",
        "caption": str(caption),
        "resolved_prompt": str(caption),
        "clip_start_idx": int(clip_start_idx),
        "clip_end_idx": int(clip_end_idx),
        "clip_num_frames": int(target_frames),
        "selection_reason": str(selection_reason),
        "clip_manifest": clip_manifest,
        "summary": {
            "frame_count": int(frame_count),
            "keyframe_count": int(keyframe_count),
            "prompt_segment_count": int(len(prompt_segments)),
            "state_segment_count": int(_as_dict(segment_obj.get("summary")).get("state_segment_count", 0) or 0),
            "aligned_segment_count": int(_as_dict(aligned_plan.get("summary")).get("segment_count", 0) or 0),
            "clip_segment_count": int(len(trace_segments)),
        },
        "sample_index": None,
    }
    return {
        "candidates": [candidate],
        "rejections": [],
        "summary": dict(candidate.get("summary") or {}),
    }


def _select_generation_unit_training_candidates(
    segment_manifest,
    exp_dir,
    profile,
    generation_units_payload=None,
    prompt_spans_payload=None,
):
    segment_obj = _as_dict(segment_manifest)
    spec = training_spec(profile)
    frame_count = int(
        _as_dict(segment_obj.get("frames")).get("frame_count_used", 0)
        or _as_dict(segment_obj.get("frames")).get("frame_count", 0)
        or 0
    )
    if frame_count < int(spec["target_num_frames"]):
        return {
            "candidates": [],
            "rejections": [{"reason": "frame_count_too_short:{}<{}".format(frame_count, int(spec["target_num_frames"]))}],
            "summary": {
                "frame_count": int(frame_count),
                "clip_count": 0,
                "skipped_short_span_count": 0,
            },
        }

    generation_units_obj, generation_units_path = (
        (dict(generation_units_payload or {}), None)
        if generation_units_payload is not None
        else _load_generation_units_payload(exp_dir)
    )
    prompt_spans_obj, prompt_spans_path = (
        (dict(prompt_spans_payload or {}), None)
        if prompt_spans_payload is not None
        else _load_prompt_spans_payload(exp_dir)
    )
    if generation_units_path is None:
        generation_units_path = (Path(exp_dir).resolve() / "segment" / "generation_units.json").resolve()
    if prompt_spans_path is None:
        prompt_spans_path = (Path(exp_dir).resolve() / "prompt" / "prompt_spans.json").resolve()

    units = list(_as_dict(generation_units_obj).get("units") or [])
    spans = list(_as_dict(prompt_spans_obj).get("spans") or [])
    if not units:
        raise RuntimeError("generation_units export requires non-empty generation_units.json")
    if not spans:
        raise RuntimeError("generation_units export requires non-empty prompt_spans.json")

    unit_map = {}
    span_units_map = {}
    for raw_unit in units:
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        if not unit_id:
            raise RuntimeError("generation unit missing unit_id")
        unit_map[unit_id] = unit
        prompt_ref = _as_dict(unit.get("prompt_ref"))
        span_id = str(prompt_ref.get("span_id", "") or "")
        if not span_id:
            raise RuntimeError("generation unit missing prompt_ref.span_id")
        span_units_map.setdefault(span_id, []).append(unit)

    candidates = []
    rejections = []
    skipped_short_span_count = 0
    sample_counter = 0

    for raw_span in spans:
        span = _as_dict(raw_span)
        span_id = str(span.get("span_id", "") or "")
        if not span_id:
            raise RuntimeError("prompt span missing span_id")
        span_units = list(span_units_map.get(span_id) or [])
        if not span_units:
            rejections.append({"reason": "span_has_no_generation_units", "source_span_id": str(span_id)})
            continue
        span_units.sort(key=lambda item: (_safe_int(item.get("anchor_start_idx"), 0), str(item.get("unit_id", "") or "")))

        span_start_idx = _safe_int(span.get("anchor_start_idx"), _safe_int(span_units[0].get("anchor_start_idx"), 0))
        span_end_idx = _safe_int(span.get("anchor_end_idx"), _safe_int(span_units[-1].get("anchor_end_idx"), -1))
        span_num_frames = int(span_end_idx - span_start_idx + 1)
        if span_num_frames < int(spec["target_num_frames"]):
            skipped_short_span_count += 1
            rejections.append(
                {
                    "reason": "span_length_insufficient",
                    "source_span_id": str(span_id),
                    "span_num_frames": int(span_num_frames),
                    "target_num_frames": int(spec["target_num_frames"]),
                }
            )
            continue

        resolved_prompt = str(span.get("resolved_prompt", "") or "")
        if not resolved_prompt:
            rejections.append({"reason": "span_missing_resolved_prompt", "source_span_id": str(span_id)})
            continue

        span_window_starts = _window_start_indices(
            span_start_idx=span_start_idx,
            span_end_idx=span_end_idx,
            target_num_frames=int(spec["target_num_frames"]),
            step_frames=int(spec["step_frames"]),
        )
        for span_sample_index, clip_start_idx in enumerate(span_window_starts):
            clip_end_idx = int(clip_start_idx + int(spec["target_num_frames"]) - 1)
            trace_units = _trace_generation_units(span_units, clip_start_idx, clip_end_idx)
            if sum([int(item.get("used_num_frames", 0) or 0) for item in trace_units]) != int(spec["target_num_frames"]):
                rejections.append(
                    {
                        "reason": "generation_unit_trace_frame_mismatch",
                        "source_span_id": str(span_id),
                        "train_start_idx": int(clip_start_idx),
                        "train_end_idx": int(clip_end_idx),
                    }
                )
                continue

            source_unit_ids = [str(item.get("unit_id", "") or "") for item in trace_units if str(item.get("unit_id", "") or "")]
            clip_id = "{}_sample_{:03d}".format(str(span_id), int(sample_counter))
            clip_manifest = {
                "version": 2,
                "schema": "train_clip_manifest.v2",
                "source": "export.clip_builder",
                "export_source": "generation_units",
                "prompt_structure": "base_scene_motion",
                "clip_id": str(clip_id),
                "train_start_idx": int(clip_start_idx),
                "train_end_idx": int(clip_end_idx),
                "target_num_frames": int(spec["target_num_frames"]),
                "target_fps": int(spec["target_fps"]),
                "resolved_prompt": str(resolved_prompt),
                "source_unit_ids": list(source_unit_ids),
                "source_span_id": str(span_id),
                "source_prompt_ref": {
                    "artifact_path": "prompt/prompt_spans.json",
                    "span_id": str(span_id),
                },
                "source_files": {
                    "generation_units": str(Path(generation_units_path).resolve()),
                    "prompt_spans": str(Path(prompt_spans_path).resolve()),
                },
                "trace": {
                    "units": trace_units,
                    "span": {
                        "span_id": str(span_id),
                        "anchor_start_idx": int(span_start_idx),
                        "anchor_end_idx": int(span_end_idx),
                        "shared_unit_count": int(_safe_int(span.get("shared_unit_count"), len(span_units))),
                        "scene_label": str(span.get("scene_label", "scene_group_000") or "scene_group_000"),
                        "motion_label": str(span.get("motion_label", "steady") or "steady"),
                    },
                },
            }
            candidates.append(
                {
                    "clip_id": str(clip_id),
                    "caption": str(resolved_prompt),
                    "resolved_prompt": str(resolved_prompt),
                    "clip_start_idx": int(clip_start_idx),
                    "clip_end_idx": int(clip_end_idx),
                    "clip_num_frames": int(spec["target_num_frames"]),
                    "selection_reason": "generation_unit_span_sliding_window",
                    "clip_manifest": clip_manifest,
                    "source_unit_ids": list(source_unit_ids),
                    "source_span_id": str(span_id),
                    "source_prompt_ref": {
                        "artifact_path": "prompt/prompt_spans.json",
                        "span_id": str(span_id),
                    },
                    "resolved_prompt_source": "prompt_spans",
                    "sample_index": int(sample_counter),
                    "summary": {
                        "frame_count": int(frame_count),
                        "generation_unit_count": int(len(units)),
                        "prompt_span_count": int(len(spans)),
                        "clip_unit_count": int(len(source_unit_ids)),
                        "span_shared_unit_count": int(_safe_int(span.get("shared_unit_count"), len(span_units))),
                    },
                }
            )
            sample_counter += 1
            _ = span_sample_index

    return {
        "candidates": candidates,
        "rejections": rejections,
        "summary": {
            "frame_count": int(frame_count),
            "generation_unit_count": int(len(units)),
            "prompt_span_count": int(len(spans)),
            "clip_count": int(len(candidates)),
            "skipped_short_span_count": int(skipped_short_span_count),
        },
    }


def select_training_candidates(
    export_source,
    segment_manifest,
    prompt_manifest,
    exp_dir,
    profile,
    generation_units_payload=None,
    prompt_spans_payload=None,
):
    export_source = str(export_source or "aligned").strip().lower() or "aligned"
    if export_source == "generation_units":
        return _select_generation_unit_training_candidates(
            segment_manifest=segment_manifest,
            exp_dir=exp_dir,
            profile=profile,
            generation_units_payload=generation_units_payload,
            prompt_spans_payload=prompt_spans_payload,
        )
    if export_source != "aligned":
        raise RuntimeError("unsupported export source: {}".format(export_source))
    return _select_aligned_training_candidates(
        segment_manifest=segment_manifest,
        prompt_manifest=prompt_manifest,
        exp_dir=exp_dir,
        profile=profile,
    )


def build_clip_filename(dataset, sequence, clip_index, start_sec, sample_index=None):
    start_tag = dot_to_p(canon_num_str(start_sec))
    dataset_token = sanitize_token(dataset, max_len=32)
    sequence_token = sanitize_token(sequence, max_len=48)
    base = "{}__{}__clip{:04d}__t{}s".format(
        dataset_token,
        sequence_token,
        int(clip_index),
        start_tag,
    )
    if sample_index is not None:
        base = "{}__sample{:03d}".format(base, int(sample_index))
    return "{}.mp4".format(base)


def write_training_clip(frames_dir, output_path, start_idx, num_frames, fps):
    frames_root = Path(frames_dir).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        output.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-start_number",
        str(int(start_idx)),
        "-i",
        str((frames_root / "%06d.png").resolve()),
        "-frames:v",
        str(int(num_frames)),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            "ffmpeg export failed for {}: {}".format(
                output,
                (completed.stderr or completed.stdout or "").strip() or "unknown error",
            )
        )
    return output
