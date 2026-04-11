from __future__ import annotations

import json
import subprocess
from pathlib import Path

from exphub.common.types import canon_num_str, dot_to_p, sanitize_token


DEFAULT_EXPORT_FPS = 24
DEFAULT_EXPORT_NUM_FRAMES = 73
DEFAULT_EXPORT_WIDTH = 832
DEFAULT_EXPORT_HEIGHT = 480


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


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

    return {
        "target_fps": int(target_fps),
        "target_num_frames": int(target_num_frames),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "target_duration_sec": float(target_duration_sec),
        "harvest_sec": float(harvest_sec),
        "stride_sec": float(stride_sec),
        "step_frames": max(1, int(round(float(stride_sec) * float(target_fps)))),
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
    return starts


def select_training_candidates(
    segment_manifest,
    exp_dir,
    profile,
    generation_units_payload=None,
    prompt_spans_payload=None,
):
    spec = training_spec(profile)
    segment_obj = _as_dict(segment_manifest)
    frame_count = int(
        _as_dict(segment_obj.get("frames")).get("frame_count_used", 0)
        or _as_dict(segment_obj.get("frames")).get("frame_count", 0)
        or 0
    )
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

    span_units_map = {}
    for raw_unit in units:
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        if not unit_id:
            raise RuntimeError("generation unit missing unit_id")
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

        for clip_start_idx in _window_start_indices(
            span_start_idx=span_start_idx,
            span_end_idx=span_end_idx,
            target_num_frames=int(spec["target_num_frames"]),
            step_frames=int(spec["step_frames"]),
        ):
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


def build_clip_filename(dataset, sequence, clip_index, start_sec, sample_index=None):
    start_tag = dot_to_p(canon_num_str(start_sec))
    dataset_token = sanitize_token(dataset, max_len=32)
    sequence_token = sanitize_token(sequence, max_len=48)
    base = "{}__{}__clip{:04d}__t{}s".format(dataset_token, sequence_token, int(clip_index), start_tag)
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
