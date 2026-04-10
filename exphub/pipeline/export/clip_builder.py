from __future__ import annotations

import json
import subprocess
from pathlib import Path

from exphub.common.types import canon_num_str, dot_to_p, sanitize_token


EXPORT_FPS = 24
EXPORT_NUM_FRAMES = 73
EXPORT_WIDTH = 832
EXPORT_HEIGHT = 480
EXPORT_DURATION_SEC = float(EXPORT_NUM_FRAMES - 1) / float(EXPORT_FPS)
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
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("aligned segment plan must be a JSON object: {}".format(plan_path))
    segments = list(payload.get("segments") or [])
    if not segments:
        raise RuntimeError("aligned segment plan contains zero segments: {}".format(plan_path))
    return payload, plan_path


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


def training_spec():
    return {
        "fps": int(EXPORT_FPS),
        "num_frames": int(EXPORT_NUM_FRAMES),
        "width": int(EXPORT_WIDTH),
        "height": int(EXPORT_HEIGHT),
        "duration_sec": float(EXPORT_DURATION_SEC),
    }


def build_training_caption(prompt_manifest):
    manifest = _as_dict(prompt_manifest)
    if str(manifest.get("prompt_structure", "") or "") != "base_scene_motion":
        raise RuntimeError("export requires prompt_structure=base_scene_motion")
    return _build_training_caption_from_segments(list(manifest.get("segments") or []))


def select_training_candidate(segment_manifest, prompt_manifest, exp_dir):
    segment_obj = _as_dict(segment_manifest)
    prompt_obj = _as_dict(prompt_manifest)
    spec = training_spec()

    frame_count = int(_as_dict(segment_obj.get("frames")).get("frame_count", 0) or 0)
    keyframe_count = int(_as_dict(segment_obj.get("keyframes")).get("count", 0) or 0)
    prompt_segments = list(prompt_obj.get("segments") or [])
    if frame_count < int(spec["num_frames"]):
        return {"accepted": False, "reason": "frame_count_too_short:{}<{}".format(frame_count, int(spec["num_frames"]))}
    if keyframe_count < 2:
        return {"accepted": False, "reason": "insufficient_keyframes:{}".format(keyframe_count)}
    if str(prompt_obj.get("prompt_structure", "") or "") != "base_scene_motion":
        return {
            "accepted": False,
            "reason": "unsupported_prompt_structure:{}".format(str(prompt_obj.get("prompt_structure", "") or "<missing>")),
        }
    if not prompt_segments:
        return {"accepted": False, "reason": "missing_prompt_segments"}

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
        return {"accepted": False, "reason": "aligned_segments_missing_prompt_support"}

    target_frames = int(spec["num_frames"])
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
                    _safe_int(current.get("aligned_start_idx"), 0)
                    + _safe_int(current.get("aligned_end_idx"), 0)
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
        return {"accepted": False, "reason": "no_aligned_exportable_clip"}

    trace_segments = _trace_segments(chosen_rows, clip_start_idx, clip_end_idx)
    if sum([int(item.get("used_num_frames", 0) or 0) for item in trace_segments]) != target_frames:
        return {"accepted": False, "reason": "clip_trace_frame_mismatch"}

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
        return {"accepted": False, "reason": "empty_training_caption"}

    clip_manifest = {
        "version": 1,
        "schema": "train_clip_manifest.v1",
        "source": "export.clip_builder",
        "prompt_structure": "base_scene_motion",
        "clip_start_idx": int(clip_start_idx),
        "clip_end_idx": int(clip_end_idx),
        "clip_num_frames": int(target_frames),
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
    return {
        "accepted": True,
        "caption": caption,
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
    }


def build_clip_filename(dataset, sequence, clip_index, start_sec):
    start_tag = dot_to_p(canon_num_str(start_sec))
    dataset_token = sanitize_token(dataset, max_len=32)
    sequence_token = sanitize_token(sequence, max_len=48)
    return "{}__{}__clip{:04d}__t{}s.mp4".format(
        dataset_token,
        sequence_token,
        int(clip_index),
        start_tag,
    )


def write_training_clip(frames_dir, output_path, start_idx, num_frames, fps=EXPORT_FPS):
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
