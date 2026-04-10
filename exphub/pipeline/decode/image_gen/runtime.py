from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from exphub.common.io import ensure_file, write_json_atomic


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _prompt_preview(text, limit=160):
    collapsed = _collapse_ws(text)
    if len(collapsed) <= int(limit):
        return collapsed
    return collapsed[: max(0, int(limit) - 3)].rstrip() + "..."


def _count_by_key(items, key):
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = _collapse_ws(item.get(key, "")) or "unknown"
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _normalize_runtime_segment(item, idx, default_prompt, default_negative_prompt, default_source, default_backend):
    segment_id = _safe_int(item.get("segment_id", item.get("state_segment_id", idx)), idx)
    raw_start_idx = _safe_int(item.get("raw_start_idx", item.get("raw_start_frame")), None)
    raw_end_idx = _safe_int(item.get("raw_end_idx", item.get("raw_end_frame")), None)
    desired_start_idx = _safe_int(item.get("desired_start_idx"), None)
    desired_end_idx = _safe_int(item.get("desired_end_idx"), None)
    desired_num_frames = _safe_int(item.get("desired_num_frames"), None)
    aligned_start_idx = _safe_int(item.get("aligned_start_idx"), None)
    aligned_end_idx = _safe_int(item.get("aligned_end_idx"), None)
    aligned_num_frames = _safe_int(item.get("aligned_num_frames"), None)
    if (
        raw_start_idx is None
        or raw_end_idx is None
        or desired_start_idx is None
        or desired_end_idx is None
        or desired_num_frames is None
        or aligned_start_idx is None
        or aligned_end_idx is None
        or aligned_num_frames is None
    ):
        raise ValueError("image gen runtime segment missing aligned contract fields for segment {}".format(int(segment_id)))
    start_idx = int(aligned_start_idx)
    end_idx = int(aligned_end_idx)
    if int(aligned_num_frames) != int(end_idx - start_idx + 1):
        raise ValueError("image gen runtime aligned frame count mismatch for segment {}".format(int(segment_id)))
    if int(desired_num_frames) != int(desired_end_idx - desired_start_idx + 1):
        raise ValueError("image gen runtime desired frame count mismatch for segment {}".format(int(segment_id)))
    if int(end_idx) < int(start_idx):
        raise ValueError("image gen runtime segment has invalid range for segment {}".format(int(segment_id)))
    deploy_start_idx = int(start_idx)
    deploy_end_idx = int(end_idx)
    raw_gap = int(raw_end_idx) - int(raw_start_idx)
    deploy_gap = int(deploy_end_idx) - int(deploy_start_idx)
    num_frames = int(aligned_num_frames)
    resolved_prompt = _collapse_ws(item.get("resolved_prompt", "")) or str(default_prompt)
    negative_prompt = _collapse_ws(item.get("negative_prompt", "")) or str(default_negative_prompt)
    prompt_source = _collapse_ws(item.get("prompt_source", "")) or str(default_source)
    execution_backend = _collapse_ws(item.get("execution_backend", "")) or str(default_backend)

    return {
        "seg": int(_safe_int(item.get("seg"), idx)),
        "segment_id": int(segment_id),
        "schedule_source": _collapse_ws(item.get("schedule_source", "")) or "segment.aligned_segment_plan",
        "execution_backend": str(execution_backend),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_frame": int(start_idx),
        "end_frame": int(end_idx),
        "raw_start_idx": int(raw_start_idx),
        "raw_end_idx": int(raw_end_idx),
        "raw_start_frame": int(raw_start_idx),
        "raw_end_frame": int(raw_end_idx),
        "desired_start_idx": int(desired_start_idx),
        "desired_end_idx": int(desired_end_idx),
        "desired_num_frames": int(desired_num_frames),
        "aligned_start_idx": int(aligned_start_idx),
        "aligned_end_idx": int(aligned_end_idx),
        "aligned_num_frames": int(aligned_num_frames),
        "deploy_start_idx": int(deploy_start_idx),
        "deploy_end_idx": int(deploy_end_idx),
        "raw_gap": int(raw_gap),
        "deploy_gap": int(deploy_gap),
        "num_frames": int(num_frames),
        "left_shift": int(_safe_int(item.get("left_shift"), 0) or 0),
        "right_shift": int(_safe_int(item.get("right_shift"), 0) or 0),
        "align_reason": str(item.get("align_reason", "") or ""),
        "state_segment_id": _safe_int(item.get("state_segment_id", segment_id), int(segment_id)),
        "state_label": _collapse_ws(item.get("state_label", "")) or "state_unlabeled",
        "risk_level": _collapse_ws(item.get("risk_level", "")),
        "is_valid_for_decode": bool(item.get("is_valid_for_decode", False)),
        "is_valid_for_export": bool(item.get("is_valid_for_export", False)),
        "prompt_source": str(prompt_source),
        "base_prompt": _collapse_ws(item.get("base_prompt", "")) or str(default_prompt),
        "resolved_prompt": str(resolved_prompt),
        "negative_prompt": str(negative_prompt),
        "scene_prompt": _collapse_ws(item.get("scene_prompt", "")),
        "motion_prompt": _collapse_ws(item.get("motion_prompt", "")),
        "scene_prompt_source": _collapse_ws(item.get("scene_prompt_source", "")),
        "motion_prompt_source": _collapse_ws(item.get("motion_prompt_source", "")),
        "scene_encoding_backend": _collapse_ws(item.get("scene_encoding_backend", "")),
        "continuity_emphasis": _collapse_ws(item.get("continuity_emphasis", "")),
        "representative_frame": dict(_as_dict(item.get("representative_frame"))),
        "num_inference_steps": item.get("num_inference_steps"),
        "guidance_scale": item.get("guidance_scale"),
    }


@dataclass
class ImageGenRequest(object):
    frames_dir: Path
    exp_dir: Path
    prompt_file_path: Path
    execution_plan_path: Path
    fps: int
    kf_gap: int
    base_idx: int
    num_segments: int
    seed_base: int
    gpus: int
    schedule_source: str
    execution_backend: str
    execution_segments: List[Dict[str, object]] = field(default_factory=list)
    infer_extra: List[str] = field(default_factory=list)


def _load_manifest(path, label):
    manifest_path = ensure_file(path, label)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("{} must be a JSON object: {}".format(label, manifest_path))
    payload["_path"] = str(manifest_path)
    return payload


def load_segment_manifest(path):
    manifest = _load_manifest(path, "segment manifest")
    state_segments = list(_as_dict(manifest.get("state_segments")).get("segments") or [])
    if not state_segments:
        raise ValueError("segment manifest must contain state_segments.segments")
    return manifest


def _load_aligned_segment_plan(segment_manifest):
    manifest_path = Path(str(segment_manifest.get("_path", "") or "")).resolve()
    exp_dir = manifest_path.parent.parent
    artifact_meta = _as_dict(segment_manifest.get("aligned_segment_plan"))
    artifact_path = str(artifact_meta.get("path", "") or _as_dict(segment_manifest.get("artifacts")).get("aligned_segment_plan", "") or "").strip()
    if not artifact_path:
        raise ValueError("segment manifest missing aligned_segment_plan artifact path")
    aligned_path = (exp_dir / artifact_path).resolve()
    payload = _load_manifest(aligned_path, "aligned segment plan")
    segments = list(payload.get("segments") or [])
    if not segments:
        raise ValueError("aligned segment plan must contain segments")
    raw_boundaries = list(payload.get("raw_boundary_indices") or [])
    aligned_boundaries = list(payload.get("aligned_boundary_indices") or [])
    if raw_boundaries and aligned_boundaries and len(raw_boundaries) != len(aligned_boundaries):
        raise ValueError("aligned segment plan boundary arrays must have the same length")
    payload["_path"] = str(aligned_path)
    return payload


def load_prompt_manifest(path):
    manifest = _load_manifest(path, "prompt manifest")
    segments = list(manifest.get("segments") or [])
    if not segments:
        raise ValueError("prompt manifest must contain segments")
    return manifest


def load_image_gen_runtime(path, default_prompt="", default_negative_prompt=""):
    payload = _load_manifest(path, "image gen runtime")
    base_prompt = _collapse_ws(payload.get("base_prompt", "")) or _collapse_ws(default_prompt)
    negative_prompt = _collapse_ws(payload.get("negative_prompt", "")) or _collapse_ws(default_negative_prompt)
    if not base_prompt:
        raise ValueError("image gen runtime must contain base_prompt")

    source = str(payload.get("source", "") or "decode.image_gen.runtime").strip()
    execution_backend = str(payload.get("execution_backend", payload.get("infer_backend", "")) or "").strip()
    segments = []
    for idx, raw_item in enumerate(list(payload.get("segments") or [])):
        item = _as_dict(raw_item)
        if not item:
            continue
        segments.append(
            _normalize_runtime_segment(
                item=item,
                idx=idx,
                default_prompt=base_prompt,
                default_negative_prompt=negative_prompt,
                default_source=source,
                default_backend=execution_backend,
            )
        )

    if not segments:
        raise ValueError("image gen runtime must contain at least one segment")

    segments.sort(key=lambda item: (int(item.get("seg", 0)), int(item.get("start_idx", 0))))
    return {
        "version": int(payload.get("version", 1) or 1),
        "schema": str(payload.get("schema", "") or "image_gen_runtime.v1"),
        "base_prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "source": str(source),
        "execution_backend": str(execution_backend),
        "segments": segments,
        "source_files": dict(payload.get("source_files", {}) or {}),
        "_raw": payload,
        "_path": payload["_path"],
    }


def build_image_gen_runtime(segment_manifest, prompt_manifest, infer_backend="wan_fun_5b_inp"):
    segment_payload = _as_dict(segment_manifest)
    prompt_payload = _as_dict(prompt_manifest)
    exp_dir = Path(segment_payload["_path"]).resolve().parent.parent
    aligned_plan = _load_aligned_segment_plan(segment_payload)
    aligned_boundaries = [int(_safe_int(item, 0) or 0) for item in list(aligned_plan.get("aligned_boundary_indices") or [])]

    state_rows = list(_as_dict(segment_payload.get("state_segments")).get("segments") or [])
    aligned_rows = list(aligned_plan.get("segments") or [])
    prompt_segments = list(prompt_payload.get("segments") or [])
    prompt_by_state = {}
    for item in prompt_segments:
        item_dict = _as_dict(item)
        state_segment_id = _safe_int(item_dict.get("state_segment_id"), None)
        if state_segment_id is None:
            continue
        prompt_by_state[int(state_segment_id)] = item_dict
    aligned_by_state = {}
    for item in aligned_rows:
        item_dict = _as_dict(item)
        aligned_by_state[int(_safe_int(item_dict.get("segment_id"), len(aligned_by_state)))] = item_dict

    segments = []
    for idx, raw_state in enumerate(state_rows):
        state_item = _as_dict(raw_state)
        state_segment_id = _safe_int(state_item.get("segment_id"), idx)
        prompt_item = _as_dict(prompt_by_state.get(int(state_segment_id)))
        aligned_item = _as_dict(aligned_by_state.get(int(state_segment_id)))
        if not prompt_item:
            raise RuntimeError("prompt manifest missing state segment {}".format(int(state_segment_id)))
        if not aligned_item:
            raise RuntimeError("aligned segment plan missing state segment {}".format(int(state_segment_id)))

        state_start = int(state_item.get("start_frame", 0) or 0)
        state_end = int(state_item.get("end_frame", 0) or 0)
        prompt_start = int(prompt_item.get("start_frame", state_start) or state_start)
        prompt_end = int(prompt_item.get("end_frame", state_end) or state_end)
        aligned_raw_start = int(aligned_item.get("raw_start_idx", state_start) or state_start)
        aligned_raw_end = int(aligned_item.get("raw_end_idx", state_end) or state_end)
        if state_start != prompt_start or state_end != prompt_end:
            raise RuntimeError(
                "prompt/segment range mismatch for state segment {}: prompt={}..{} segment={}..{}".format(
                    int(state_segment_id),
                    int(prompt_start),
                    int(prompt_end),
                    int(state_start),
                    int(state_end),
                )
            )
        if state_start != aligned_raw_start or state_end != aligned_raw_end:
            raise RuntimeError(
                "aligned/raw range mismatch for state segment {}: aligned_raw={}..{} segment={}..{}".format(
                    int(state_segment_id),
                    int(aligned_raw_start),
                    int(aligned_raw_end),
                    int(state_start),
                    int(state_end),
                )
            )
        if not bool(aligned_item.get("is_valid_for_decode", False)):
            raise RuntimeError("aligned segment plan marks state segment {} invalid for decode".format(int(state_segment_id)))

        desired_start_idx = int(aligned_item.get("desired_start_idx", state_start) or state_start)
        desired_end_idx = int(aligned_item.get("desired_end_idx", state_end) or state_end)
        desired_num_frames = int(aligned_item.get("desired_num_frames", desired_end_idx - desired_start_idx + 1) or 0)
        aligned_start_idx = int(aligned_item.get("aligned_start_idx", desired_start_idx) or desired_start_idx)
        aligned_end_idx = int(aligned_item.get("aligned_end_idx", desired_end_idx) or desired_end_idx)
        aligned_num_frames = int(aligned_item.get("aligned_num_frames", aligned_end_idx - aligned_start_idx + 1) or 0)
        if aligned_num_frames <= 0:
            raise RuntimeError("aligned segment {} resolves to zero frames".format(int(state_segment_id)))

        segments.append(
            {
                "seg": int(idx),
                "segment_id": int(state_segment_id),
                "schedule_source": "segment.aligned_segment_plan",
                "execution_backend": str(infer_backend),
                "state_segment_id": int(state_segment_id),
                "state_label": str(state_item.get("state_label", prompt_item.get("state_label", "state_unlabeled")) or "state_unlabeled"),
                "risk_level": str(state_item.get("risk_level", aligned_item.get("risk_level", "")) or ""),
                "start_idx": int(aligned_start_idx),
                "end_idx": int(aligned_end_idx),
                "raw_start_idx": int(state_start),
                "raw_end_idx": int(state_end),
                "desired_start_idx": int(desired_start_idx),
                "desired_end_idx": int(desired_end_idx),
                "desired_num_frames": int(desired_num_frames),
                "aligned_start_idx": int(aligned_start_idx),
                "aligned_end_idx": int(aligned_end_idx),
                "aligned_num_frames": int(aligned_num_frames),
                "deploy_start_idx": int(aligned_start_idx),
                "deploy_end_idx": int(aligned_end_idx),
                "raw_gap": int(state_end - state_start),
                "deploy_gap": int(aligned_end_idx - aligned_start_idx),
                "num_frames": int(aligned_num_frames),
                "left_shift": int(aligned_item.get("left_shift", 0) or 0),
                "right_shift": int(aligned_item.get("right_shift", 0) or 0),
                "align_reason": str(aligned_item.get("align_reason", "") or ""),
                "is_valid_for_decode": True,
                "is_valid_for_export": bool(aligned_item.get("is_valid_for_export", False)),
                "prompt_source": "prompt_manifest.base_scene_motion",
                "base_prompt": str(prompt_payload.get("base_prompt", "") or ""),
                "resolved_prompt": str(prompt_item.get("resolved_prompt", "") or ""),
                "negative_prompt": str(prompt_item.get("negative_prompt", prompt_payload.get("negative_prompt", "")) or ""),
                "scene_prompt": str(prompt_item.get("scene_prompt", "") or ""),
                "motion_prompt": str(prompt_item.get("motion_prompt", "") or ""),
                "scene_prompt_source": str(prompt_item.get("scene_prompt_source", "") or ""),
                "motion_prompt_source": str(prompt_item.get("motion_prompt_source", "") or ""),
                "scene_encoding_backend": str(prompt_item.get("scene_encoding_backend", "") or ""),
                "continuity_emphasis": str(prompt_item.get("continuity_emphasis", "balanced") or "balanced"),
                "representative_frame": dict(_as_dict(prompt_item.get("representative_frame"))),
            }
        )

    for idx, seg_item in enumerate(segments):
        if idx == 0:
            continue
        prev = segments[idx - 1]
        if int(seg_item["aligned_start_idx"]) != int(prev["aligned_end_idx"]):
            raise RuntimeError(
                "aligned segment plan must use globally snapped shared boundaries: prev_end={} current_start={} state_segment={}".format(
                    int(prev["aligned_end_idx"]),
                    int(seg_item["aligned_start_idx"]),
                    int(seg_item["state_segment_id"]),
                )
            )
    if aligned_boundaries:
        expected_boundary_count = int(len(segments) + 1)
        if len(aligned_boundaries) != expected_boundary_count:
            raise RuntimeError(
                "aligned segment plan boundary count mismatch: expected={} actual={}".format(
                    int(expected_boundary_count),
                    int(len(aligned_boundaries)),
                )
            )

    return {
        "version": 1,
        "schema": "image_gen_runtime.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "decode.image_gen.runtime",
        "prompt_structure": str(prompt_payload.get("prompt_structure", "base_scene_motion") or "base_scene_motion"),
        "base_prompt": str(prompt_payload.get("base_prompt", "") or ""),
        "negative_prompt": str(prompt_payload.get("negative_prompt", "") or ""),
        "execution_backend": str(infer_backend),
        "segments": segments,
        "source_files": {
            "segment_manifest": _relative_path(exp_dir, segment_payload["_path"]),
            "aligned_segment_plan": _relative_path(exp_dir, aligned_plan["_path"]),
            "prompt_manifest": _relative_path(exp_dir, prompt_payload["_path"]),
        },
        "summary": {
            "segment_count": int(len(segments)),
            "boundary_count": int(len(aligned_boundaries)),
            "prompt_source_counts": dict(_count_by_key(segments, "prompt_source")),
            "state_label_counts": dict(_count_by_key(segments, "state_label")),
        },
    }


def build_execution_plan(image_gen_runtime):
    runtime_payload = _as_dict(image_gen_runtime)
    segments = []
    for item in list(runtime_payload.get("segments") or []):
        seg_item = _as_dict(item)
        segments.append(
            {
                "seg": int(seg_item.get("seg", len(segments)) or len(segments)),
                "segment_id": int(seg_item.get("segment_id", seg_item.get("state_segment_id", len(segments))) or len(segments)),
                "schedule_source": str(seg_item.get("schedule_source", "segment.aligned_segment_plan") or "segment.aligned_segment_plan"),
                "execution_backend": str(seg_item.get("execution_backend", runtime_payload.get("execution_backend", "")) or ""),
                "raw_start_idx": int(seg_item.get("raw_start_idx", seg_item.get("start_idx", 0)) or 0),
                "raw_end_idx": int(seg_item.get("raw_end_idx", seg_item.get("end_idx", 0)) or 0),
                "desired_start_idx": int(seg_item.get("desired_start_idx", seg_item.get("start_idx", 0)) or 0),
                "desired_end_idx": int(seg_item.get("desired_end_idx", seg_item.get("end_idx", 0)) or 0),
                "desired_num_frames": int(seg_item.get("desired_num_frames", seg_item.get("num_frames", 0)) or 0),
                "aligned_start_idx": int(seg_item.get("aligned_start_idx", seg_item.get("start_idx", 0)) or 0),
                "aligned_end_idx": int(seg_item.get("aligned_end_idx", seg_item.get("end_idx", 0)) or 0),
                "aligned_num_frames": int(seg_item.get("aligned_num_frames", seg_item.get("num_frames", 0)) or 0),
                "deploy_start_idx": int(seg_item.get("deploy_start_idx", seg_item.get("start_idx", 0)) or 0),
                "deploy_end_idx": int(seg_item.get("deploy_end_idx", seg_item.get("end_idx", 0)) or 0),
                "start_idx": int(seg_item.get("start_idx", 0) or 0),
                "end_idx": int(seg_item.get("end_idx", 0) or 0),
                "raw_gap": int(seg_item.get("raw_gap", 0) or 0),
                "deploy_gap": int(seg_item.get("deploy_gap", 0) or 0),
                "num_frames": int(seg_item.get("num_frames", 0) or 0),
                "left_shift": int(seg_item.get("left_shift", 0) or 0),
                "right_shift": int(seg_item.get("right_shift", 0) or 0),
                "align_reason": str(seg_item.get("align_reason", "") or ""),
                "is_valid_for_decode": bool(seg_item.get("is_valid_for_decode", False)),
                "is_valid_for_export": bool(seg_item.get("is_valid_for_export", False)),
            }
        )
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schedule_source": "segment.aligned_segment_plan",
        "execution_backend": str(runtime_payload.get("execution_backend", "") or ""),
        "segments": segments,
    }


def resolve_image_gen_runtime_segments(
    image_gen_runtime,
    num_segments,
    default_prompt,
    default_negative_prompt,
    default_num_inference_steps,
    default_guidance_scale,
):
    count = max(0, int(num_segments))
    runtime_payload = _as_dict(image_gen_runtime)
    prompt = _collapse_ws(runtime_payload.get("base_prompt", "")) or _collapse_ws(default_prompt)
    negative_prompt = _collapse_ws(runtime_payload.get("negative_prompt", "")) or _collapse_ws(default_negative_prompt)
    source = str(runtime_payload.get("source", "") or "decode.image_gen.runtime")
    override_map = {}
    for raw_item in list(runtime_payload.get("segments") or []):
        item = _as_dict(raw_item)
        seg = _safe_int(item.get("seg"), -1)
        if seg is None or int(seg) < 0:
            continue
        override_map[int(seg)] = item

    resolved = []
    for seg in range(count):
        override_item = _as_dict(override_map.get(int(seg), {}))
        num_inference_steps = override_item.get("num_inference_steps", default_num_inference_steps)
        if num_inference_steps in (None, ""):
            num_inference_steps = default_num_inference_steps
        guidance_scale = override_item.get("guidance_scale", default_guidance_scale)
        if guidance_scale in (None, ""):
            guidance_scale = default_guidance_scale
        resolved_prompt_text = _collapse_ws(override_item.get("resolved_prompt", "")) or str(prompt)
        resolved_negative_text = _collapse_ws(override_item.get("negative_prompt", "")) or str(negative_prompt)
        resolved.append(
            {
                "seg": int(seg),
                "resolved_prompt": str(resolved_prompt_text),
                "negative_prompt": str(resolved_negative_text),
                "prompt_source": _collapse_ws(override_item.get("prompt_source", "")) or str(source),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "state_segment_id": override_item.get("state_segment_id"),
                "state_label": override_item.get("state_label"),
                "base_prompt": override_item.get("base_prompt"),
                "aligned_start_idx": override_item.get("aligned_start_idx"),
                "aligned_end_idx": override_item.get("aligned_end_idx"),
                "aligned_num_frames": override_item.get("aligned_num_frames"),
            }
        )
    return resolved


def build_prompt_resolution(image_gen_runtime, execution_segments, exp_dir):
    runtime_payload = _as_dict(image_gen_runtime)
    exp_root = Path(exp_dir).resolve()
    runtime_segments = {}
    for item in list(runtime_payload.get("segments") or []):
        seg_item = _as_dict(item)
        runtime_segments[int(seg_item.get("segment_id", seg_item.get("state_segment_id", 0)) or 0)] = seg_item

    resolution_items = []
    for idx, raw_exec_segment in enumerate(list(execution_segments or [])):
        exec_segment = _as_dict(raw_exec_segment)
        seg = _safe_int(exec_segment.get("seg", idx), idx)
        segment_id = _safe_int(exec_segment.get("segment_id", seg), seg)
        runtime_item = _as_dict(runtime_segments.get(int(segment_id), {}))
        if not runtime_item:
            raise ValueError("image gen runtime missing segment {}".format(int(segment_id)))
        resolution_items.append(
            {
                "seg": int(seg),
                "segment_id": int(segment_id),
                "start_frame": int(runtime_item.get("aligned_start_idx", runtime_item.get("start_idx", 0)) or 0),
                "end_frame": int(runtime_item.get("aligned_end_idx", runtime_item.get("end_idx", 0)) or 0),
                "aligned_num_frames": int(runtime_item.get("aligned_num_frames", runtime_item.get("num_frames", 0)) or 0),
                "desired_start_idx": int(runtime_item.get("desired_start_idx", runtime_item.get("raw_start_idx", 0)) or 0),
                "desired_end_idx": int(runtime_item.get("desired_end_idx", runtime_item.get("raw_end_idx", 0)) or 0),
                "desired_num_frames": int(runtime_item.get("desired_num_frames", runtime_item.get("num_frames", 0)) or 0),
                "raw_start_idx": int(runtime_item.get("raw_start_idx", 0) or 0),
                "raw_end_idx": int(runtime_item.get("raw_end_idx", 0) or 0),
                "left_shift": int(runtime_item.get("left_shift", 0) or 0),
                "right_shift": int(runtime_item.get("right_shift", 0) or 0),
                "align_reason": str(runtime_item.get("align_reason", "") or ""),
                "is_valid_for_decode": bool(runtime_item.get("is_valid_for_decode", False)),
                "is_valid_for_export": bool(runtime_item.get("is_valid_for_export", False)),
                "state_segment_id": _safe_int(runtime_item.get("state_segment_id"), None),
                "state_label": runtime_item.get("state_label"),
                "prompt_source": str(runtime_item.get("prompt_source", "") or "prompt_manifest.base_scene_motion"),
                "base_prompt": str(runtime_item.get("base_prompt", runtime_payload.get("base_prompt", "")) or ""),
                "resolved_prompt": str(runtime_item.get("resolved_prompt", "") or runtime_payload.get("base_prompt", "")),
                "negative_prompt": str(runtime_item.get("negative_prompt", "") or runtime_payload.get("negative_prompt", "")),
                "prompt_preview": _prompt_preview(runtime_item.get("resolved_prompt", "") or runtime_payload.get("base_prompt", "")),
            }
        )

    prompt_source_counts = _count_by_key(resolution_items, "prompt_source")
    state_label_counts = _count_by_key([item for item in resolution_items if item.get("state_label")], "state_label")
    return {
        "state_prompt_enabled": bool(resolution_items),
        "state_prompt_segment_count": int(len(list(runtime_payload.get("segments") or []))),
        "matched_execution_segment_count": int(len(resolution_items)),
        "image_gen_runtime_version": int(runtime_payload.get("version", 1) or 1),
        "image_gen_runtime_schema": str(runtime_payload.get("schema", "") or "image_gen_runtime.v1"),
        "image_gen_runtime_source": str(runtime_payload.get("source", "") or "decode.image_gen.runtime"),
        "prompt_source_counts": dict(prompt_source_counts),
        "state_label_counts": dict(state_label_counts),
        "segment_resolutions": list(resolution_items),
        "prompt_resolution": {
            "version": int(runtime_payload.get("version", 1) or 1),
            "schema": str(runtime_payload.get("schema", "") or "image_gen_runtime.v1"),
            "prompt_source_counts": dict(prompt_source_counts),
            "state_label_counts": dict(state_label_counts),
            "source_files": {
                key: _relative_path(exp_root, Path(exp_root / str(value)).resolve())
                for key, value in dict(runtime_payload.get("source_files", {}) or {}).items()
                if str(value).strip()
            },
            "warnings": [],
        },
        "warnings": [],
    }


def merge_prompt_resolution_into_runs_plan(plan_obj, segment_resolutions):
    plan = dict(plan_obj or {})
    plan_segments = list(plan.get("segments", []) or [])
    resolution_map = {}
    for raw_item in list(segment_resolutions or []):
        if not isinstance(raw_item, dict):
            continue
        seg = raw_item.get("seg")
        if seg is None:
            continue
        resolution_map[int(seg)] = dict(raw_item)

    for idx, raw_segment in enumerate(plan_segments):
        if not isinstance(raw_segment, dict):
            continue
        seg_item = raw_segment
        seg_key = seg_item.get("seg", idx)
        resolved = resolution_map.get(int(seg_key), {})
        if not resolved:
            continue
        seg_item["prompt_source"] = str(resolved.get("prompt_source", seg_item.get("prompt_source", "")) or "")
        seg_item["state_segment_id"] = resolved.get("state_segment_id")
        seg_item["state_label"] = resolved.get("state_label")
        seg_item["base_prompt"] = resolved.get("base_prompt")
        seg_item["resolved_prompt"] = resolved.get("resolved_prompt")
        seg_item["negative_prompt"] = resolved.get("negative_prompt")
        seg_item["prompt_preview"] = resolved.get("prompt_preview")
        seg_item["aligned_start_idx"] = resolved.get("start_frame")
        seg_item["aligned_end_idx"] = resolved.get("end_frame")
        seg_item["aligned_num_frames"] = resolved.get("aligned_num_frames")
        seg_item["desired_start_idx"] = resolved.get("desired_start_idx")
        seg_item["desired_end_idx"] = resolved.get("desired_end_idx")
        seg_item["desired_num_frames"] = resolved.get("desired_num_frames")
        seg_item["raw_start_idx"] = resolved.get("raw_start_idx")
        seg_item["raw_end_idx"] = resolved.get("raw_end_idx")
        seg_item["left_shift"] = resolved.get("left_shift")
        seg_item["right_shift"] = resolved.get("right_shift")
        seg_item["align_reason"] = resolved.get("align_reason")
        seg_item["is_valid_for_decode"] = resolved.get("is_valid_for_decode")
        seg_item["is_valid_for_export"] = resolved.get("is_valid_for_export")
    plan["schedule_source"] = "segment.aligned_segment_plan"
    plan["segments"] = plan_segments
    return plan


def write_backend_runtime_files(infer_dir, image_gen_runtime, execution_plan):
    infer_root = Path(infer_dir).resolve()
    infer_root.mkdir(parents=True, exist_ok=True)

    prompt_fd, prompt_tmp = tempfile.mkstemp(prefix="image_gen_runtime_", suffix=".json", dir=str(infer_root))
    execution_fd, execution_tmp = tempfile.mkstemp(prefix="execution_plan_", suffix=".json", dir=str(infer_root))
    os.close(prompt_fd)
    os.close(execution_fd)
    Path(prompt_tmp).unlink()
    Path(execution_tmp).unlink()
    write_json_atomic(Path(prompt_tmp), image_gen_runtime, indent=2)
    write_json_atomic(Path(execution_tmp), execution_plan, indent=2)
    return Path(prompt_tmp).resolve(), Path(execution_tmp).resolve()
