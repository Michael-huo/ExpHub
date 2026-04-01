from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from exphub.common.io import ensure_file, read_json_dict


STATE_PROMPT_PRESETS = {
    "low_state": {
        "prompt_text": (
            "Within this interval, keep motion transitions gentle and preserve stable first-person continuity, "
            "local geometry, and temporal consistency."
        ),
        "negative_prompt_delta": "",
        "prompt_strength": 0.35,
        "motion_trend": "stable_interval",
    },
    "high_state": {
        "prompt_text": (
            "Within this interval, treat the motion as high-risk and preserve first-person continuity, "
            "local geometry, exposure stability, and temporal coherence under stronger viewpoint change."
        ),
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "prompt_strength": 0.75,
        "motion_trend": "risk_interval",
    },
    "unknown": {
        "prompt_text": (
            "Within this interval, preserve first-person continuity, stable geometry, and temporal coherence."
        ),
        "negative_prompt_delta": "",
        "prompt_strength": 0.50,
        "motion_trend": "unknown_interval",
    },
}  # type: Dict[str, Dict[str, object]]


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _relative_to_exp(exp_dir, target_path):
    # type: (Path, Path) -> str
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _resolve_artifact_path(exp_dir, segment_manifest_path, artifact_rel, default_path):
    # type: (Path, Path, str, Path) -> Path
    raw_rel = str(artifact_rel or "").strip()
    if raw_rel:
        candidate = (Path(exp_dir).resolve() / raw_rel).resolve()
        if candidate.is_file():
            return candidate
    return Path(default_path).resolve()


def _preset_for_state_label(state_label):
    # type: (str) -> Dict[str, object]
    name = str(state_label or "unknown").strip() or "unknown"
    return dict(STATE_PROMPT_PRESETS.get(name, STATE_PROMPT_PRESETS["unknown"]))


def load_segment_prompt_inputs(segment_manifest_path):
    # type: (Path) -> Dict[str, object]
    manifest_path = ensure_file(segment_manifest_path, "segment manifest")
    manifest = read_json_dict(manifest_path)
    if not manifest:
        raise RuntimeError("invalid segment manifest: {}".format(manifest_path))

    exp_dir = manifest_path.parent.parent.resolve()
    artifacts = _as_dict(manifest.get("artifacts"))
    state_segments_payload = _as_dict(manifest.get("state_segments"))
    deploy_schedule_payload = _as_dict(manifest.get("deploy_schedule"))

    state_segments_path = _resolve_artifact_path(
        exp_dir,
        manifest_path,
        artifacts.get("state_segments_compat", ""),
        manifest_path.parent / "state_segmentation" / "state_segments.json",
    )
    deploy_schedule_path = _resolve_artifact_path(
        exp_dir,
        manifest_path,
        artifacts.get("deploy_schedule", ""),
        manifest_path.parent / "deploy_schedule.json",
    )

    if not state_segments_payload:
        state_segments_payload = read_json_dict(state_segments_path)
    if not deploy_schedule_payload:
        deploy_schedule_payload = read_json_dict(deploy_schedule_path)

    if not state_segments_payload:
        raise RuntimeError("segment manifest missing state_segments payload: {}".format(manifest_path))
    if not deploy_schedule_payload:
        raise RuntimeError("segment manifest missing deploy_schedule payload: {}".format(manifest_path))

    frames_meta = _as_dict(manifest.get("frames"))
    return {
        "segment_manifest_path": manifest_path,
        "segment_manifest": manifest,
        "state_segments_path": state_segments_path,
        "state_segments_payload": state_segments_payload,
        "deploy_schedule_path": deploy_schedule_path,
        "deploy_schedule_payload": deploy_schedule_payload,
        "frame_count": int(frames_meta.get("frame_count", 0) or 0),
        "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
        "exp_dir": exp_dir,
        "source_files": {
            "segment_manifest": _relative_to_exp(exp_dir, manifest_path),
            "state_segments": _relative_to_exp(exp_dir, state_segments_path),
            "deploy_schedule": _relative_to_exp(exp_dir, deploy_schedule_path),
        },
    }


def build_state_prompt_manifest(segment_inputs, prompt_dir, base_prompt_path):
    # type: (Dict[str, object], Path, Path) -> Dict[str, object]
    payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("state_segments payload has no segments")

    segments = []
    for idx, raw_item in enumerate(state_rows):
        item = _as_dict(raw_item)
        start_frame = int(item.get("start_frame", 0) or 0)
        end_frame = int(item.get("end_frame", 0) or 0)
        if end_frame < start_frame:
            raise RuntimeError("invalid state segment range at index {}".format(idx))
        state_label = str(item.get("state_label", "unknown") or "unknown")
        preset = _preset_for_state_label(state_label)
        segments.append(
            {
                "state_segment_id": int(item.get("segment_id", idx) or idx),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "state_label": state_label,
                "prompt_text": str(preset.get("prompt_text", "") or ""),
                "negative_prompt_delta": str(preset.get("negative_prompt_delta", "") or ""),
                "prompt_strength": float(preset.get("prompt_strength", 0.5) or 0.5),
                "motion_trend": str(preset.get("motion_trend", "unknown_interval") or "unknown_interval"),
                "source_segment_id": int(item.get("segment_id", idx) or idx),
            }
        )

    exp_dir = Path(segment_inputs.get("exp_dir")).resolve()
    prompt_root = Path(prompt_dir).resolve()
    return {
        "version": 1,
        "schema": "state_prompt_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_type": "interval_prompt_manifest",
        "base_prompt_path": _relative_to_exp(exp_dir, base_prompt_path),
        "state_segment_count": int(len(segments)),
        "source_files": dict(segment_inputs.get("source_files") or {}),
        "state_segments": segments,
        "summary": {
            "state_segment_count": int(len(segments)),
            "frame_count": int(segment_inputs.get("frame_count", 0) or 0),
            "frame_count_used": int(segment_inputs.get("frame_count_used", 0) or 0),
            "prompt_dir": _relative_to_exp(exp_dir, prompt_root),
        },
    }
