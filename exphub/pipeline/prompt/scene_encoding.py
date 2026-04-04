from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from exphub.pipeline.prompt.backends import create_backend


SCENE_PROMPT_INSTRUCTION = (
    "Encode this single representative frame from a first-person video segment as one short scene prompt. "
    "Describe only stable visible scene semantics such as environment type, layout, floor or path, lighting, "
    "major obstacles or moving objects, and forward visibility. Do not mention camera motion, risk, image quality, "
    "style, feelings, or uncertainty. Do not write full sentences. Output only one concise scene prompt."
)

_SCENE_PROMPT_PREFIXES = [
    "scene prompt:",
    "prompt:",
    "output:",
    "the image shows",
    "this image shows",
    "this frame shows",
    "a first-person view of",
    "first-person view of",
    "a first person view of",
    "first person view of",
    "a view of",
    "view of",
]
_SCENE_PROMPT_WORD_LIMIT = 24


def _as_dict(value):
    # type: (object) -> Dict[str, object]
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def _relative_to_exp(exp_dir, target_path):
    # type: (Path, Path) -> str
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _safe_int(value, default=0):
    # type: (object, int) -> int
    try:
        return int(value)
    except Exception:
        return int(default)


def _trim_scene_prompt(text):
    # type: (object) -> str
    prompt = _collapse_ws(text).strip(" \t\r\n\"'`")
    prompt_lower = prompt.lower()
    for prefix in _SCENE_PROMPT_PREFIXES:
        if prompt_lower.startswith(prefix):
            prompt = prompt[len(prefix) :].strip(" \t\r\n:,-")
            prompt_lower = prompt.lower()
            break
    for suffix in (".", ";", ",", ":"):
        if prompt.endswith(suffix):
            prompt = prompt[:-1].rstrip()
    words = [part for part in prompt.split(" ") if part]
    if len(words) > int(_SCENE_PROMPT_WORD_LIMIT):
        prompt = " ".join(words[: int(_SCENE_PROMPT_WORD_LIMIT)]).rstrip(" ,;:.")
    return _collapse_ws(prompt)


def _resolve_frame_path(frames_dir, start_frame, end_frame, preferred_frame):
    # type: (Path, int, int, int) -> Path
    frames_root = Path(frames_dir).resolve()
    start_idx = int(start_frame)
    end_idx = int(end_frame)
    preferred_idx = max(int(start_idx), min(int(end_idx), int(preferred_frame)))

    search_order = [int(preferred_idx)]
    max_offset = int(max(abs(preferred_idx - start_idx), abs(end_idx - preferred_idx)))
    for offset in range(1, max_offset + 1):
        left_idx = int(preferred_idx - offset)
        right_idx = int(preferred_idx + offset)
        if left_idx >= int(start_idx):
            search_order.append(int(left_idx))
        if right_idx <= int(end_idx):
            search_order.append(int(right_idx))

    for frame_idx in search_order:
        image_path = frames_root / "{:06d}.png".format(int(frame_idx))
        if image_path.is_file():
            return image_path
    raise RuntimeError(
        "cannot resolve representative frame under {} for state range [{}:{}]".format(
            frames_root,
            int(start_idx),
            int(end_idx),
        )
    )


def _pick_primary_frame(state_row, keyframe_indices, frames_dir):
    # type: (Dict[str, object], List[int], Path) -> Dict[str, object]
    start_frame = int(state_row.get("start_frame", 0) or 0)
    end_frame = int(state_row.get("end_frame", 0) or 0)
    midpoint_frame = int(start_frame + max(0, int(end_frame - start_frame)) // 2)
    candidate_keyframes = [int(idx) for idx in list(keyframe_indices or []) if int(start_frame) <= int(idx) <= int(end_frame)]

    selection_source = "state_segment_midpoint_fallback"
    selected_frame = int(midpoint_frame)
    if candidate_keyframes:
        selected_frame = min(candidate_keyframes, key=lambda idx: (abs(int(idx) - int(midpoint_frame)), int(idx)))
        selection_source = "segment_keyframe_nearest_midpoint"

    image_path = _resolve_frame_path(
        frames_dir=frames_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        preferred_frame=selected_frame,
    )
    frame_idx = int(image_path.stem)
    return {
        "frame_idx": int(frame_idx),
        "image_path": image_path,
        "selection_source": str(selection_source),
        "midpoint_frame": int(midpoint_frame),
        "candidate_keyframe_count": int(len(candidate_keyframes)),
    }


def build_state_scene_encoding(segment_inputs, frames_dir, prompt_model_ref=""):
    # type: (Dict[str, object], Path, str) -> Dict[str, object]
    manifest = _as_dict(segment_inputs.get("segment_manifest"))
    state_payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(state_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("state_segments payload has no segments for scene encoding")

    keyframe_indices = sorted(set(int(v) for v in list(_as_dict(manifest.get("keyframes")).get("indices") or [])))
    exp_dir = Path(segment_inputs.get("exp_dir")).resolve()
    frames_root = Path(frames_dir).resolve()

    backend = create_backend(model_ref=str(prompt_model_ref or ""))
    backend.load()
    backend_meta = dict(backend.meta() or {})
    backend_name = str(backend_meta.get("backend", "smolvlm2") or "smolvlm2")

    encoded_segments = []
    for idx, raw_item in enumerate(state_rows):
        state_row = _as_dict(raw_item)
        primary_frame = _pick_primary_frame(
            state_row=state_row,
            keyframe_indices=keyframe_indices,
            frames_dir=frames_root,
        )
        raw_prompt = backend.generate([str(primary_frame["image_path"])], SCENE_PROMPT_INSTRUCTION)
        scene_prompt = _trim_scene_prompt(raw_prompt)
        if not scene_prompt:
            raise RuntimeError(
                "scene encoding returned empty prompt for state segment {}".format(
                    _safe_int(state_row.get("segment_id"), idx)
                )
            )
        encoded_segments.append(
            {
                "state_segment_id": _safe_int(state_row.get("segment_id"), idx),
                "source_segment_id": _safe_int(state_row.get("segment_id"), idx),
                "state_label": str(state_row.get("state_label", "unknown") or "unknown"),
                "start_frame": int(state_row.get("start_frame", 0) or 0),
                "end_frame": int(state_row.get("end_frame", 0) or 0),
                "scene_prompt": str(scene_prompt),
                "scene_prompt_source": "state_v2t_primary_frame",
                "scene_encoding_backend": str(backend_name),
                "representative_frame": {
                    "frame_idx": int(primary_frame["frame_idx"]),
                    "image_path": _relative_to_exp(exp_dir, primary_frame["image_path"]),
                    "selection_source": str(primary_frame["selection_source"]),
                    "midpoint_frame": int(primary_frame["midpoint_frame"]),
                    "candidate_keyframe_count": int(primary_frame["candidate_keyframe_count"]),
                },
            }
        )

    return {
        "version": 1,
        "source": "state_level_scene_encoding_v1",
        "scene_prompt_mode": "state_v2t_primary_frame",
        "backend": str(backend_name),
        "backend_meta": dict(backend_meta),
        "state_segments": encoded_segments,
    }
