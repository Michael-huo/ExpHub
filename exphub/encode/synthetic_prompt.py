from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from exphub.common.io import read_json_dict, write_json_atomic
from exphub.common.logging import log_info


DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
CAPTION_INSTRUCTION = "Briefly describe the visible scene for first-person video generation."
PROMPT_STRATEGY = "four_part_blip2_semantic_v1"

PROMPT_BASE = (
    "Maintain first-person viewpoint continuity across the full sequence. "
    "Preserve stable scene geometry, perspective, and camera alignment. "
    "Keep exposure and white balance stable over time. "
    "Preserve temporal coherence without flicker or drifting structure."
)

PROMPT_NEGATIVE = (
    "blurry, low detail, low quality, distorted geometry, warped structure, flicker, temporal inconsistency, "
    "drifting objects, duplicated objects, broken perspective, unstable camera, sudden viewpoint change, "
    "ghosting, artifacts, oversmoothing"
)

MOTION_PROMPTS = {
    "stop": "Motion: near-static camera pose, preserve still-scene stability and fine geometry.",
    "forward": "Motion: smooth forward egomotion, preserve clear depth progression and stable perspective.",
    "left_turn": "Motion: smooth left turn, keep rotation continuous and geometry aligned.",
    "right_turn": "Motion: smooth right turn, keep rotation continuous and geometry aligned.",
    "mixed": "Motion: mixed egomotion, keep transitions coherent and camera movement readable.",
}

_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_LOW_VALUE_PREFIXES = (
    "the scene is",
    "this scene is",
    "a picture of",
    "an image of",
    "a photo of",
    "a black and white photo of",
    "a color photo of",
    "the image shows",
    "the picture shows",
    "for video generation is",
)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _collapse_ws(value):
    return " ".join(str(value or "").strip().split()).strip()


def _python_cmd_exists(cmd) -> bool:
    text = str(cmd or "").strip()
    if not text:
        return False
    if os.path.isabs(text) or os.sep in text:
        path = Path(text).expanduser()
        return path.is_file() and os.access(str(path), os.X_OK)
    return bool(shutil.which(text))


def _frame_path(frames_dir, idx):
    frame_root = Path(frames_dir).resolve()
    stem = "{:06d}".format(int(idx))
    for ext in _FRAME_EXTS:
        candidate = frame_root / "{}{}".format(stem, ext)
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("prepare frame not found for index {} under {}".format(int(idx), frame_root))


def _display_frame_path(frame_path, out_path):
    target = Path(frame_path).resolve()
    if out_path is None:
        return str(target)
    prompt_path = Path(out_path).resolve()
    exp_dir = prompt_path.parent.parent
    try:
        return target.relative_to(exp_dir).as_posix()
    except Exception:
        return str(target)


def _sample_indices(unit):
    start_idx = int(unit.get("start_idx"))
    end_idx = int(unit.get("end_idx"))
    mid_idx = int((start_idx + end_idx) // 2)
    out = []
    for idx in (start_idx, mid_idx, end_idx):
        if idx not in out:
            out.append(idx)
    return out


def _clean_caption(value):
    text = _collapse_ws(value).strip(" .")
    text = re.sub(r"^(question|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(question|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    lowered = text.lower()
    for prefix in _LOW_VALUE_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" ,:;-")
            lowered = text.lower()
            break
    text = re.sub(r"^(there is|there are)\s+", "", text, flags=re.IGNORECASE).strip()
    text = _collapse_ws(text).strip(" .")
    return text


def _fuse_semantic(captions):
    cleaned = []
    seen = set()
    for caption in captions:
        text = _clean_caption(caption)
        key = text.lower()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
        if len(cleaned) >= 3:
            break
    if cleaned:
        scene_text = "; ".join(cleaned)
    else:
        scene_text = "continuous first-person scene content"
    max_len = 220
    if len(scene_text) > max_len:
        scene_text = scene_text[:max_len].rsplit(" ", 1)[0].rstrip(" ,;")
    return _collapse_ws(
        "Semantic: {}. Preserve building edges, ground plane, depth cues, and stable foreground-background layout.".format(
            scene_text.rstrip(".")
        )
    )


def _build_caption_plan(generation_units, frames_dir, out_path):
    units = []
    unique_by_path = {}
    unique_items = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        caption_frames = []
        for frame_idx in _sample_indices(unit):
            frame_path = _frame_path(frames_dir, frame_idx)
            abs_path = str(frame_path)
            if abs_path not in unique_by_path:
                item = {
                    "frame_key": "frame_{:06d}".format(int(frame_idx)),
                    "unit_id": unit_id,
                    "frame_idx": int(frame_idx),
                    "frame_path": abs_path,
                }
                unique_by_path[abs_path] = item
                unique_items.append(item)
            caption_frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "frame_path": _display_frame_path(frame_path, out_path),
                    "_abs_frame_path": abs_path,
                }
            )
        unit_copy = dict(unit)
        unit_copy["_caption_frames"] = caption_frames
        units.append(unit_copy)
    return units, unique_items


def _run_blip2_caption_backend(unique_items, prompt_python, prompt_blip2_model, out_path, exphub_root):
    prompt_python_text = str(prompt_python or "").strip()
    if not _python_cmd_exists(prompt_python_text):
        raise RuntimeError(
            "prompt python not found or not executable: {}. Create the blip2 conda environment or pass --prompt-python.".format(
                prompt_python_text or "<empty>"
            )
        )
    prompt_path = Path(out_path).resolve() if out_path is not None else Path.cwd().resolve() / "prompts.json"
    input_json = prompt_path.with_name("prompts_blip2_input.json")
    output_json = prompt_path.with_name("prompts_blip2_output.json")
    write_json_atomic(
        input_json,
        {
            "items": list(unique_items),
            "instruction": CAPTION_INSTRUCTION,
        },
        indent=2,
    )

    repo_root = Path(exphub_root).resolve() if exphub_root else Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    old_pythonpath = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = str(repo_root) if not old_pythonpath else "{}:{}".format(repo_root, old_pythonpath)
    cmd = [
        prompt_python_text,
        "-m",
        "exphub.encode._prompt_backend_blip2",
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
        "--model",
        str(prompt_blip2_model or DEFAULT_BLIP2_MODEL),
        "--device",
        "cuda:0",
        "--max-new-tokens",
        "40",
        "--num-beams",
        "3",
    ]
    log_info("BLIP-2 caption backend start unique_frames={} model={}".format(len(unique_items), prompt_blip2_model))
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "BLIP-2 caption backend failed rc={} cmd={} output:\n{}".format(
                proc.returncode,
                " ".join(cmd),
                str(proc.stdout or "").strip(),
            )
        )
    payload = read_json_dict(output_json)
    if not payload:
        raise RuntimeError("BLIP-2 caption backend produced invalid JSON: {}".format(output_json))
    caption_by_path = {}
    for raw_item in list(payload.get("items") or []):
        item = _as_dict(raw_item)
        frame_path = str(Path(str(item.get("frame_path", "") or "")).resolve())
        caption_by_path[frame_path] = _clean_caption(item.get("caption"))
    missing = [str(item.get("frame_path")) for item in unique_items if str(Path(str(item.get("frame_path"))).resolve()) not in caption_by_path]
    if missing:
        raise RuntimeError("BLIP-2 caption output missing frame(s): {}".format(", ".join(missing[:5])))
    return payload, caption_by_path


def build_prompts(
    generation_units,
    motion_segments,
    semantic_anchors,
    frames_dir=None,
    prompt_python="",
    prompt_backend="blip2",
    prompt_blip2_model=DEFAULT_BLIP2_MODEL,
    out_path=None,
    exphub_root=None,
):
    del motion_segments, semantic_anchors
    backend = str(prompt_backend or "blip2").strip().lower()
    if backend != "blip2":
        raise RuntimeError("unsupported prompt backend '{}'; pass1 supports only blip2".format(prompt_backend))
    if frames_dir is None:
        raise RuntimeError("frames_dir is required for BLIP-2 semantic captions")

    planned_units, unique_items = _build_caption_plan(generation_units, frames_dir, out_path)
    blip2_payload, caption_by_path = _run_blip2_caption_backend(
        unique_items=unique_items,
        prompt_python=prompt_python,
        prompt_blip2_model=prompt_blip2_model,
        out_path=out_path,
        exphub_root=exphub_root,
    )

    units = []
    for unit in planned_units:
        motion_label = str(unit.get("motion_label", "mixed") or "mixed")
        prompt_motion = MOTION_PROMPTS.get(motion_label, MOTION_PROMPTS["mixed"])
        caption_frames = []
        captions = []
        for raw_frame in list(unit.get("_caption_frames") or []):
            frame = dict(raw_frame)
            abs_path = str(frame.pop("_abs_frame_path"))
            caption = caption_by_path.get(abs_path, "")
            frame["caption"] = str(caption)
            caption_frames.append(frame)
            captions.append(caption)
        prompt_semantic = _fuse_semantic(captions)
        prompt_positive = _collapse_ws("{} {} {}".format(PROMPT_BASE, prompt_motion, prompt_semantic))
        units.append(
            {
                "unit_id": str(unit.get("unit_id", "") or ""),
                "start_idx": int(unit.get("start_idx")),
                "end_idx": int(unit.get("end_idx")),
                "motion_label": str(motion_label),
                "prompt_negative": str(PROMPT_NEGATIVE),
                "prompt_base": str(PROMPT_BASE),
                "prompt_motion": str(prompt_motion),
                "prompt_semantic": str(prompt_semantic),
                "prompt_positive": str(prompt_positive),
                "caption_backend": "blip2",
                "caption_frames": caption_frames,
            }
        )

    payload = {
        "schema": "prompts.v2",
        "prompt_strategy": PROMPT_STRATEGY,
        "semantic_backend": {
            "name": "blip2",
            "model": str(blip2_payload.get("model", prompt_blip2_model or DEFAULT_BLIP2_MODEL)),
            "sample_policy": "start_mid_end",
            "caption_instruction": CAPTION_INSTRUCTION,
            "unique_caption_frame_count": int(len(unique_items)),
        },
        "prompt_negative": str(PROMPT_NEGATIVE),
        "units": units,
        "summary": {
            "unit_count": int(len(units)),
            "prompt_positive_source": "prompt_base + prompt_motion + prompt_semantic",
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
    return payload
