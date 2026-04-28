from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info


DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
CAPTION_INSTRUCTION = "Briefly describe the visible scene for first-person video generation."
DEFAULT_CANDIDATE_STRIDE = 24
DEFAULT_TEXT_IMAGE_DROP_THRESHOLD = 0.05
DEFAULT_MIN_SEMANTIC_UPDATE_GAP = 24
DEFAULT_LOW_SIMILARITY_PATIENCE = 2
DEFAULT_IMAGE_NOVELTY_THRESHOLD = 0.08
DEFAULT_TURN_IMAGE_NOVELTY_THRESHOLD = 0.12
DEFAULT_COVERAGE_GAP_PATIENCE = 2
DEFAULT_MAX_UNIT_SPAN_FRAMES = 96

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
    "a view of",
)

_STABILITY_PHRASES = (
    "stable building edges",
    "ground plane",
    "depth cues",
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


def _abs_time_at(prepare_result, idx):
    values = list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])
    if int(idx) < 0 or int(idx) >= len(values):
        raise RuntimeError("prepare_result frame_index_map missing abs time for frame {}".format(int(idx)))
    return float(values[int(idx)])


def _display_frame_path(frame_path, out_path):
    target = Path(frame_path).resolve()
    if out_path is None:
        return str(target)
    semantic_path = Path(out_path).resolve()
    exp_dir = semantic_path.parent.parent
    try:
        return target.relative_to(exp_dir).as_posix()
    except Exception:
        return str(target)


def _normalize_embeddings(embeddings):
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


class _OpenClipSession:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        import torch
        import open_clip

        self.torch = torch
        self.open_clip = open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)

    def encode_images(self, frame_paths, batch_size=16):
        from PIL import Image

        encoded = []
        batch = []
        batch_size = max(1, int(batch_size))
        for path in frame_paths:
            with Image.open(str(path)) as image_obj:
                batch.append(self.preprocess(image_obj.convert("RGB")))
            if len(batch) < batch_size:
                continue
            tensor = self.torch.stack(batch, dim=0).to(self.device)
            with self.torch.no_grad():
                encoded.append(self.model.encode_image(tensor).detach().cpu().float().numpy())
            batch = []
        if batch:
            tensor = self.torch.stack(batch, dim=0).to(self.device)
            with self.torch.no_grad():
                encoded.append(self.model.encode_image(tensor).detach().cpu().float().numpy())
        if not encoded:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(encoded, axis=0).astype(np.float32)

    def encode_texts(self, texts):
        items = [str(item or "") for item in list(texts or [])]
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        tokens = self.open_clip.tokenize(items).to(self.device)
        with self.torch.no_grad():
            encoded = self.model.encode_text(tokens).detach().cpu().float().numpy()
        return encoded.astype(np.float32)

    def meta(self, embedding_dim):
        return {
            "caption_backend": "blip2",
            "caption_model": "",
            "embedding_backend": "openclip",
            "embedding_model": self.model_name,
            "embedding_pretrained": self.pretrained,
            "embedding_device": self.device,
            "embedding_dim": int(embedding_dim),
            "semantic_state_policy": "text_image_similarity_drop",
        }


def clean_caption(value):
    text = _collapse_ws(value).strip(" .")
    text = re.sub(r"^(question|answer|semantic|motion|base)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(there is|there are)\s+", "", text, flags=re.IGNORECASE).strip()
    lowered = text.lower()
    for prefix in _LOW_VALUE_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" ,:;-")
            lowered = text.lower()
            break
    text = re.sub(r"\s*[,;]\s*", ", ", text)
    return _collapse_ws(text).strip(" .,:;-")


def normalize_caption_to_semantic(caption):
    text = clean_caption(caption).lower()
    if not text:
        text = "continuous first-person scene layout"
    text = text.replace(" and ", ", ")
    text = re.sub(r"\bwood and brick\b", "wood, brick", text)
    raw_parts = [part.strip(" .,:;-") for part in re.split(r"[,;]", text)]
    parts = []
    seen = set()
    dynamic_parts = []
    for part in raw_parts:
        part = _collapse_ws(part)
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        if re.search(r"\b(person|people|pedestrian|car|vehicle|truck|bus|bicycle|cyclist)\b", key):
            dynamic_parts.append(part)
        else:
            parts.append(part)
    ordered = parts[:4] + dynamic_parts[:1]
    for phrase in _STABILITY_PHRASES:
        key = phrase.lower()
        if key not in seen:
            ordered.append(phrase)
            seen.add(key)
    result = ", ".join(ordered)
    if len(result) > 220:
        result = result[:220].rsplit(",", 1)[0].strip(" ,")
    return _collapse_ws(result)


def _legal_between(legal_positions, start_idx, end_idx):
    return [int(idx) for idx in legal_positions if int(start_idx) <= int(idx) <= int(end_idx)]


def _candidate_positions(candidates, start_idx, end_idx, stride):
    out = []
    last = None
    stride = max(1, int(stride))
    for idx in candidates:
        if int(idx) <= int(start_idx) or int(idx) >= int(end_idx):
            continue
        if last is None or int(idx) - int(last) >= stride:
            out.append(int(idx))
            last = int(idx)
    return out


def _image_novelty(normalized_images, pos_to_embed, reference_idx, frame_idx):
    ref_pos = pos_to_embed.get(int(reference_idx))
    frame_pos = pos_to_embed.get(int(frame_idx))
    if ref_pos is None or frame_pos is None:
        return 0.0
    cosine = float(np.dot(normalized_images[ref_pos], normalized_images[frame_pos]))
    return float(1.0 - max(-1.0, min(1.0, cosine)))


def _motion_image_novelty_threshold(motion_label, policy):
    label = str(motion_label or "mixed")
    if label in ("left_turn", "right_turn", "mixed"):
        return float(policy["turn_image_novelty_threshold"])
    return float(policy["image_novelty_threshold"])


def _plan_semantic_events_for_motion_state(
    motion_state_id,
    motion_label,
    motion_state_start,
    motion_state_end,
    candidates,
    normalized_images,
    pos_to_embed,
    policy,
):
    if (
        not candidates
        or int(candidates[0]) != int(motion_state_start)
        or int(candidates[-1]) != int(motion_state_end)
    ):
        raise RuntimeError("motion_state boundaries must be legal: {}".format(motion_state_id))
    events = [
        {
            "frame_idx": int(motion_state_start),
            "event_type": "semantic_state_start",
            "reason": "semantic_state_start",
            "image_novelty": 0.0,
            "threshold": _motion_image_novelty_threshold(motion_label, policy),
            "patience": int(policy["coverage_gap_patience"]),
            "max_image_novelty_before_update": 0.0,
        }
    ]
    current_semantic_start = int(motion_state_start)
    threshold = _motion_image_novelty_threshold(motion_label, policy)
    patience = max(1, int(policy["coverage_gap_patience"]))
    min_gap = max(1, int(policy["min_semantic_update_gap"]))
    consecutive = []
    max_novelty = 0.0
    for frame_idx in _candidate_positions(candidates, motion_state_start, motion_state_end, policy["candidate_stride"]):
        frame_idx = int(frame_idx)
        novelty = _image_novelty(normalized_images, pos_to_embed, current_semantic_start, frame_idx)
        max_novelty = max(float(max_novelty), float(novelty))
        far_enough_from_start = int(frame_idx) - int(current_semantic_start) >= int(min_gap)
        far_enough_from_end = int(motion_state_end) - int(frame_idx) >= int(min_gap)
        if novelty > threshold and far_enough_from_start and far_enough_from_end:
            consecutive.append((int(frame_idx), float(novelty), float(max_novelty)))
        else:
            consecutive = []
        if len(consecutive) < patience:
            continue
        events.append(
            {
                "frame_idx": int(frame_idx),
                "event_type": "semantic_update",
                "reason": "coverage_gap",
                "image_novelty": float(novelty),
                "threshold": float(threshold),
                "patience": int(patience),
                "max_image_novelty_before_update": float(max_novelty),
                "frames_from_semantic_state_start": int(frame_idx) - int(current_semantic_start),
            }
        )
        current_semantic_start = int(frame_idx)
        consecutive = []
        max_novelty = 0.0
    return events


def _run_blip2_caption_backend(unique_items, prompt_python, prompt_blip2_model, exphub_root):
    prompt_python_text = str(prompt_python or "").strip()
    if not _python_cmd_exists(prompt_python_text):
        raise RuntimeError(
            "prompt python not found or not executable: {}. Create the blip2 conda environment or pass --prompt-python.".format(
                prompt_python_text or "<empty>"
            )
        )
    repo_root = Path(exphub_root).resolve() if exphub_root else Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    old_pythonpath = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = str(repo_root) if not old_pythonpath else "{}:{}".format(repo_root, old_pythonpath)

    with tempfile.TemporaryDirectory(prefix="exphub_blip2_") as tmp_dir:
        input_json = Path(tmp_dir) / "semantic_state_blip2_input.json"
        output_json = Path(tmp_dir) / "semantic_state_blip2_output.json"
        write_json_atomic(
            input_json,
            {
                "items": list(unique_items),
                "instruction": CAPTION_INSTRUCTION,
            },
            indent=2,
        )
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
        log_info("BLIP-2 caption backend start semantic_state_frames={} model={}".format(len(unique_items), prompt_blip2_model))
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
        raise RuntimeError("BLIP-2 caption backend produced invalid JSON")
    caption_by_path = {}
    for raw_item in list(payload.get("items") or []):
        item = _as_dict(raw_item)
        frame_path = str(Path(str(item.get("frame_path", "") or "")).resolve())
        caption_by_path[frame_path] = clean_caption(item.get("caption"))
    missing = [str(item.get("frame_path")) for item in unique_items if str(Path(str(item.get("frame_path"))).resolve()) not in caption_by_path]
    if missing:
        raise RuntimeError("BLIP-2 caption output missing frame(s): {}".format(", ".join(missing[:5])))
    return payload, caption_by_path


def _similarity_rows(candidates, normalized_images, pos_to_embed, text_emb, reference_sim, semantic_state_start_idx):
    rows = []
    min_similarity = float(reference_sim)
    max_image_novelty = 0.0
    if text_emb is None:
        return rows, float(min_similarity), float(max_image_novelty)
    for frame_idx in candidates:
        pos = pos_to_embed.get(int(frame_idx))
        if pos is None:
            continue
        sim = float(np.dot(normalized_images[pos], text_emb))
        novelty = _image_novelty(normalized_images, pos_to_embed, semantic_state_start_idx, frame_idx)
        min_similarity = min(float(min_similarity), float(sim))
        max_image_novelty = max(float(max_image_novelty), float(novelty))
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "text_image_similarity": float(sim),
                "text_image_drop": float(reference_sim - sim),
                "image_novelty": float(novelty),
                "frames_from_semantic_state_start": int(frame_idx) - int(semantic_state_start_idx),
            }
        )
    return rows, float(min_similarity), float(max_image_novelty)


def build_semantic_anchors(
    prepare_result,
    motion_segments,
    frames_dir,
    out_path=None,
    prompt_python="",
    prompt_blip2_model=DEFAULT_BLIP2_MODEL,
    candidate_stride=DEFAULT_CANDIDATE_STRIDE,
    text_image_drop_threshold=DEFAULT_TEXT_IMAGE_DROP_THRESHOLD,
    min_semantic_update_gap=DEFAULT_MIN_SEMANTIC_UPDATE_GAP,
    low_similarity_patience=DEFAULT_LOW_SIMILARITY_PATIENCE,
    image_novelty_threshold=DEFAULT_IMAGE_NOVELTY_THRESHOLD,
    turn_image_novelty_threshold=DEFAULT_TURN_IMAGE_NOVELTY_THRESHOLD,
    coverage_gap_patience=DEFAULT_COVERAGE_GAP_PATIENCE,
    max_unit_span_frames=DEFAULT_MAX_UNIT_SPAN_FRAMES,
    exphub_root=None,
):
    started = time.time()
    prepare = _as_dict(prepare_result)
    frame_dir = ensure_dir(frames_dir, "prepare frames dir")
    frames = list_frames_sorted(frame_dir)
    legal_grid = _as_dict(prepare.get("legal_grid"))
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    if not legal_positions:
        raise RuntimeError("prepare_result legal_grid missing legal_positions")

    policy = {
        "candidate_stride": int(candidate_stride),
        "text_image_drop_threshold": float(text_image_drop_threshold),
        "min_semantic_update_gap": int(min_semantic_update_gap),
        "low_similarity_patience": int(low_similarity_patience),
        "image_novelty_threshold": float(image_novelty_threshold),
        "turn_image_novelty_threshold": float(turn_image_novelty_threshold),
        "coverage_gap_patience": int(coverage_gap_patience),
        "max_unit_span_frames": int(max_unit_span_frames),
    }

    clip = _OpenClipSession()
    legal_frame_paths = [frames[idx] for idx in legal_positions]
    image_embeddings = clip.encode_images(legal_frame_paths)
    normalized_images = _normalize_embeddings(image_embeddings)
    pos_to_embed = {int(frame_idx): int(pos) for pos, frame_idx in enumerate(legal_positions)}
    backend_meta = clip.meta(image_embeddings.shape[1] if image_embeddings.ndim == 2 else 0)
    backend_meta["caption_model"] = str(prompt_blip2_model or DEFAULT_BLIP2_MODEL)
    backend_meta["semantic_state_policy"] = "image_novelty_coverage_gap"

    planned_motion_states = []
    unique_by_path = {}
    unique_items = []
    for raw_motion_state in list(_as_dict(motion_segments).get("motion_states") or []):
        motion_state = _as_dict(raw_motion_state)
        motion_state_id = str(motion_state.get("motion_state_id", "") or "")
        motion_state_start = int(motion_state.get("start_idx"))
        motion_state_end = int(motion_state.get("end_idx"))
        candidates = _legal_between(legal_positions, motion_state_start, motion_state_end)
        semantic_events = _plan_semantic_events_for_motion_state(
            motion_state_id,
            str(motion_state.get("motion_label", "") or "mixed"),
            motion_state_start,
            motion_state_end,
            candidates,
            normalized_images,
            pos_to_embed,
            policy,
        )
        planned_motion_states.append(
            {
                "motion_state_id": str(motion_state_id),
                "motion_label": str(motion_state.get("motion_label", "") or "mixed"),
                "start_idx": int(motion_state_start),
                "end_idx": int(motion_state_end),
                "candidates": candidates,
                "semantic_events": semantic_events,
            }
        )
        for event in semantic_events:
            frame_idx = int(event["frame_idx"])
            frame_path = Path(frames[frame_idx]).resolve()
            abs_path = str(frame_path)
            if abs_path not in unique_by_path:
                unique_by_path[abs_path] = {
                    "frame_key": "frame_{:06d}".format(int(frame_idx)),
                    "frame_idx": int(frame_idx),
                    "frame_path": abs_path,
                }
                unique_items.append(unique_by_path[abs_path])

    blip2_payload, caption_by_path = _run_blip2_caption_backend(
        unique_items=unique_items,
        prompt_python=prompt_python,
        prompt_blip2_model=prompt_blip2_model,
        exphub_root=exphub_root,
    )
    prompt_semantics = [normalize_caption_to_semantic(caption_by_path[str(item["frame_path"])]) for item in unique_items]
    text_embeddings = _normalize_embeddings(clip.encode_texts(prompt_semantics))
    text_by_path = {str(item["frame_path"]): text_embeddings[idx] for idx, item in enumerate(unique_items)}
    semantic_by_path = {str(item["frame_path"]): prompt_semantics[idx] for idx, item in enumerate(unique_items)}

    motion_state_payloads = []
    all_similarity_rows = []
    coverage_gap_count = 0
    state_counter = 0
    for planned in planned_motion_states:
        motion_state_id = str(planned["motion_state_id"])
        motion_state_start = int(planned["start_idx"])
        motion_state_end = int(planned["end_idx"])
        candidates = list(planned["candidates"])
        candidate_rows = _candidate_positions(candidates, motion_state_start, motion_state_end, policy["candidate_stride"])
        semantic_events = list(planned["semantic_events"])
        semantic_states = []
        event_items = []
        for local_idx, event in enumerate(semantic_events):
            semantic_state_start_idx = int(event["frame_idx"])
            semantic_state_end_idx = (
                int(semantic_events[local_idx + 1]["frame_idx"])
                if local_idx + 1 < len(semantic_events)
                else int(motion_state_end)
            )
            frame_path = str(Path(frames[semantic_state_start_idx]).resolve())
            caption = caption_by_path.get(frame_path, "")
            prompt_semantic = semantic_by_path.get(frame_path, normalize_caption_to_semantic(caption))
            text_emb = text_by_path.get(frame_path)
            state_id = "sem_{:04d}".format(state_counter)
            image_pos = pos_to_embed[int(semantic_state_start_idx)]
            text_image_similarity = float(np.dot(normalized_images[image_pos], text_emb)) if text_emb is not None else 0.0
            scan_candidates = [
                idx for idx in candidate_rows if int(semantic_state_start_idx) <= int(idx) <= int(semantic_state_end_idx)
            ]
            rows, min_similarity, max_image_novelty = _similarity_rows(
                scan_candidates,
                normalized_images,
                pos_to_embed,
                text_emb,
                text_image_similarity,
                semantic_state_start_idx,
            )
            for row in rows:
                all_similarity_rows.append(
                    {"motion_state_id": str(motion_state_id), "semantic_state_id": str(state_id), **row}
                )
            state_counter += 1
            event_type = str(event.get("event_type", "semantic_state_start") or "semantic_state_start")
            state_reason = str(event.get("reason", "semantic_state_start") or "semantic_state_start")
            image_novelty_at_start = float(event.get("image_novelty", 0.0) or 0.0)
            max_image_novelty_before_update = float(event.get("max_image_novelty_before_update", 0.0) or 0.0)
            if state_reason == "coverage_gap":
                coverage_gap_count += 1
            semantic_states.append(
                {
                    "semantic_state_id": str(state_id),
                    "motion_state_id": str(motion_state_id),
                    "start_idx": int(semantic_state_start_idx),
                    "end_idx": int(semantic_state_end_idx),
                    "semantic_state_start_idx": int(semantic_state_start_idx),
                    "caption": str(caption),
                    "prompt_semantic": str(prompt_semantic),
                    "reason": str(state_reason),
                    "text_image_similarity_at_start": float(text_image_similarity),
                    "min_text_image_similarity": float(min_similarity),
                    "image_novelty_at_start": float(image_novelty_at_start),
                    "max_image_novelty_before_update": float(max_image_novelty_before_update),
                    "caption_frame": {
                        "frame_idx": int(semantic_state_start_idx),
                        "frame_path": _display_frame_path(frames[semantic_state_start_idx], out_path),
                    },
                    "diagnostics": {
                        "candidate_count": int(len(scan_candidates)),
                        "reference_text_image_similarity": float(text_image_similarity),
                        "low_similarity_count": int(
                            len(
                                [
                                    row
                                    for row in rows
                                    if float(row["text_image_drop"]) > float(policy["text_image_drop_threshold"])
                                ]
                            )
                        ),
                        "max_image_novelty": float(max_image_novelty),
                    },
                }
            )
            event_item = {
                "frame_idx": int(semantic_state_start_idx),
                "abs_time_sec": _abs_time_at(prepare, int(semantic_state_start_idx)),
                "event_type": str(event_type),
                "reason": str(state_reason),
                "semantic_state_id": str(state_id),
                "text_image_similarity": float(text_image_similarity),
                "text_image_drop": 0.0,
                "image_novelty": float(image_novelty_at_start),
            }
            if state_reason == "coverage_gap":
                event_item.update(
                    {
                        "threshold": float(event.get("threshold", policy["image_novelty_threshold"]) or 0.0),
                        "patience": int(event.get("patience", policy["coverage_gap_patience"]) or 0),
                    }
                )
            event_items.append(
                event_item
            )
        motion_state_payloads.append(
            {
                "motion_state_id": str(motion_state_id),
                "start_idx": int(motion_state_start),
                "end_idx": int(motion_state_end),
                "motion_label": str(planned.get("motion_label", "") or "mixed"),
                "semantic_states": semantic_states,
                "semantic_events": event_items,
                "similarity_rows": [row for row in all_similarity_rows if str(row.get("motion_state_id")) == str(motion_state_id)],
            }
        )

    payload = {
        "version": 2,
        "source": "encode.semantic_state.image_novelty.v2",
        "backend_meta": {
            **backend_meta,
            "caption_model": str(blip2_payload.get("model", prompt_blip2_model or DEFAULT_BLIP2_MODEL)),
        },
        "policy": policy,
        "motion_states": motion_state_payloads,
        "similarity_rows": all_similarity_rows,
        "summary": {
            "motion_state_count": int(len(motion_state_payloads)),
            "semantic_state_count": int(sum(len(item.get("semantic_states") or []) for item in motion_state_payloads)),
            "blip2_caption_count": int(len(unique_items)),
            "blip2_batch_count": int(1 if unique_items else 0),
            "coverage_gap_count": int(coverage_gap_count),
            "text_image_drop_count": int(0),
            "elapsed_sec": float(time.time() - started),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
        log_info(
            "semantic states generated: states={} captions={} coverage_gap_count={} path={}".format(
                payload["summary"]["semantic_state_count"],
                payload["summary"]["blip2_caption_count"],
                payload["summary"]["coverage_gap_count"],
                Path(out_path).resolve(),
            )
        )
    return payload


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mainline", action="store_true")
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--motion_segments", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--prompt_python", default=os.environ.get("EXPHUB_PROMPT_PYTHON", ""))
    parser.add_argument("--prompt_blip2_model", default=os.environ.get("EXPHUB_BLIP2_MODEL", DEFAULT_BLIP2_MODEL))
    parser.add_argument("--candidate_stride", type=int, default=DEFAULT_CANDIDATE_STRIDE)
    parser.add_argument("--text_image_drop_threshold", type=float, default=DEFAULT_TEXT_IMAGE_DROP_THRESHOLD)
    parser.add_argument("--min_semantic_update_gap", type=int, default=DEFAULT_MIN_SEMANTIC_UPDATE_GAP)
    parser.add_argument("--low_similarity_patience", type=int, default=DEFAULT_LOW_SIMILARITY_PATIENCE)
    parser.add_argument("--image_novelty_threshold", type=float, default=DEFAULT_IMAGE_NOVELTY_THRESHOLD)
    parser.add_argument("--turn_image_novelty_threshold", type=float, default=DEFAULT_TURN_IMAGE_NOVELTY_THRESHOLD)
    parser.add_argument("--coverage_gap_patience", type=int, default=DEFAULT_COVERAGE_GAP_PATIENCE)
    parser.add_argument("--max_unit_span_frames", type=int, default=DEFAULT_MAX_UNIT_SPAN_FRAMES)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_mainline:
        raise SystemExit("--run-mainline is required")
    build_semantic_anchors(
        prepare_result=read_json_dict(ensure_file(args.prepare_result, "prepare result")),
        motion_segments=read_json_dict(ensure_file(args.motion_segments, "motion segments")),
        frames_dir=args.frames_dir,
        out_path=args.out_path,
        prompt_python=str(args.prompt_python or ""),
        prompt_blip2_model=str(args.prompt_blip2_model or DEFAULT_BLIP2_MODEL),
        candidate_stride=int(args.candidate_stride),
        text_image_drop_threshold=float(args.text_image_drop_threshold),
        min_semantic_update_gap=int(args.min_semantic_update_gap),
        low_similarity_patience=int(args.low_similarity_patience),
        image_novelty_threshold=float(args.image_novelty_threshold),
        turn_image_novelty_threshold=float(args.turn_image_novelty_threshold),
        coverage_gap_patience=int(args.coverage_gap_patience),
        max_unit_span_frames=int(args.max_unit_span_frames),
    )


if __name__ == "__main__":
    main()
