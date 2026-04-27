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
DEFAULT_MIN_ANCHOR_GAP = 24
DEFAULT_LOW_SIMILARITY_PATIENCE = 2
DEFAULT_MAX_STATE_DURATION = 96

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
            "anchor_policy": "text_image_similarity_drop",
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


def _choose_duration_anchor(candidates, current_idx, end_idx, max_state_duration):
    limit = min(int(end_idx), int(current_idx) + int(max_state_duration))
    viable = [int(idx) for idx in candidates if int(current_idx) < int(idx) < int(end_idx) and int(idx) <= limit]
    if not viable:
        return None
    return int(max(viable))


def _plan_state_anchors_for_segment(seg_id, seg_start, seg_end, candidates, policy):
    if not candidates or int(candidates[0]) != int(seg_start) or int(candidates[-1]) != int(seg_end):
        raise RuntimeError("motion segment boundaries must be legal: {}".format(seg_id))
    anchors = [{"frame_idx": int(seg_start), "reason": "segment_start", "score": 0.0}]
    current = int(seg_start)
    while int(seg_end) - int(current) > int(policy["max_state_duration"]):
        chosen = _choose_duration_anchor(candidates, current, seg_end, policy["max_state_duration"])
        if chosen is None:
            break
        anchors.append({"frame_idx": int(chosen), "reason": "duration_fallback", "score": 0.0})
        current = int(chosen)
    return anchors


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
        input_json = Path(tmp_dir) / "semantic_anchor_blip2_input.json"
        output_json = Path(tmp_dir) / "semantic_anchor_blip2_output.json"
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
        log_info("BLIP-2 caption backend start semantic_anchor_frames={} model={}".format(len(unique_items), prompt_blip2_model))
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


def _similarity_rows(candidates, normalized_images, pos_to_embed, text_emb, reference_sim, current_anchor_idx):
    rows = []
    min_similarity = float(reference_sim)
    if text_emb is None:
        return rows, float(min_similarity)
    for frame_idx in candidates:
        pos = pos_to_embed.get(int(frame_idx))
        if pos is None:
            continue
        sim = float(np.dot(normalized_images[pos], text_emb))
        min_similarity = min(float(min_similarity), float(sim))
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "similarity": float(sim),
                "drop": float(reference_sim - sim),
                "frames_from_anchor": int(frame_idx) - int(current_anchor_idx),
            }
        )
    return rows, float(min_similarity)


def build_semantic_anchors(
    prepare_result,
    motion_segments,
    frames_dir,
    out_path=None,
    prompt_python="",
    prompt_blip2_model=DEFAULT_BLIP2_MODEL,
    candidate_stride=DEFAULT_CANDIDATE_STRIDE,
    text_image_drop_threshold=DEFAULT_TEXT_IMAGE_DROP_THRESHOLD,
    min_anchor_gap=DEFAULT_MIN_ANCHOR_GAP,
    low_similarity_patience=DEFAULT_LOW_SIMILARITY_PATIENCE,
    max_state_duration=DEFAULT_MAX_STATE_DURATION,
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
        "min_anchor_gap": int(min_anchor_gap),
        "low_similarity_patience": int(low_similarity_patience),
        "max_state_duration": int(max_state_duration),
    }

    clip = _OpenClipSession()
    legal_frame_paths = [frames[idx] for idx in legal_positions]
    image_embeddings = clip.encode_images(legal_frame_paths)
    normalized_images = _normalize_embeddings(image_embeddings)
    pos_to_embed = {int(frame_idx): int(pos) for pos, frame_idx in enumerate(legal_positions)}
    backend_meta = clip.meta(image_embeddings.shape[1] if image_embeddings.ndim == 2 else 0)
    backend_meta["caption_model"] = str(prompt_blip2_model or DEFAULT_BLIP2_MODEL)

    planned_segments = []
    unique_by_path = {}
    unique_items = []
    for raw_segment in list(_as_dict(motion_segments).get("segments") or []):
        segment = _as_dict(raw_segment)
        seg_id = str(segment.get("seg_id", "") or "")
        seg_start = int(segment.get("start_idx"))
        seg_end = int(segment.get("end_idx"))
        candidates = _legal_between(legal_positions, seg_start, seg_end)
        state_anchors = _plan_state_anchors_for_segment(seg_id, seg_start, seg_end, candidates, policy)
        planned_segments.append(
            {
                "seg_id": str(seg_id),
                "start_idx": int(seg_start),
                "end_idx": int(seg_end),
                "candidates": candidates,
                "state_anchors": state_anchors,
            }
        )
        for anchor in state_anchors:
            frame_idx = int(anchor["frame_idx"])
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

    segment_payloads = []
    all_similarity_rows = []
    state_counter = 0
    for planned in planned_segments:
        seg_id = str(planned["seg_id"])
        seg_start = int(planned["start_idx"])
        seg_end = int(planned["end_idx"])
        candidates = list(planned["candidates"])
        candidate_rows = _candidate_positions(candidates, seg_start, seg_end, policy["candidate_stride"])
        state_anchors = list(planned["state_anchors"])
        semantic_states = []
        anchor_items = []
        for local_idx, anchor in enumerate(state_anchors):
            anchor_idx = int(anchor["frame_idx"])
            next_anchor_idx = int(state_anchors[local_idx + 1]["frame_idx"]) if local_idx + 1 < len(state_anchors) else int(seg_end)
            frame_path = str(Path(frames[anchor_idx]).resolve())
            caption = caption_by_path.get(frame_path, "")
            prompt_semantic = semantic_by_path.get(frame_path, normalize_caption_to_semantic(caption))
            text_emb = text_by_path.get(frame_path)
            image_pos = pos_to_embed[int(anchor_idx)]
            anchor_sim = float(np.dot(normalized_images[image_pos], text_emb)) if text_emb is not None else 0.0
            scan_candidates = [idx for idx in candidate_rows if int(anchor_idx) <= int(idx) <= int(next_anchor_idx)]
            rows, min_similarity = _similarity_rows(scan_candidates, normalized_images, pos_to_embed, text_emb, anchor_sim, anchor_idx)
            for row in rows:
                all_similarity_rows.append({"seg_id": str(seg_id), "semantic_state_id": "sem_{:04d}".format(state_counter), **row})
            state_id = "sem_{:04d}".format(state_counter)
            state_counter += 1
            semantic_states.append(
                {
                    "semantic_state_id": str(state_id),
                    "seg_id": str(seg_id),
                    "start_idx": int(anchor_idx),
                    "end_idx": int(next_anchor_idx),
                    "anchor_idx": int(anchor_idx),
                    "caption": str(caption),
                    "prompt_semantic": str(prompt_semantic),
                    "reason": str(anchor.get("reason", "segment_start") or "segment_start"),
                    "similarity_at_anchor": float(anchor_sim),
                    "min_similarity": float(min_similarity),
                    "caption_frame": {
                        "frame_idx": int(anchor_idx),
                        "frame_path": _display_frame_path(frames[anchor_idx], out_path),
                    },
                    "diagnostics": {
                        "candidate_count": int(len(scan_candidates)),
                        "reference_similarity": float(anchor_sim),
                        "low_similarity_count": int(
                            len([row for row in rows if float(row["drop"]) > float(policy["text_image_drop_threshold"])])
                        ),
                    },
                }
            )
            anchor_items.append(
                {
                    "frame_idx": int(anchor_idx),
                    "abs_time_sec": _abs_time_at(prepare, int(anchor_idx)),
                    "reason": str(anchor.get("reason", "segment_start") or "segment_start"),
                    "semantic_state_id": str(state_id),
                    "score": float(anchor_sim),
                }
            )
        anchor_items.append(
            {
                "frame_idx": int(seg_end),
                "abs_time_sec": _abs_time_at(prepare, int(seg_end)),
                "reason": "segment_boundary",
            }
        )
        segment_payloads.append(
            {
                "seg_id": str(seg_id),
                "start_idx": int(seg_start),
                "end_idx": int(seg_end),
                "semantic_states": semantic_states,
                "anchors": anchor_items,
                "anchor_indices": [int(item["frame_idx"]) for item in anchor_items],
                "anchor_items": anchor_items,
                "similarity_rows": [row for row in all_similarity_rows if str(row.get("seg_id")) == str(seg_id)],
            }
        )

    payload = {
        "version": 2,
        "source": "encode.semantic_anchor.text_image.v1",
        "backend_meta": {
            **backend_meta,
            "caption_model": str(blip2_payload.get("model", prompt_blip2_model or DEFAULT_BLIP2_MODEL)),
        },
        "policy": policy,
        "segments": segment_payloads,
        "similarity_rows": all_similarity_rows,
        "summary": {
            "motion_segment_count": int(len(segment_payloads)),
            "semantic_state_count": int(sum(len(item.get("semantic_states") or []) for item in segment_payloads)),
            "blip2_caption_count": int(len(unique_items)),
            "elapsed_sec": float(time.time() - started),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
        log_info(
            "semantic states generated: states={} captions={} path={}".format(
                payload["summary"]["semantic_state_count"],
                payload["summary"]["blip2_caption_count"],
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
    parser.add_argument("--min_anchor_gap", type=int, default=DEFAULT_MIN_ANCHOR_GAP)
    parser.add_argument("--low_similarity_patience", type=int, default=DEFAULT_LOW_SIMILARITY_PATIENCE)
    parser.add_argument("--max_state_duration", type=int, default=DEFAULT_MAX_STATE_DURATION)
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
        min_anchor_gap=int(args.min_anchor_gap),
        low_similarity_patience=int(args.low_similarity_patience),
        max_state_duration=int(args.max_state_duration),
    )


if __name__ == "__main__":
    main()
