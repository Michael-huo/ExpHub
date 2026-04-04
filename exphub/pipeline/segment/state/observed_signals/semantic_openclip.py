#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exphub.common.logging import log_info
from .kinematics import compute_velocity, resolve_dt, series_stats

try:
    import open_clip
except Exception as e:
    open_clip = None
    _OPEN_CLIP_IMPORT_ERROR = e
else:
    _OPEN_CLIP_IMPORT_ERROR = None


DEFAULT_SEMANTIC_MODEL_NAME = "ViT-B-32"
DEFAULT_SEMANTIC_PRETRAINED = "openai"
DEFAULT_SEMANTIC_BATCH_SIZE = 16


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_embeddings(embeddings):
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def _encode_embeddings(frame_paths, model_name, pretrained, device, batch_size):
    if open_clip is None:
        raise RuntimeError("import open_clip failed: {}".format(_OPEN_CLIP_IMPORT_ERROR))

    log_info("semantic openclip load start: model={} pretrained={} device={}".format(model_name, pretrained, device))
    load_t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    log_info("semantic openclip load done: {:.3f}s".format(time.time() - load_t0))

    log_info("semantic openclip encode start: frames={} batch_size={}".format(len(frame_paths), batch_size))
    encode_t0 = time.time()
    encoded = []
    batch_images = []
    batch_size = max(1, int(batch_size))

    for frame_path in frame_paths:
        with Image.open(str(frame_path)) as img:
            image = img.convert("RGB")
            batch_images.append(preprocess(image))

        if len(batch_images) < batch_size:
            continue

        batch_tensor = torch.stack(batch_images, dim=0).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
        encoded.append(features.detach().cpu().float().numpy())
        batch_images = []

    if batch_images:
        batch_tensor = torch.stack(batch_images, dim=0).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
        encoded.append(features.detach().cpu().float().numpy())

    if not encoded:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    embeddings = np.concatenate(encoded, axis=0).astype(np.float32)
    encode_sec = float(time.time() - encode_t0)
    log_info(
        "semantic openclip encode done: frames={} dim={} elapsed={:.3f}s".format(
            embeddings.shape[0],
            embeddings.shape[1] if embeddings.ndim == 2 else 0,
            encode_sec,
        )
    )
    return embeddings, encode_sec


def _compute_semantic_delta(embeddings):
    num_frames = int(embeddings.shape[0])
    deltas = [0.0 for _ in range(num_frames)]
    if num_frames <= 1:
        return deltas

    normalized = _normalize_embeddings(embeddings)
    for idx in range(1, num_frames):
        cosine = float(np.dot(normalized[idx], normalized[idx - 1]))
        cosine = max(-1.0, min(1.0, cosine))
        deltas[idx] = float(1.0 - cosine)
    return deltas


def compute_semantic_rows(
    frame_paths,
    model_name=DEFAULT_SEMANTIC_MODEL_NAME,
    pretrained=DEFAULT_SEMANTIC_PRETRAINED,
    batch_size=DEFAULT_SEMANTIC_BATCH_SIZE,
    timestamps=None,
    fps=None,
):
    frame_paths = [Path(p).resolve() for p in frame_paths]
    device = _select_device()

    embeddings, encode_sec = _encode_embeddings(
        frame_paths=frame_paths,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
    )

    dt_sec, dt_source = resolve_dt(timestamps=timestamps, fps=fps)
    semantic_displacement = _compute_semantic_delta(embeddings)
    semantic_velocity = compute_velocity(semantic_displacement, dt_sec)
    rows = []
    for idx, frame_path in enumerate(frame_paths):
        rows.append(
            {
                "frame_idx": int(idx),
                "file_name": Path(frame_path).name,
                "semantic_delta": float(semantic_displacement[idx]),
                "semantic_displacement": float(semantic_displacement[idx]),
                "semantic_velocity": float(semantic_velocity[idx]),
            }
        )

    meta = {
        "enabled": True,
        "backend": "openclip",
        "model_name": str(model_name),
        "pretrained": str(pretrained),
        "device": str(device),
        "encode_sec": float(encode_sec),
        "dt_sec": float(dt_sec),
        "dt_source": str(dt_source),
        "num_frames": int(len(frame_paths)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 0,
        "delta_method": "1 - cosine_similarity(e_t, e_{t-1}) using L2-normalized OpenCLIP image embeddings",
        "signal_stats": {
            "semantic_displacement": series_stats(semantic_displacement),
            "semantic_velocity": series_stats(semantic_velocity),
        },
    }
    return rows, meta
