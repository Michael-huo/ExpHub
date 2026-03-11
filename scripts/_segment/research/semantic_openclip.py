#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from scripts._common import log_info, log_warn

try:
    import open_clip
except Exception as e:
    open_clip = None
    _OPEN_CLIP_IMPORT_ERROR = e
else:
    _OPEN_CLIP_IMPORT_ERROR = None


DEFAULT_SEMANTIC_MODEL_NAME = "ViT-B-32"
DEFAULT_SEMANTIC_PRETRAINED = "openai"
DEFAULT_SEMANTIC_CACHE_NAME = "semantic_embeddings.npz"
DEFAULT_SEMANTIC_BATCH_SIZE = 16
DEFAULT_SEMANTIC_DENSITY_EPS = 0.03
DEFAULT_SEMANTIC_DENSITY_ALPHA = 0.7
DEFAULT_SEMANTIC_DENSITY_BETA = 0.3


def _moving_average(values, window_size):
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    out = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        window = values[left:right]
        if not window:
            out.append(0.0)
        else:
            out.append(float(sum(window) / float(len(window))))
    return out, window_size


def _minmax_normalize(values):
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    arr = (arr - vmin) / float(vmax - vmin)
    return [float(x) for x in arr.tolist()]


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


def _resolve_dt(timestamps=None, fps=None):
    if fps is not None:
        try:
            fps_value = float(fps)
        except Exception:
            fps_value = 0.0
        if fps_value > 0.0:
            return float(1.0 / fps_value), "fps"

    if timestamps:
        diffs = []
        prev_t = None
        for ts_sec in timestamps:
            cur_t = float(ts_sec)
            if prev_t is not None:
                dt = float(cur_t - prev_t)
                if dt > 1e-9:
                    diffs.append(dt)
            prev_t = cur_t
        if diffs:
            return float(np.median(np.asarray(diffs, dtype=np.float32))), "timestamps_median"

    return 1.0, "unit_dt_fallback"


def _cache_matches(cache_obj, frame_paths, model_name, pretrained):
    try:
        frame_idx = cache_obj["frame_idx"]
        file_name = cache_obj["file_name"]
        embeddings = cache_obj["embeddings"]
    except Exception:
        return False

    if len(frame_idx) != len(frame_paths) or len(file_name) != len(frame_paths):
        return False
    if embeddings.ndim != 2 or embeddings.shape[0] != len(frame_paths):
        return False

    expected_indices = np.arange(len(frame_paths), dtype=np.int32)
    if not np.array_equal(frame_idx.astype(np.int32), expected_indices):
        return False

    cached_names = [str(x) for x in file_name.tolist()]
    expected_names = [Path(p).name for p in frame_paths]
    if cached_names != expected_names:
        return False

    cached_model_name = cache_obj["model_name"] if "model_name" in cache_obj.files else None
    cached_pretrained = cache_obj["pretrained"] if "pretrained" in cache_obj.files else None
    if cached_model_name is not None and str(cached_model_name.tolist()) != str(model_name):
        return False
    if cached_pretrained is not None and str(cached_pretrained.tolist()) != str(pretrained):
        return False
    return True


def _load_embedding_cache(cache_path, frame_paths, model_name, pretrained):
    if not Path(cache_path).is_file():
        return None

    try:
        with np.load(str(cache_path), allow_pickle=False) as cache_obj:
            if not _cache_matches(cache_obj, frame_paths, model_name, pretrained):
                return None
            return np.asarray(cache_obj["embeddings"], dtype=np.float32)
    except Exception as e:
        log_warn("semantic cache load failed, will rebuild: {} ({})".format(cache_path, e))
        return None


def _write_embedding_cache(cache_path, frame_paths, embeddings, model_name, pretrained):
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp_path),
        frame_idx=np.arange(len(frame_paths), dtype=np.int32),
        file_name=np.asarray([Path(p).name for p in frame_paths]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_name=np.asarray(model_name),
        pretrained=np.asarray(pretrained),
    )
    os.replace(str(tmp_path), str(cache_path))


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
    log_info("semantic openclip encode done: frames={} dim={} elapsed={:.3f}s".format(
        embeddings.shape[0],
        embeddings.shape[1] if embeddings.ndim == 2 else 0,
        encode_sec,
    ))
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


def _compute_velocity(displacement, dt):
    dt = max(float(dt), 1e-9)
    values = []
    for value in displacement:
        values.append(float(value) / dt)
    return values


def _compute_acceleration(velocity, dt):
    dt = max(float(dt), 1e-9)
    values = [0.0 for _ in velocity]
    for idx in range(1, len(velocity)):
        values[idx] = float(abs(float(velocity[idx]) - float(velocity[idx - 1])) / dt)
    return values


def _cumulative_sum(values):
    out = []
    total = 0.0
    for value in values:
        total += float(value)
        out.append(float(total))
    return out


def _series_stats(values):
    if not values:
        return {
            "mean": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def compute_semantic_rows(
    frame_paths,
    cache_dir,
    smooth_window=5,
    model_name=DEFAULT_SEMANTIC_MODEL_NAME,
    pretrained=DEFAULT_SEMANTIC_PRETRAINED,
    batch_size=DEFAULT_SEMANTIC_BATCH_SIZE,
    timestamps=None,
    fps=None,
    density_eps=DEFAULT_SEMANTIC_DENSITY_EPS,
    density_alpha=DEFAULT_SEMANTIC_DENSITY_ALPHA,
    density_beta=DEFAULT_SEMANTIC_DENSITY_BETA,
):
    frame_paths = [Path(p).resolve() for p in frame_paths]
    cache_dir = Path(cache_dir).resolve()
    cache_path = cache_dir / DEFAULT_SEMANTIC_CACHE_NAME
    device = _select_device()

    cache_hit = False
    cache_t0 = time.time()
    embeddings = _load_embedding_cache(cache_path, frame_paths, model_name, pretrained)
    cache_lookup_sec = float(time.time() - cache_t0)
    encode_sec = 0.0

    if embeddings is not None:
        cache_hit = True
        log_info("semantic cache hit: {}".format(cache_path))
    else:
        log_info("semantic cache miss: {}".format(cache_path))
        embeddings, encode_sec = _encode_embeddings(
            frame_paths=frame_paths,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
        )
        _write_embedding_cache(cache_path, frame_paths, embeddings, model_name, pretrained)
        log_info("semantic cache write: {}".format(cache_path))

    dt_sec, dt_source = _resolve_dt(timestamps=timestamps, fps=fps)
    semantic_displacement = _compute_semantic_delta(embeddings)
    semantic_smooth, actual_window = _moving_average(semantic_displacement, smooth_window)
    semantic_velocity = _compute_velocity(semantic_displacement, dt_sec)
    semantic_velocity_smooth, velocity_window = _moving_average(semantic_velocity, smooth_window)
    semantic_acceleration = _compute_acceleration(semantic_velocity, dt_sec)
    semantic_acceleration_smooth, acceleration_window = _moving_average(semantic_acceleration, smooth_window)
    semantic_velocity_norm = _minmax_normalize(semantic_velocity_smooth)
    semantic_acceleration_norm = _minmax_normalize(semantic_acceleration_smooth)
    semantic_density = []
    for idx in range(len(frame_paths)):
        semantic_density.append(
            float(density_eps)
            + float(density_alpha) * float(semantic_velocity_norm[idx])
            + float(density_beta) * float(semantic_acceleration_norm[idx])
        )
    semantic_action = _cumulative_sum(semantic_density)
    rows = []
    for idx, frame_path in enumerate(frame_paths):
        rows.append(
            {
                "frame_idx": int(idx),
                "file_name": Path(frame_path).name,
                "semantic_delta": float(semantic_displacement[idx]),
                "semantic_smooth": float(semantic_smooth[idx]),
                "semantic_displacement": float(semantic_displacement[idx]),
                "semantic_velocity": float(semantic_velocity[idx]),
                "semantic_velocity_smooth": float(semantic_velocity_smooth[idx]),
                "semantic_velocity_norm": float(semantic_velocity_norm[idx]),
                "semantic_acceleration": float(semantic_acceleration[idx]),
                "semantic_acceleration_smooth": float(semantic_acceleration_smooth[idx]),
                "semantic_acceleration_norm": float(semantic_acceleration_norm[idx]),
                "semantic_density": float(semantic_density[idx]),
                "semantic_action": float(semantic_action[idx]),
            }
        )

    meta = {
        "enabled": True,
        "backend": "openclip",
        "model_name": str(model_name),
        "pretrained": str(pretrained),
        "device": str(device),
        "cache_path": str(cache_path),
        "cache_hit": bool(cache_hit),
        "cache_lookup_sec": float(cache_lookup_sec),
        "encode_sec": float(encode_sec),
        "dt_sec": float(dt_sec),
        "dt_source": str(dt_source),
        "num_frames": int(len(frame_paths)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 0,
        "delta_method": "1 - cosine_similarity(e_t, e_{t-1}) using L2-normalized OpenCLIP image embeddings",
        "smoothing": {
            "method": "moving_average",
            "window_size": int(actual_window),
        },
        "velocity_smoothing": {
            "method": "moving_average",
            "window_size": int(velocity_window),
        },
        "acceleration_smoothing": {
            "method": "moving_average",
            "window_size": int(acceleration_window),
        },
        "density": {
            "eps": float(density_eps),
            "alpha": float(density_alpha),
            "beta": float(density_beta),
        },
        "signal_stats": {
            "semantic_displacement": _series_stats(semantic_displacement),
            "semantic_velocity": _series_stats(semantic_velocity),
            "semantic_acceleration": _series_stats(semantic_acceleration),
            "semantic_density": _series_stats(semantic_density),
        },
        "semantic_peak_enabled": False,
    }
    return rows, meta
