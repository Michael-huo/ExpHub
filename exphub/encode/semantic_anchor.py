from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_warn


DEFAULT_MAX_ANCHORS_PER_SEGMENT = 8


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _normalize_embeddings(embeddings):
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _encode_openclip(frame_paths, model_name="ViT-B-32", pretrained="openai", batch_size=16):
    import torch
    from PIL import Image
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    encoded = []
    batch = []
    batch_size = max(1, int(batch_size))
    for path in frame_paths:
        with Image.open(str(path)) as image_obj:
            batch.append(preprocess(image_obj.convert("RGB")))
        if len(batch) < batch_size:
            continue
        tensor = torch.stack(batch, dim=0).to(device)
        with torch.no_grad():
            encoded.append(model.encode_image(tensor).detach().cpu().float().numpy())
        batch = []
    if batch:
        tensor = torch.stack(batch, dim=0).to(device)
        with torch.no_grad():
            encoded.append(model.encode_image(tensor).detach().cpu().float().numpy())
    if not encoded:
        return np.zeros((0, 0), dtype=np.float32), {"backend": "openclip", "device": device, "embedding_dim": 0}
    embeddings = np.concatenate(encoded, axis=0).astype(np.float32)
    return embeddings, {"backend": "openclip", "device": device, "embedding_dim": int(embeddings.shape[1])}


def _encode_fallback(frame_paths):
    from PIL import Image

    vectors = []
    for path in frame_paths:
        with Image.open(str(path)) as image_obj:
            image = image_obj.convert("RGB").resize((64, 36))
            arr = np.asarray(image, dtype=np.float32) / 255.0
        hist = []
        for channel in range(3):
            values, _ = np.histogram(arr[..., channel], bins=16, range=(0.0, 1.0), density=False)
            hist.extend(values.astype(np.float32).tolist())
        vectors.append(np.asarray(hist, dtype=np.float32))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), {"backend": "rgb_histogram_fallback", "embedding_dim": 0}
    embeddings = np.stack(vectors, axis=0).astype(np.float32)
    return embeddings, {"backend": "rgb_histogram_fallback", "embedding_dim": int(embeddings.shape[1])}


def _semantic_gap(normalized_embeddings, left_pos, cur_pos, right_pos):
    left_gap = 0.0
    right_gap = 0.0
    if cur_pos > left_pos:
        left_gap = float(1.0 - np.dot(normalized_embeddings[cur_pos], normalized_embeddings[left_pos]))
    if right_pos > cur_pos:
        right_gap = float(1.0 - np.dot(normalized_embeddings[right_pos], normalized_embeddings[cur_pos]))
    return max(0.0, left_gap), max(0.0, right_gap), max(0.0, left_gap + right_gap)


def _legal_between(legal_positions, start_idx, end_idx):
    return [idx for idx in legal_positions if int(start_idx) <= int(idx) <= int(end_idx)]


def _add_duration_fallback(anchor_map, candidates, max_delta):
    ordered = sorted(int(idx) for idx in anchor_map.keys())
    changed = True
    while changed:
        changed = False
        ordered = sorted(int(idx) for idx in anchor_map.keys())
        for left, right in zip(ordered[:-1], ordered[1:]):
            if int(right) - int(left) <= int(max_delta):
                continue
            viable = [idx for idx in candidates if int(left) < int(idx) < int(right) and int(idx) - int(left) <= int(max_delta)]
            if not viable:
                raise RuntimeError(
                    "duration_fallback cannot split anchor span: left={} right={} max_delta={}".format(left, right, max_delta)
                )
            chosen = max(viable)
            anchor_map[int(chosen)] = {
                "frame_idx": int(chosen),
                "score": float(anchor_map.get(int(chosen), {}).get("score", 0.0)),
                "reason": "duration_fallback",
            }
            changed = True
            break


def build_semantic_anchors(
    prepare_result,
    motion_segments,
    frames_dir,
    out_path=None,
    max_anchors_per_segment=DEFAULT_MAX_ANCHORS_PER_SEGMENT,
):
    started = time.time()
    prepare = _as_dict(prepare_result)
    frame_dir = ensure_dir(frames_dir, "prepare frames dir")
    frames = list_frames_sorted(frame_dir)
    legal_grid = _as_dict(prepare.get("legal_grid"))
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    allowed_deltas = [int(item) for item in list(legal_grid.get("allowed_delta_indices") or [])]
    if not legal_positions or not allowed_deltas:
        raise RuntimeError("prepare_result legal_grid missing legal_positions or allowed_delta_indices")

    legal_frame_paths = [frames[idx] for idx in legal_positions]
    try:
        embeddings, backend_meta = _encode_openclip(legal_frame_paths)
    except Exception as exc:
        log_warn("semantic OpenCLIP backend failed; using deterministic image fallback: {}".format(exc))
        embeddings, backend_meta = _encode_fallback(legal_frame_paths)
    normalized = _normalize_embeddings(embeddings)
    pos_to_embed = {int(frame_idx): int(pos) for pos, frame_idx in enumerate(legal_positions)}
    max_delta = int(max(allowed_deltas))

    segment_payloads = []
    all_gap_rows = []
    for raw_segment in list(_as_dict(motion_segments).get("segments") or []):
        segment = _as_dict(raw_segment)
        seg_id = str(segment.get("seg_id", "") or "")
        seg_start = int(segment.get("start_idx"))
        seg_end = int(segment.get("end_idx"))
        candidates = _legal_between(legal_positions, seg_start, seg_end)
        if candidates[0] != seg_start or candidates[-1] != seg_end:
            raise RuntimeError("motion segment boundaries must be legal: {}".format(seg_id))
        anchor_map = {
            int(seg_start): {"frame_idx": int(seg_start), "score": 0.0, "reason": "segment_boundary"},
            int(seg_end): {"frame_idx": int(seg_end), "score": 0.0, "reason": "segment_boundary"},
        }

        details = []
        internal = candidates[1:-1]
        scores = []
        for idx in internal:
            pos = pos_to_embed[int(idx)]
            left_pos = max(0, pos - 1)
            right_pos = min(len(legal_positions) - 1, pos + 1)
            left_gap, right_gap, score = _semantic_gap(normalized, left_pos, pos, right_pos)
            row = {
                "frame_idx": int(idx),
                "left_gap": float(left_gap),
                "right_gap": float(right_gap),
                "score": float(score),
            }
            details.append(row)
            all_gap_rows.append({"seg_id": str(seg_id), **row})
            scores.append(float(score))

        threshold = 0.18
        if scores:
            threshold = max(0.08, float(np.mean(scores)) + float(np.std(scores)) * 0.75)
        semantic_rows = [row for row in details if float(row["score"]) >= threshold]
        semantic_rows.sort(key=lambda row: float(row["score"]), reverse=True)
        max_semantic = max(0, int(max_anchors_per_segment) - 2)
        for row in semantic_rows[:max_semantic]:
            frame_idx = int(row["frame_idx"])
            anchor_map[frame_idx] = {
                "frame_idx": int(frame_idx),
                "score": float(row["score"]),
                "reason": "semantic_gain",
            }

        _add_duration_fallback(anchor_map, candidates, max_delta=max_delta)

        fallback_indices = {idx for idx, item in anchor_map.items() if item.get("reason") == "duration_fallback"}
        boundary_indices = {seg_start, seg_end}
        semantic_indices = [
            idx
            for idx, item in sorted(anchor_map.items(), key=lambda pair: float(pair[1].get("score", 0.0)), reverse=True)
            if item.get("reason") == "semantic_gain"
        ]
        keep = set(boundary_indices) | set(fallback_indices)
        for idx in semantic_indices:
            if len(keep) >= int(max_anchors_per_segment):
                break
            keep.add(int(idx))
        anchor_items = [dict(anchor_map[idx]) for idx in sorted(keep)]
        segment_payloads.append(
            {
                "seg_id": str(seg_id),
                "start_idx": int(seg_start),
                "end_idx": int(seg_end),
                "anchors": anchor_items,
                "anchor_indices": [int(item["frame_idx"]) for item in anchor_items],
                "anchor_items": anchor_items,
                "details": details,
            }
        )

    payload = {
        "version": 1,
        "source": "encode.semantic_anchor.v1",
        "segments": segment_payloads,
        "gap_rows": all_gap_rows,
        "backend_meta": backend_meta,
        "policy": {
            "max_anchors_per_segment": int(max_anchors_per_segment),
            "duration_fallback_max_delta": int(max_delta),
            "anchor_reasons": ["segment_boundary", "semantic_gain", "duration_fallback"],
        },
        "summary": {
            "segment_count": int(len(segment_payloads)),
            "anchor_count": int(sum(len(item.get("anchor_items") or []) for item in segment_payloads)),
            "elapsed_sec": float(time.time() - started),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
        log_info("semantic anchors generated: count={} path={}".format(payload["summary"]["anchor_count"], Path(out_path).resolve()))
    return payload


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mainline", action="store_true")
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--motion_segments", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--max_anchors_per_segment", type=int, default=DEFAULT_MAX_ANCHORS_PER_SEGMENT)
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
        max_anchors_per_segment=int(args.max_anchors_per_segment),
    )


if __name__ == "__main__":
    main()
