from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info


MOTION_LABELS = ("stop", "forward", "left_turn", "right_turn", "mixed")


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _time_at(prepare_result, idx):
    values = list(_as_dict(prepare_result.get("frame_index_map")).get("prepared_to_abs_time_sec") or [])
    if idx < 0 or idx >= len(values):
        raise RuntimeError("prepare_result frame_index_map missing abs time for frame {}".format(int(idx)))
    return float(values[int(idx)])


def _read_gray(path, max_width=320):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError("failed to read prepared frame: {}".format(path))
    h, w = image.shape[:2]
    if w > int(max_width):
        scale = float(max_width) / float(w)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _flow_pair(prev_gray, cur_gray):
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        cur_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    fx = flow[..., 0].astype(np.float32)
    fy = flow[..., 1].astype(np.float32)
    mag = np.sqrt(fx * fx + fy * fy)

    h, w = fx.shape
    xs = (np.arange(w, dtype=np.float32) - float(w - 1) / 2.0) / max(float(w), 1.0)
    ys = (np.arange(h, dtype=np.float32) - float(h - 1) / 2.0) / max(float(h), 1.0)
    xg, yg = np.meshgrid(xs, ys)
    radial = fx * xg + fy * yg
    rotational = fx * (-yg) + fy * xg

    return {
        "magnitude": float(np.median(mag)),
        "mean_dx": float(np.mean(fx)),
        "abs_dx": float(np.mean(np.abs(fx))),
        "radial": float(np.mean(radial)),
        "rotational": float(np.mean(rotational)),
    }


def _label_window(magnitude, mean_dx, abs_dx):
    magnitude = float(magnitude)
    mean_dx = float(mean_dx)
    abs_dx = max(float(abs_dx), 1e-6)
    directionality = abs(float(mean_dx)) / abs_dx
    if magnitude < 0.18:
        return "stop", 0.90
    if directionality >= 0.34 and magnitude >= 0.24:
        if mean_dx > 0.0:
            return "left_turn", min(0.95, 0.55 + directionality * 0.35)
        return "right_turn", min(0.95, 0.55 + directionality * 0.35)
    if magnitude >= 0.22 and directionality < 0.22:
        return "forward", 0.72
    return "mixed", 0.58


def _aggregate_features(rows):
    if not rows:
        return {
            "magnitude": 0.0,
            "mean_dx": 0.0,
            "abs_dx": 0.0,
            "radial": 0.0,
            "rotational": 0.0,
        }
    keys = ["magnitude", "mean_dx", "abs_dx", "radial", "rotational"]
    return {key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys}


def build_motion_segments(prepare_result, frames_dir, out_path=None):
    started = time.time()
    prepare = _as_dict(prepare_result)
    frame_dir = ensure_dir(frames_dir, "prepare frames dir")
    frames = list_frames_sorted(frame_dir)
    num_frames = int(prepare.get("num_frames", len(frames)) or len(frames))
    if len(frames) != num_frames:
        raise RuntimeError("prepare frame count mismatch: files={} prepare_result={}".format(len(frames), num_frames))

    legal_grid = _as_dict(prepare.get("legal_grid"))
    legal_positions = [int(item) for item in list(legal_grid.get("legal_positions") or [])]
    if len(legal_positions) < 2:
        raise RuntimeError("prepare_result legal_grid requires at least two legal positions")
    if legal_positions[0] != 0:
        raise RuntimeError("legal positions must start at 0")
    if legal_positions[-1] != num_frames - 1:
        raise RuntimeError(
            "last legal position must match final prepared frame: last_legal={} final_frame={}".format(
                legal_positions[-1],
                num_frames - 1,
            )
        )

    pair_features = [None for _ in range(num_frames)]
    prev_gray = _read_gray(frames[0])
    for idx in range(1, num_frames):
        cur_gray = _read_gray(frames[idx])
        pair_features[idx] = _flow_pair(prev_gray, cur_gray)
        prev_gray = cur_gray

    windows = []
    for win_idx in range(len(legal_positions) - 1):
        start_idx = int(legal_positions[win_idx])
        end_idx = int(legal_positions[win_idx + 1])
        rows = [row for row in pair_features[start_idx + 1 : end_idx + 1] if row is not None]
        agg = _aggregate_features(rows)
        label, confidence = _label_window(agg["magnitude"], agg["mean_dx"], agg["abs_dx"])
        windows.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "motion_label": str(label),
                "scores": {
                    "translational": float(agg["magnitude"]),
                    "directional": float(agg["mean_dx"]),
                    "confidence": float(confidence),
                    "radial_diagnostic": float(agg["radial"]),
                    "rotational_diagnostic": float(agg["rotational"]),
                },
            }
        )

    segments = []
    current = None
    current_scores = []
    for window in windows:
        if current is None or window["motion_label"] != current["motion_label"]:
            if current is not None:
                _finish_motion_segment(current, current_scores, segments, prepare)
            current = {
                "start_idx": int(window["start_idx"]),
                "end_idx": int(window["end_idx"]),
                "motion_label": str(window["motion_label"]),
            }
            current_scores = [dict(window["scores"])]
        else:
            current["end_idx"] = int(window["end_idx"])
            current_scores.append(dict(window["scores"]))
    if current is not None:
        _finish_motion_segment(current, current_scores, segments, prepare)

    for idx, item in enumerate(segments):
        item["seg_id"] = "seg_{:04d}".format(int(idx))

    payload = {
        "version": 1,
        "source": "encode.motion_segment.v1",
        "fps": int(prepare.get("target_fps", legal_grid.get("fps", 0)) or 0),
        "motion_labels": list(MOTION_LABELS),
        "segments": segments,
        "summary": {
            "segment_count": int(len(segments)),
            "legal_position_count": int(len(legal_positions)),
            "elapsed_sec": float(time.time() - started),
        },
    }
    if out_path is not None:
        write_json_atomic(out_path, payload, indent=2)
        log_info("motion segments generated: count={} path={}".format(len(segments), Path(out_path).resolve()))
    return payload


def _finish_motion_segment(current, score_rows, segments, prepare_result):
    scores = {}
    for key in ["translational", "directional", "confidence", "radial_diagnostic", "rotational_diagnostic"]:
        scores[key] = float(np.mean([float(row.get(key, 0.0)) for row in score_rows])) if score_rows else 0.0
    segments.append(
        {
            "seg_id": "",
            "start_idx": int(current["start_idx"]),
            "end_idx": int(current["end_idx"]),
            "start_abs_time_sec": _time_at(prepare_result, int(current["start_idx"])),
            "end_abs_time_sec": _time_at(prepare_result, int(current["end_idx"])),
            "motion_label": str(current["motion_label"]),
            "scores": scores,
        }
    )


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mainline", action="store_true")
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--out_path", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_mainline:
        raise SystemExit("--run-mainline is required")
    prepare_result = read_json_dict(ensure_file(args.prepare_result, "prepare result"))
    build_motion_segments(prepare_result, args.frames_dir, args.out_path)


if __name__ == "__main__":
    main()
