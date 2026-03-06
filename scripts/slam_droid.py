#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slam_droid.py

修复点：让 ExpHub 脚本在行为上与 my_exp.py（DROID 官方 demo 改版）一致，避免“同一数据集结果差异巨大”。

你遇到的“my_exp.py 正常直线 / slam_droid.py 轨迹混乱”，最核心原因通常是：
- my_exp.py：只要 calib 里带畸变系数，就默认执行 cv2.undistort
- slam_droid.py：默认不 undistort（需显式 --undistort）

此外，本脚本还提供两处“对齐 demo 行为”的开关：
- resize 插值默认用 linear（与 my_exp.py 相同；原版脚本下采样会用 area）
- intrinsics 缩放默认使用“resize 后、裁 8 倍数之前的 w1/h1”（与 my_exp.py 相同）

输出结构遵循 ExpHub track 目录规范：
<slam_out_dir>/traj_est.tum
<slam_out_dir>/traj_est.npz
<slam_out_dir>/run_meta.json

用法：
python run_droid_exphub_v2.py \
  --segment_dir /path/to/ExpHub/experiments/scand/your_seq/your_exp/segment \
  --droid_repo  /path/to/DROID-SLAM \
  --weights     droid.pth \
  --out_dir     /path/to/ExpHub/experiments/scand/gt_baseline_scand_seq01_dur16s_w768_h480_fps25_v1 \
  --slam_out_dir /path/to/ExpHub/experiments/scand/gt_baseline_scand_seq01_dur16s_w768_h480_fps25_v1/slam/ori \
  --disable_vis

若你明确想关去畸变：
  --undistort_mode off
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import lietorch
from _common import get_platform_config, log_err, log_info, log_prog, log_warn

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


_DIGITS_RE = re.compile(r"(\d+)")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _parse_frame_index(name: str) -> int:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = _DIGITS_RE.search(stem)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _sorted_frame_files(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    items = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    items.sort(key=lambda p: (_parse_frame_index(p.name), p.name))
    return items


def _load_calib(calib_path: Path) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    arr = np.loadtxt(str(calib_path), delimiter=" ")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"calib must have >=4 numbers, got {arr.size}: {calib_path}")
    fx, fy, cx, cy = map(float, arr[:4])
    dist = arr[4:].astype(np.float64) if arr.size > 4 else None
    return fx, fy, cx, cy, dist


def _load_timestamps_list(ts_path: Path) -> List[float]:
    out: List[float] = []
    for line in _read_text(ts_path).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        out.append(float(parts[-1]))
    return out


def _resolve_timestamp(seq_i: int, file_frame_idx: int, timestamps: Optional[List[float]], fps_fallback: float) -> float:
    if timestamps:
        if 0 <= file_frame_idx < len(timestamps):
            return float(timestamps[file_frame_idx])
        if 0 <= seq_i < len(timestamps):
            return float(timestamps[seq_i])

    if fps_fallback and fps_fallback > 0:
        idx = file_frame_idx if file_frame_idx >= 0 else seq_i
        return float(idx) / float(fps_fallback)

    return float(seq_i)

def _maybe_tqdm(it: Iterable, disable: bool, desc: str, total: Optional[int] = None):
    if disable or tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, bar_format=_BAR_FORMAT)

def mat2quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    q = np.empty((4,), dtype=np.float64)
    tr = np.trace(R)

    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q[:] = [qx, qy, qz, qw]
    q /= np.linalg.norm(q)
    return q


def save_trajectory(traj_est, timestamps_sec: List[float], tum_path: Path, npz_path: Path) -> None:
    traj_tensor = torch.from_numpy(traj_est) if isinstance(traj_est, np.ndarray) else traj_est
    se3 = lietorch.SE3(traj_tensor)
    T = se3.matrix().cpu().numpy()

    if T.shape[1:] == (3, 4):
        last_row = np.array([0, 0, 0, 1], dtype=T.dtype)
        last_row = np.broadcast_to(last_row, (T.shape[0], 1, 4))
        T = np.concatenate([T, last_row], axis=1)

    ts = np.asarray(timestamps_sec, dtype=float)
    if T.shape[0] != ts.shape[0]:
        raise RuntimeError(f"pose 数量 {T.shape[0]} 和时间戳数量 {ts.shape[0]} 不一致")

    _ensure_dir(tum_path.parent)
    _ensure_dir(npz_path.parent)

    np.savez(str(npz_path), tstamps=ts, poses=T)

    with open(str(tum_path), "w", encoding="utf-8") as f:
        for t, Ti in zip(ts, T):
            R = Ti[:3, :3]
            tvec = Ti[:3, 3]
            qx, qy, qz, qw = mat2quat_xyzw(R)
            f.write(
                f"{t:.9f} "
                f"{tvec[0]:.9f} {tvec[1]:.9f} {tvec[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def show_image(image_chw: torch.Tensor) -> None:
    img = image_chw.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", img / 255.0)
    cv2.waitKey(1)


@dataclass
class StreamConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[np.ndarray]
    timestamps: Optional[List[float]]
    fps_fallback: float
    undistort_mode: str  # auto/on/off
    resize_interp: str   # linear/area
    intr_scale_mode: str # demo/correct
    target_area: int = 384 * 512
    divisible: int = 8


def _need_undistort(cfg: StreamConfig) -> bool:
    has_dist = cfg.dist is not None and cfg.dist.size > 0
    if cfg.undistort_mode == "on":
        return has_dist
    if cfg.undistort_mode == "off":
        return False
    return has_dist  # auto


def _resize_interp_flag(cfg: StreamConfig) -> int:
    return cv2.INTER_AREA if cfg.resize_interp == "area" else cv2.INTER_LINEAR


def _make_intrinsics_tensor_demo(cfg: StreamConfig, w0: int, h0: int, w1_pre: int, h1_pre: int) -> torch.Tensor:
    intr = torch.as_tensor([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=torch.float32)
    intr[0::2] *= (w1_pre / w0)
    intr[1::2] *= (h1_pre / h0)
    return intr


def _make_intrinsics_tensor_correct(cfg: StreamConfig, w0: int, h0: int, w1: int, h1: int) -> torch.Tensor:
    intr = torch.as_tensor([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=torch.float32)
    intr[0::2] *= (w1 / w0)
    intr[1::2] *= (h1 / h0)
    return intr


def droid_stream(files: List[Path], cfg: StreamConfig):
    for seq_i, p in enumerate(files):
        frame_idx = _parse_frame_index(p.name)

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if _need_undistort(cfg):
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = cfg.fx
            K[1, 1] = cfg.fy
            K[0, 2] = cfg.cx
            K[1, 2] = cfg.cy
            img = cv2.undistort(img, K, cfg.dist)

        h0, w0 = img.shape[:2]

        s = np.sqrt(cfg.target_area / float(h0 * w0))
        h1_pre = int(h0 * s)
        w1_pre = int(w0 * s)

        img = cv2.resize(img, (w1_pre, h1_pre), interpolation=_resize_interp_flag(cfg))

        h1 = h1_pre - (h1_pre % cfg.divisible)
        w1 = w1_pre - (w1_pre % cfg.divisible)
        img = img[:h1, :w1]

        image = torch.as_tensor(img).permute(2, 0, 1)

        if cfg.intr_scale_mode == "demo":
            intrinsics = _make_intrinsics_tensor_demo(cfg, w0, h0, w1_pre, h1_pre)
        else:
            intrinsics = _make_intrinsics_tensor_correct(cfg, w0, h0, w1, h1)

        ts = _resolve_timestamp(seq_i, frame_idx, cfg.timestamps, cfg.fps_fallback)
        yield seq_i, image[None], intrinsics, float(ts)


def _resolve_segment_paths(segment_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Resolve frames/calib/timestamps.

    Layout (no meta/):
      - <root>/frames
      - <root>/calib.txt
      - <root>/timestamps.txt (optional)

    Also accepts passing <root>/frames directly.
    """
    if segment_dir.name == "frames":
        frames_dir = segment_dir
        root = segment_dir.parent
    else:
        frames_dir = segment_dir / "frames"
        root = segment_dir

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    calib = root / "calib.txt"
    if not calib.exists():
        raise FileNotFoundError(f"calib.txt not found: {calib}")

    ts = root / "timestamps.txt"
    if not ts.exists():
        ts = None
    return frames_dir, calib, ts


def _setup_droid_import(droid_repo: Path) -> None:
    droid_repo = droid_repo.resolve()
    if not droid_repo.exists():
        raise FileNotFoundError(f"droid_repo not found: {droid_repo}")

    droid_slam_pkg = droid_repo / "droid_slam"
    sys.path.append(str(droid_slam_pkg if droid_slam_pkg.exists() else droid_repo))


def main():
    try:
        cfg = get_platform_config()
        default_repo = cfg.get("repos", {}).get("droid_slam", "")
        default_weights = cfg.get("models", {}).get("droid", {}).get("path", "")
    except Exception:
        default_repo = ""
        default_weights = "droid.pth"

    parser = argparse.ArgumentParser()

    parser.add_argument("--segment_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--slam_out_dir",
        type=str,
        default="",
        help="Final SLAM output directory. If empty, fallback to <out_dir>/slam for compatibility.",
    )

    parser.add_argument("--droid_repo", type=str, default=default_repo)
    parser.add_argument("--weights", type=str, default=default_weights)

    parser.add_argument("--t0", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)

    parser.add_argument("--fps", type=float, default=0.0)

    parser.add_argument("--undistort_mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--resize_interp", choices=["linear", "area"], default="linear")
    parser.add_argument("--intr_scale_mode", choices=["demo", "correct"], default="demo")

    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--no_tqdm", action="store_true")

    args = parser.parse_args()
    args.stereo = False

    if not str(args.droid_repo).strip():
        raise SystemExit(
            "[ERR] --droid_repo is empty. Set repos.droid_slam in config/platform.yaml or pass --droid_repo."
        )
    if not str(args.weights).strip():
        raise SystemExit(
            "[ERR] --weights is empty. Set models.droid.path in config/platform.yaml or pass --weights."
        )

    segment_dir = Path(args.segment_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    slam_out_dir = Path(args.slam_out_dir).resolve() if str(args.slam_out_dir).strip() else (out_dir / "slam")
    droid_repo = Path(args.droid_repo).resolve()

    frames_dir, calib_path, timestamps_path = _resolve_segment_paths(segment_dir)

    fx, fy, cx, cy, dist = _load_calib(calib_path)
    timestamps = _load_timestamps_list(timestamps_path) if timestamps_path else None

    files = _sorted_frame_files(frames_dir)
    if args.t0 > 0:
        files = files[args.t0:]
    if args.stride > 1:
        files = files[::args.stride]
    if args.max_frames and args.max_frames > 0:
        files = files[:args.max_frames]

    if not files:
        raise RuntimeError(f"No images found under: {frames_dir}")

    _setup_droid_import(droid_repo)
    from droid import Droid  # noqa

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = droid_repo / weights_path
    args.weights = str(weights_path)

    slam_dir = slam_out_dir
    _ensure_dir(slam_dir)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "segment_dir": str(segment_dir),
        "frames_dir": str(frames_dir),
        "calib": str(calib_path),
        "timestamps": str(timestamps_path) if timestamps_path else "",
        "out_dir": str(out_dir),
        "slam_out_dir": str(slam_dir),
        "droid_repo": str(droid_repo),
        "weights": str(weights_path),
        "t0": int(args.t0),
        "stride": int(args.stride),
        "max_frames": int(args.max_frames),
        "fx_fy_cx_cy": [fx, fy, cx, cy],
        "dist_len": int(dist.size) if dist is not None else 0,
        "undistort_mode": args.undistort_mode,
        "resize_interp": args.resize_interp,
        "intr_scale_mode": args.intr_scale_mode,
    }

    torch.multiprocessing.set_start_method("spawn", force=True)

    cfg = StreamConfig(
        fx=fx, fy=fy, cx=cx, cy=cy,
        dist=dist,
        timestamps=timestamps,
        fps_fallback=float(args.fps),
        undistort_mode=str(args.undistort_mode),
        resize_interp=str(args.resize_interp),
        intr_scale_mode=str(args.intr_scale_mode),
    )

    droid = None
    timestamps_used: List[float] = []
    log_info(
        "slam tracking start: frames={} (C++ backend running, please wait...)".format(
            len(files)
        )
    )


    for t_int, image, intrinsics, ts in _maybe_tqdm(
        droid_stream(files, cfg),
        disable=True,
        desc="DROID track",
        total=len(files),
    ):
        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(int(t_int), image, intrinsics=intrinsics)
        timestamps_used.append(float(ts))

    if droid is None:
        raise RuntimeError("No frames processed. Check segment_dir/frames or your index settings.")

    def stream_for_terminate():
        for t_int, image, intrinsics, _ts in droid_stream(files, cfg):
            yield int(t_int), image, intrinsics

    traj_est = droid.terminate(stream_for_terminate())

    tum_path = slam_dir / "traj_est.tum"
    npz_path = slam_dir / "traj_est.npz"
    save_trajectory(traj_est, timestamps_used, tum_path=tum_path, npz_path=npz_path)

    run_meta.update({
        "frames_processed": len(timestamps_used),
        "timestamp_first": float(timestamps_used[0]) if timestamps_used else None,
        "timestamp_last": float(timestamps_used[-1]) if timestamps_used else None,
        "tum_path": str(tum_path),
        "npz_path": str(npz_path),
    })
    _json_dump(slam_dir / "run_meta.json", run_meta)

    log_prog("traj saved: {}".format(tum_path))
    log_info("npz saved: {}".format(npz_path))
    log_info("meta saved: {}".format(slam_dir / "run_meta.json"))


if __name__ == "__main__":
    main()
