from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_warn

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
_DIGITS_RE = re.compile(r"(\d+)")


@dataclass
class StreamConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Optional[object]
    timestamps: Optional[List[float]]
    fps_fallback: float
    undistort_mode: str
    resize_interp: str
    intr_scale_mode: str
    target_area: int = 384 * 512
    divisible: int = 8


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _read_text(path_obj):
    return Path(path_obj).read_text(encoding="utf-8")


def _parse_frame_index(name):
    match = _DIGITS_RE.search(str(Path(name).stem))
    if match is None:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _load_calib(calib_source):
    import numpy as np

    arr = np.loadtxt(str(calib_source), delimiter=" ")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError("calib must have >=4 numbers: {}".format(calib_source))
    fx, fy, cx, cy = [float(item) for item in arr[:4]]
    dist = arr[4:].astype(np.float64) if arr.size > 4 else None
    return fx, fy, cx, cy, dist


def _load_timestamps_list(ts_source):
    out = []
    for line in _read_text(ts_source).splitlines():
        text = str(line).strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        out.append(float(parts[-1]))
    return out


def _resolve_timestamp(seq_i, file_frame_idx, timestamps, fps_fallback):
    if timestamps:
        if 0 <= file_frame_idx < len(timestamps):
            return float(timestamps[file_frame_idx])
        if 0 <= seq_i < len(timestamps):
            return float(timestamps[seq_i])
    if fps_fallback and fps_fallback > 0:
        idx = file_frame_idx if file_frame_idx >= 0 else seq_i
        return float(idx) / float(fps_fallback)
    return float(seq_i)


def _maybe_tqdm(iterable, disable, desc, total=None):
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, bar_format=_BAR_FORMAT)


def _need_undistort(cfg):
    has_dist = cfg.dist is not None and cfg.dist.size > 0
    if cfg.undistort_mode == "on":
        return has_dist
    if cfg.undistort_mode == "off":
        return False
    return has_dist


def _resize_interp_flag(cfg):
    import cv2

    return cv2.INTER_AREA if cfg.resize_interp == "area" else cv2.INTER_LINEAR


def _make_intrinsics_tensor_demo(cfg, w0, h0, w1_pre, h1_pre):
    import torch

    intrinsics = torch.as_tensor([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=torch.float32)
    intrinsics[0::2] *= float(w1_pre) / float(w0)
    intrinsics[1::2] *= float(h1_pre) / float(h0)
    return intrinsics


def _make_intrinsics_tensor_correct(cfg, w0, h0, w1, h1):
    import torch

    intrinsics = torch.as_tensor([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=torch.float32)
    intrinsics[0::2] *= float(w1) / float(w0)
    intrinsics[1::2] *= float(h1) / float(h0)
    return intrinsics


def droid_stream(files, cfg):
    import cv2
    import numpy as np
    import torch

    for seq_i, path_obj in enumerate(files):
        frame_idx = _parse_frame_index(path_obj.name)
        image_bgr = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        if _need_undistort(cfg):
            camera = np.eye(3, dtype=np.float64)
            camera[0, 0] = cfg.fx
            camera[1, 1] = cfg.fy
            camera[0, 2] = cfg.cx
            camera[1, 2] = cfg.cy
            image_bgr = cv2.undistort(image_bgr, camera, cfg.dist)

        h0, w0 = image_bgr.shape[:2]
        scale = (float(cfg.target_area) / float(h0 * w0)) ** 0.5
        h1_pre = int(h0 * scale)
        w1_pre = int(w0 * scale)

        image_bgr = cv2.resize(image_bgr, (w1_pre, h1_pre), interpolation=_resize_interp_flag(cfg))
        h1 = h1_pre - (h1_pre % int(cfg.divisible))
        w1 = w1_pre - (w1_pre % int(cfg.divisible))
        image_bgr = image_bgr[:h1, :w1]
        image = torch.as_tensor(image_bgr).permute(2, 0, 1)

        if cfg.intr_scale_mode == "demo":
            intrinsics = _make_intrinsics_tensor_demo(cfg, w0, h0, w1_pre, h1_pre)
        else:
            intrinsics = _make_intrinsics_tensor_correct(cfg, w0, h0, w1, h1)

        timestamp = _resolve_timestamp(seq_i, frame_idx, cfg.timestamps, cfg.fps_fallback)
        yield seq_i, image[None], intrinsics, float(timestamp)


def _quat_xyzw_from_rot(rotation):
    import numpy as np

    R = np.asarray(rotation, dtype=np.float64)
    q = np.empty((4,), dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (R[2, 1] - R[1, 2]) * scale
        qy = (R[0, 2] - R[2, 0]) * scale
        qz = (R[1, 0] - R[0, 1]) * scale
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            scale = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / scale
            qx = 0.25 * scale
            qy = (R[0, 1] + R[1, 0]) / scale
            qz = (R[0, 2] + R[2, 0]) / scale
        elif R[1, 1] > R[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / scale
            qx = (R[0, 1] + R[1, 0]) / scale
            qy = 0.25 * scale
            qz = (R[1, 2] + R[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / scale
            qx = (R[0, 2] + R[2, 0]) / scale
            qy = (R[1, 2] + R[2, 1]) / scale
            qz = 0.25 * scale
    q[:] = [qx, qy, qz, qw]
    q /= np.linalg.norm(q)
    return q


def _save_trajectory(traj_est, timestamps_sec, tum_path, npz_path):
    import lietorch
    import numpy as np
    import torch

    traj_tensor = torch.from_numpy(traj_est) if isinstance(traj_est, np.ndarray) else traj_est
    se3 = lietorch.SE3(traj_tensor)
    matrices = se3.matrix().cpu().numpy()

    if matrices.shape[1:] == (3, 4):
        last_row = np.array([0, 0, 0, 1], dtype=matrices.dtype)
        last_row = np.broadcast_to(last_row, (matrices.shape[0], 1, 4))
        matrices = np.concatenate([matrices, last_row], axis=1)

    timestamps = np.asarray(timestamps_sec, dtype=float)
    if matrices.shape[0] != timestamps.shape[0]:
        raise RuntimeError("pose count {} does not match timestamp count {}".format(matrices.shape[0], timestamps.shape[0]))

    Path(tum_path).parent.mkdir(parents=True, exist_ok=True)
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz_path), tstamps=timestamps, poses=matrices)

    with open(str(tum_path), "w", encoding="utf-8") as handle:
        for timestamp, pose in zip(timestamps, matrices):
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            qx, qy, qz, qw = _quat_xyzw_from_rot(rotation)
            handle.write(
                "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                    float(timestamp),
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                    float(qx),
                    float(qy),
                    float(qz),
                    float(qw),
                )
            )


def _show_image(image_chw):
    import cv2

    image = image_chw.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def _setup_droid_import(droid_repo):
    repo = Path(droid_repo).resolve()
    if not repo.is_dir():
        raise RuntimeError("droid repo not found: {}".format(repo))
    droid_pkg = repo / "droid_slam"
    sys.path.append(str(droid_pkg if droid_pkg.exists() else repo))


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _track_args(config):
    args = argparse.Namespace(**dict(config))
    args.stereo = False
    return args


def _run_track(exp_dir, track_name, frames_dir, args):
    import torch

    started = time.time()
    frames_dir = ensure_dir(frames_dir, "{} frames dir".format(track_name))
    calib_path = ensure_file(args.decode_calib, "decode calib")
    timestamps_path = ensure_file(args.decode_timestamps, "decode timestamps")

    files = list_frames_sorted(frames_dir)
    if args.t0 > 0:
        files = files[args.t0 :]
    if args.stride > 1:
        files = files[:: args.stride]
    if args.max_frames > 0:
        files = files[: args.max_frames]
    if not files:
        raise RuntimeError("no frames available for slam track {}: {}".format(track_name, frames_dir))

    fx, fy, cx, cy, dist = _load_calib(calib_path)
    timestamps = _load_timestamps_list(timestamps_path)

    _setup_droid_import(args.droid_repo)
    from droid import Droid  # noqa

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(args.droid_repo).resolve() / weights_path

    track_dir = Path(args.out_dir).resolve() / str(track_name)
    track_dir.mkdir(parents=True, exist_ok=True)

    torch.multiprocessing.set_start_method("spawn", force=True)
    stream_cfg = StreamConfig(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist=dist,
        timestamps=timestamps,
        fps_fallback=float(args.fps),
        undistort_mode=str(args.undistort_mode),
        resize_interp=str(args.resize_interp),
        intr_scale_mode=str(args.intr_scale_mode),
    )

    droid = None
    timestamps_used = []
    log_info("slam tracking start: track={} frames={}".format(track_name, len(files)))

    for t_int, image, intrinsics, timestamp in _maybe_tqdm(
        droid_stream(files, stream_cfg),
        disable=bool(args.no_tqdm),
        desc="DROID {}".format(track_name),
        total=len(files),
    ):
        if not bool(args.disable_vis):
            _show_image(image[0])
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(int(t_int), image, intrinsics=intrinsics)
        timestamps_used.append(float(timestamp))

    if droid is None:
        raise RuntimeError("no frames processed for slam track {}".format(track_name))

    def stream_for_terminate():
        for t_int, image, intrinsics, _timestamp in droid_stream(files, stream_cfg):
            yield int(t_int), image, intrinsics

    termination_mode = "backend_optimized"
    try:
        traj_est = droid.terminate(stream_for_terminate())
    except ValueError as exc:
        if "not enough values to unpack" not in str(exc):
            raise
        log_warn("slam backend produced no proximity factors; using trajectory filler only")
        camera_trajectory = droid.traj_filler(stream_for_terminate())
        traj_est = camera_trajectory.inv().data.cpu().numpy()
        termination_mode = "traj_filler_only"

    tum_path = (track_dir / "traj_est.tum").resolve()
    npz_path = (track_dir / "traj_est.npz").resolve()
    _save_trajectory(traj_est, timestamps_used, tum_path=tum_path, npz_path=npz_path)

    run_meta = {
        "version": 1,
        "source": "eval.slam_run.track",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "track": str(track_name),
        "frames_dir": _relative_path(exp_dir, frames_dir),
        "num_frames": int(len(files)),
        "frames_processed": int(len(timestamps_used)),
        "calib_source": _relative_path(exp_dir, calib_path),
        "timestamps_source": _relative_path(exp_dir, timestamps_path),
        "trajectory_path": _relative_path(exp_dir, tum_path),
        "npz_path": _relative_path(exp_dir, npz_path),
        "runtime_sec": float(time.time() - started),
        "termination_mode": termination_mode,
        "droid_repo": str(Path(args.droid_repo).resolve()),
        "weights": str(weights_path.resolve()),
        "timestamp_first": float(timestamps_used[0]) if timestamps_used else None,
        "timestamp_last": float(timestamps_used[-1]) if timestamps_used else None,
    }
    run_meta_path = (track_dir / "run_meta.json").resolve()
    write_json_atomic(run_meta_path, run_meta, indent=2)
    log_prog("slam summary: track={} frames_processed={}".format(track_name, int(len(timestamps_used))))

    return {
        "frames_dir": _relative_path(exp_dir, frames_dir),
        "num_frames": int(len(files)),
        "frames_processed": int(len(timestamps_used)),
        "calib_source": _relative_path(exp_dir, calib_path),
        "timestamps_source": _relative_path(exp_dir, timestamps_path),
        "trajectory_path": _relative_path(exp_dir, tum_path),
        "npz_path": _relative_path(exp_dir, npz_path),
        "run_meta_path": _relative_path(exp_dir, run_meta_path),
        "runtime_sec": float(run_meta["runtime_sec"]),
        "status": "success",
        "termination_mode": termination_mode,
    }


def run_slam(config):
    args = _track_args(config)
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_result_path = ensure_file(args.prepare_result, "prepare result")
    generation_units_path = ensure_file(args.generation_units, "generation units")
    encode_result_path = ensure_file(args.encode_result, "encode result")
    decode_report_path = ensure_file(args.decode_report, "decode report")
    decode_merge_report_path = ensure_file(args.decode_merge_report, "decode merge report")

    seq = str(args.seq or "both").strip().lower()
    if seq == "auto":
        seq = "both"
    if seq not in ("ori", "gen", "both"):
        raise RuntimeError("unsupported slam seq: {}".format(seq))

    tracks = {}
    if seq in ("ori", "both"):
        tracks["ori"] = _run_track(exp_dir, "ori", args.prepare_frames_dir, args)
    if seq in ("gen", "both"):
        tracks["gen"] = _run_track(exp_dir, "gen", args.decode_frames_dir, args)

    inputs = {
        "prepare_result": _relative_path(exp_dir, prepare_result_path),
        "prepare_frames_dir": _relative_path(exp_dir, Path(args.prepare_frames_dir).resolve()),
        "generation_units": _relative_path(exp_dir, generation_units_path),
        "encode_result": _relative_path(exp_dir, encode_result_path),
        "decode_frames_dir": _relative_path(exp_dir, Path(args.decode_frames_dir).resolve()),
        "decode_calib": _relative_path(exp_dir, Path(args.decode_calib).resolve()),
        "decode_timestamps": _relative_path(exp_dir, Path(args.decode_timestamps).resolve()),
        "decode_report": _relative_path(exp_dir, decode_report_path),
        "decode_merge_report": _relative_path(exp_dir, decode_merge_report_path),
    }
    report = {
        "version": 1,
        "source": "eval.slam_run",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "success",
        "requested_seq": str(seq),
        "input_source_kind": "native",
        "native_inputs": list(inputs.values()),
        "fallback_used": False,
        "inputs": inputs,
        "ori": dict(tracks.get("ori") or {}),
        "gen": dict(tracks.get("gen") or {}),
        "warnings": [],
    }
    report_path = out_dir / "eval_slam_report.json"
    write_json_atomic(report_path, report, indent=2)
    log_info("eval slam report: {}".format(report_path))
    return {
        "report_path": report_path,
        "report": report,
        "out_dir": out_dir,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-native-slam", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--decode_frames_dir", required=True)
    parser.add_argument("--decode_calib", required=True)
    parser.add_argument("--decode_timestamps", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--decode_merge_report", required=True)
    parser.add_argument("--seq", default="both", choices=["auto", "ori", "gen", "both"])
    parser.add_argument("--droid_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--t0", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--undistort_mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--resize_interp", choices=["linear", "area"], default="linear")
    parser.add_argument("--intr_scale_mode", choices=["demo", "correct"], default="demo")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.0)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--no_tqdm", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_native_slam:
        raise SystemExit("eval slam helper requires --run-native-slam")
    run_slam(vars(args))


if __name__ == "__main__":
    main()
