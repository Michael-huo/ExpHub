from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
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

    if isinstance(calib_source, (list, tuple)):
        arr = np.asarray(list(calib_source), dtype=np.float64).reshape(-1)
    else:
        arr = np.loadtxt(str(calib_source), delimiter=" ")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError("calib must have >=4 numbers: {}".format(calib_source))
    fx, fy, cx, cy = [float(item) for item in arr[:4]]
    dist = arr[4:].astype(np.float64) if arr.size > 4 else None
    return fx, fy, cx, cy, dist


def _load_timestamps_list(ts_source):
    if isinstance(ts_source, (list, tuple)):
        out = []
        for item in list(ts_source):
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    if ts_source is None:
        return []
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
    intrinsics[0::2] *= (float(w1_pre) / float(w0))
    intrinsics[1::2] *= (float(h1_pre) / float(h0))
    return intrinsics


def _make_intrinsics_tensor_correct(cfg, w0, h0, w1, h1):
    import torch

    intrinsics = torch.as_tensor([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=torch.float32)
    intrinsics[0::2] *= (float(w1) / float(w0))
    intrinsics[1::2] *= (float(h1) / float(h0))
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


def _resolve_input_paths(root_dir, prepare_result_path=""):
    root = Path(root_dir).resolve()
    frames_dir = ensure_dir(root / "frames", "slam input frames")
    calib_file = root / "calib.txt"
    timestamps_file = root / "timestamps.txt"
    if calib_file.is_file() and timestamps_file.is_file():
        source_path = root / "decode_merge_report.json"
        return frames_dir, calib_file, timestamps_file, source_path if source_path.is_file() else timestamps_file

    manifest_candidates = [
        root / "prepare_result.json",
        Path(prepare_result_path).resolve() if str(prepare_result_path or "").strip() else None,
        root.parent / "prepare" / "prepare_result.json",
        root / "input_report.json",
        root.parent / "encode" / "legacy_segment_manifest.json",
        root.parent / "input" / "input_report.json",
    ]
    manifest_path = None
    for candidate in manifest_candidates:
        if candidate is not None and Path(candidate).is_file():
            manifest_path = Path(candidate).resolve()
            break
    if manifest_path is None:
        raise RuntimeError("missing required slam input report: {}".format(manifest_candidates[0].resolve()))
    manifest_obj = dict(read_json_dict(manifest_path) or {})
    camera_obj = dict(manifest_obj.get("camera") or {})
    calib_source = list(camera_obj.get("calib") or [])
    timestamps_source = list(camera_obj.get("timestamps") or [])
    if not calib_source:
        intrinsics = dict(manifest_obj.get("normalized_intrinsics") or {})
        if intrinsics:
            calib_source = [
                intrinsics.get("fx"),
                intrinsics.get("fy"),
                intrinsics.get("cx"),
                intrinsics.get("cy"),
            ] + list(intrinsics.get("dist") or [])
    if not timestamps_source:
        frame_index_map = dict(manifest_obj.get("frame_index_map") or {})
        timestamps_source = list(
            frame_index_map.get("prepared_to_rel_time_sec")
            or frame_index_map.get("prepared_to_time_sec")
            or []
        )
    return frames_dir, calib_source, timestamps_source, manifest_path


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


def _copy_if_exists(src_path, dst_path):
    src = Path(src_path).resolve()
    if not src.is_file():
        return ""
    dst = Path(dst_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return str(dst)


def _run_track(exp_dir, track_name, input_root, args):
    import torch

    frames_dir, calib_source, timestamps_source, manifest_path = _resolve_input_paths(
        input_root,
        getattr(args, "prepare_result", ""),
    )
    files = list_frames_sorted(frames_dir)
    if args.t0 > 0:
        files = files[args.t0 :]
    if args.stride > 1:
        files = files[:: args.stride]
    if args.max_frames > 0:
        files = files[: args.max_frames]
    if not files:
        raise RuntimeError("no frames available for slam track {}: {}".format(track_name, frames_dir))

    fx, fy, cx, cy, dist = _load_calib(calib_source)
    timestamps = _load_timestamps_list(timestamps_source)

    _setup_droid_import(args.droid_repo)
    from droid import Droid  # noqa

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(args.droid_repo).resolve() / weights_path

    track_dir = Path(args.out_dir).resolve() / str(track_name)
    track_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "track": str(track_name),
        "input_root": str(Path(input_root).resolve()),
        "frames_dir": str(frames_dir),
        "calib": f"{manifest_path}#camera.calib",
        "timestamps": f"{manifest_path}#camera.timestamps",
        "slam_out_dir": str(track_dir),
        "droid_repo": str(Path(args.droid_repo).resolve()),
        "weights": str(weights_path.resolve()),
        "t0": int(args.t0),
        "stride": int(args.stride),
        "max_frames": int(args.max_frames),
        "fx_fy_cx_cy": [fx, fy, cx, cy],
        "dist_len": int(dist.size) if dist is not None else 0,
        "undistort_mode": str(args.undistort_mode),
        "resize_interp": str(args.resize_interp),
        "intr_scale_mode": str(args.intr_scale_mode),
    }

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
        # DROID can fail to build any backend proximity edges on very short clips.
        # In that case we still have a valid tracked keyframe trajectory, so fall
        # back to the trajectory filler without swallowing unrelated errors.
        if "not enough values to unpack" not in str(exc):
            raise
        log_warn("slam backend produced no proximity factors; using trajectory filler only")
        camera_trajectory = droid.traj_filler(stream_for_terminate())
        traj_est = camera_trajectory.inv().data.cpu().numpy()
        termination_mode = "traj_filler_only"
    tum_path = (track_dir / "traj_est.tum").resolve()
    npz_path = (track_dir / "traj_est.npz").resolve()
    _save_trajectory(traj_est, timestamps_used, tum_path=tum_path, npz_path=npz_path)

    run_meta.update(
        {
            "frames_processed": int(len(timestamps_used)),
            "timestamp_first": float(timestamps_used[0]) if timestamps_used else None,
            "timestamp_last": float(timestamps_used[-1]) if timestamps_used else None,
            "tum_path": str(tum_path),
            "npz_path": str(npz_path),
            "termination_mode": termination_mode,
        }
    )
    run_meta_path = (track_dir / "run_meta.json").resolve()
    write_json_atomic(run_meta_path, run_meta, indent=2)
    log_prog("slam summary: track={} frames_processed={}".format(track_name, int(len(timestamps_used))))

    return {
        "track": str(track_name),
        "input_root": _relative_path(exp_dir, input_root),
        "frames_dir": _relative_path(exp_dir, frames_dir),
        "traj_path": _relative_path(exp_dir, tum_path),
        "npz_path": _relative_path(exp_dir, npz_path),
        "run_meta_path": _relative_path(exp_dir, run_meta_path),
        "frames_processed": int(len(timestamps_used)),
        "status": "success",
        "termination_mode": termination_mode,
    }


def _write_slam_report(report_path, report_obj):
    payload = dict(report_obj or {})
    payload["report_path"] = str(Path(report_path).resolve())
    write_json_atomic(report_path, payload, indent=2)
    return Path(report_path).resolve()


def run_slam_substage(args):
    exp_dir = Path(args.exp_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(str(out_dir), ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = str(args.seq or "both").strip().lower()
    if seq == "auto":
        seq = "both"
    if seq not in ("ori", "gen", "both"):
        raise RuntimeError("unsupported slam seq: {}".format(seq))

    infer_report = dict(read_json_dict(Path(args.infer_report)) or {})
    encode_result = dict(read_json_dict(Path(getattr(args, "encode_result", ""))) or {})
    merge_report = dict(read_json_dict(Path(args.merge_report)) or {})
    merge_manifest = dict(read_json_dict(Path(args.merge_manifest)) or {})
    merge_summary = dict(merge_manifest.get("summary") or {})
    if not merge_summary:
        merge_summary = dict(merge_report.get("summary") or {})

    track_specs = []
    if seq in ("ori", "both"):
        track_specs.append(("ori", Path(args.segment_dir).resolve()))
    if seq in ("gen", "both"):
        track_specs.append(("gen", Path(args.merge_dir).resolve()))

    track_reports = {}
    for track_name, input_root in track_specs:
        track_reports[track_name] = _run_track(exp_dir, track_name, input_root, args)

    primary_track = "gen" if "gen" in track_reports else "ori"
    primary_src = (out_dir / primary_track / "traj_est.tum").resolve()
    primary_dst = (out_dir / "traj_est.txt").resolve()
    primary_path = _copy_if_exists(primary_src, primary_dst)

    reference_path = ""
    if "ori" in track_reports:
        reference_path = str((out_dir / "ori" / "traj_est.tum").resolve())

    report = {
        "report_schema_version": "eval_slam_report.v1",
        "step": "eval",
        "substage": "slam",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "slam_status": "success",
        "planner": "generation_units",
        "prompt_strategy": str(infer_report.get("prompt_strategy", "") or encode_result.get("prompt_mode", "") or "prompts"),
        "source_unit_count": int(merge_summary.get("source_unit_count", 0) or encode_result.get("num_generation_units", 0) or 0),
        "source_span_count": int(merge_summary.get("source_span_count", 0) or 0),
        "shared_anchor_count": int(merge_summary.get("shared_anchor_count", 0) or 0),
        "requested_seq": str(seq),
        "primary_track": str(primary_track),
        "primary_trajectory_path": _relative_path(exp_dir, primary_path) if primary_path else "",
        "reference_track": "ori" if "ori" in track_reports else "",
        "reference_trajectory_path": _relative_path(exp_dir, reference_path) if reference_path else "",
        "inputs": {
            "infer_report": _relative_path(exp_dir, Path(args.infer_report).resolve()),
            "merge_report": _relative_path(exp_dir, Path(args.merge_report).resolve()),
            "merge_manifest": _relative_path(exp_dir, Path(args.merge_manifest).resolve()),
            "prepare_result": _relative_path(exp_dir, Path(getattr(args, "prepare_result", "")).resolve())
            if str(getattr(args, "prepare_result", "") or "").strip()
            else "",
            "generation_units": _relative_path(exp_dir, Path(getattr(args, "generation_units", "")).resolve())
            if str(getattr(args, "generation_units", "") or "").strip()
            else "",
            "encode_result": _relative_path(exp_dir, Path(getattr(args, "encode_result", "")).resolve())
            if str(getattr(args, "encode_result", "") or "").strip()
            else "",
            "segment_dir": _relative_path(exp_dir, Path(args.segment_dir).resolve()),
            "merge_dir": _relative_path(exp_dir, Path(args.merge_dir).resolve()),
        },
        "tracks": track_reports,
        "artifact_contract": {
            "formal_files": [
                "eval_slam_report.json",
                "traj_est.txt",
            ],
            "formal_track_files": [
                "<track>/traj_est.tum",
                "<track>/traj_est.npz",
                "<track>/run_meta.json",
            ],
            "track_dirs": sorted(track_reports.keys()),
        },
        "warnings": [],
    }
    report_path = _write_slam_report(out_dir / "eval_slam_report.json", report)
    log_info("eval slam report: {}".format(report_path))
    return {
        "report_path": report_path,
        "report": report,
        "out_dir": out_dir,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--prepare_result", default="")
    parser.add_argument("--generation_units", default="")
    parser.add_argument("--encode_result", default="")
    parser.add_argument("--infer_dir", required=True)
    parser.add_argument("--infer_report", required=True)
    parser.add_argument("--merge_dir", required=True)
    parser.add_argument("--merge_report", required=True)
    parser.add_argument("--merge_manifest", required=True)
    parser.add_argument("--seq", default="both")
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
    if not args.run_formal_mainline:
        raise SystemExit("eval slam helper requires --run-formal-mainline")
    args.stereo = False
    run_slam_substage(args)


if __name__ == "__main__":
    main()
