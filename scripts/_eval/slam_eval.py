#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import io
import math
import re
import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np

from _common import list_frames_sorted, log_info
from _eval.io import (
    append_warning,
    empty_stats,
    metric_stats,
    read_json,
    read_timestamps,
    resolve_slam_eval_inputs,
    write_csv,
    write_json,
)


_DIGITS_RE = re.compile(r"(\d+)")
_RATIO_TEST = 0.75
_RANSAC_THRESHOLD_PX = 1.0
_RANSAC_PROB = 0.999
_ROT_SUCCESS_DEG = 10.0
_TRANS_SUCCESS_DEG = 10.0
_POSE_EPS = 1e-9
_TRACK_TS_TOL_SEC = 1e-3
_PLOT_DPI = 220
_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_INLIER_COLOR = "#1f4e79"
_POSE_COLOR = "#4d7f4b"
_FAIL_COLOR = "#b35c2e"
_REF_LINE_COLOR = "#6a7480"


def _base_metrics(exp_dir):
    paths = resolve_slam_eval_inputs(exp_dir)
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "reference_source": "unavailable",
        "uses_proxy_reference": False,
        "reference_path": "",
        "reference_run_meta_path": "",
        "pair_selection_source": "",
        "selected_pair_count": 0,
        "valid_pair_count": 0,
        "valid_pose_pair_count": 0,
        "successful_pose_pair_count": 0,
        "pose_success_rate": None,
        "metric_units": {
            "inlier_ratio": "ratio",
            "pose_success_rate": "ratio",
            "rotation_error_deg": "deg",
            "translation_direction_error_deg": "deg",
        },
        "thresholds": {
            "rotation_error_deg_lt": float(_ROT_SUCCESS_DEG),
            "translation_direction_error_deg_lt": float(_TRANS_SUCCESS_DEG),
        },
        "segment_timestamps_path": str(paths["segment_timestamps_path"]),
        "merge_frames_dir": str(paths["merge_frames_dir"]),
        "runs_plan_path": str(paths["runs_plan_path"]),
        "merge_meta_path": str(paths["merge_meta_path"]),
        "inlier_ratio": empty_stats(),
        "rotation_error_deg": empty_stats(),
        "translation_direction_error_deg": empty_stats(),
    }


def _parse_frame_index(name):
    match = _DIGITS_RE.search(str(Path(name).stem))
    if match is None:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _load_cv_runtime(metrics_obj):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                import cv2
    except Exception as exc:
        append_warning(metrics_obj, "OpenCV unavailable; slam-friendly eval unavailable: {}".format(exc))
        return None

    if not hasattr(cv2, "SIFT_create"):
        append_warning(metrics_obj, "OpenCV SIFT unavailable; slam-friendly eval unavailable")
        return None

    try:
        sift = cv2.SIFT_create()
    except Exception as exc:
        append_warning(metrics_obj, "failed to initialize OpenCV SIFT: {}".format(exc))
        return None

    return {
        "cv2": cv2,
        "sift": sift,
        "matcher": cv2.BFMatcher(cv2.NORM_L2),
    }


def _load_camera_matrix(exp_dir, metrics_obj):
    paths = resolve_slam_eval_inputs(exp_dir)
    calib_candidates = [paths["merge_calib_path"], paths["segment_calib_path"]]
    calib_path = None
    for item in calib_candidates:
        if Path(item).is_file():
            calib_path = Path(item)
            break
    if calib_path is None:
        append_warning(metrics_obj, "missing calib.txt; slam-friendly eval unavailable")
        return None

    try:
        arr = np.loadtxt(str(calib_path), delimiter=" ")
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception as exc:
        append_warning(metrics_obj, "failed to load calib.txt for slam-friendly eval: {}".format(exc))
        return None

    if arr.size < 4:
        append_warning(metrics_obj, "invalid calib.txt; expected >=4 numbers: {}".format(calib_path))
        return None

    fx, fy, cx, cy = [float(v) for v in arr[:4]]
    if fx <= 0.0 or fy <= 0.0:
        append_warning(metrics_obj, "invalid focal length in calib.txt: {}".format(calib_path))
        return None

    metrics_obj["calib_path"] = str(calib_path)
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _segments_from_runs_plan(plan_obj):
    if not isinstance(plan_obj, dict):
        return []
    raw_segments = list(plan_obj.get("segments") or [])
    out = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        if "start_idx" not in item or "end_idx" not in item:
            continue
        try:
            out.append(
                {
                    "start_idx": int(item.get("start_idx")),
                    "end_idx": int(item.get("end_idx")),
                }
            )
        except Exception:
            continue
    out.sort(key=lambda item: (int(item["start_idx"]), int(item["end_idx"])))
    return out


def _segments_from_schedule_and_merge_meta(schedule_obj, merge_meta_obj):
    if not isinstance(schedule_obj, dict) or not isinstance(merge_meta_obj, dict):
        return []

    merged_start = merge_meta_obj.get("merged_start_idx")
    merged_end = merge_meta_obj.get("merged_end_idx")
    if merged_start is None or merged_end is None:
        return []

    try:
        merged_start = int(merged_start)
        merged_end = int(merged_end)
    except Exception:
        return []

    raw_segments = list(schedule_obj.get("segments") or [])
    out = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start_idx = int(item.get("deploy_start_idx"))
            end_idx = int(item.get("deploy_end_idx"))
        except Exception:
            continue
        if end_idx < merged_start or start_idx > merged_end:
            continue
        out.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )
    out.sort(key=lambda item: (int(item["start_idx"]), int(item["end_idx"])))
    return out


def _resolve_pair_context(exp_dir, metrics_obj):
    paths = resolve_slam_eval_inputs(exp_dir)
    merge_frames_dir = Path(paths["merge_frames_dir"]).resolve()
    if not merge_frames_dir.is_dir():
        append_warning(metrics_obj, "missing merge frames directory: {}".format(merge_frames_dir))
        return None

    merge_frames = list_frames_sorted(merge_frames_dir)
    if not merge_frames:
        append_warning(metrics_obj, "no merge frames found for slam-friendly eval: {}".format(merge_frames_dir))
        return None

    pair_source = ""
    segments = _segments_from_runs_plan(read_json(paths["runs_plan_path"]))
    if segments:
        pair_source = "infer/runs_plan.json"
    else:
        segments = _segments_from_schedule_and_merge_meta(
            read_json(paths["deploy_schedule_path"]),
            read_json(paths["merge_meta_path"]),
        )
        if segments:
            pair_source = "segment/deploy_schedule.json + merge/merge_meta.json"

    if not segments:
        append_warning(
            metrics_obj,
            "cannot resolve generated-frame schedule for slam-friendly eval; require runs_plan or deploy_schedule+merge_meta",
        )
        return None

    merged_start = int(segments[0]["start_idx"])
    merged_end = int(segments[-1]["end_idx"])
    expected_count = int(merged_end - merged_start + 1)
    if expected_count != len(merge_frames):
        append_warning(
            metrics_obj,
            "merge frame count mismatch for slam-friendly eval: merge={} schedule={}".format(
                len(merge_frames), expected_count
            ),
        )

    frame_map = {}
    max_count = min(len(merge_frames), expected_count)
    for seq_idx in range(max_count):
        frame_idx = int(merged_start + seq_idx)
        frame_map[frame_idx] = merge_frames[seq_idx]

    anchor_indices = set()
    for item in segments:
        anchor_indices.add(int(item["start_idx"]))
        anchor_indices.add(int(item["end_idx"]))

    generated_indices = []
    for frame_idx in sorted(frame_map.keys()):
        if frame_idx in anchor_indices:
            continue
        generated_indices.append(int(frame_idx))

    generated_set = set(generated_indices)
    pairs = []
    for frame_idx in generated_indices:
        next_idx = int(frame_idx + 1)
        if next_idx not in generated_set:
            continue
        if frame_idx not in frame_map or next_idx not in frame_map:
            continue
        pairs.append(
            {
                "frame_idx_0": int(frame_idx),
                "frame_idx_1": int(next_idx),
                "image_path_0": str(frame_map[frame_idx]),
                "image_path_1": str(frame_map[next_idx]),
            }
        )

    metrics_obj["pair_selection_source"] = str(pair_source)
    metrics_obj["selected_pair_count"] = int(len(pairs))

    if not pairs:
        append_warning(metrics_obj, "no consecutive generated frame pairs available for slam-friendly eval")
        return None
    return pairs


def _quat_xyzw_to_rot(qx, qy, qz, qw):
    q = np.asarray([float(qx), float(qy), float(qz), float(qw)], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= _POSE_EPS:
        raise ValueError("invalid quaternion norm")
    q = q / norm
    x, y, z, w = [float(v) for v in q]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix_from_tum_row(row):
    values = [float(v) for v in row]
    if len(values) < 8:
        raise ValueError("invalid TUM row")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_xyzw_to_rot(values[4], values[5], values[6], values[7])
    mat[:3, 3] = np.asarray(values[1:4], dtype=np.float64)
    return float(values[0]), mat


def _load_pose_trajectory_npz(path):
    try:
        payload = np.load(str(path))
        timestamps = np.asarray(payload["tstamps"], dtype=np.float64).reshape(-1)
        poses = np.asarray(payload["poses"], dtype=np.float64)
    except Exception:
        return None

    if poses.ndim != 3 or poses.shape[0] != timestamps.shape[0]:
        return None
    if poses.shape[1:] == (3, 4):
        last_row = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        poses = np.concatenate([poses, np.repeat(last_row[None, :, :], poses.shape[0], axis=0)], axis=1)
    if poses.shape[1:] != (4, 4):
        return None
    return {
        "timestamps": timestamps,
        "poses": poses,
    }


def _load_pose_trajectory_tum(path):
    timestamps = []
    poses = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    for line in lines:
        text = str(line).strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 8:
            continue
        try:
            ts, pose = _pose_matrix_from_tum_row(parts[:8])
        except Exception:
            continue
        timestamps.append(float(ts))
        poses.append(pose)

    if not poses:
        return None
    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "poses": np.asarray(poses, dtype=np.float64),
    }


def _load_reference_frames(run_meta_obj):
    if not isinstance(run_meta_obj, dict):
        return []
    frames_dir = str(run_meta_obj.get("frames_dir", "") or "").strip()
    if not frames_dir:
        return []
    path = Path(frames_dir).resolve()
    if not path.is_dir():
        return []
    files = list_frames_sorted(path)
    if not files:
        return []

    t0 = 0
    stride = 1
    max_frames = 0
    try:
        t0 = int(run_meta_obj.get("t0", 0) or 0)
    except Exception:
        t0 = 0
    try:
        stride = int(run_meta_obj.get("stride", 1) or 1)
    except Exception:
        stride = 1
    try:
        max_frames = int(run_meta_obj.get("max_frames", 0) or 0)
    except Exception:
        max_frames = 0

    if t0 > 0:
        files = files[t0:]
    if stride > 1:
        files = files[::stride]
    if max_frames > 0:
        files = files[:max_frames]
    return files


def _build_pose_map_from_frames(payload, run_meta_obj):
    files = _load_reference_frames(run_meta_obj)
    poses = np.asarray(payload.get("poses"), dtype=np.float64)
    if not files or poses.ndim != 3:
        return {}

    usable = min(len(files), poses.shape[0])
    pose_map = {}
    for idx in range(usable):
        frame_idx = _parse_frame_index(files[idx].name)
        if frame_idx < 0:
            continue
        pose_map[int(frame_idx)] = poses[idx]
    return pose_map


def _build_pose_map_from_timestamps(payload, timestamps_path):
    if not timestamps_path:
        return {}
    timestamps = read_timestamps(timestamps_path)
    if not timestamps:
        return {}

    reference_ts = []
    for item in timestamps:
        try:
            reference_ts.append(float(item))
        except Exception:
            reference_ts.append(None)

    pose_map = {}
    pose_timestamps = np.asarray(payload.get("timestamps"), dtype=np.float64).reshape(-1)
    poses = np.asarray(payload.get("poses"), dtype=np.float64)
    if poses.shape[0] != pose_timestamps.shape[0]:
        return {}

    for pose_idx in range(pose_timestamps.shape[0]):
        ts_value = float(pose_timestamps[pose_idx])
        best_frame_idx = None
        best_diff = None
        for frame_idx, ref_ts in enumerate(reference_ts):
            if ref_ts is None:
                continue
            diff = abs(float(ref_ts) - ts_value)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_frame_idx = frame_idx
        if best_frame_idx is None or best_diff is None or best_diff > _TRACK_TS_TOL_SEC:
            continue
        pose_map[int(best_frame_idx)] = poses[pose_idx]
    return pose_map


def _load_reference_track(paths, metrics_obj, source_name, uses_proxy, npz_path, tum_path, run_meta_path):
    payload = None
    resolved_path = None
    if Path(npz_path).is_file():
        payload = _load_pose_trajectory_npz(npz_path)
        resolved_path = Path(npz_path).resolve()
    if payload is None and Path(tum_path).is_file():
        payload = _load_pose_trajectory_tum(tum_path)
        resolved_path = Path(tum_path).resolve()
    if payload is None:
        return None

    run_meta_obj = read_json(run_meta_path) or {}
    pose_map = _build_pose_map_from_frames(payload, run_meta_obj)
    if not pose_map:
        timestamps_path = str((run_meta_obj or {}).get("timestamps", "") or "")
        if not timestamps_path and str(source_name) in ("gt", "ori_proxy"):
            timestamps_path = str(paths["segment_timestamps_path"])
        pose_map = _build_pose_map_from_timestamps(payload, timestamps_path)

    if not pose_map:
        return None

    return {
        "reference_source": str(source_name),
        "uses_proxy_reference": bool(uses_proxy),
        "reference_path": str(resolved_path),
        "reference_run_meta_path": str(Path(run_meta_path).resolve()) if Path(run_meta_path).is_file() else "",
        "poses_by_frame_idx": pose_map,
    }


def _resolve_reference_context(exp_dir, metrics_obj):
    paths = resolve_slam_eval_inputs(exp_dir)
    candidates = [
        (
            "gt",
            False,
            paths["slam_gt_npz_path"],
            paths["slam_gt_tum_path"],
            paths["slam_gt_run_meta_path"],
        ),
        (
            "ori_proxy",
            True,
            paths["slam_ori_npz_path"],
            paths["slam_ori_tum_path"],
            paths["slam_ori_run_meta_path"],
        ),
    ]

    for source_name, uses_proxy, npz_path, tum_path, run_meta_path in candidates:
        context = _load_reference_track(
            paths,
            metrics_obj,
            source_name,
            uses_proxy,
            npz_path,
            tum_path,
            run_meta_path,
        )
        if context is None:
            continue
        metrics_obj["reference_source"] = str(context["reference_source"])
        metrics_obj["uses_proxy_reference"] = bool(context["uses_proxy_reference"])
        metrics_obj["reference_path"] = str(context["reference_path"])
        metrics_obj["reference_run_meta_path"] = str(context["reference_run_meta_path"])
        return context

    append_warning(metrics_obj, "reference trajectory unavailable for pose_success_rate; require slam/gt or slam/ori")
    return None


def _load_gray_image(cv2, path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError("failed to read image")
    return image


def _match_features(runtime, image0, image1):
    sift = runtime["sift"]
    matcher = runtime["matcher"]
    kp0, desc0 = sift.detectAndCompute(image0, None)
    kp1, desc1 = sift.detectAndCompute(image1, None)
    if desc0 is None or desc1 is None or len(kp0) <= 0 or len(kp1) <= 0:
        return [], None, None

    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    good = []
    for item in list(knn_matches or []):
        if len(item) < 2:
            continue
        m0, m1 = item[0], item[1]
        if m0.distance < float(_RATIO_TEST) * m1.distance:
            good.append(m0)
    return good, kp0, kp1


def _estimate_relative_pose(runtime, camera_matrix, kp0, kp1, matches):
    cv2 = runtime["cv2"]
    if len(matches) <= 0:
        return {
            "raw_matches": 0,
            "inlier_matches": 0,
            "inlier_ratio": None,
            "rotation_matrix": None,
            "translation_vec": None,
        }

    raw_matches = int(len(matches))
    pts0 = np.asarray([kp0[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts1 = np.asarray([kp1[m.trainIdx].pt for m in matches], dtype=np.float64)

    if raw_matches < 5:
        return {
            "raw_matches": raw_matches,
            "inlier_matches": 0,
            "inlier_ratio": 0.0,
            "rotation_matrix": None,
            "translation_vec": None,
        }

    essential, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=camera_matrix,
        method=cv2.RANSAC,
        prob=float(_RANSAC_PROB),
        threshold=float(_RANSAC_THRESHOLD_PX),
    )

    if essential is None or mask is None:
        return {
            "raw_matches": raw_matches,
            "inlier_matches": 0,
            "inlier_ratio": 0.0,
            "rotation_matrix": None,
            "translation_vec": None,
        }

    essential = np.asarray(essential, dtype=np.float64)
    if essential.ndim == 2 and essential.shape[0] > 3:
        essential = essential[:3, :3]
    if essential.shape != (3, 3):
        return {
            "raw_matches": raw_matches,
            "inlier_matches": 0,
            "inlier_ratio": 0.0,
            "rotation_matrix": None,
            "translation_vec": None,
        }

    inlier_mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
    inlier_matches = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(inlier_matches) / float(raw_matches) if raw_matches > 0 else None

    if inlier_matches < 5:
        return {
            "raw_matches": raw_matches,
            "inlier_matches": inlier_matches,
            "inlier_ratio": inlier_ratio,
            "rotation_matrix": None,
            "translation_vec": None,
        }

    try:
        _, rot, trans, _ = cv2.recoverPose(
            essential,
            pts0,
            pts1,
            cameraMatrix=camera_matrix,
            mask=mask,
        )
    except Exception:
        rot = None
        trans = None

    return {
        "raw_matches": raw_matches,
        "inlier_matches": inlier_matches,
        "inlier_ratio": inlier_ratio,
        "rotation_matrix": None if rot is None else np.asarray(rot, dtype=np.float64),
        "translation_vec": None if trans is None else np.asarray(trans, dtype=np.float64).reshape(-1),
    }


def _relative_pose_from_reference(reference_context, frame_idx_0, frame_idx_1):
    pose_map = dict(reference_context.get("poses_by_frame_idx") or {})
    pose0 = pose_map.get(int(frame_idx_0))
    pose1 = pose_map.get(int(frame_idx_1))
    if pose0 is None or pose1 is None:
        return None

    try:
        rel = np.matmul(np.linalg.inv(np.asarray(pose1, dtype=np.float64)), np.asarray(pose0, dtype=np.float64))
    except Exception:
        return None

    return {
        "rotation_matrix": np.asarray(rel[:3, :3], dtype=np.float64),
        "translation_vec": np.asarray(rel[:3, 3], dtype=np.float64).reshape(-1),
    }


def _rotation_error_deg(rot_est, rot_ref):
    delta = np.matmul(np.asarray(rot_est, dtype=np.float64), np.asarray(rot_ref, dtype=np.float64).T)
    trace = float(np.trace(delta))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(math.degrees(math.acos(cos_angle)))


def _direction_error_deg(vec_a, vec_b):
    arr_a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a <= _POSE_EPS or norm_b <= _POSE_EPS:
        return None
    cos_angle = float(np.dot(arr_a, arr_b) / float(norm_a * norm_b))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def _pose_success(rotation_error_deg, translation_error_deg):
    if rotation_error_deg is None or translation_error_deg is None:
        return None
    return bool(
        float(rotation_error_deg) < float(_ROT_SUCCESS_DEG)
        and float(translation_error_deg) < float(_TRANS_SUCCESS_DEG)
    )


def _csv_rows(records):
    rows = []
    for item in records:
        pose_success = item.get("pose_success")
        rows.append(
            {
                "frame_idx_0": int(item["frame_idx_0"]),
                "frame_idx_1": int(item["frame_idx_1"]),
                "raw_matches": "" if item.get("raw_matches") is None else int(item.get("raw_matches")),
                "inlier_matches": "" if item.get("inlier_matches") is None else int(item.get("inlier_matches")),
                "inlier_ratio": "" if item.get("inlier_ratio") is None else "{:.8f}".format(float(item.get("inlier_ratio"))),
                "pose_success": "" if pose_success is None else int(bool(pose_success)),
                "rotation_error_deg": "" if item.get("rotation_error_deg") is None else "{:.8f}".format(float(item.get("rotation_error_deg"))),
                "translation_direction_error_deg": "" if item.get("translation_direction_error_deg") is None else "{:.8f}".format(float(item.get("translation_direction_error_deg"))),
            }
        )
    return rows


def _setup_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="exphub_mpl_")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax):
    ax.set_facecolor(_FIG_FACE)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#2f3b4a", labelsize=9.5)


def _plot_unavailable(ax, title, ylabel, text):
    _style_axes(ax)
    ax.set_title(title, fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Pair Index", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#6a7480",
    )


def _plot_inlier_axis(ax, records, metrics_obj):
    xs = list(range(len(records)))
    values = []
    has_value = False
    for item in records:
        value = item.get("inlier_ratio")
        if value is None:
            values.append(np.nan)
            continue
        has_value = True
        values.append(float(value))

    if not has_value:
        _plot_unavailable(ax, "Inlier Ratio", "Ratio", "Inlier ratio unavailable")
        ax.set_ylim(0.0, 1.02)
        return

    _style_axes(ax)
    ax.set_title("Inlier Ratio", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Pair Index", fontsize=10.5)
    ax.set_ylabel("Ratio", fontsize=10.5)
    ax.plot(
        xs,
        values,
        color=_INLIER_COLOR,
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        markerfacecolor=_INLIER_COLOR,
        markeredgewidth=0.0,
    )
    ref_value = (metrics_obj or {}).get("inlier_ratio", {}).get("mean")
    if ref_value is not None:
        ax.axhline(float(ref_value), color=_REF_LINE_COLOR, linestyle="--", linewidth=1.1, alpha=0.9)
    ax.set_ylim(0.0, 1.02)


def _plot_pose_axis(ax, records):
    xs = []
    ys = []
    for idx, item in enumerate(records):
        value = item.get("pose_success")
        if value is None:
            continue
        xs.append(int(idx))
        ys.append(1 if bool(value) else 0)

    if not xs:
        _plot_unavailable(ax, "Pose Success", "State", "Pose success unavailable")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["fail", "success"])
        return

    _style_axes(ax)
    ax.set_title("Pose Success", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Pair Index", fontsize=10.5)
    ax.set_ylabel("State", fontsize=10.5)
    ax.step(xs, ys, where="mid", color=_POSE_COLOR, linewidth=1.5, alpha=0.95)
    fail_x = [xs[pos] for pos, value in enumerate(ys) if int(value) == 0]
    succ_x = [xs[pos] for pos, value in enumerate(ys) if int(value) == 1]
    if fail_x:
        ax.scatter(fail_x, [0 for _ in fail_x], color=_FAIL_COLOR, s=18, zorder=3)
    if succ_x:
        ax.scatter(succ_x, [1 for _ in succ_x], color=_POSE_COLOR, s=18, zorder=3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fail", "success"])


def _write_plot(out_path, records, metrics_obj):
    plt = _setup_matplotlib()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), dpi=_PLOT_DPI, sharex=False)
    fig.patch.set_facecolor(_FIG_FACE)
    fig.suptitle("SLAM-friendly Metrics", fontsize=13, color=_TEXT_COLOR, y=0.99)

    _plot_inlier_axis(axes[0], records, metrics_obj)
    _plot_pose_axis(axes[1], records)

    fig.tight_layout(pad=0.8, rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _update_status(metrics_obj):
    has_inlier = metrics_obj["inlier_ratio"].get("mean") is not None
    has_pose = metrics_obj.get("pose_success_rate") is not None
    warnings_list = list(metrics_obj.get("warnings", []) or [])
    if has_inlier and has_pose:
        metrics_obj["eval_status"] = "success" if not warnings_list else "partial"
        return
    if has_inlier or has_pose:
        metrics_obj["eval_status"] = "partial"
        return
    metrics_obj["eval_status"] = "failed"


def write_slam_outputs(out_dir, metrics_obj, records):
    metrics_path = out_dir / "slam_metrics.json"
    pairs_path = out_dir / "slam_pairs.csv"
    plot_path = out_dir / "plots" / "slam_metrics_curve.png"
    write_json(metrics_path, metrics_obj, indent=2)
    write_csv(
        pairs_path,
        [
            "frame_idx_0",
            "frame_idx_1",
            "raw_matches",
            "inlier_matches",
            "inlier_ratio",
            "pose_success",
            "rotation_error_deg",
            "translation_direction_error_deg",
        ],
        _csv_rows(records),
    )
    try:
        _write_plot(plot_path, records, metrics_obj)
    except Exception as exc:
        append_warning(metrics_obj, "failed to generate slam plot: {}".format(exc))
        write_json(metrics_path, metrics_obj, indent=2)


def run_slam_eval(exp_dir, out_dir):
    exp_root = Path(exp_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_obj = _base_metrics(exp_root)
    log_info("slam-friendly eval start: exp_dir={} out_dir={}".format(exp_root, out_path))

    pairs = _resolve_pair_context(exp_root, metrics_obj)
    runtime = _load_cv_runtime(metrics_obj)
    camera_matrix = _load_camera_matrix(exp_root, metrics_obj)
    reference_context = _resolve_reference_context(exp_root, metrics_obj)

    records = []
    missing_reference_pairs = 0
    pose_unresolved_pairs = 0

    if pairs is not None and runtime is not None and camera_matrix is not None:
        cv2 = runtime["cv2"]
        for pair in pairs:
            record = {
                "frame_idx_0": int(pair["frame_idx_0"]),
                "frame_idx_1": int(pair["frame_idx_1"]),
                "raw_matches": None,
                "inlier_matches": None,
                "inlier_ratio": None,
                "pose_success": None,
                "rotation_error_deg": None,
                "translation_direction_error_deg": None,
            }
            try:
                image0 = _load_gray_image(cv2, pair["image_path_0"])
                image1 = _load_gray_image(cv2, pair["image_path_1"])
            except Exception:
                records.append(record)
                continue

            try:
                matches, kp0, kp1 = _match_features(runtime, image0, image1)
                pose_payload = _estimate_relative_pose(runtime, camera_matrix, kp0, kp1, matches)
            except Exception:
                pose_payload = {
                    "raw_matches": 0,
                    "inlier_matches": 0,
                    "inlier_ratio": 0.0,
                    "rotation_matrix": None,
                    "translation_vec": None,
                }

            record["raw_matches"] = int(pose_payload.get("raw_matches") or 0)
            record["inlier_matches"] = int(pose_payload.get("inlier_matches") or 0)
            record["inlier_ratio"] = pose_payload.get("inlier_ratio")

            if reference_context is None:
                records.append(record)
                continue

            ref_pose = _relative_pose_from_reference(
                reference_context,
                pair["frame_idx_0"],
                pair["frame_idx_1"],
            )
            if ref_pose is None:
                missing_reference_pairs += 1
                records.append(record)
                continue

            rot_est = pose_payload.get("rotation_matrix")
            trans_est = pose_payload.get("translation_vec")
            if rot_est is None or trans_est is None:
                pose_unresolved_pairs += 1
                records.append(record)
                continue

            rotation_error_deg = _rotation_error_deg(rot_est, ref_pose["rotation_matrix"])
            translation_error_deg = _direction_error_deg(trans_est, ref_pose["translation_vec"])
            record["rotation_error_deg"] = rotation_error_deg
            record["translation_direction_error_deg"] = translation_error_deg
            record["pose_success"] = _pose_success(rotation_error_deg, translation_error_deg)
            records.append(record)

    if missing_reference_pairs > 0:
        append_warning(
            metrics_obj,
            "reference poses missing for {} selected slam pairs".format(int(missing_reference_pairs)),
        )
    if pose_unresolved_pairs > 0:
        append_warning(
            metrics_obj,
            "pose estimation unresolved for {} selected slam pairs".format(int(pose_unresolved_pairs)),
        )

    inlier_values = [item.get("inlier_ratio") for item in records if item.get("inlier_ratio") is not None]
    rotation_values = [item.get("rotation_error_deg") for item in records if item.get("rotation_error_deg") is not None]
    translation_values = [
        item.get("translation_direction_error_deg")
        for item in records
        if item.get("translation_direction_error_deg") is not None
    ]
    pose_success_values = [item.get("pose_success") for item in records if item.get("pose_success") is not None]

    metrics_obj["valid_pair_count"] = int(len(inlier_values))
    metrics_obj["valid_pose_pair_count"] = int(len(pose_success_values))
    metrics_obj["successful_pose_pair_count"] = int(sum(1 for item in pose_success_values if item))
    metrics_obj["pose_success_rate"] = (
        float(metrics_obj["successful_pose_pair_count"]) / float(metrics_obj["valid_pose_pair_count"])
        if metrics_obj["valid_pose_pair_count"] > 0
        else None
    )
    metrics_obj["inlier_ratio"] = metric_stats(inlier_values)
    metrics_obj["rotation_error_deg"] = metric_stats(rotation_values)
    metrics_obj["translation_direction_error_deg"] = metric_stats(translation_values)
    _update_status(metrics_obj)
    write_slam_outputs(out_path, metrics_obj, records)

    return {
        "metrics": metrics_obj,
        "records": records,
        "metrics_path": out_path / "slam_metrics.json",
        "pairs_path": out_path / "slam_pairs.csv",
        "plot_path": out_path / "plots" / "slam_metrics_curve.png",
    }
