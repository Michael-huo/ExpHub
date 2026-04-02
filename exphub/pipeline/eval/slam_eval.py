from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np

from exphub.common.io import list_frames_sorted
from .reporting import append_warning, empty_stats, metric_stats, read_json, read_timestamps, resolve_formal_eval_inputs


_RATIO_TEST = 0.75
_RANSAC_THRESHOLD_PX = 1.0
_RANSAC_PROB = 0.999
_ROT_SUCCESS_DEG = 10.0
_TRANS_SUCCESS_DEG = 10.0
_POSE_EPS = 1e-9
_TRACK_TS_TOL_SEC = 1e-3


def _base_metrics(exp_dir):
    inputs = resolve_formal_eval_inputs(exp_dir)
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "reference_source": "unavailable",
        "uses_proxy_reference": False,
        "reference_path": "",
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
        "merge_manifest_path": str(inputs["merge_manifest_path"]),
        "slam_report_path": str(inputs["slam_report_path"]),
        "inlier_ratio": empty_stats(),
        "rotation_error_deg": empty_stats(),
        "translation_direction_error_deg": empty_stats(),
    }


def _load_cv_runtime(metrics_obj):
    try:
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
    inputs = resolve_formal_eval_inputs(exp_dir)
    calib_path = None
    for candidate in [inputs["merge_calib_path"], inputs["segment_calib_path"]]:
        if Path(candidate).is_file():
            calib_path = Path(candidate)
            break
    if calib_path is None:
        append_warning(metrics_obj, "missing calib.txt; slam-friendly eval unavailable")
        return None
    try:
        arr = np.loadtxt(str(calib_path), delimiter=" ")
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception as exc:
        append_warning(metrics_obj, "failed to load calib.txt: {}".format(exc))
        return None
    if arr.size < 4:
        append_warning(metrics_obj, "invalid calib.txt: {}".format(calib_path))
        return None
    fx, fy, cx, cy = [float(item) for item in arr[:4]]
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_pose_trajectory_npz(path_obj):
    try:
        payload = np.load(str(path_obj))
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
    return {"timestamps": timestamps, "poses": poses}


def _quat_xyzw_to_rot(qx, qy, qz, qw):
    q = np.asarray([float(qx), float(qy), float(qz), float(qw)], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= _POSE_EPS:
        raise ValueError("invalid quaternion norm")
    q = q / norm
    x, y, z, w = [float(item) for item in q]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_pose_trajectory_tum(path_obj):
    timestamps = []
    poses = []
    try:
        lines = Path(path_obj).read_text(encoding="utf-8", errors="ignore").splitlines()
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
            timestamp = float(parts[0])
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = _quat_xyzw_to_rot(parts[4], parts[5], parts[6], parts[7])
            pose[:3, 3] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        except Exception:
            continue
        timestamps.append(timestamp)
        poses.append(pose)
    if not poses:
        return None
    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "poses": np.asarray(poses, dtype=np.float64),
    }


def _build_pose_map_from_frames(payload, run_meta_obj):
    frames_dir = Path(str((run_meta_obj or {}).get("frames_dir", "") or "")).resolve()
    if not frames_dir.is_dir():
        return {}
    files = list_frames_sorted(frames_dir)
    t0 = int((run_meta_obj or {}).get("t0", 0) or 0)
    stride = int((run_meta_obj or {}).get("stride", 1) or 1)
    max_frames = int((run_meta_obj or {}).get("max_frames", 0) or 0)
    if t0 > 0:
        files = files[t0:]
    if stride > 1:
        files = files[::stride]
    if max_frames > 0:
        files = files[:max_frames]

    poses = np.asarray(payload.get("poses"), dtype=np.float64)
    usable = min(len(files), poses.shape[0])
    pose_map = {}
    for idx in range(usable):
        stem = Path(files[idx]).stem
        try:
            frame_idx = int(stem)
        except Exception:
            continue
        pose_map[int(frame_idx)] = poses[idx]
    return pose_map


def _build_pose_map_from_timestamps(payload, timestamps_path):
    timestamps = read_timestamps(timestamps_path)
    if not timestamps:
        return {}
    pose_ts = np.asarray(payload.get("timestamps"), dtype=np.float64).reshape(-1)
    poses = np.asarray(payload.get("poses"), dtype=np.float64)
    if poses.shape[0] != pose_ts.shape[0]:
        return {}
    out = {}
    for pose_idx, ts_value in enumerate(pose_ts):
        best_idx = None
        best_diff = None
        for frame_idx, ref_ts in enumerate(timestamps):
            diff = abs(float(ref_ts) - float(ts_value))
            if best_diff is None or diff < best_diff:
                best_idx = frame_idx
                best_diff = diff
        if best_idx is None or best_diff is None or best_diff > _TRACK_TS_TOL_SEC:
            continue
        out[int(best_idx)] = poses[pose_idx]
    return out


def _load_reference_context(exp_dir, metrics_obj):
    slam_report = read_json(resolve_formal_eval_inputs(exp_dir)["slam_report_path"]) or {}
    tracks = dict(slam_report.get("tracks") or {})
    candidates = [("gt", False), ("ori", True)]
    for track_name, uses_proxy in candidates:
        track_obj = tracks.get(track_name)
        if not isinstance(track_obj, dict):
            continue
        traj_rel = str(track_obj.get("npz_path", "") or "")
        tum_rel = str(track_obj.get("traj_path", "") or "")
        run_meta_rel = str(track_obj.get("run_meta_path", "") or "")
        traj_path = (Path(exp_dir).resolve() / traj_rel).resolve() if traj_rel else None
        tum_path = (Path(exp_dir).resolve() / tum_rel).resolve() if tum_rel else None
        run_meta_path = (Path(exp_dir).resolve() / run_meta_rel).resolve() if run_meta_rel else None
        payload = None
        if traj_path is not None and traj_path.is_file():
            payload = _load_pose_trajectory_npz(traj_path)
        if payload is None and tum_path is not None and tum_path.is_file():
            payload = _load_pose_trajectory_tum(tum_path)
        if payload is None:
            continue
        run_meta_obj = read_json(run_meta_path) if run_meta_path is not None else {}
        pose_map = _build_pose_map_from_frames(payload, run_meta_obj)
        if not pose_map:
            timestamps_path = str((run_meta_obj or {}).get("timestamps", "") or "")
            if timestamps_path:
                pose_map = _build_pose_map_from_timestamps(payload, timestamps_path)
        if not pose_map:
            continue
        metrics_obj["reference_source"] = str(track_name)
        metrics_obj["uses_proxy_reference"] = bool(uses_proxy)
        metrics_obj["reference_path"] = str(traj_path if traj_path is not None and traj_path.is_file() else tum_path)
        return {
            "poses_by_frame_idx": pose_map,
            "reference_source": str(track_name),
        }
    append_warning(metrics_obj, "reference trajectory unavailable for slam-friendly eval; require slam/ori or slam/gt")
    return None


def _resolve_pair_context(exp_dir, metrics_obj):
    inputs = resolve_formal_eval_inputs(exp_dir)
    merge_frames_dir = Path(inputs["merge_frames_dir"]).resolve()
    merge_manifest = read_json(inputs["merge_manifest_path"]) or {}
    if not merge_frames_dir.is_dir():
        append_warning(metrics_obj, "missing merge frames directory: {}".format(merge_frames_dir))
        return None
    merge_frames = list_frames_sorted(merge_frames_dir)
    if not merge_frames:
        append_warning(metrics_obj, "no merge frames found for slam-friendly eval")
        return None

    summary = dict(merge_manifest.get("summary") or {})
    segments = list(merge_manifest.get("segments") or [])
    if summary.get("merged_start_idx") is None or summary.get("merged_end_idx") is None:
        append_warning(metrics_obj, "merge_manifest missing merged_start_idx/merged_end_idx")
        return None
    merged_start = int(summary.get("merged_start_idx"))
    merged_end = int(summary.get("merged_end_idx"))
    metrics_obj["pair_selection_source"] = "merge/merge_manifest.json"

    frame_map = {}
    expected_count = int(merged_end - merged_start + 1)
    usable = min(expected_count, len(merge_frames))
    for seq_idx in range(usable):
        frame_map[int(merged_start + seq_idx)] = merge_frames[seq_idx]

    anchor_indices = set()
    for item in segments:
        if not isinstance(item, dict):
            continue
        anchor_indices.add(int(item.get("start_idx", -1)))
        anchor_indices.add(int(item.get("end_idx", -1)))

    generated_indices = [frame_idx for frame_idx in sorted(frame_map.keys()) if frame_idx not in anchor_indices]
    pairs = []
    generated_set = set(generated_indices)
    for frame_idx in generated_indices:
        next_idx = int(frame_idx + 1)
        if next_idx not in generated_set:
            continue
        pairs.append(
            {
                "frame_idx_0": int(frame_idx),
                "frame_idx_1": int(next_idx),
                "image_path_0": str(frame_map[frame_idx]),
                "image_path_1": str(frame_map[next_idx]),
            }
        )
    metrics_obj["selected_pair_count"] = int(len(pairs))
    if not pairs:
        append_warning(metrics_obj, "no consecutive generated frame pairs available for slam-friendly eval")
        return None
    return pairs


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
    raw_matches = int(len(matches))
    if raw_matches < 5:
        return {
            "raw_matches": raw_matches,
            "inlier_matches": 0,
            "inlier_ratio": 0.0 if raw_matches > 0 else None,
            "rotation_matrix": None,
            "translation_vec": None,
        }
    pts0 = np.asarray([kp0[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts1 = np.asarray([kp1[m.trainIdx].pt for m in matches], dtype=np.float64)
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
        _, rot, trans, _ = cv2.recoverPose(essential, pts0, pts1, cameraMatrix=camera_matrix, mask=mask)
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
    pose0 = reference_context["poses_by_frame_idx"].get(int(frame_idx_0))
    pose1 = reference_context["poses_by_frame_idx"].get(int(frame_idx_1))
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


def run_slam_eval(exp_dir, out_dir):
    metrics_obj = _base_metrics(exp_dir)
    runtime = _load_cv_runtime(metrics_obj)
    camera_matrix = _load_camera_matrix(exp_dir, metrics_obj)
    reference_context = _load_reference_context(exp_dir, metrics_obj)
    pair_context = _resolve_pair_context(exp_dir, metrics_obj)
    if runtime is None or camera_matrix is None or reference_context is None or pair_context is None:
        return {"metrics": metrics_obj, "records": []}

    cv2 = runtime["cv2"]
    records = []
    inlier_values = []
    rotation_errors = []
    translation_errors = []

    for item in pair_context:
        try:
            image0 = cv2.imread(str(item["image_path_0"]), cv2.IMREAD_GRAYSCALE)
            image1 = cv2.imread(str(item["image_path_1"]), cv2.IMREAD_GRAYSCALE)
        except Exception:
            image0 = None
            image1 = None
        if image0 is None or image1 is None:
            continue

        matches, kp0, kp1 = _match_features(runtime, image0, image1)
        pose_est = _estimate_relative_pose(runtime, camera_matrix, kp0, kp1, matches)
        pose_ref = _relative_pose_from_reference(reference_context, item["frame_idx_0"], item["frame_idx_1"])

        rotation_error = None
        translation_error = None
        pose_success = None
        if pose_ref is not None and pose_est.get("rotation_matrix") is not None and pose_est.get("translation_vec") is not None:
            rotation_error = _rotation_error_deg(pose_est["rotation_matrix"], pose_ref["rotation_matrix"])
            translation_error = _direction_error_deg(pose_est["translation_vec"], pose_ref["translation_vec"])
            if rotation_error is not None and translation_error is not None:
                pose_success = bool(rotation_error < float(_ROT_SUCCESS_DEG) and translation_error < float(_TRANS_SUCCESS_DEG))

        if pose_est.get("inlier_ratio") is not None:
            inlier_values.append(float(pose_est["inlier_ratio"]))
        if rotation_error is not None:
            rotation_errors.append(float(rotation_error))
        if translation_error is not None:
            translation_errors.append(float(translation_error))

        records.append(
            {
                "frame_idx_0": int(item["frame_idx_0"]),
                "frame_idx_1": int(item["frame_idx_1"]),
                "raw_matches": int(pose_est.get("raw_matches", 0) or 0),
                "inlier_matches": int(pose_est.get("inlier_matches", 0) or 0),
                "inlier_ratio": pose_est.get("inlier_ratio"),
                "pose_success": pose_success,
                "rotation_error_deg": rotation_error,
                "translation_direction_error_deg": translation_error,
            }
        )

    metrics_obj["valid_pair_count"] = int(len(records))
    metrics_obj["valid_pose_pair_count"] = int(len([item for item in records if item.get("pose_success") is not None]))
    metrics_obj["successful_pose_pair_count"] = int(len([item for item in records if item.get("pose_success") is True]))
    if metrics_obj["valid_pose_pair_count"] > 0:
        metrics_obj["pose_success_rate"] = float(metrics_obj["successful_pose_pair_count"]) / float(metrics_obj["valid_pose_pair_count"])
    metrics_obj["inlier_ratio"] = metric_stats(inlier_values)
    metrics_obj["rotation_error_deg"] = metric_stats(rotation_errors)
    metrics_obj["translation_direction_error_deg"] = metric_stats(translation_errors)
    metrics_obj["eval_status"] = "success" if records else "failed"
    return {
        "metrics": metrics_obj,
        "records": records,
    }
