from __future__ import annotations

import copy
import datetime
import os
import tempfile
from pathlib import Path

import numpy as np

from exphub.common.logging import log_info
from .reporting import append_warning, read_json, read_timestamps


_STAT_KEYS = ["rmse", "mean", "median", "std", "min", "max"]
_PLOT_DPI = 220
_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_REF_COLOR = "#1f4e79"
_EST_COLOR = "#c56a2d"
_REF_BG_COLOR = "#b8c3cf"
_KEYFRAME_MARKER_EDGE = "#6e7c86"
_KEYFRAME_MARKER_SIZE = 22


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _empty_stats():
    return dict((key, None) for key in _STAT_KEYS)


def _float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _stats_dict(stats_obj):
    out = _empty_stats()
    for key in _STAT_KEYS:
        out[key] = _float_or_none((stats_obj or {}).get(key))
    return out


def _base_metrics(config):
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "reference_name": str(_get_arg(config, "reference_name", "ori")),
        "estimate_name": str(_get_arg(config, "estimate_name", "gen")),
        "reference_path": str(Path(_get_arg(config, "reference")).resolve()),
        "estimate_path": str(Path(_get_arg(config, "estimate")).resolve()),
        "reference_pose_count": 0,
        "estimate_pose_count": 0,
        "matched_pose_count": 0,
        "alignment_mode": str(_get_arg(config, "alignment_mode", "se3")),
        "rpe_delta": float(_get_arg(config, "delta", 1.0)),
        "rpe_delta_unit": str(_get_arg(config, "delta_unit", "frames")),
        "sync_t_max_diff_sec": float(_get_arg(config, "t_max_diff", 0.01)),
        "sync_t_offset_sec": float(_get_arg(config, "t_offset", 0.0)),
        "ori_path_length_m": None,
        "gen_path_length_m": None,
        "metric_units": {
            "ape_trans": "m",
            "rpe_trans": "m",
            "rpe_rot": "deg",
        },
        "ape_trans": _empty_stats(),
        "rpe_trans": _empty_stats(),
        "rpe_rot": _empty_stats(),
    }


def _alignment_flags(alignment_mode):
    mode = str(alignment_mode or "se3").strip().lower()
    if mode == "sim3":
        return True, True, False
    if mode == "origin":
        return False, False, True
    if mode == "none":
        return False, False, False
    return True, False, False


def _load_evo_modules():
    from evo.core import metrics
    from evo.core import sync
    from evo.core import units
    from evo.main_ape import ape as evo_ape
    from evo.main_rpe import rpe as evo_rpe
    from evo.tools import file_interface

    return file_interface, metrics, sync, units, evo_ape, evo_rpe


def _path_length(points_xyz):
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        return None
    diffs = np.diff(points[:, :3], axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    finite = lengths[np.isfinite(lengths)]
    if finite.size <= 0:
        return None
    return float(np.sum(finite))


def _curve_payload(result):
    if result is None:
        return None
    errors = np.asarray(result.np_arrays.get("error_array", []), dtype=np.float64)
    if errors.size <= 0:
        return None
    seconds = result.np_arrays.get("seconds_from_start")
    if seconds is not None:
        xs = np.asarray(seconds, dtype=np.float64)
        xlabel = "Time From Start (s)"
    else:
        xs = np.arange(errors.shape[0], dtype=np.float64)
        xlabel = "Pose Index"
    return {
        "x": xs.tolist(),
        "y": errors.tolist(),
        "xlabel": xlabel,
        "x_kind": "time" if seconds is not None else "index",
    }


def _traj_detail_records(ref_traj, est_traj, ape_result):
    ref_xyz = np.asarray(getattr(ref_traj, "positions_xyz", None), dtype=np.float64)
    est_xyz = np.asarray(getattr(est_traj, "positions_xyz", None), dtype=np.float64)
    timestamps = np.asarray(getattr(ref_traj, "timestamps", []), dtype=np.float64).reshape(-1)
    errors = np.asarray(getattr(ape_result, "np_arrays", {}).get("error_array", []), dtype=np.float64).reshape(-1)
    size = min(ref_xyz.shape[0], est_xyz.shape[0], timestamps.shape[0], errors.shape[0])
    if size <= 0:
        return []
    rows = []
    for idx in range(size):
        rows.append(
            {
                "pose_idx": int(idx),
                "timestamp": float(timestamps[idx]) if np.isfinite(timestamps[idx]) else None,
                "ape_trans_m": float(errors[idx]) if np.isfinite(errors[idx]) else None,
                "ref_x": float(ref_xyz[idx, 0]) if ref_xyz.shape[1] > 0 and np.isfinite(ref_xyz[idx, 0]) else None,
                "ref_y": float(ref_xyz[idx, 1]) if ref_xyz.shape[1] > 1 and np.isfinite(ref_xyz[idx, 1]) else None,
                "ref_z": float(ref_xyz[idx, 2]) if ref_xyz.shape[1] > 2 and np.isfinite(ref_xyz[idx, 2]) else None,
                "est_x": float(est_xyz[idx, 0]) if est_xyz.shape[1] > 0 and np.isfinite(est_xyz[idx, 0]) else None,
                "est_y": float(est_xyz[idx, 1]) if est_xyz.shape[1] > 1 and np.isfinite(est_xyz[idx, 1]) else None,
                "est_z": float(est_xyz[idx, 2]) if est_xyz.shape[1] > 2 and np.isfinite(est_xyz[idx, 2]) else None,
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


def _box_props(edge_color):
    return {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": edge_color,
        "linewidth": 0.9,
        "alpha": 0.96,
    }


def _style_axes(ax):
    ax.set_facecolor(_FIG_FACE)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#2f3b4a", labelsize=10)


def _style_figure(fig):
    fig.patch.set_facecolor(_FIG_FACE)


def _stat_line(label, value, unit):
    if value is None:
        return "{}: n/a".format(label)
    return "{}: {:.4f} {}".format(label, float(value), unit).strip()


def _metrics_box_text(metrics_obj):
    lines = [
        _stat_line("APE RMSE", metrics_obj["ape_trans"].get("rmse"), "m"),
        _stat_line("RPE trans RMSE", metrics_obj["rpe_trans"].get("rmse"), "m"),
        "Matched: {}".format(int(metrics_obj.get("matched_pose_count") or 0)),
    ]
    return "\n".join(lines)


def _candidate_exp_roots(path_candidates):
    seen = set()
    out = []
    for raw_path in list(path_candidates or []):
        if raw_path is None:
            continue
        text = str(raw_path).strip()
        if not text:
            continue
        try:
            path_obj = Path(text).resolve()
        except Exception:
            continue
        for candidate in [path_obj] + list(path_obj.parents):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def _resolve_eval_exp_root(path_candidates):
    for candidate in _candidate_exp_roots(path_candidates):
        if (candidate / "segment" / "segment_manifest.json").is_file():
            return candidate
    return None


def _segment_timestamp_map(exp_root):
    if exp_root is None:
        return {}
    timestamps = read_timestamps(Path(exp_root).resolve() / "segment" / "timestamps.txt")
    out = {}
    for idx, value in enumerate(timestamps):
        try:
            out[int(idx)] = float(value)
        except Exception:
            continue
    return out


def _load_formal_keyframe_context(config, metrics_obj):
    exp_root = _resolve_eval_exp_root(
        [
            _get_arg(config, "exp_dir", None),
            _get_arg(config, "out_dir", None),
            _get_arg(config, "reference", None),
            _get_arg(config, "estimate", None),
        ]
    )
    empty_context = {
        "exp_root": None,
        "manifest_path": "",
        "frame_indices": [],
        "timestamps_by_frame": {},
    }
    if exp_root is None:
        append_warning(metrics_obj, "eval plot keyframes unavailable: missing segment formal artifacts")
        return empty_context

    exp_root = Path(exp_root).resolve()
    manifest_path = exp_root / "segment" / "segment_manifest.json"
    manifest_obj = read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest_obj, dict):
        manifest_obj = {}

    frame_indices = []
    seen = set()
    raw_indices = []
    if isinstance(manifest_obj.get("keyframes"), dict):
        raw_indices = list(manifest_obj["keyframes"].get("indices") or [])
    for value in raw_indices:
        try:
            frame_idx = int(value)
        except Exception:
            continue
        if frame_idx < 0 or frame_idx in seen:
            continue
        seen.add(frame_idx)
        frame_indices.append(frame_idx)
    frame_indices.sort()
    if not frame_indices:
        append_warning(metrics_obj, "eval plot keyframes unavailable: empty keyframe indices in formal segment artifacts")
        return empty_context

    timestamps_by_frame = {}
    fallback = _segment_timestamp_map(exp_root)
    for frame_idx in frame_indices:
        if frame_idx in fallback:
            timestamps_by_frame[frame_idx] = float(fallback[frame_idx])

    return {
        "exp_root": exp_root,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "frame_indices": list(frame_indices),
        "timestamps_by_frame": dict(timestamps_by_frame),
    }


def _finite_timestamps(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _timestamp_match_tolerance(sample_timestamps):
    finite = _finite_timestamps(sample_timestamps)
    if finite.shape[0] < 2:
        return 1e-4
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0.0]
    if diffs.size <= 0:
        return 1e-4
    return max(1e-4, float(np.median(diffs)) * 0.45)


def _nearest_timestamp_indices(sample_timestamps, target_timestamps, tolerance):
    if tolerance is None or tolerance < 0.0:
        return []
    sample_arr = np.asarray(sample_timestamps, dtype=np.float64).reshape(-1)
    if sample_arr.size <= 0:
        return []
    out = []
    seen = set()
    for target in list(target_timestamps or []):
        try:
            ts_value = float(target)
        except Exception:
            continue
        if not np.isfinite(ts_value):
            continue
        pos = int(np.searchsorted(sample_arr, ts_value))
        candidates = []
        if pos < sample_arr.shape[0]:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best_idx = None
        best_diff = None
        for idx in candidates:
            diff = abs(float(sample_arr[idx]) - ts_value)
            if best_diff is None or diff < best_diff:
                best_idx = int(idx)
                best_diff = float(diff)
        if best_idx is None or best_diff is None or best_diff > tolerance:
            continue
        if best_idx in seen:
            continue
        seen.add(best_idx)
        out.append(best_idx)
    out.sort()
    return out


def _keyframe_timestamps(keyframe_context):
    if not isinstance(keyframe_context, dict):
        return []
    timestamps_map = dict(keyframe_context.get("timestamps_by_frame") or {})
    out = []
    for frame_idx in list(keyframe_context.get("frame_indices") or []):
        if frame_idx not in timestamps_map:
            continue
        try:
            value = float(timestamps_map[frame_idx])
        except Exception:
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _traj_keyframe_sample_indices(ref_timestamps, keyframe_context, sample_count):
    if sample_count <= 0:
        return []
    tolerance = _timestamp_match_tolerance(ref_timestamps)
    sample_indices = _nearest_timestamp_indices(ref_timestamps, _keyframe_timestamps(keyframe_context), tolerance)
    out = []
    for idx in sample_indices:
        if idx <= 0 or idx >= int(sample_count) - 1:
            continue
        out.append(int(idx))
    return out


def _normalized_points(points_xy):
    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.size == 0:
        return arr
    if arr.ndim != 2:
        return np.asarray([], dtype=np.float64).reshape(0, 2)
    finite_mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_mask]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64).reshape(0, 2)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    return (arr - mins) / span


def _corner_candidates(points_xy):
    norm_pts = _normalized_points(points_xy)
    if norm_pts.size == 0:
        return ["upper left", "upper right", "lower left", "lower right"]
    corners = [
        ("upper left", (0.00, 0.58, 0.42, 0.42)),
        ("upper right", (0.58, 0.58, 0.42, 0.42)),
        ("lower left", (0.00, 0.00, 0.42, 0.42)),
        ("lower right", (0.58, 0.00, 0.42, 0.42)),
    ]
    ranked = []
    for name, rect in corners:
        x0, y0, width, height = rect
        x1 = x0 + width
        y1 = y0 + height
        inside = (
            (norm_pts[:, 0] >= x0)
            & (norm_pts[:, 0] <= x1)
            & (norm_pts[:, 1] >= y0)
            & (norm_pts[:, 1] <= y1)
        )
        ranked.append((int(np.count_nonzero(inside)), name))
    ranked.sort(key=lambda item: item[0])
    return [name for _, name in ranked]


def _corner_anchor(corner_name):
    mapping = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }
    return mapping.get(corner_name, mapping["upper left"])


def _add_corner_box(ax, text, corner_name, edge_color, fontsize=9.4):
    x, y, ha, va = _corner_anchor(corner_name)
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=_TEXT_COLOR,
        bbox=_box_props(edge_color),
        zorder=6,
    )


def _error_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "exphub_pose_error",
        ["#334d69", "#5f778b", "#8e9ea6", "#b4a48f", "#a57d5e", "#845640"],
    )


def _xy_projection(points_xyz):
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2:
        return np.asarray([], dtype=np.float64).reshape(0, 2)
    dims = min(points.shape[1], 2)
    projected = points[:, :dims]
    if dims == 2:
        return projected
    zeros = np.zeros((points.shape[0], 2 - dims), dtype=np.float64)
    return np.hstack([projected, zeros])


def _project_traj_to_view_plane(ref_xyz, est_xyz):
    ref_points = np.asarray(ref_xyz, dtype=np.float64)
    est_points = np.asarray(est_xyz, dtype=np.float64)
    fallback = {
        "ref_xy": _xy_projection(ref_points),
        "est_xy": _xy_projection(est_points),
        "projection_name": "xy",
        "used_pca": False,
    }
    if ref_points.ndim != 2 or est_points.ndim != 2:
        return fallback
    if ref_points.shape[0] < 2 or ref_points.shape[1] < 3 or est_points.shape[1] < 3:
        return fallback
    finite_mask = np.all(np.isfinite(ref_points[:, :3]), axis=1)
    ref_valid = ref_points[finite_mask, :3]
    if ref_valid.shape[0] < 2:
        return fallback
    center = np.mean(ref_valid, axis=0)
    centered = ref_valid - center
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return fallback
    if vh.shape[0] < 2:
        return fallback
    basis = vh[:2, :].T
    if basis.shape != (3, 2) or not np.all(np.isfinite(basis)):
        return fallback
    for idx in range(basis.shape[1]):
        axis = basis[:, idx]
        max_entry = int(np.argmax(np.abs(axis)))
        if axis[max_entry] < 0.0:
            basis[:, idx] *= -1.0
    ref_xy = np.dot(ref_points[:, :3] - center, basis)
    est_xy = np.dot(est_points[:, :3] - center, basis)
    if not np.any(np.isfinite(ref_xy)) or not np.any(np.isfinite(est_xy)):
        return fallback
    return {
        "ref_xy": ref_xy,
        "est_xy": est_xy,
        "projection_name": "pca",
        "used_pca": True,
    }


def _plot_traj_xy(plt, out_path, ref_traj, est_traj, ape_result, ref_label, est_label, metrics_obj, keyframe_sample_indices=None):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    ref_xyz = np.asarray(ref_traj.positions_xyz, dtype=np.float64)
    est_xyz = np.asarray(est_traj.positions_xyz, dtype=np.float64)
    errors = np.asarray(ape_result.np_arrays.get("error_array", []), dtype=np.float64)
    projection = _project_traj_to_view_plane(ref_xyz, est_xyz)
    ref_xy = projection["ref_xy"]
    est_xy = projection["est_xy"]
    if ref_xy.shape[0] <= 0 or est_xy.shape[0] <= 0:
        append_warning(metrics_obj, "trajectory plot unavailable: insufficient aligned positions")
        return
    all_xy = np.vstack([ref_xy, est_xy])
    corners = _corner_candidates(all_xy)
    info_corner = corners[0]
    legend_corner = corners[1] if len(corners) > 1 else "upper right"

    fig, ax = plt.subplots(figsize=(9.2, 6.6), dpi=_PLOT_DPI)
    _style_figure(fig)
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.12, top=0.90)
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color=_REF_BG_COLOR, linewidth=1.8, alpha=0.96, zorder=1)

    colorbar = None
    est_line_drawn = False
    if est_xyz.shape[0] >= 2 and errors.size > 0:
        size = min(est_xyz.shape[0], errors.shape[0])
        est_xy_view = est_xy[:size]
        if est_xy_view.shape[0] >= 2:
            seg_errors = 0.5 * (errors[:-1] + errors[1:]) if errors.size >= 2 else errors
            if seg_errors.size == 0:
                seg_errors = np.asarray([errors[0]], dtype=np.float64)
            points = est_xy_view.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_errors = np.asarray(seg_errors[: segments.shape[0]], dtype=np.float64)
            finite_errors = seg_errors[np.isfinite(seg_errors)]
            if finite_errors.size > 0:
                fill_value = float(np.median(finite_errors))
                seg_errors = np.where(np.isfinite(seg_errors), seg_errors, fill_value)
                vmax = float(np.percentile(finite_errors, 98.0))
                vmax = max(vmax, float(finite_errors.max()), 1e-6)
                ax.plot(est_xy[:, 0], est_xy[:, 1], color="#6e7c86", linewidth=0.9, alpha=0.22, zorder=2)
                lc = LineCollection(
                    segments,
                    cmap=_error_colormap(),
                    norm=Normalize(vmin=0.0, vmax=vmax),
                    linewidths=2.9,
                    capstyle="round",
                    zorder=3,
                )
                lc.set_array(seg_errors)
                ax.add_collection(lc)
                colorbar_ax = fig.add_axes([0.865, 0.16, 0.022, 0.66])
                colorbar = fig.colorbar(lc, cax=colorbar_ax)
                colorbar.set_label("APE (m)", fontsize=10.5, color=_TEXT_COLOR)
                colorbar.ax.tick_params(labelsize=9.4, colors=_TEXT_COLOR)
                colorbar.outline.set_edgecolor(_SPINE_COLOR)
                colorbar.outline.set_linewidth(0.8)
                est_line_drawn = True
    if not est_line_drawn:
        ax.plot(est_xy[:, 0], est_xy[:, 1], color=_EST_COLOR, linewidth=2.2, alpha=0.96, zorder=3)

    show_keyframe_legend = False
    if keyframe_sample_indices:
        valid_indices = [int(idx) for idx in keyframe_sample_indices if 0 <= int(idx) < ref_xy.shape[0]]
        if valid_indices:
            keyframe_xy = ref_xy[valid_indices]
            ax.scatter(
                keyframe_xy[:, 0],
                keyframe_xy[:, 1],
                s=_KEYFRAME_MARKER_SIZE,
                facecolors="white",
                edgecolors=_KEYFRAME_MARKER_EDGE,
                linewidths=0.8,
                alpha=0.92,
                zorder=4,
            )
            show_keyframe_legend = True

    marker_size = 40
    ax.scatter(ref_xy[0, 0], ref_xy[0, 1], color=_REF_COLOR, s=marker_size, marker="o", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(ref_xy[-1, 0], ref_xy[-1, 1], color=_REF_COLOR, s=marker_size + 10, marker="D", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(est_xy[0, 0], est_xy[0, 1], color=_EST_COLOR, s=marker_size, marker="o", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(est_xy[-1, 0], est_xy[-1, 1], color=_EST_COLOR, s=marker_size + 10, marker="D", edgecolors="white", linewidths=0.9, zorder=5)

    ax.set_title("Trajectory Projection", fontsize=13, color=_TEXT_COLOR, pad=10)
    ax.set_xlabel("View Axis 1 (m)", fontsize=11)
    ax.set_ylabel("View Axis 2 (m)", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.06, y=0.08)
    _style_axes(ax)

    legend_handles = [
        Line2D([0], [0], color=_REF_BG_COLOR, linewidth=1.8, label=ref_label),
        Line2D(
            [0],
            [0],
            color="#6e7c86" if colorbar is not None else _EST_COLOR,
            linewidth=2.4,
            label="{} (APE-colored)".format(est_label) if colorbar is not None else est_label,
        ),
    ]
    if show_keyframe_legend:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=_KEYFRAME_MARKER_EDGE,
                markeredgewidth=0.8,
                markersize=5.0,
                label="keyframes",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc=legend_corner,
        frameon=True,
        facecolor="white",
        edgecolor=_SPINE_COLOR,
        framealpha=0.96,
        fontsize=9.3,
    )
    _add_corner_box(ax, _metrics_box_text(metrics_obj), info_corner, _REF_COLOR)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _save_traj_plot(out_dir, ref_traj, est_traj, ape_result, metrics_obj, keyframe_context):
    if ref_traj is None or est_traj is None:
        return None
    ref_xyz = np.asarray(getattr(ref_traj, "positions_xyz", None), dtype=np.float64)
    est_xyz = np.asarray(getattr(est_traj, "positions_xyz", None), dtype=np.float64)
    if ref_xyz.ndim != 2 or est_xyz.ndim != 2 or ref_xyz.shape[0] <= 0 or est_xyz.shape[0] <= 0:
        append_warning(metrics_obj, "trajectory plot unavailable: insufficient aligned positions")
        return None

    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "plots" / "traj_xy.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    ref_timestamps = np.asarray(getattr(ref_traj, "timestamps", []), dtype=np.float64).reshape(-1)
    _plot_traj_xy(
        plt,
        plot_path,
        ref_traj,
        est_traj,
        ape_result,
        "{} (reference)".format(metrics_obj.get("reference_name")),
        "{} (estimate)".format(metrics_obj.get("estimate_name")),
        metrics_obj,
        keyframe_sample_indices=_traj_keyframe_sample_indices(ref_timestamps, keyframe_context, ref_timestamps.shape[0]),
    )
    return plot_path


def run_traj_eval(config):
    metrics_obj = _base_metrics(config)
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(metrics_obj["reference_path"]).resolve()
    est_path = Path(metrics_obj["estimate_path"]).resolve()
    if not ref_path.is_file():
        append_warning(metrics_obj, "missing reference trajectory: {}".format(ref_path))
        return {"metrics": metrics_obj, "overview": {}, "records": []}
    if not est_path.is_file():
        append_warning(metrics_obj, "missing estimate trajectory: {}".format(est_path))
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    try:
        file_interface, evo_metrics, evo_sync, evo_units, evo_ape, evo_rpe = _load_evo_modules()
    except Exception as exc:
        append_warning(metrics_obj, "evo unavailable for trajectory eval: {}".format(exc))
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    try:
        log_info("eval load trajectories")
        traj_ref = file_interface.read_tum_trajectory_file(str(ref_path))
        traj_est = file_interface.read_tum_trajectory_file(str(est_path))
    except Exception as exc:
        append_warning(metrics_obj, "failed to load trajectories: {}".format(exc))
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    metrics_obj["reference_pose_count"] = int(getattr(traj_ref, "num_poses", 0))
    metrics_obj["estimate_pose_count"] = int(getattr(traj_est, "num_poses", 0))
    if metrics_obj["reference_pose_count"] <= 0 or metrics_obj["estimate_pose_count"] <= 0:
        append_warning(metrics_obj, "trajectory pose count is zero after loading")
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    try:
        log_info("eval synchronize trajectories")
        sync_ref, sync_est = evo_sync.associate_trajectories(
            traj_ref,
            traj_est,
            float(metrics_obj["sync_t_max_diff_sec"]),
            float(metrics_obj["sync_t_offset_sec"]),
            first_name=str(metrics_obj["reference_name"]),
            snd_name=str(metrics_obj["estimate_name"]),
        )
    except Exception as exc:
        append_warning(metrics_obj, "trajectory synchronization failed: {}".format(exc))
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    metrics_obj["matched_pose_count"] = int(getattr(sync_ref, "num_poses", 0))
    if metrics_obj["matched_pose_count"] <= 0:
        append_warning(metrics_obj, "no matched poses after synchronization")
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    align, correct_scale, align_origin = _alignment_flags(metrics_obj["alignment_mode"])
    try:
        ape_result = evo_ape(
            copy.deepcopy(sync_ref),
            copy.deepcopy(sync_est),
            evo_metrics.PoseRelation.translation_part,
            align=align,
            correct_scale=correct_scale,
            align_origin=align_origin,
            ref_name=str(metrics_obj["reference_name"]),
            est_name=str(metrics_obj["estimate_name"]),
        )
        metrics_obj["ape_trans"] = _stats_dict(ape_result.stats)
    except Exception as exc:
        append_warning(metrics_obj, "failed to compute APE translation: {}".format(exc))
        return {"metrics": metrics_obj, "overview": {}, "records": []}

    trajectories = getattr(ape_result, "trajectories", {}) or {}
    aligned_ref = trajectories.get(str(metrics_obj["reference_name"])) or sync_ref
    aligned_est = trajectories.get(str(metrics_obj["estimate_name"])) or sync_est
    metrics_obj["ori_path_length_m"] = _path_length(getattr(aligned_ref, "positions_xyz", None))
    metrics_obj["gen_path_length_m"] = _path_length(getattr(aligned_est, "positions_xyz", None))

    rpe_trans_result = None
    rpe_rot_result = None
    if metrics_obj["matched_pose_count"] >= 2:
        try:
            delta_unit = getattr(evo_units.Unit, str(metrics_obj["rpe_delta_unit"]))
            rpe_trans_result = evo_rpe(
                copy.deepcopy(sync_ref),
                copy.deepcopy(sync_est),
                evo_metrics.PoseRelation.translation_part,
                float(metrics_obj["rpe_delta"]),
                delta_unit,
                align=align,
                correct_scale=correct_scale,
                align_origin=align_origin,
                ref_name=str(metrics_obj["reference_name"]),
                est_name=str(metrics_obj["estimate_name"]),
            )
            metrics_obj["rpe_trans"] = _stats_dict(rpe_trans_result.stats)
        except Exception as exc:
            append_warning(metrics_obj, "failed to compute RPE translation: {}".format(exc))
        try:
            delta_unit = getattr(evo_units.Unit, str(metrics_obj["rpe_delta_unit"]))
            rpe_rot_result = evo_rpe(
                copy.deepcopy(sync_ref),
                copy.deepcopy(sync_est),
                evo_metrics.PoseRelation.rotation_angle_deg,
                float(metrics_obj["rpe_delta"]),
                delta_unit,
                align=align,
                correct_scale=correct_scale,
                align_origin=align_origin,
                ref_name=str(metrics_obj["reference_name"]),
                est_name=str(metrics_obj["estimate_name"]),
            )
            metrics_obj["rpe_rot"] = _stats_dict(rpe_rot_result.stats)
        except Exception as exc:
            append_warning(metrics_obj, "failed to compute RPE rotation: {}".format(exc))
    else:
        append_warning(metrics_obj, "matched pose count < 2; skip RPE")

    metrics_obj["eval_status"] = "success" if metrics_obj["ape_trans"].get("rmse") is not None else "partial"
    traj_plot = None
    if not bool(_get_arg(config, "skip_plots", False)):
        keyframe_context = _load_formal_keyframe_context(config, metrics_obj)
        traj_plot = _save_traj_plot(out_dir, aligned_ref, aligned_est, ape_result, metrics_obj, keyframe_context)

    overview = {
        "ape_curve": _curve_payload(ape_result),
        "rpe_trans_curve": _curve_payload(rpe_trans_result),
        "rpe_rot_curve": _curve_payload(rpe_rot_result),
    }
    return {
        "metrics": metrics_obj,
        "overview": overview,
        "records": _traj_detail_records(aligned_ref, aligned_est, ape_result),
        "artifacts": {
            "traj_xy_plot": str(traj_plot) if traj_plot is not None else "",
        },
    }
