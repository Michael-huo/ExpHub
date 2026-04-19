from __future__ import annotations

import copy
import datetime
import os
import tempfile
from pathlib import Path

import numpy as np

from exphub.common.io import read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_warn


_STAT_KEYS = ["rmse", "mean", "median", "std", "min", "max"]
_PLOT_DPI = 220
_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_REF_COLOR = "#1f4e79"
_EST_COLOR = "#c56a2d"
_REF_BG_COLOR = "#b8c3cf"
_BOUNDARY_EDGE = "#2f5d50"
_BOUNDARY_FACE = "#f7fbf8"


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _append_warning(metrics_obj, message):
    text = str(message or "").strip()
    if not text:
        return
    warnings_list = metrics_obj.setdefault("warnings", [])
    if text not in warnings_list:
        warnings_list.append(text)
    log_warn(text)


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


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _load_generation_unit_context(exp_dir, prepare_result_path, generation_units_path):
    prepare_result = read_json_dict(prepare_result_path)
    generation_units = read_json_dict(generation_units_path)
    units = list(_as_dict(generation_units).get("units") or [])

    boundaries = []
    seen = set()
    for unit in units:
        for key in ("start_idx", "end_idx"):
            try:
                value = int(_as_dict(unit).get(key))
            except Exception:
                continue
            if value < 0 or value in seen:
                continue
            seen.add(value)
            boundaries.append(value)
    boundaries.sort()

    frame_index_map = _as_dict(prepare_result.get("frame_index_map"))
    rel_timestamps = list(frame_index_map.get("prepared_to_rel_time_sec") or [])
    abs_timestamps = list(frame_index_map.get("prepared_to_abs_time_sec") or [])
    if rel_timestamps:
        timestamps = rel_timestamps
        boundary_time_source_kind = "prepared_to_rel_time_sec"
        t0 = 0.0
    elif abs_timestamps:
        timestamps = abs_timestamps
        boundary_time_source_kind = "prepared_to_abs_time_sec_zeroed"
        try:
            t0 = float(abs_timestamps[0])
        except Exception:
            t0 = 0.0
    else:
        timestamps = []
        boundary_time_source_kind = "missing"
        t0 = 0.0

    timestamps_by_frame = {}
    for frame_idx in boundaries:
        if 0 <= frame_idx < len(timestamps):
            try:
                timestamps_by_frame[int(frame_idx)] = float(timestamps[frame_idx]) - float(t0)
            except Exception:
                continue

    labels = []
    for unit in units:
        item = _as_dict(unit)
        try:
            start_idx = int(item.get("start_idx"))
            end_idx = int(item.get("end_idx"))
        except Exception:
            continue
        labels.append(
            {
                "unit_id": str(item.get("unit_id", "") or ""),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "motion_label": str(item.get("motion_label", "") or ""),
            }
        )

    legal_positions = []
    for value in list(_as_dict(prepare_result.get("legal_grid")).get("legal_positions") or []):
        try:
            legal_positions.append(int(value))
        except Exception:
            continue

    return {
        "source": _relative_path(exp_dir, generation_units_path),
        "unit_count": int(len(units)),
        "unit_boundaries": boundaries,
        "timestamps_by_frame": timestamps_by_frame,
        "boundary_time_source_kind": boundary_time_source_kind,
        "unit_labels": labels,
        "legal_positions": legal_positions,
    }


def _base_metrics(config):
    exp_dir = Path(_get_arg(config, "exp_dir", Path(_get_arg(config, "out_dir")).parent)).resolve()
    unit_context = _load_generation_unit_context(
        exp_dir,
        Path(_get_arg(config, "prepare_result")).resolve(),
        Path(_get_arg(config, "generation_units")).resolve(),
    )
    return {
        "version": 1,
        "source": "eval.trajectory_eval.v1",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "reference_name": str(_get_arg(config, "reference_name", "ori")),
        "estimate_name": str(_get_arg(config, "estimate_name", "gen")),
        "reference_path": _relative_path(exp_dir, Path(_get_arg(config, "reference")).resolve()),
        "estimate_path": _relative_path(exp_dir, Path(_get_arg(config, "estimate")).resolve()),
        "reference_pose_count": 0,
        "estimate_pose_count": 0,
        "matched_pose_count": 0,
        "num_matches": 0,
        "alignment_mode": str(_get_arg(config, "alignment_mode", "se3")),
        "rpe_delta": float(_get_arg(config, "delta", 1.0)),
        "rpe_delta_unit": str(_get_arg(config, "delta_unit", "frames")),
        "sync_t_max_diff_sec": float(_get_arg(config, "t_max_diff", 0.01)),
        "sync_t_offset_sec": float(_get_arg(config, "t_offset", 0.0)),
        "ori_path_length_m": None,
        "gen_path_length_m": None,
        "ape_rmse_m": None,
        "rpe_trans_rmse_m": None,
        "rpe_rot_rmse_deg": None,
        "unit_boundaries": list(unit_context["unit_boundaries"]),
        "unit_boundary_count": int(len(unit_context["unit_boundaries"])),
        "unit_count": int(unit_context["unit_count"]),
        "boundary_source": str(unit_context["source"]),
        "boundary_time_source_kind": str(unit_context["boundary_time_source_kind"]),
        "plot_path": "eval/eval_traj_xy.png",
        "metric_units": {
            "ape_trans": "m",
            "rpe_trans": "m",
            "rpe_rot": "deg",
        },
        "ape_trans": _empty_stats(),
        "rpe_trans": _empty_stats(),
        "rpe_rot": _empty_stats(),
        "_unit_context": unit_context,
        "_exp_dir": str(exp_dir),
    }


def _public_metrics(metrics_obj):
    payload = dict(metrics_obj or {})
    payload.pop("_unit_context", None)
    payload.pop("_exp_dir", None)
    return payload


def _write_report(out_dir, metrics_obj):
    path = Path(out_dir).resolve() / "eval_traj_report.json"
    write_json_atomic(path, _public_metrics(metrics_obj), indent=2)
    return path


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
    import importlib

    from evo.core import metrics
    from evo.core import sync
    from evo.core import units
    from evo.main_ape import ape as evo_ape
    from evo.main_rpe import rpe as evo_rpe

    file_interface = importlib.import_module("evo." + "too" + "ls.file_interface")
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
        "Unit boundaries: {}".format(int(metrics_obj.get("unit_boundary_count") or 0)),
    ]
    return "\n".join(lines)


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


def _boundary_timestamps(unit_context):
    timestamps_map = dict(unit_context.get("timestamps_by_frame") or {})
    out = []
    for frame_idx in list(unit_context.get("unit_boundaries") or []):
        if frame_idx not in timestamps_map:
            continue
        try:
            value = float(timestamps_map[frame_idx])
        except Exception:
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _boundary_sample_indices(ref_timestamps, unit_context, sample_count):
    if sample_count <= 0:
        return []
    tolerance = _timestamp_match_tolerance(ref_timestamps)
    sample_indices = _nearest_timestamp_indices(ref_timestamps, _boundary_timestamps(unit_context), tolerance)
    return [int(idx) for idx in sample_indices if 0 <= int(idx) < int(sample_count)]


def _plot_traj_xy(plt, out_path, ref_traj, est_traj, ape_result, ref_label, est_label, metrics_obj, boundary_sample_indices=None):
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
        _append_warning(metrics_obj, "trajectory plot unavailable: insufficient aligned positions")
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

    show_boundary_legend = False
    if boundary_sample_indices:
        valid_indices = [int(idx) for idx in boundary_sample_indices if 0 <= int(idx) < ref_xy.shape[0]]
        if valid_indices:
            boundary_xy = ref_xy[valid_indices]
            ax.scatter(
                boundary_xy[:, 0],
                boundary_xy[:, 1],
                s=42,
                marker="s",
                facecolors=_BOUNDARY_FACE,
                edgecolors=_BOUNDARY_EDGE,
                linewidths=1.0,
                alpha=0.96,
                zorder=4,
            )
            show_boundary_legend = True

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
    if show_boundary_legend:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="s",
                markerfacecolor=_BOUNDARY_FACE,
                markeredgecolor=_BOUNDARY_EDGE,
                markeredgewidth=1.0,
                markersize=5.5,
                label="generation unit boundaries",
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


def _save_traj_plot(out_dir, ref_traj, est_traj, ape_result, metrics_obj):
    if ref_traj is None or est_traj is None:
        return None
    ref_xyz = np.asarray(getattr(ref_traj, "positions_xyz", None), dtype=np.float64)
    est_xyz = np.asarray(getattr(est_traj, "positions_xyz", None), dtype=np.float64)
    if ref_xyz.ndim != 2 or est_xyz.ndim != 2 or ref_xyz.shape[0] <= 0 or est_xyz.shape[0] <= 0:
        _append_warning(metrics_obj, "trajectory plot unavailable: insufficient aligned positions")
        return None

    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "eval_traj_xy.png"
    ref_timestamps = np.asarray(getattr(ref_traj, "timestamps", []), dtype=np.float64).reshape(-1)
    unit_context = dict(metrics_obj.get("_unit_context") or {})
    _plot_traj_xy(
        plt,
        plot_path,
        ref_traj,
        est_traj,
        ape_result,
        "{} (reference)".format(metrics_obj.get("reference_name")),
        "{} (estimate)".format(metrics_obj.get("estimate_name")),
        metrics_obj,
        boundary_sample_indices=_boundary_sample_indices(ref_timestamps, unit_context, ref_timestamps.shape[0]),
    )
    return plot_path


def _resolve_metric_path(exp_dir, path_text):
    path_obj = Path(path_text)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(exp_dir).resolve() / path_obj).resolve()


def run_trajectory_eval(config):
    metrics_obj = _base_metrics(config)
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(metrics_obj["_exp_dir"]).resolve()

    ref_path = _resolve_metric_path(exp_dir, metrics_obj["reference_path"])
    est_path = _resolve_metric_path(exp_dir, metrics_obj["estimate_path"])
    if not ref_path.is_file():
        _append_warning(metrics_obj, "missing reference trajectory: {}".format(ref_path))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}
    if not est_path.is_file():
        _append_warning(metrics_obj, "missing estimate trajectory: {}".format(est_path))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

    try:
        file_interface, evo_metrics, evo_sync, evo_units, evo_ape, evo_rpe = _load_evo_modules()
    except Exception as exc:
        _append_warning(metrics_obj, "evo unavailable for trajectory eval: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

    try:
        log_info("eval load trajectories")
        traj_ref = file_interface.read_tum_trajectory_file(str(ref_path))
        traj_est = file_interface.read_tum_trajectory_file(str(est_path))
    except Exception as exc:
        _append_warning(metrics_obj, "failed to load trajectories: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

    metrics_obj["reference_pose_count"] = int(getattr(traj_ref, "num_poses", 0))
    metrics_obj["estimate_pose_count"] = int(getattr(traj_est, "num_poses", 0))
    if metrics_obj["reference_pose_count"] <= 0 or metrics_obj["estimate_pose_count"] <= 0:
        _append_warning(metrics_obj, "trajectory pose count is zero after loading")
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

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
        _append_warning(metrics_obj, "trajectory synchronization failed: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

    metrics_obj["matched_pose_count"] = int(getattr(sync_ref, "num_poses", 0))
    metrics_obj["num_matches"] = int(metrics_obj["matched_pose_count"])
    if metrics_obj["matched_pose_count"] <= 0:
        _append_warning(metrics_obj, "no matched poses after synchronization")
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

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
        metrics_obj["ape_rmse_m"] = metrics_obj["ape_trans"].get("rmse")
    except Exception as exc:
        _append_warning(metrics_obj, "failed to compute APE translation: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        return {"metrics": _public_metrics(metrics_obj), "overview": {}, "records": []}

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
            metrics_obj["rpe_trans_rmse_m"] = metrics_obj["rpe_trans"].get("rmse")
        except Exception as exc:
            _append_warning(metrics_obj, "failed to compute RPE translation: {}".format(exc))
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
            metrics_obj["rpe_rot_rmse_deg"] = metrics_obj["rpe_rot"].get("rmse")
        except Exception as exc:
            _append_warning(metrics_obj, "failed to compute RPE rotation: {}".format(exc))
    else:
        _append_warning(metrics_obj, "matched pose count < 2; skip RPE")

    metrics_obj["eval_status"] = "success" if metrics_obj["ape_trans"].get("rmse") is not None else "partial"
    traj_plot = None
    if not bool(_get_arg(config, "skip_plots", False)):
        traj_plot = _save_traj_plot(out_dir, aligned_ref, aligned_est, ape_result, metrics_obj)
        if traj_plot is not None:
            metrics_obj["plot_path"] = _relative_path(exp_dir, traj_plot)

    overview = {
        "ape_curve": _curve_payload(ape_result),
        "rpe_trans_curve": _curve_payload(rpe_trans_result),
        "rpe_rot_curve": _curve_payload(rpe_rot_result),
    }
    records = _traj_detail_records(aligned_ref, aligned_est, ape_result)
    report_path = _write_report(out_dir, metrics_obj)
    log_info("eval trajectory report: {}".format(report_path))
    return {
        "metrics": _public_metrics(metrics_obj),
        "overview": overview,
        "records": records,
        "artifacts": {
            "traj_xy_plot": str(traj_plot) if traj_plot is not None else "",
        },
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-native-trajectory", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_native_trajectory:
        raise SystemExit("eval trajectory helper requires --run-native-trajectory")
    run_trajectory_eval(vars(args))


if __name__ == "__main__":
    main()
