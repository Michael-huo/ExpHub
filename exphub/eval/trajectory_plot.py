from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import numpy as np


def _setup_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="exphub_mpl_")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _load_tum_trajectory(path_obj):
    from evo.tools import file_interface

    path = str(Path(path_obj).resolve())
    if hasattr(file_interface, "read_tum_trajectory_file"):
        return file_interface.read_tum_trajectory_file(path)
    if hasattr(file_interface, "load_tum_trajectory_file"):
        return file_interface.load_tum_trajectory_file(path)
    raise RuntimeError("No compatible evo TUM trajectory loader found")


def _positions(traj):
    value = getattr(traj, "positions_xyz", None)
    arr = np.asarray(value, dtype=np.float64) if value is not None else None
    if arr is None or arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    return arr[:, :3]


def _timestamps(traj):
    value = getattr(traj, "timestamps", None)
    arr = np.asarray(value, dtype=np.float64) if value is not None else None
    if arr is None or arr.ndim != 1 or arr.shape[0] == 0:
        return None
    return arr


def _num_poses(traj):
    value = getattr(traj, "num_poses", None)
    if value is not None:
        return int(value)
    positions = _positions(traj)
    if positions is None:
        return 0
    return int(positions.shape[0])


def _select_plane(ref_positions):
    ranges = np.ptp(np.asarray(ref_positions, dtype=np.float64), axis=0)
    scores = {
        "xy": float(ranges[0] * ranges[1]),
        "xz": float(ranges[0] * ranges[2]),
        "yz": float(ranges[1] * ranges[2]),
    }
    if max(scores.values()) > 1e-12:
        return max(scores.items(), key=lambda item: item[1])[0]

    order = list(np.argsort(ranges))
    axes = set(order[-2:])
    if axes == {0, 1}:
        return "xy"
    if axes == {0, 2}:
        return "xz"
    return "yz"


def _plane_columns(plane):
    if plane == "xy":
        return 0, 1, "x (m)", "y (m)"
    if plane == "xz":
        return 0, 2, "x (m)", "z (m)"
    if plane == "yz":
        return 1, 2, "y (m)", "z (m)"
    raise RuntimeError("unsupported plot plane: {}".format(plane))


def _fixed_equal_limits(xy_arrays, axes_aspect, padding_ratio=0.06, eps=1e-9):
    axes_aspect = float(axes_aspect)
    if not np.isfinite(axes_aspect) or axes_aspect <= eps:
        raise RuntimeError("invalid plot axes aspect: {}".format(axes_aspect))

    finite_arrays = []
    for xy in xy_arrays:
        arr = np.asarray(xy, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        if np.any(mask):
            finite_arrays.append(arr[mask])
    if not finite_arrays:
        raise RuntimeError("trajectory plot has no finite 2D coordinates")

    points = np.vstack(finite_arrays)
    xmin = float(np.min(points[:, 0]))
    xmax = float(np.max(points[:, 0]))
    ymin = float(np.min(points[:, 1]))
    ymax = float(np.max(points[:, 1]))
    x_center = 0.5 * (xmin + xmax)
    y_center = 0.5 * (ymin + ymax)
    x_span = max(float(xmax - xmin), 0.0)
    y_span = max(float(ymax - ymin), 0.0)

    if x_span <= eps and y_span <= eps:
        y_span = 1.0
        x_span = y_span * axes_aspect
    elif x_span <= eps:
        x_span = max(y_span * axes_aspect, eps)
    elif y_span <= eps:
        y_span = max(x_span / axes_aspect, eps)

    pad_factor = 1.0 + 2.0 * float(padding_ratio)
    x_range = x_span * pad_factor
    y_range = y_span * pad_factor
    current_aspect = x_range / y_range
    if current_aspect < axes_aspect:
        x_range = y_range * axes_aspect
    elif current_aspect > axes_aspect:
        y_range = x_range / axes_aspect

    return (
        (x_center - 0.5 * x_range, x_center + 0.5 * x_range),
        (y_center - 0.5 * y_range, y_center + 0.5 * y_range),
    )


def _mask_for_time_range(timestamps, start, end):
    values = np.asarray(timestamps, dtype=np.float64)
    return (values >= float(start)) & (values <= float(end))


def _clipped_positions(traj, start, end, label):
    positions = _positions(traj)
    timestamps = _timestamps(traj)
    if positions is None or timestamps is None:
        raise RuntimeError("{} trajectory positions/timestamps are unavailable for plotting".format(label))
    mask = _mask_for_time_range(timestamps, start, end)
    if not np.any(mask):
        raise RuntimeError("{} trajectory has no samples in the common plot window".format(label))
    return positions[mask], mask


def _warn_pair_mismatch(label, plot_pairs, evo_pairs, warnings):
    if evo_pairs is None:
        return
    if int(plot_pairs) != int(evo_pairs):
        warnings.append(
            "trajectory plot {label} association pairs differ from evo_ape pose pairs: plot={plot}, evo={evo}".format(
                label=str(label).upper(),
                plot=int(plot_pairs),
                evo=int(evo_pairs),
            )
        )


def _plot_endpoint_markers(ax, xy, color, zorder):
    if xy is None or len(xy) == 0:
        return
    ax.plot(
        xy[0, 0],
        xy[0, 1],
        marker="o",
        linestyle="None",
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=5.5,
        label="_nolegend_",
        zorder=zorder,
    )
    ax.plot(
        xy[-1, 0],
        xy[-1, 1],
        marker="s",
        linestyle="None",
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=5.5,
        label="_nolegend_",
        zorder=zorder,
    )


def _save_trajectory_overlay(
    plt,
    output_path,
    gt_xy,
    ori_xy,
    rec_xy,
    xlabel,
    ylabel,
    xlim,
    ylim,
    show_legend,
    fig_size,
    fig_dpi,
    save_dpi,
    margins,
):
    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    try:
        fig.subplots_adjust(**margins)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(
            gt_xy[:, 0],
            gt_xy[:, 1],
            label="GT",
            color="#222222",
            linestyle="-",
            linewidth=2.4,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        ax.plot(
            ori_xy[:, 0],
            ori_xy[:, 1],
            label="ORI",
            color="#4C78A8",
            linestyle="--",
            linewidth=1.9,
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
            zorder=2,
        )
        ax.plot(
            rec_xy[:, 0],
            rec_xy[:, 1],
            label="REC",
            color="#C06C5B",
            linestyle="-",
            linewidth=2.05,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        _plot_endpoint_markers(ax, gt_xy, "#222222", zorder=4)
        _plot_endpoint_markers(ax, ori_xy, "#4C78A8", zorder=3)
        _plot_endpoint_markers(ax, rec_xy, "#C06C5B", zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="both", labelsize=9, width=0.7, color="#444444")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.16)
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.7)
        if show_legend:
            legend = ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.97, 0.95),
                frameon=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#D0D0D0",
                fontsize=10,
                handlelength=2.4,
                borderpad=0.4,
                labelspacing=0.35,
            )
            legend.set_zorder(10)
        fig.savefig(str(output_path), dpi=save_dpi, facecolor="white")
    finally:
        plt.close(fig)


def generate_trajectory_overlay(
    out_dir,
    exp_dir,
    gt_path,
    ori_path,
    rec_path,
    t_max_diff,
    ori_pose_pairs=None,
    rec_pose_pairs=None,
    plot_plane="auto",
):
    from evo.core import sync

    out_dir = Path(out_dir).resolve()
    exp_dir = Path(exp_dir).resolve()
    plot_path = out_dir / "trajectory_overlay_auto2d.png"
    paper_path = out_dir / "trajectory_overlay_paper.png"
    paper_pdf_path = out_dir / "trajectory_overlay_paper.pdf"
    warnings = []

    try:
        gt = _load_tum_trajectory(gt_path)
        ori = _load_tum_trajectory(ori_path)
        rec = _load_tum_trajectory(rec_path)

        gt_ori, ori_assoc = sync.associate_trajectories(gt, ori, max_diff=float(t_max_diff))
        gt_rec, rec_assoc = sync.associate_trajectories(gt, rec, max_diff=float(t_max_diff))
        gt_ori_pairs = _num_poses(gt_ori)
        gt_rec_pairs = _num_poses(gt_rec)
        _warn_pair_mismatch("ori", gt_ori_pairs, ori_pose_pairs, warnings)
        _warn_pair_mismatch("rec", gt_rec_pairs, rec_pose_pairs, warnings)

        gt_ori_times = _timestamps(gt_ori)
        gt_rec_times = _timestamps(gt_rec)
        if gt_ori_times is None or gt_rec_times is None:
            raise RuntimeError("associated GT timestamps are unavailable for plotting")
        common_start = max(float(gt_ori_times[0]), float(gt_rec_times[0]))
        common_end = min(float(gt_ori_times[-1]), float(gt_rec_times[-1]))
        if common_end <= common_start:
            raise RuntimeError(
                "no valid common trajectory plot window: start={} end={}".format(common_start, common_end)
            )

        ori_aligned = copy.deepcopy(ori_assoc)
        ori_aligned.align(gt_ori, correct_scale=True)
        rec_aligned = copy.deepcopy(rec_assoc)
        rec_aligned.align(gt_rec, correct_scale=True)

        gt_positions, _gt_mask = _clipped_positions(gt, common_start, common_end, "GT")
        ori_mask = _mask_for_time_range(gt_ori_times, common_start, common_end)
        rec_mask = _mask_for_time_range(gt_rec_times, common_start, common_end)
        ori_positions_all = _positions(ori_aligned)
        rec_positions_all = _positions(rec_aligned)
        if ori_positions_all is None or rec_positions_all is None:
            raise RuntimeError("aligned ORI/REC trajectory positions are unavailable for plotting")
        if not np.any(ori_mask) or not np.any(rec_mask):
            raise RuntimeError("ORI/REC aligned trajectories have no samples in the common plot window")
        ori_positions = ori_positions_all[ori_mask]
        rec_positions = rec_positions_all[rec_mask]

        selected_plane = _select_plane(gt_positions) if plot_plane == "auto" else str(plot_plane)
        i, j, xlabel, ylabel = _plane_columns(selected_plane)
        gt_xy = gt_positions[:, [i, j]]
        ori_xy = ori_positions[:, [i, j]]
        rec_xy = rec_positions[:, [i, j]]
        fig_size = (4.8, 3.8)
        fig_dpi = 180
        save_dpi = 300
        margins = {"left": 0.16, "right": 0.98, "bottom": 0.17, "top": 0.97}
        axes_aspect = (fig_size[0] * (margins["right"] - margins["left"])) / (
            fig_size[1] * (margins["top"] - margins["bottom"])
        )
        xlim, ylim = _fixed_equal_limits((gt_xy, ori_xy, rec_xy), axes_aspect=axes_aspect, padding_ratio=0.05)
        plt = _setup_matplotlib()
        for path, show_legend in ((plot_path, True), (paper_path, False), (paper_pdf_path, False)):
            _save_trajectory_overlay(
                plt=plt,
                output_path=path,
                gt_xy=gt_xy,
                ori_xy=ori_xy,
                rec_xy=rec_xy,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                show_legend=show_legend,
                fig_size=fig_size,
                fig_dpi=fig_dpi,
                save_dpi=save_dpi,
                margins=margins,
            )

        return {
            "plot_status": "success",
            "trajectory_overlay_path": _relative_path(exp_dir, plot_path),
            "trajectory_overlay_paper_path": _relative_path(exp_dir, paper_path),
            "trajectory_overlay_paper_pdf_path": _relative_path(exp_dir, paper_pdf_path),
            "selected_plot_plane": selected_plane,
            "gt_plot_mode": "common_overlap_segment",
            "plot_common_start": common_start,
            "plot_common_end": common_end,
            "warnings": warnings,
        }
    except Exception as exc:
        reason = "trajectory overlay skipped: {}".format(exc)
        warnings.append(reason)
        return {
            "plot_status": "skipped",
            "trajectory_overlay_path": None,
            "trajectory_overlay_paper_path": None,
            "trajectory_overlay_paper_pdf_path": None,
            "selected_plot_plane": None,
            "gt_plot_mode": None,
            "plot_common_start": None,
            "plot_common_end": None,
            "warnings": warnings,
        }
