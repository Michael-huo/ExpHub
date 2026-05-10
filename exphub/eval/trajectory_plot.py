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
                label=label.upper(),
                plot=int(plot_pairs),
                evo=int(evo_pairs),
            )
        )


def generate_trajectory_overlay(
    out_dir,
    exp_dir,
    gt_path,
    ori_path,
    gen_path,
    t_max_diff,
    ori_pose_pairs=None,
    gen_pose_pairs=None,
    plot_plane="auto",
):
    from evo.core import sync

    out_dir = Path(out_dir).resolve()
    exp_dir = Path(exp_dir).resolve()
    plot_path = out_dir / "trajectory_overlay_auto2d.png"
    warnings = []

    try:
        gt = _load_tum_trajectory(gt_path)
        ori = _load_tum_trajectory(ori_path)
        gen = _load_tum_trajectory(gen_path)

        gt_ori, ori_assoc = sync.associate_trajectories(gt, ori, max_diff=float(t_max_diff))
        gt_gen, gen_assoc = sync.associate_trajectories(gt, gen, max_diff=float(t_max_diff))
        gt_ori_pairs = _num_poses(gt_ori)
        gt_gen_pairs = _num_poses(gt_gen)
        _warn_pair_mismatch("ori", gt_ori_pairs, ori_pose_pairs, warnings)
        _warn_pair_mismatch("gen", gt_gen_pairs, gen_pose_pairs, warnings)

        gt_ori_times = _timestamps(gt_ori)
        gt_gen_times = _timestamps(gt_gen)
        if gt_ori_times is None or gt_gen_times is None:
            raise RuntimeError("associated GT timestamps are unavailable for plotting")
        common_start = max(float(gt_ori_times[0]), float(gt_gen_times[0]))
        common_end = min(float(gt_ori_times[-1]), float(gt_gen_times[-1]))
        if common_end <= common_start:
            raise RuntimeError(
                "no valid common trajectory plot window: start={} end={}".format(common_start, common_end)
            )

        ori_aligned = copy.deepcopy(ori_assoc)
        ori_aligned.align(gt_ori, correct_scale=True)
        gen_aligned = copy.deepcopy(gen_assoc)
        gen_aligned.align(gt_gen, correct_scale=True)

        gt_positions, _gt_mask = _clipped_positions(gt, common_start, common_end, "GT")
        ori_mask = _mask_for_time_range(gt_ori_times, common_start, common_end)
        gen_mask = _mask_for_time_range(gt_gen_times, common_start, common_end)
        ori_positions_all = _positions(ori_aligned)
        gen_positions_all = _positions(gen_aligned)
        if ori_positions_all is None or gen_positions_all is None:
            raise RuntimeError("aligned ORI/GEN trajectory positions are unavailable for plotting")
        if not np.any(ori_mask) or not np.any(gen_mask):
            raise RuntimeError("ORI/GEN aligned trajectories have no samples in the common plot window")
        ori_positions = ori_positions_all[ori_mask]
        gen_positions = gen_positions_all[gen_mask]

        selected_plane = _select_plane(gt_positions) if plot_plane == "auto" else str(plot_plane)
        i, j, xlabel, ylabel = _plane_columns(selected_plane)
        plt = _setup_matplotlib()
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(gt_positions[:, i], gt_positions[:, j], label="GT", color="#202020", linewidth=2.0)
        ax.plot(ori_positions[:, i], ori_positions[:, j], label="ORI", color="#c56a2d", linewidth=1.8)
        ax.plot(gen_positions[:, i], gen_positions[:, j], label="GEN", color="#2f7d5c", linewidth=1.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Evo-synchronized Sim3-aligned trajectory overlay\nplane={}".format(selected_plane))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        return {
            "plot_status": "success",
            "trajectory_overlay_path": _relative_path(exp_dir, plot_path),
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
            "selected_plot_plane": None,
            "gt_plot_mode": None,
            "plot_common_start": None,
            "plot_common_end": None,
            "warnings": warnings,
        }
