#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import datetime
import os
import tempfile
from pathlib import Path

import numpy as np

from _common import log_err, log_info, log_warn
from _eval.io import append_warning, fmt_value, write_json


_STAT_KEYS = ["rmse", "mean", "median", "std", "min", "max"]
_PLOT_DPI = 220
_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_REF_COLOR = "#1f4e79"
_EST_COLOR = "#c56a2d"
_REF_BG_COLOR = "#b8c3cf"
_FILL_ALPHA = 0.06


def add_traj_eval_args(parser):
    parser.add_argument("--reference", required=True)
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reference_name", default="ori")
    parser.add_argument("--estimate_name", default="gen")
    parser.add_argument(
        "--alignment_mode",
        default="se3",
        choices=["none", "se3", "sim3", "origin"],
    )
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument(
        "--delta_unit",
        default="frames",
        choices=["frames", "meters", "seconds"],
    )
    parser.add_argument("--t_max_diff", type=float, default=0.01)
    parser.add_argument("--t_offset", type=float, default=0.0)
    parser.add_argument("--skip_plots", action="store_true")


def _empty_stats():
    return {key: None for key in _STAT_KEYS}


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _stats_dict(stats):
    out = _empty_stats()
    for key in _STAT_KEYS:
        out[key] = _float_or_none(stats.get(key))
    return out


def _curve_payload(result):
    if result is None:
        return None

    errors = np.asarray(result.np_arrays.get("error_array", []), dtype=np.float64)
    if errors.size == 0:
        return None

    xs = None
    xlabel = "Pose Index"
    seconds = result.np_arrays.get("seconds_from_start")
    if seconds is not None:
        seconds_arr = np.asarray(seconds, dtype=np.float64)
        if seconds_arr.shape[0] == errors.shape[0]:
            xs = seconds_arr
            xlabel = "Time From Start (s)"

    if xs is None:
        xs = np.arange(errors.shape[0], dtype=np.float64)

    return {
        "x": xs,
        "y": errors,
        "xlabel": xlabel,
    }


def _base_metrics(args):
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "reference_name": str(args.reference_name),
        "estimate_name": str(args.estimate_name),
        "reference_path": str(Path(args.reference).resolve()),
        "estimate_path": str(Path(args.estimate).resolve()),
        "reference_pose_count": 0,
        "estimate_pose_count": 0,
        "matched_pose_count": 0,
        "alignment_mode": str(args.alignment_mode),
        "rpe_delta": float(args.delta),
        "rpe_delta_unit": str(args.delta_unit),
        "sync_t_max_diff_sec": float(args.t_max_diff),
        "sync_t_offset_sec": float(args.t_offset),
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


def _evaluate(args, metrics_obj):
    file_interface, evo_metrics, evo_sync, evo_units, evo_ape, evo_rpe = _load_evo_modules()

    ref_path = Path(args.reference).resolve()
    est_path = Path(args.estimate).resolve()
    missing_inputs = False
    if not ref_path.is_file():
        append_warning(metrics_obj, "missing reference trajectory: {}".format(ref_path))
        missing_inputs = True
    if not est_path.is_file():
        append_warning(metrics_obj, "missing estimate trajectory: {}".format(est_path))
        missing_inputs = True
    if missing_inputs:
        return None

    log_info("eval load trajectories")
    traj_ref = file_interface.read_tum_trajectory_file(str(ref_path))
    traj_est = file_interface.read_tum_trajectory_file(str(est_path))
    metrics_obj["reference_pose_count"] = int(getattr(traj_ref, "num_poses", 0))
    metrics_obj["estimate_pose_count"] = int(getattr(traj_est, "num_poses", 0))

    if metrics_obj["reference_pose_count"] <= 0:
        append_warning(metrics_obj, "reference trajectory has no poses: {}".format(ref_path))
        return None
    if metrics_obj["estimate_pose_count"] <= 0:
        append_warning(metrics_obj, "estimate trajectory has no poses: {}".format(est_path))
        return None

    log_info("eval synchronize trajectories")
    sync_ref, sync_est = evo_sync.associate_trajectories(
        traj_ref,
        traj_est,
        float(args.t_max_diff),
        float(args.t_offset),
        first_name=str(args.reference_name),
        snd_name=str(args.estimate_name),
    )
    metrics_obj["matched_pose_count"] = int(getattr(sync_ref, "num_poses", 0))
    if metrics_obj["matched_pose_count"] <= 0:
        append_warning(metrics_obj, "no matched poses after synchronization")
        return None

    align, correct_scale, align_origin = _alignment_flags(args.alignment_mode)

    log_info("eval compute APE translation")
    ape_result = evo_ape(
        copy.deepcopy(sync_ref),
        copy.deepcopy(sync_est),
        evo_metrics.PoseRelation.translation_part,
        align=align,
        correct_scale=correct_scale,
        align_origin=align_origin,
        ref_name=str(args.reference_name),
        est_name=str(args.estimate_name),
    )
    metrics_obj["ape_trans"] = _stats_dict(ape_result.stats)

    rpe_trans_result = None
    rpe_rot_result = None
    if metrics_obj["matched_pose_count"] < 2:
        append_warning(metrics_obj, "matched pose count < 2; skip RPE")
    else:
        delta_unit = getattr(evo_units.Unit, str(args.delta_unit))

        log_info("eval compute RPE translation")
        rpe_trans_result = evo_rpe(
            copy.deepcopy(sync_ref),
            copy.deepcopy(sync_est),
            evo_metrics.PoseRelation.translation_part,
            float(args.delta),
            delta_unit,
            align=align,
            correct_scale=correct_scale,
            align_origin=align_origin,
            ref_name=str(args.reference_name),
            est_name=str(args.estimate_name),
        )
        metrics_obj["rpe_trans"] = _stats_dict(rpe_trans_result.stats)

        try:
            log_info("eval compute RPE rotation")
            rpe_rot_result = evo_rpe(
                copy.deepcopy(sync_ref),
                copy.deepcopy(sync_est),
                evo_metrics.PoseRelation.rotation_angle_deg,
                float(args.delta),
                delta_unit,
                align=align,
                correct_scale=correct_scale,
                align_origin=align_origin,
                ref_name=str(args.reference_name),
                est_name=str(args.estimate_name),
            )
            metrics_obj["rpe_rot"] = _stats_dict(rpe_rot_result.stats)
        except Exception as exc:
            append_warning(metrics_obj, "failed to compute RPE rotation: {}".format(exc))

    return {
        "ape": ape_result,
        "rpe_trans": rpe_trans_result,
        "rpe_rot": rpe_rot_result,
    }


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
        _stat_line("RPE RMSE", metrics_obj["rpe_trans"].get("rmse"), "m"),
        "Matched: {}".format(int(metrics_obj.get("matched_pose_count") or 0)),
    ]
    return "\n".join(lines)


def _curve_stats_text(stats_obj, unit):
    return "\n".join(
        [
            _stat_line("RMSE", stats_obj.get("rmse"), unit),
            _stat_line("Mean", stats_obj.get("mean"), unit),
            _stat_line("Max", stats_obj.get("max"), unit),
        ]
    )


def _normalized_points(points_xy):
    arr = np.asarray(points_xy, dtype=np.float64)
    if arr.size == 0:
        return arr
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
        [
            "#334d69",
            "#5f778b",
            "#8e9ea6",
            "#b4a48f",
            "#a57d5e",
            "#845640",
        ],
    )


def _plot_traj_xy(plt, out_path, ref_traj, est_traj, ape_result, ref_label, est_label, metrics_obj):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    ref_xyz = np.asarray(ref_traj.positions_xyz, dtype=np.float64)
    est_xyz = np.asarray(est_traj.positions_xyz, dtype=np.float64)
    errors = np.asarray(ape_result.np_arrays.get("error_array", []), dtype=np.float64)
    all_xy = np.vstack([ref_xyz[:, :2], est_xyz[:, :2]])
    corners = _corner_candidates(all_xy)
    info_corner = corners[0]
    legend_corner = corners[1] if len(corners) > 1 else "upper right"

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=_PLOT_DPI)
    _style_figure(fig)
    ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], color=_REF_BG_COLOR, linewidth=1.8, alpha=0.96, zorder=1)

    colorbar = None
    est_line_drawn = False
    if est_xyz.shape[0] >= 2 and errors.size > 0:
        size = min(est_xyz.shape[0], errors.shape[0])
        est_xy = est_xyz[:size, :2]
        if est_xy.shape[0] >= 2:
            seg_errors = 0.5 * (errors[:-1] + errors[1:]) if errors.size >= 2 else errors
            if seg_errors.size == 0:
                seg_errors = np.asarray([errors[0]], dtype=np.float64)
            points = est_xy.reshape(-1, 1, 2)
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
                colorbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
                colorbar.set_label("APE (m)", fontsize=10.5, color=_TEXT_COLOR)
                colorbar.ax.tick_params(labelsize=9.4, colors=_TEXT_COLOR)
                colorbar.outline.set_edgecolor(_SPINE_COLOR)
                colorbar.outline.set_linewidth(0.8)
                est_line_drawn = True

    if not est_line_drawn:
        ax.plot(est_xyz[:, 0], est_xyz[:, 1], color=_EST_COLOR, linewidth=2.2, alpha=0.96, zorder=3)

    marker_size = 40
    ax.scatter(ref_xyz[0, 0], ref_xyz[0, 1], color=_REF_COLOR, s=marker_size, marker="o", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(ref_xyz[-1, 0], ref_xyz[-1, 1], color=_REF_COLOR, s=marker_size + 10, marker="D", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(est_xyz[0, 0], est_xyz[0, 1], color=_EST_COLOR, s=marker_size, marker="o", edgecolors="white", linewidths=0.9, zorder=5)
    ax.scatter(est_xyz[-1, 0], est_xyz[-1, 1], color=_EST_COLOR, s=marker_size + 10, marker="D", edgecolors="white", linewidths=0.9, zorder=5)

    ax.set_title("Trajectory XY", fontsize=13, color=_TEXT_COLOR, pad=10)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    _style_axes(ax)
    legend_handles = [
        Line2D([0], [0], color=_REF_BG_COLOR, linewidth=1.8, label=ref_label),
        Line2D([0], [0], color="#6e7c86" if colorbar is not None else _EST_COLOR, linewidth=2.4, label="{} (APE-colored)".format(est_label) if colorbar is not None else est_label),
    ]
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
    fig.tight_layout(pad=0.7)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _plot_ape_curve(plt, out_path, curve_data, metrics_obj):
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=_PLOT_DPI)
    _style_figure(fig)
    ax.plot(curve_data["x"], curve_data["y"], color=_REF_COLOR, linewidth=1.9, zorder=3)
    ax.fill_between(curve_data["x"], curve_data["y"], color=_REF_COLOR, alpha=_FILL_ALPHA, zorder=2)
    ax.set_title("APE", fontsize=13, color=_TEXT_COLOR, pad=8)
    ax.set_xlabel(str(curve_data["xlabel"]), fontsize=11)
    ax.set_ylabel("Translation Error (m)", fontsize=11)
    _style_axes(ax)
    _add_corner_box(ax, _curve_stats_text(metrics_obj["ape_trans"], "m"), "upper right", _REF_COLOR, fontsize=9.2)
    fig.tight_layout(pad=0.6)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _plot_rpe_curve(plt, out_path, trans_curve, rot_curve, metrics_obj):
    if rot_curve is None:
        fig, axes = plt.subplots(1, 1, figsize=(8.8, 4.5), dpi=_PLOT_DPI)
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.3), dpi=_PLOT_DPI, sharex=False)
    _style_figure(fig)
    fig.suptitle("RPE", fontsize=13, color=_TEXT_COLOR, y=0.99)

    ax0 = axes[0]
    ax0.plot(trans_curve["x"], trans_curve["y"], color=_EST_COLOR, linewidth=1.9, zorder=3)
    ax0.fill_between(trans_curve["x"], trans_curve["y"], color=_EST_COLOR, alpha=_FILL_ALPHA, zorder=2)
    ax0.set_title("Translation", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax0.set_xlabel(str(trans_curve["xlabel"]), fontsize=11)
    ax0.set_ylabel("Translation Error (m)", fontsize=11)
    _style_axes(ax0)
    _add_corner_box(ax0, _curve_stats_text(metrics_obj["rpe_trans"], "m"), "upper right", _EST_COLOR, fontsize=9.0)

    if rot_curve is not None:
        ax1 = axes[1]
        ax1.plot(rot_curve["x"], rot_curve["y"], color="#546d8c", linewidth=1.8, zorder=3)
        ax1.fill_between(rot_curve["x"], rot_curve["y"], color="#546d8c", alpha=_FILL_ALPHA, zorder=2)
        ax1.set_title("Rotation", fontsize=11.5, color=_TEXT_COLOR, pad=6)
        ax1.set_xlabel(str(rot_curve["xlabel"]), fontsize=11)
        ax1.set_ylabel("Rotation Error (deg)", fontsize=11)
        _style_axes(ax1)
        _add_corner_box(ax1, _curve_stats_text(metrics_obj["rpe_rot"], "deg"), "upper right", "#546d8c", fontsize=9.0)

    fig.tight_layout(pad=0.55, rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _generate_plots(args, plot_dir, eval_payload, metrics_obj):
    if args.skip_plots:
        log_info("eval plots skipped by flag")
        return

    ape_result = eval_payload.get("ape")
    rpe_trans_result = eval_payload.get("rpe_trans")
    if ape_result is None or rpe_trans_result is None:
        append_warning(metrics_obj, "skip plots because core metrics are unavailable")
        return

    plt = _setup_matplotlib()
    plot_dir.mkdir(parents=True, exist_ok=True)

    ref_traj = ape_result.trajectories.get(str(args.reference_name))
    est_traj = ape_result.trajectories.get(str(args.estimate_name))
    if ref_traj is None or est_traj is None:
        append_warning(metrics_obj, "aligned trajectories missing in APE result; skip traj plot")
    else:
        _plot_traj_xy(
            plt,
            plot_dir / "traj_xy.png",
            ref_traj,
            est_traj,
            ape_result,
            "{} (reference)".format(args.reference_name),
            "{} (estimate)".format(args.estimate_name),
            metrics_obj,
        )

    ape_curve = _curve_payload(ape_result)
    if ape_curve is None:
        append_warning(metrics_obj, "APE error curve unavailable")
    else:
        _plot_ape_curve(
            plt,
            plot_dir / "ape_curve.png",
            ape_curve,
            metrics_obj,
        )

    rpe_trans_curve = _curve_payload(rpe_trans_result)
    if rpe_trans_curve is None:
        append_warning(metrics_obj, "RPE translation curve unavailable")
    else:
        rpe_rot_curve = _curve_payload(eval_payload.get("rpe_rot"))
        _plot_rpe_curve(plt, plot_dir / "rpe_curve.png", rpe_trans_curve, rpe_rot_curve, metrics_obj)


def update_traj_eval_status(metrics_obj):
    ape_ok = metrics_obj["ape_trans"].get("rmse") is not None
    rpe_ok = metrics_obj["rpe_trans"].get("rmse") is not None
    warnings_list = list(metrics_obj.get("warnings", []) or [])
    if ape_ok and rpe_ok:
        metrics_obj["eval_status"] = "success" if not warnings_list else "partial"
    elif ape_ok or rpe_ok:
        metrics_obj["eval_status"] = "partial"
    else:
        metrics_obj["eval_status"] = "failed"


def write_traj_outputs(out_dir, metrics_obj):
    metrics_path = out_dir / "traj_metrics.json"
    write_json(metrics_path, metrics_obj, indent=2)


def log_traj_terminal_summary(metrics_obj, out_dir):
    status = str(metrics_obj.get("eval_status"))
    matched = int(metrics_obj.get("matched_pose_count") or 0)
    prefix = log_info if status in ("success", "partial") else log_warn

    prefix("eval summary: status={} matched_poses={} out={}".format(status, matched, out_dir))
    prefix("eval metric: APE RMSE={}".format(fmt_value(metrics_obj["ape_trans"].get("rmse"), "m")))
    prefix("eval metric: RPE trans RMSE={}".format(fmt_value(metrics_obj["rpe_trans"].get("rmse"), "m")))
    prefix("eval metric: RPE rot RMSE={}".format(fmt_value(metrics_obj["rpe_rot"].get("rmse"), "deg")))
    if metrics_obj.get("warnings"):
        prefix("eval warnings: {}".format(len(metrics_obj["warnings"])))


def run_traj_eval(args, emit_terminal_summary=True):
    out_dir = Path(args.out_dir).resolve()
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_obj = _base_metrics(args)

    log_info(
        "eval start: reference={} estimate={} out_dir={}".format(
            Path(args.reference).resolve(),
            Path(args.estimate).resolve(),
            out_dir,
        )
    )

    eval_payload = None
    try:
        eval_payload = _evaluate(args, metrics_obj)
        if eval_payload is not None:
            try:
                _generate_plots(args, plot_dir, eval_payload, metrics_obj)
            except Exception as exc:
                append_warning(metrics_obj, "failed to generate plots: {}".format(exc))
    except ImportError as exc:
        append_warning(metrics_obj, "missing eval dependency: {}".format(exc))
    except Exception as exc:
        log_err("unexpected eval failure: {}".format(exc))
        append_warning(metrics_obj, "unexpected eval failure: {}".format(exc))

    update_traj_eval_status(metrics_obj)
    write_traj_outputs(out_dir, metrics_obj)
    if emit_terminal_summary:
        log_traj_terminal_summary(metrics_obj, out_dir)

    return {
        "metrics": metrics_obj,
        "metrics_path": out_dir / "traj_metrics.json",
        "plot_dir": plot_dir,
    }
