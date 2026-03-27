#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _eval.io import write_json


REPORT_FILENAME = "report.json"
DETAILS_FILENAME = "details.csv"
METRICS_OVERVIEW_FILENAME = "metrics_overview.png"
LEGACY_EVAL_OUTPUT_NAMES = [
    "traj_metrics.json",
    "image_metrics.json",
    "slam_metrics.json",
    "summary.txt",
    "image_per_frame.csv",
    "slam_pairs.csv",
    "plots/ape_curve.png",
    "plots/rpe_curve.png",
    "plots/image_metrics_curve.png",
    "plots/slam_metrics_curve.png",
]

_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_APE_COLOR = "#1f4e79"
_RPE_TRANS_COLOR = "#c56a2d"
_RPE_ROT_COLOR = "#546d8c"
_PSNR_COLOR = "#1f4e79"
_MSSSIM_COLOR = "#4d7f4b"
_LPIPS_COLOR = "#b35c2e"
_INLIER_COLOR = "#1f4e79"
_POSE_COLOR = "#4d7f4b"


def _collect_warnings(traj_metrics, image_metrics, slam_metrics):
    warnings_list = []
    for prefix, metrics_obj in [
        ("traj", traj_metrics),
        ("image", image_metrics),
        ("slam", slam_metrics),
    ]:
        for item in list((metrics_obj or {}).get("warnings", []) or []):
            text = "{}: {}".format(prefix, item)
            if text not in warnings_list:
                warnings_list.append(text)
    return warnings_list


def _overall_status(traj_metrics, image_metrics, slam_metrics):
    statuses = []
    for metrics_obj in [traj_metrics, image_metrics, slam_metrics]:
        status = str((metrics_obj or {}).get("eval_status", "") or "").strip()
        if status:
            statuses.append(status)
    if not statuses:
        return "failed"
    if all(status == "success" for status in statuses):
        return "success"
    if any(status in ("success", "partial") for status in statuses):
        return "partial"
    return "failed"


def build_eval_report(traj_metrics, image_metrics, slam_metrics, summary_text):
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": _overall_status(traj_metrics, image_metrics, slam_metrics),
        "warnings": _collect_warnings(traj_metrics, image_metrics, slam_metrics),
        "traj_eval": dict(traj_metrics or {}),
        "image_eval": dict(image_metrics or {}),
        "slam_friendly_eval": dict(slam_metrics or {}),
        "summary_text": str(summary_text or ""),
        "artifact_contract": {
            "default_files": [
                REPORT_FILENAME,
                DETAILS_FILENAME,
                "plots/traj_xy.png",
                "plots/{}".format(METRICS_OVERVIEW_FILENAME),
            ],
            "legacy_default_outputs_replaced": list(LEGACY_EVAL_OUTPUT_NAMES),
        },
    }


def _detail_fieldnames():
    return [
        "row_type",
        "row_id",
        "seq_idx",
        "frame_idx",
        "timestamp",
        "psnr",
        "ms_ssim",
        "lpips",
        "frame_idx_0",
        "frame_idx_1",
        "raw_matches",
        "inlier_matches",
        "inlier_ratio",
        "pose_success",
        "rotation_error_deg",
        "translation_direction_error_deg",
    ]


def _image_detail_rows(records):
    rows = []
    for idx, item in enumerate(list(records or [])):
        rows.append(
            {
                "row_type": "image_frame",
                "row_id": int(idx),
                "seq_idx": item.get("seq_idx", ""),
                "frame_idx": item.get("frame_idx", ""),
                "timestamp": item.get("timestamp", ""),
                "psnr": item.get("psnr", ""),
                "ms_ssim": item.get("ms_ssim", ""),
                "lpips": item.get("lpips", ""),
                "frame_idx_0": "",
                "frame_idx_1": "",
                "raw_matches": "",
                "inlier_matches": "",
                "inlier_ratio": "",
                "pose_success": "",
                "rotation_error_deg": "",
                "translation_direction_error_deg": "",
            }
        )
    return rows


def _slam_detail_rows(records):
    rows = []
    for idx, item in enumerate(list(records or [])):
        rows.append(
            {
                "row_type": "slam_pair",
                "row_id": int(idx),
                "seq_idx": "",
                "frame_idx": "",
                "timestamp": "",
                "psnr": "",
                "ms_ssim": "",
                "lpips": "",
                "frame_idx_0": item.get("frame_idx_0", ""),
                "frame_idx_1": item.get("frame_idx_1", ""),
                "raw_matches": item.get("raw_matches", ""),
                "inlier_matches": item.get("inlier_matches", ""),
                "inlier_ratio": item.get("inlier_ratio", ""),
                "pose_success": item.get("pose_success", ""),
                "rotation_error_deg": item.get("rotation_error_deg", ""),
                "translation_direction_error_deg": item.get("translation_direction_error_deg", ""),
            }
        )
    return rows


def write_eval_details(out_dir, image_records, slam_records):
    out_path = Path(out_dir).resolve() / DETAILS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _detail_fieldnames()
    rows = _image_detail_rows(image_records) + _slam_detail_rows(slam_records)
    with out_path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=_GRID_COLOR, linewidth=0.9, alpha=0.75)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_color(_SPINE_COLOR)
    ax.tick_params(labelsize=9, colors=_TEXT_COLOR)


def _plot_unavailable(ax, title, xlabel, ylabel, text):
    _style_axes(ax)
    ax.set_title(title, fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#6a7480")


def _curve_xy(curve):
    if not isinstance(curve, dict):
        return [], []
    raw_x = curve.get("x")
    raw_y = curve.get("y")
    xs = list(raw_x) if raw_x is not None else []
    ys = list(raw_y) if raw_y is not None else []
    return xs, ys


def _plot_curve(ax, curve, title, ylabel, color):
    xs, ys = _curve_xy(curve)
    if not xs or not ys:
        _plot_unavailable(ax, title, "x", ylabel, "{} unavailable".format(title))
        return
    _style_axes(ax)
    ax.set_title(title, fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel(str(curve.get("xlabel", "Index")), fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.plot(xs, ys, color=color, linewidth=1.7)


def _plot_rpe(ax, trans_curve, rot_curve):
    trans_x, trans_y = _curve_xy(trans_curve)
    rot_x, rot_y = _curve_xy(rot_curve)
    has_trans = bool(trans_x and trans_y)
    has_rot = bool(rot_x and rot_y)
    if not has_trans and not has_rot:
        _plot_unavailable(ax, "RPE Curve", "x", "error", "RPE curve unavailable")
        return

    _style_axes(ax)
    ax.set_title("RPE Curve", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    twin = None
    if has_trans:
        ax.set_xlabel(str(trans_curve.get("xlabel", "Index")), fontsize=10.5)
        ax.set_ylabel("Translation Error (m)", fontsize=10.5)
        ax.plot(trans_x, trans_y, color=_RPE_TRANS_COLOR, linewidth=1.7, label="RPE trans")
    if has_rot:
        axis_target = ax if not has_trans else ax.twinx()
        twin = axis_target if has_trans else None
        axis_target.set_ylabel("Rotation Error (deg)", fontsize=10.5, color=_RPE_ROT_COLOR if has_trans else _TEXT_COLOR)
        axis_target.plot(rot_x, rot_y, color=_RPE_ROT_COLOR, linewidth=1.5, label="RPE rot")
        if has_trans:
            axis_target.tick_params(axis="y", colors=_RPE_ROT_COLOR)
            for side in ["top", "right", "left", "bottom"]:
                axis_target.spines[side].set_color(_SPINE_COLOR)
    handles, labels = ax.get_legend_handles_labels()
    if twin is not None:
        handles2, labels2 = twin.get_legend_handles_labels()
        if handles2:
            handles.extend(handles2)
            labels.extend(labels2)
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)


def _xy_pairs(records, key):
    xs = []
    ys = []
    for item in list(records or []):
        value = item.get(key)
        x_value = item.get("frame_idx", item.get("seq_idx"))
        if value is None or x_value is None:
            continue
        xs.append(float(x_value))
        ys.append(float(value))
    return xs, ys


def _plot_image(ax, records):
    psnr_x, psnr_y = _xy_pairs(records, "psnr")
    msssim_x, msssim_y = _xy_pairs(records, "ms_ssim")
    lpips_x, lpips_y = _xy_pairs(records, "lpips")
    if not psnr_y and not msssim_y and not lpips_y:
        _plot_unavailable(ax, "Image Metrics", "frame_idx", "metric", "Image metrics unavailable")
        return

    _style_axes(ax)
    ax.set_title("Image Metrics", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Frame Index", fontsize=10.5)
    if psnr_y:
        ax.set_ylabel("PSNR (dB)", fontsize=10.5)
        ax.plot(psnr_x, psnr_y, color=_PSNR_COLOR, linewidth=1.7, label="PSNR")
    twin = ax.twinx()
    twin_has_data = False
    if msssim_y:
        twin.plot(msssim_x, msssim_y, color=_MSSSIM_COLOR, linewidth=1.5, label="MS-SSIM")
        twin_has_data = True
    if lpips_y:
        twin.plot(lpips_x, lpips_y, color=_LPIPS_COLOR, linewidth=1.5, label="LPIPS")
        twin_has_data = True
    twin.set_ylabel("Score", fontsize=10.5)
    twin.tick_params(labelsize=9, colors=_TEXT_COLOR)
    for side in ["top", "right", "left", "bottom"]:
        twin.spines[side].set_color(_SPINE_COLOR)
    handles, labels = ax.get_legend_handles_labels()
    if twin_has_data:
        handles2, labels2 = twin.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)


def _plot_slam(ax, records):
    xs = list(range(len(list(records or []))))
    inlier_y = []
    pose_x = []
    pose_y = []
    for idx, item in enumerate(list(records or [])):
        inlier_value = item.get("inlier_ratio")
        inlier_y.append(None if inlier_value is None else float(inlier_value))
        pose_value = item.get("pose_success")
        if pose_value is not None:
            pose_x.append(int(idx))
            pose_y.append(1.0 if bool(pose_value) else 0.0)
    if not xs:
        _plot_unavailable(ax, "SLAM-friendly Metrics", "pair_idx", "score", "SLAM-friendly metrics unavailable")
        return

    _style_axes(ax)
    ax.set_title("SLAM-friendly Metrics", fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Pair Index", fontsize=10.5)
    ax.set_ylabel("Inlier Ratio", fontsize=10.5)
    if any(value is not None for value in inlier_y):
        y_values = [0.0 if value is None else float(value) for value in inlier_y]
        ax.plot(xs, y_values, color=_INLIER_COLOR, linewidth=1.7, label="inlier_ratio")
    twin = ax.twinx()
    if pose_x:
        twin.step(pose_x, pose_y, where="mid", color=_POSE_COLOR, linewidth=1.4, label="pose_success")
        twin.set_ylim(-0.1, 1.1)
        twin.set_yticks([0.0, 1.0])
        twin.set_yticklabels(["fail", "success"])
    twin.set_ylabel("Pose Success", fontsize=10.5)
    twin.tick_params(labelsize=9, colors=_TEXT_COLOR)
    for side in ["top", "right", "left", "bottom"]:
        twin.spines[side].set_color(_SPINE_COLOR)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)


def save_metrics_overview(out_dir, traj_payload, image_records, slam_records):
    plot_path = Path(out_dir).resolve() / "plots" / METRICS_OVERVIEW_FILENAME
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180)
    fig.patch.set_facecolor(_FIG_FACE)
    fig.suptitle("Evaluation Metrics Overview", fontsize=14, color=_TEXT_COLOR, y=0.99)

    traj_payload = dict(traj_payload or {})
    _plot_curve(axes[0][0], traj_payload.get("ape_curve"), "APE Curve", "Translation Error (m)", _APE_COLOR)
    _plot_rpe(axes[0][1], traj_payload.get("rpe_trans_curve"), traj_payload.get("rpe_rot_curve"))
    _plot_image(axes[1][0], image_records)
    _plot_slam(axes[1][1], slam_records)

    fig.tight_layout(pad=0.9, rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return plot_path


def cleanup_legacy_eval_outputs(out_dir):
    out_path = Path(out_dir).resolve()
    for rel_path in LEGACY_EVAL_OUTPUT_NAMES:
        path = out_path / rel_path
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception:
            continue


def write_eval_artifacts(out_dir, traj_result, image_result, slam_result, summary_text):
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    report = build_eval_report(
        (traj_result or {}).get("metrics", {}),
        (image_result or {}).get("metrics", {}),
        (slam_result or {}).get("metrics", {}),
        summary_text,
    )
    report_path = out_path / REPORT_FILENAME
    write_json(report_path, report, indent=2)
    details_path = write_eval_details(
        out_path,
        (image_result or {}).get("records", []),
        (slam_result or {}).get("records", []),
    )
    metrics_plot_path = save_metrics_overview(
        out_path,
        (traj_result or {}).get("overview", {}),
        (image_result or {}).get("records", []),
        (slam_result or {}).get("records", []),
    )
    cleanup_legacy_eval_outputs(out_path)
    return {
        "report_path": report_path,
        "details_path": details_path,
        "metrics_overview_path": metrics_plot_path,
        "report": report,
    }
