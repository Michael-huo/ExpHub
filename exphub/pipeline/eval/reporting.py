from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from exphub.common.io import write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_warn


STAT_KEYS = ["mean", "median", "std", "min", "max"]


def empty_stats():
    return dict((key, None) for key in STAT_KEYS)


def metric_stats(values):
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return empty_stats()
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def append_warning(metrics_obj, message):
    text = str(message or "").strip()
    if not text:
        return
    warnings_list = metrics_obj.setdefault("warnings", [])
    if text not in warnings_list:
        warnings_list.append(text)
    log_warn(text)


def read_json(path_obj):
    path = Path(path_obj).resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def read_timestamps(path_obj):
    path = Path(path_obj).resolve()
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = str(line).strip()
        if not text:
            continue
        try:
            out.append(float(text.split()[-1]))
        except Exception:
            continue
    return out


def write_json(path_obj, payload, indent=2):
    write_json_atomic(path_obj, payload, indent=indent)


def write_text(path_obj, text):
    write_text_atomic(path_obj, text)


def write_csv(path_obj, fieldnames, rows):
    path = Path(path_obj).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow(dict(row))
    os.replace(str(tmp_path), str(path))


def fmt_value(value, unit=""):
    if value is None:
        return "n/a"
    text = "{:.6f}".format(float(value))
    unit_text = str(unit or "").strip()
    if unit_text:
        return "{} {}".format(text, unit_text)
    return text


def resolve_formal_eval_inputs(exp_dir):
    exp_root = Path(exp_dir).resolve()
    return {
        "exp_dir": exp_root,
        "segment_frames_dir": exp_root / "segment" / "frames",
        "segment_calib_path": exp_root / "segment" / "calib.txt",
        "segment_timestamps_path": exp_root / "segment" / "timestamps.txt",
        "merge_frames_dir": exp_root / "merge" / "frames",
        "merge_calib_path": exp_root / "merge" / "calib.txt",
        "merge_timestamps_path": exp_root / "merge" / "timestamps.txt",
        "merge_manifest_path": exp_root / "merge" / "merge_manifest.json",
        "runs_plan_path": exp_root / "infer" / "runs_plan.json",
        "slam_report_path": exp_root / "slam" / "report.json",
        "eval_dir": exp_root / "eval",
    }


def _warning_lines(metrics_obj):
    warnings_list = list((metrics_obj or {}).get("warnings", []) or [])
    if not warnings_list:
        return ["warnings: 0"]
    lines = ["warnings: {}".format(len(warnings_list))]
    for item in warnings_list:
        lines.append("- {}".format(item))
    return lines


def _bool_text(value):
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def build_summary_lines(traj_metrics, image_metrics, slam_metrics):
    traj = traj_metrics or {}
    image = image_metrics or {}
    slam = slam_metrics or {}
    lines = [
        "=== Trajectory Eval ===",
        "status: {}".format(traj.get("eval_status", "failed")),
        "APE RMSE: {}".format(fmt_value((traj.get("ape_trans") or {}).get("rmse"), "m")),
        "RPE trans RMSE: {}".format(fmt_value((traj.get("rpe_trans") or {}).get("rmse"), "m")),
        "RPE rot RMSE: {}".format(fmt_value((traj.get("rpe_rot") or {}).get("rmse"), "deg")),
        "matched poses: {}".format(int(traj.get("matched_pose_count") or 0)),
        "ori_path_length_m: {}".format(fmt_value(traj.get("ori_path_length_m"), "m")),
        "gen_path_length_m: {}".format(fmt_value(traj.get("gen_path_length_m"), "m")),
    ]
    lines.extend(_warning_lines(traj))
    lines.extend(
        [
            "",
            "=== Image Eval ===",
            "status: {}".format(image.get("eval_status", "failed")),
            "PSNR mean: {}".format(fmt_value((image.get("psnr") or {}).get("mean"), "dB")),
            "MS-SSIM mean: {}".format(fmt_value((image.get("ms_ssim") or {}).get("mean"))),
            "LPIPS mean: {}".format(fmt_value((image.get("lpips") or {}).get("mean"))),
            "frame_count: {}".format(int(image.get("frame_count") or 0)),
        ]
    )
    lines.extend(_warning_lines(image))
    lines.extend(
        [
            "",
            "=== SLAM-Friendly Eval ===",
            "status: {}".format(slam.get("eval_status", "failed")),
            "reference_source: {}".format(slam.get("reference_source", "unavailable")),
            "uses_proxy_reference: {}".format(_bool_text(slam.get("uses_proxy_reference"))),
            "inlier_ratio mean: {}".format(fmt_value((slam.get("inlier_ratio") or {}).get("mean"))),
            "pose_success_rate: {}".format(fmt_value(slam.get("pose_success_rate"))),
            "valid_pair_count: {}".format(int(slam.get("valid_pair_count") or 0)),
        ]
    )
    lines.extend(_warning_lines(slam))
    return lines


def build_summary_text(traj_metrics, image_metrics, slam_metrics):
    return "\n".join(build_summary_lines(traj_metrics, image_metrics, slam_metrics))


def log_eval_terminal_summary(traj_metrics, image_metrics, slam_metrics, out_dir):
    traj_status = str((traj_metrics or {}).get("eval_status", "failed"))
    matched = int((traj_metrics or {}).get("matched_pose_count") or 0)
    prefix = log_info if traj_status in ("success", "partial") else log_warn
    prefix("eval summary: traj/image/slam evaluation completed")
    log_info("eval traj: status={} matched_poses={}".format(traj_status, matched))
    log_info("eval image: status={} frame_count={}".format(str((image_metrics or {}).get("eval_status", "failed")), int((image_metrics or {}).get("frame_count") or 0)))
    log_info("eval slam: status={} valid_pairs={}".format(str((slam_metrics or {}).get("eval_status", "failed")), int((slam_metrics or {}).get("valid_pair_count") or 0)))
    log_info("eval out_dir: {}".format(Path(out_dir).resolve()))


def _collect_warnings(traj_metrics, image_metrics, slam_metrics):
    warnings_list = []
    for prefix, metrics_obj in [("traj", traj_metrics), ("image", image_metrics), ("slam", slam_metrics)]:
        for item in list((metrics_obj or {}).get("warnings", []) or []):
            text = "{}: {}".format(prefix, item)
            if text not in warnings_list:
                warnings_list.append(text)
    return warnings_list


def _overall_status(traj_metrics, image_metrics, slam_metrics):
    statuses = []
    for metrics_obj in [traj_metrics, image_metrics, slam_metrics]:
        value = str((metrics_obj or {}).get("eval_status", "") or "").strip()
        if value:
            statuses.append(value)
    if not statuses:
        return "failed"
    if all(value == "success" for value in statuses):
        return "success"
    if any(value in ("success", "partial") for value in statuses):
        return "partial"
    return "failed"


def build_eval_report(traj_metrics, image_metrics, slam_metrics, summary_text):
    return {
        "created_at": str((traj_metrics or {}).get("created_at", "") or ""),
        "eval_status": _overall_status(traj_metrics, image_metrics, slam_metrics),
        "warnings": _collect_warnings(traj_metrics, image_metrics, slam_metrics),
        "traj_eval": dict(traj_metrics or {}),
        "image_eval": dict(image_metrics or {}),
        "slam_friendly_eval": dict(slam_metrics or {}),
        "summary_text": str(summary_text or ""),
        "artifact_contract": {
            "formal_files": [
                "report.json",
                "metrics/traj_eval.json",
                "metrics/image_eval.json",
                "metrics/slam_eval.json",
                "plots/traj_xy.png",
                "plots/metrics_overview.png",
            ],
            "compat_files": [
                "details.csv",
            ],
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


def write_eval_details(out_dir, image_records, slam_records):
    rows = []
    for idx, item in enumerate(list(image_records or [])):
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
    for idx, item in enumerate(list(slam_records or [])):
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
    details_path = Path(out_dir).resolve() / "details.csv"
    write_csv(details_path, _detail_fieldnames(), rows)
    return details_path


def _setup_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="exphub_mpl_")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _curve_xy(curve):
    if not isinstance(curve, dict):
        return [], []
    return list(curve.get("x") or []), list(curve.get("y") or [])


def save_metrics_overview(out_dir, traj_overview, image_records, slam_records):
    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "plots" / "metrics_overview.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=180)
    fig.patch.set_facecolor("white")

    ape_x, ape_y = _curve_xy((traj_overview or {}).get("ape_curve"))
    if ape_x and ape_y:
        axes[0][0].plot(ape_x, ape_y, color="#1f4e79", linewidth=1.6)
    axes[0][0].set_title("APE Curve")

    rpe_tx, rpe_ty = _curve_xy((traj_overview or {}).get("rpe_trans_curve"))
    rpe_rx, rpe_ry = _curve_xy((traj_overview or {}).get("rpe_rot_curve"))
    if rpe_tx and rpe_ty:
        axes[0][1].plot(rpe_tx, rpe_ty, color="#c56a2d", linewidth=1.5, label="rpe_trans")
    if rpe_rx and rpe_ry:
        axes[0][1].plot(rpe_rx, rpe_ry, color="#546d8c", linewidth=1.5, label="rpe_rot")
    if rpe_tx or rpe_rx:
        axes[0][1].legend(loc="upper right", fontsize=8)
    axes[0][1].set_title("RPE Curves")

    image_x = [int(item.get("frame_idx")) for item in list(image_records or []) if item.get("frame_idx") is not None]
    image_psnr = [float(item.get("psnr")) for item in list(image_records or []) if item.get("psnr") not in (None, "")]
    image_msssim = [float(item.get("ms_ssim")) for item in list(image_records or []) if item.get("ms_ssim") not in (None, "")]
    if image_x and image_psnr:
        axes[1][0].plot(image_x[: len(image_psnr)], image_psnr, color="#1f4e79", linewidth=1.5, label="psnr")
    if image_x and image_msssim:
        axes[1][0].plot(image_x[: len(image_msssim)], image_msssim, color="#4d7f4b", linewidth=1.5, label="ms_ssim")
    if image_psnr or image_msssim:
        axes[1][0].legend(loc="upper right", fontsize=8)
    axes[1][0].set_title("Image Metrics")

    slam_x = list(range(len(list(slam_records or []))))
    slam_inlier = []
    slam_pose = []
    for item in list(slam_records or []):
        slam_inlier.append(None if item.get("inlier_ratio") in (None, "") else float(item.get("inlier_ratio")))
        slam_pose.append(None if item.get("pose_success") in (None, "") else (1.0 if bool(item.get("pose_success")) else 0.0))
    if slam_x and any(value is not None for value in slam_inlier):
        axes[1][1].plot(
            slam_x,
            [0.0 if value is None else float(value) for value in slam_inlier],
            color="#1f4e79",
            linewidth=1.5,
            label="inlier_ratio",
        )
    if slam_x and any(value is not None for value in slam_pose):
        axes[1][1].step(
            slam_x,
            [0.0 if value is None else float(value) for value in slam_pose],
            where="mid",
            color="#4d7f4b",
            linewidth=1.4,
            label="pose_success",
        )
    if slam_x:
        axes[1][1].legend(loc="upper right", fontsize=8)
    axes[1][1].set_title("SLAM-Friendly Metrics")

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return plot_path


def write_eval_artifacts(out_dir, traj_result, image_result, slam_result, summary_text):
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    traj_metrics = dict((traj_result or {}).get("metrics") or {})
    image_metrics = dict((image_result or {}).get("metrics") or {})
    slam_metrics = dict((slam_result or {}).get("metrics") or {})
    report = build_eval_report(traj_metrics, image_metrics, slam_metrics, summary_text)

    report_path = out_path / "report.json"
    write_json(report_path, report, indent=2)
    write_json(metrics_dir / "traj_eval.json", traj_metrics, indent=2)
    write_json(metrics_dir / "image_eval.json", image_metrics, indent=2)
    write_json(metrics_dir / "slam_eval.json", slam_metrics, indent=2)
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
    write_text(out_path / "summary.txt", str(summary_text or "") + "\n")
    return {
        "report_path": report_path,
        "details_path": details_path,
        "metrics_overview_path": metrics_plot_path,
        "report": report,
    }
