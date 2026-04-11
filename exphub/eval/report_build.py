from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from exphub.common.io import write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_warn


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


def _warning_lines(metrics_obj):
    warnings_list = list((metrics_obj or {}).get("warnings", []) or [])
    if not warnings_list:
        return ["warnings: 0"]
    lines = ["warnings: {}".format(len(warnings_list))]
    for item in warnings_list:
        lines.append("- {}".format(item))
    return lines


def build_summary_lines(traj_metrics):
    traj = traj_metrics or {}
    lines = [
        "=== Trajectory Eval ===",
        "status: {}".format(traj.get("eval_status", "failed")),
        "reference: {}".format(traj.get("reference_name", "ori")),
        "estimate: {}".format(traj.get("estimate_name", "gen")),
        "APE RMSE: {}".format(fmt_value((traj.get("ape_trans") or {}).get("rmse"), "m")),
        "RPE trans RMSE: {}".format(fmt_value((traj.get("rpe_trans") or {}).get("rmse"), "m")),
        "RPE rot RMSE: {}".format(fmt_value((traj.get("rpe_rot") or {}).get("rmse"), "deg")),
        "matched poses: {}".format(int(traj.get("matched_pose_count") or 0)),
        "ori_path_length_m: {}".format(fmt_value(traj.get("ori_path_length_m"), "m")),
        "gen_path_length_m: {}".format(fmt_value(traj.get("gen_path_length_m"), "m")),
    ]
    lines.extend(_warning_lines(traj))
    return lines


def build_summary_text(traj_metrics):
    return "\n".join(build_summary_lines(traj_metrics))


def log_eval_terminal_summary(traj_metrics, out_dir):
    traj_status = str((traj_metrics or {}).get("eval_status", "failed"))
    matched = int((traj_metrics or {}).get("matched_pose_count") or 0)
    prefix = log_info if traj_status in ("success", "partial") else log_warn
    prefix("eval summary: trajectory evaluation completed")
    log_info("eval traj: status={} matched_poses={}".format(traj_status, matched))
    log_info("eval out_dir: {}".format(Path(out_dir).resolve()))


def build_eval_report(traj_metrics, summary_text):
    return {
        "created_at": str((traj_metrics or {}).get("created_at", "") or ""),
        "eval_status": str((traj_metrics or {}).get("eval_status", "failed") or "failed"),
        "warnings": list((traj_metrics or {}).get("warnings", []) or []),
        "traj_eval": dict(traj_metrics or {}),
        "summary_text": str(summary_text or ""),
        "artifact_contract": {
            "formal_files": [
                "report.json",
                "summary.txt",
                "details.csv",
                "metrics/traj_eval.json",
                "plots/traj_xy.png",
                "plots/metrics_overview.png",
            ],
        },
    }


def _detail_fieldnames():
    return [
        "pose_idx",
        "timestamp",
        "ape_trans_m",
        "ref_x",
        "ref_y",
        "ref_z",
        "est_x",
        "est_y",
        "est_z",
    ]


def write_eval_details(out_dir, traj_records):
    rows = []
    for item in list(traj_records or []):
        rows.append(
            {
                "pose_idx": item.get("pose_idx", ""),
                "timestamp": item.get("timestamp", ""),
                "ape_trans_m": item.get("ape_trans_m", ""),
                "ref_x": item.get("ref_x", ""),
                "ref_y": item.get("ref_y", ""),
                "ref_z": item.get("ref_z", ""),
                "est_x": item.get("est_x", ""),
                "est_y": item.get("est_y", ""),
                "est_z": item.get("est_z", ""),
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


def save_metrics_overview(out_dir, traj_overview):
    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "plots" / "metrics_overview.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    ape_x, ape_y = _curve_xy((traj_overview or {}).get("ape_curve"))
    if ape_x and ape_y:
        axes[0].plot(ape_x, ape_y, color="#1f4e79", linewidth=1.6)
    axes[0].set_title("APE Curve")

    rpe_tx, rpe_ty = _curve_xy((traj_overview or {}).get("rpe_trans_curve"))
    rpe_rx, rpe_ry = _curve_xy((traj_overview or {}).get("rpe_rot_curve"))
    if rpe_tx and rpe_ty:
        axes[1].plot(rpe_tx, rpe_ty, color="#c56a2d", linewidth=1.5, label="rpe_trans")
    if rpe_rx and rpe_ry:
        axes[1].plot(rpe_rx, rpe_ry, color="#546d8c", linewidth=1.5, label="rpe_rot")
    if rpe_tx or rpe_rx:
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("RPE Curves")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return plot_path


def write_eval_artifacts(out_dir, traj_result, summary_text):
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    traj_metrics = dict((traj_result or {}).get("metrics") or {})
    report = build_eval_report(traj_metrics, summary_text)

    report_path = out_path / "report.json"
    write_json(report_path, report, indent=2)
    write_json(metrics_dir / "traj_eval.json", traj_metrics, indent=2)
    details_path = write_eval_details(out_path, (traj_result or {}).get("records", []))
    metrics_plot_path = save_metrics_overview(out_path, (traj_result or {}).get("overview", {}))
    write_text(out_path / "summary.txt", str(summary_text or "") + "\n")
    return {
        "report_path": report_path,
        "details_path": details_path,
        "metrics_overview_path": metrics_plot_path,
        "report": report,
    }
