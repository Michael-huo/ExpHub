from __future__ import annotations

import argparse
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

from exphub.common.io import read_json_dict, write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_prog, log_warn


_FRAME_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _read_required_json(path_obj, label, warnings):
    path = Path(path_obj).resolve()
    payload = read_json_dict(path)
    if not payload:
        message = "missing or invalid {}: {}".format(label, path)
        warnings.append(message)
        log_warn(message)
    return payload


def _fmt_value(value, unit=""):
    if value is None:
        return "n/a"
    try:
        text = "{:.6f}".format(float(value))
    except Exception:
        return "n/a"
    unit_text = str(unit or "").strip()
    if unit_text:
        return "{} {}".format(text, unit_text)
    return text


def _pick_float(obj, keys):
    current = obj
    for key in list(keys or []):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        if current is None:
            return None
        return float(current)
    except Exception:
        return None


def _pick_int(obj, keys):
    current = obj
    for key in list(keys or []):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        if current is None:
            return None
        return int(current)
    except Exception:
        return None


def _input_paths(exp_dir, config):
    return {
        "prepare_frames_dir": Path(_get_arg(config, "prepare_frames_dir")).resolve(),
        "prepare_result": Path(_get_arg(config, "prepare_result")).resolve(),
        "generation_units": Path(_get_arg(config, "generation_units")).resolve(),
        "prompts": Path(_get_arg(config, "prompts")).resolve(),
        "encode_result": Path(_get_arg(config, "encode_result")).resolve(),
        "decode_report": Path(_get_arg(config, "decode_report")).resolve(),
        "decode_merge_report": Path(_get_arg(config, "decode_merge_report")).resolve(),
        "eval_slam_report": Path(_get_arg(config, "slam_report")).resolve(),
        "eval_traj_report": Path(_get_arg(config, "traj_report")).resolve(),
    }


def _frame_sort_key(path_obj):
    item = Path(path_obj)
    if item.stem.isdigit():
        return int(item.stem)
    return 10**12


def _image_files(frames_dir):
    root = Path(frames_dir).resolve()
    if not root.is_dir():
        return []
    out = [item for item in root.iterdir() if item.is_file() and item.suffix.lower() in _FRAME_EXTS]
    out.sort(key=lambda item: (_frame_sort_key(item), item.name))
    return out


def _file_size(path_obj):
    path = Path(path_obj).resolve()
    if not path.is_file():
        return 0
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _boundary_indices(generation_units):
    out = []
    seen = set()
    for unit in list(_as_dict(generation_units).get("units") or []):
        for key in ("start_idx", "end_idx"):
            try:
                value = int(_as_dict(unit).get(key))
            except Exception:
                continue
            if value < 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
    out.sort()
    return out


def _frame_path_for_idx(frames_dir, frame_idx):
    root = Path(frames_dir).resolve()
    stem = "{:06d}".format(int(frame_idx))
    for ext in sorted(_FRAME_EXTS):
        candidate = root / "{}{}".format(stem, ext)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _build_compression_report(exp_dir, out_dir, inputs, reports, warnings):
    prepare_frames_dir = inputs["prepare_frames_dir"]
    generation_units_path = inputs["generation_units"]
    prompts_path = inputs["prompts"]
    generation_units = reports["generation_units"]

    frame_files = _image_files(prepare_frames_dir)
    orig_size_bytes = int(sum(_file_size(item) for item in frame_files))
    boundaries = _boundary_indices(generation_units)
    boundary_frame_paths = []
    boundary_frame_bytes = 0
    missing_boundaries = []
    for frame_idx in boundaries:
        frame_path = _frame_path_for_idx(prepare_frames_dir, frame_idx)
        if frame_path is None:
            missing_boundaries.append(int(frame_idx))
            continue
        boundary_frame_paths.append(frame_path)
        boundary_frame_bytes += _file_size(frame_path)

    if missing_boundaries:
        message = "compression boundary frames missing: {}".format(missing_boundaries)
        warnings.append(message)
        log_warn(message)

    json_payload_paths = [generation_units_path, prompts_path]
    json_payload_bytes = int(sum(_file_size(path) for path in json_payload_paths))
    comp_size_bytes = int(boundary_frame_bytes + json_payload_bytes)
    ratio = float(comp_size_bytes) / float(orig_size_bytes) if orig_size_bytes > 0 else None
    reduction_pct = (1.0 - float(ratio)) * 100.0 if ratio is not None else None

    report = {
        "version": 1,
        "source": "eval.compression_report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "definition": "unique generation unit boundary frames plus native prompt/unit JSON payload",
        "orig_size_bytes": int(orig_size_bytes),
        "comp_size_bytes": int(comp_size_bytes),
        "ratio": ratio,
        "reduction_pct": reduction_pct,
        "unit_count": int(len(list(_as_dict(generation_units).get("units") or []))),
        "unit_boundary_count": int(len(boundaries)),
        "unit_boundaries": list(boundaries),
        "boundary_frame_bytes": int(boundary_frame_bytes),
        "json_payload_bytes": int(json_payload_bytes),
        "json_payload_files": [_relative_path(exp_dir, path) for path in json_payload_paths],
        "boundary_frame_files": [_relative_path(exp_dir, path) for path in boundary_frame_paths],
        "source_inputs": {
            "prepare_frames_dir": _relative_path(exp_dir, prepare_frames_dir),
            "generation_units": _relative_path(exp_dir, generation_units_path),
            "prompts": _relative_path(exp_dir, prompts_path),
        },
        "excluded_from_comp_size": {
            "encode_result": "local validation/summary metadata, not required by backend execution payload",
            "decode_frames": "generated output, not compressed input",
        },
        "warnings": list(warnings),
    }
    report_path = Path(out_dir).resolve() / "eval_compression_report.json"
    write_json_atomic(report_path, report, indent=2)
    return report_path, report


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


def _write_csv(path_obj, fieldnames, rows):
    path = Path(path_obj).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow(dict(row))
    os.replace(str(tmp_path), str(path))


def _write_details(out_dir, traj_records):
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
    details_path = Path(out_dir).resolve() / "eval_details.csv"
    _write_csv(details_path, _detail_fieldnames(), rows)
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


def _save_metrics_overview(out_dir, traj_overview):
    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "eval_metrics_overview.png"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    ape_x, ape_y = _curve_xy((traj_overview or {}).get("ape_curve"))
    if ape_x and ape_y:
        axes[0].plot(ape_x, ape_y, color="#1f4e79", linewidth=1.6)
    axes[0].set_title("APE Curve")
    axes[0].set_xlabel("pose")
    axes[0].set_ylabel("m")

    rpe_tx, rpe_ty = _curve_xy((traj_overview or {}).get("rpe_trans_curve"))
    rpe_rx, rpe_ry = _curve_xy((traj_overview or {}).get("rpe_rot_curve"))
    if rpe_tx and rpe_ty:
        axes[1].plot(rpe_tx, rpe_ty, color="#c56a2d", linewidth=1.5, label="rpe_trans")
    if rpe_rx and rpe_ry:
        axes[1].plot(rpe_rx, rpe_ry, color="#546d8c", linewidth=1.5, label="rpe_rot")
    if rpe_tx or rpe_rx:
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("RPE Curves")
    axes[1].set_xlabel("pose")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return plot_path


def _summary_lines(exp_dir, inputs, reports, compression_report, warnings):
    prepare = reports["prepare_result"]
    generation_units = reports["generation_units"]
    slam_report = reports["eval_slam_report"]
    traj_report = reports["eval_traj_report"]

    units = list(_as_dict(generation_units).get("units") or [])
    lines = [
        "=== ExpHub Eval Native Summary ===",
        "created_at: {}".format(datetime.now().isoformat(timespec="seconds")),
        "workflow: slam_run -> trajectory_eval -> summary_build",
        "",
        "[Native Inputs]",
    ]
    for name in [
        "prepare_result",
        "generation_units",
        "prompts",
        "encode_result",
        "decode_report",
        "decode_merge_report",
        "eval_slam_report",
        "eval_traj_report",
    ]:
        lines.append("{}: {}".format(name, _relative_path(exp_dir, inputs[name])))

    lines.extend(
        [
            "",
            "[Headline Metrics]",
            "status: {}".format(str(traj_report.get("eval_status", "failed") or "failed")),
            "matched_poses: {}".format(int(traj_report.get("num_matches", traj_report.get("matched_pose_count", 0)) or 0)),
            "APE RMSE: {}".format(_fmt_value(traj_report.get("ape_rmse_m"), "m")),
            "RPE trans RMSE: {}".format(_fmt_value(traj_report.get("rpe_trans_rmse_m"), "m")),
            "RPE rot RMSE: {}".format(_fmt_value(traj_report.get("rpe_rot_rmse_deg"), "deg")),
            "ori_path_length: {}".format(_fmt_value(traj_report.get("ori_path_length_m"), "m")),
            "gen_path_length: {}".format(_fmt_value(traj_report.get("gen_path_length_m"), "m")),
            "",
            "[Generation Units]",
            "unit_count: {}".format(int(traj_report.get("unit_count", 0) or 0)),
            "unit_boundaries: {}".format(list(traj_report.get("unit_boundaries") or [])),
            "boundary_source: {}".format(str(traj_report.get("boundary_source", "") or "")),
            "boundary_time_source_kind: {}".format(str(traj_report.get("boundary_time_source_kind", "") or "")),
            "",
            "[Compression]",
            "orig_size: {} bytes".format(int(compression_report.get("orig_size_bytes", 0) or 0)),
            "comp_size: {} bytes".format(int(compression_report.get("comp_size_bytes", 0) or 0)),
            "ratio: {}".format(_fmt_value(compression_report.get("ratio"))),
            "reduction_pct: {}".format(_fmt_value(compression_report.get("reduction_pct"), "%")),
            "unit_boundaries: {}".format(int(compression_report.get("unit_boundary_count", 0) or 0)),
            "boundary_frame_bytes: {} bytes".format(int(compression_report.get("boundary_frame_bytes", 0) or 0)),
            "json_payload_bytes: {} bytes".format(int(compression_report.get("json_payload_bytes", 0) or 0)),
            "",
            "[Output Files]",
            "slam_report: eval/eval_slam_report.json",
            "traj_report: eval/eval_traj_report.json",
            "compression_report: eval/eval_compression_report.json",
            "summary: eval/eval_summary.txt",
            "details: eval/eval_details.csv",
            "trajectory_plot: {}".format(str(traj_report.get("plot_path", "eval/eval_traj_xy.png") or "eval/eval_traj_xy.png")),
            "metrics_overview: eval/eval_metrics_overview.png",
            "raw_ori_traj: {}".format(_as_dict(slam_report.get("ori")).get("trajectory_path", "")),
            "raw_gen_traj: {}".format(_as_dict(slam_report.get("gen")).get("trajectory_path", "")),
        ]
    )
    if warnings or list(traj_report.get("warnings") or []):
        lines.extend(["", "[Warnings]"])
        for warning in list(warnings) + list(traj_report.get("warnings") or []):
            lines.append("- {}".format(warning))
    return lines


def build_eval_summary(config):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = _input_paths(exp_dir, config)

    warnings = []
    reports = {
        "prepare_result": _read_required_json(inputs["prepare_result"], "prepare result", warnings),
        "generation_units": _read_required_json(inputs["generation_units"], "generation units", warnings),
        "prompts": _read_required_json(inputs["prompts"], "prompts", warnings),
        "encode_result": _read_required_json(inputs["encode_result"], "encode result", warnings),
        "decode_report": _read_required_json(inputs["decode_report"], "decode report", warnings),
        "decode_merge_report": _read_required_json(inputs["decode_merge_report"], "decode merge report", warnings),
        "eval_slam_report": _read_required_json(inputs["eval_slam_report"], "eval slam report", warnings),
        "eval_traj_report": _read_required_json(inputs["eval_traj_report"], "eval trajectory report", warnings),
    }

    compression_path, compression_report = _build_compression_report(exp_dir, out_dir, inputs, reports, warnings)
    summary_text = "\n".join(_summary_lines(exp_dir, inputs, reports, compression_report, warnings))
    summary_path = out_dir / "eval_summary.txt"
    details_path = _write_details(out_dir, _get_arg(config, "traj_records", []))
    overview_path = _save_metrics_overview(out_dir, _get_arg(config, "traj_overview", {}))
    write_text_atomic(summary_path, summary_text + "\n")

    log_prog("eval summary generated")
    log_info("eval summary: {}".format(summary_path))
    log_info("eval compression report: {}".format(compression_path))
    log_info("eval details: {}".format(details_path))
    log_info("eval metrics overview: {}".format(overview_path))
    return {
        "summary_path": summary_path,
        "compression_path": compression_path,
        "details_path": details_path,
        "metrics_overview_path": overview_path,
        "summary_text": summary_text,
        "compression": compression_report,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-native-summary", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--decode_merge_report", required=True)
    parser.add_argument("--slam_report", required=True)
    parser.add_argument("--traj_report", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_native_summary:
        raise SystemExit("eval summary helper requires --run-native-summary")
    build_eval_summary(vars(args))


if __name__ == "__main__":
    main()
