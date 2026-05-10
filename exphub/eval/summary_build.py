from __future__ import annotations

import argparse
import csv
import os
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


def _resolve_report_path(exp_dir, path_text):
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path.resolve()
    return (Path(exp_dir).resolve() / path).resolve()


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
        "ori_run_meta": Path(_get_arg(config, "ori_run_meta")).resolve(),
        "gen_run_meta": Path(_get_arg(config, "gen_run_meta")).resolve(),
        "evo_summary": Path(_get_arg(config, "evo_summary")).resolve(),
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
        "definition": "unique generation unit boundary frames plus prompt/unit JSON payload",
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
        "comparison",
        "metric_source",
        "alignment",
        "rmse",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "sse",
        "pose_pairs",
        "result_zip",
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


def _write_details(out_dir, evo_summary):
    rows = []
    for prefix, name in [("ori", "ori_vs_gt"), ("gen", "gen_vs_gt")]:
        item = _as_dict(evo_summary.get("{}_stats".format(prefix)))
        rows.append(
            {
                "comparison": name,
                "metric_source": evo_summary.get("metric_source", "evo_ape"),
                "alignment": evo_summary.get("alignment", "sim3"),
                "rmse": item.get("rmse", ""),
                "mean": item.get("mean", ""),
                "median": item.get("median", ""),
                "std": item.get("std", ""),
                "min": item.get("min", ""),
                "max": item.get("max", ""),
                "sse": item.get("sse", ""),
                "pose_pairs": evo_summary.get("{}_pose_pairs".format(prefix), ""),
                "result_zip": evo_summary.get("{}_result_zip".format(prefix), ""),
            }
        )
    details_path = Path(out_dir).resolve() / "eval_details.csv"
    _write_csv(details_path, _detail_fieldnames(), rows)
    return details_path


def _summary_lines(exp_dir, inputs, reports, compression_report, warnings):
    generation_units = reports["generation_units"]
    ori_run_meta = reports["ori_run_meta"]
    gen_run_meta = reports["gen_run_meta"]
    evo_summary = reports["evo_summary"]

    units = list(_as_dict(generation_units).get("units") or [])
    alignment = str(evo_summary.get("alignment") or "sim3").strip().lower()
    alignment_text = "Sim3 Umeyama (-a -s)" if alignment == "sim3" else (alignment or "n/a")
    overlay_path = evo_summary.get("trajectory_overlay_path")
    overlay_file = _resolve_report_path(exp_dir, overlay_path)
    plot_status = str(evo_summary.get("plot_status") or "skipped")
    plot_text = "n/a"
    if plot_status == "success" and overlay_file is not None and overlay_file.is_file():
        plot_text = "eval/trajectory_overlay_auto2d.png"
    failure_log = Path(inputs["evo_summary"]).resolve().parent / "evo_failure.log"
    lines = [
        "=== ExpHub Eval Summary ===",
        "created_at: {}".format(datetime.now().isoformat(timespec="seconds")),
        "workflow: slam_run -> evo_ape -> summary_build",
        "",
        "[Inputs]",
    ]
    for name in [
        "prepare_result",
        "generation_units",
        "prompts",
        "encode_result",
        "decode_report",
        "decode_merge_report",
        "ori_run_meta",
        "gen_run_meta",
        "evo_summary",
    ]:
        lines.append("{}: {}".format(name, _relative_path(exp_dir, inputs[name])))

    lines.extend(
        [
            "",
            "[Headline Metrics]",
            "status: {}".format(str(evo_summary.get("status", "failed") or "failed")),
            "",
            "[Trajectory Accuracy: evo_ape]",
            "Metric source: {}".format(str(evo_summary.get("metric_source") or "evo_ape")),
            "Alignment mode: {}".format(str(evo_summary.get("alignment") or "sim3")),
            "Alignment: {}".format(alignment_text),
            "Timestamp max difference: {} s".format(_fmt_value(evo_summary.get("t_max_diff"))),
            "ORI APE RMSE: {}".format(_fmt_value(evo_summary.get("ori_ape_rmse"), "m")),
            "GEN APE RMSE: {}".format(_fmt_value(evo_summary.get("gen_ape_rmse"), "m")),
            "Delta GEN-ORI: {}".format(_fmt_value(evo_summary.get("rmse_delta_gen_minus_ori"), "m")),
            "RMSE Increase: {}".format(_fmt_value(evo_summary.get("rmse_increase_pct"), "%")),
            "RMSE Ratio GEN/ORI: {}".format(_fmt_value(evo_summary.get("rmse_ratio_gen_over_ori"))),
            "ORI pose pairs: {}".format(evo_summary.get("ori_pose_pairs") if evo_summary.get("ori_pose_pairs") is not None else "n/a"),
            "GEN pose pairs: {}".format(evo_summary.get("gen_pose_pairs") if evo_summary.get("gen_pose_pairs") is not None else "n/a"),
            "Trajectory plot status: {}".format(str(evo_summary.get("plot_status") or "skipped")),
            "Selected plot plane: {}".format(str(evo_summary.get("selected_plot_plane") or "n/a")),
            "GT plot mode: {}".format(str(evo_summary.get("gt_plot_mode") or "n/a")),
            "Plot common start: {}".format(_fmt_value(evo_summary.get("plot_common_start"), "s")),
            "Plot common end: {}".format(_fmt_value(evo_summary.get("plot_common_end"), "s")),
            "",
            "[Generation Units]",
            "unit_count: {}".format(int(len(units))),
            "unit_boundaries: {}".format(_boundary_indices(generation_units)),
            "boundary_source: {}".format(_relative_path(exp_dir, inputs["generation_units"])),
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
            "evo_summary: eval/evo_summary.json",
            "compression_report: eval/eval_compression_report.json",
            "details: eval/eval_details.csv",
            "trajectory_overlay: {}".format(plot_text),
            "ori_traj: {}".format(str(ori_run_meta.get("trajectory_path") or "eval/ori/traj_est.tum")),
            "gen_traj: {}".format(str(gen_run_meta.get("trajectory_path") or "eval/gen/traj_est.tum")),
            "ori_evo_zip: {}".format(str(evo_summary.get("ori_result_zip") or "eval/ori/evo_ape.zip")),
            "gen_evo_zip: {}".format(str(evo_summary.get("gen_result_zip") or "eval/gen/evo_ape.zip")),
        ]
    )
    if failure_log.is_file():
        lines.append("evo_failure_log: eval/evo_failure.log")
    if warnings or list(evo_summary.get("warnings") or []):
        lines.extend(["", "[Warnings]"])
        for warning in list(warnings) + list(evo_summary.get("warnings") or []):
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
        "ori_run_meta": _read_required_json(inputs["ori_run_meta"], "ORI run meta", warnings),
        "gen_run_meta": _read_required_json(inputs["gen_run_meta"], "GEN run meta", warnings),
        "evo_summary": _read_required_json(inputs["evo_summary"], "evo summary", warnings),
    }

    compression_path, compression_report = _build_compression_report(exp_dir, out_dir, inputs, reports, warnings)
    summary_text = "\n".join(_summary_lines(exp_dir, inputs, reports, compression_report, warnings))
    summary_path = out_dir / "eval_summary.txt"
    details_path = _write_details(out_dir, reports["evo_summary"])
    write_text_atomic(summary_path, summary_text + "\n")

    log_prog("eval summary generated")
    log_info("eval summary: {}".format(summary_path))
    log_info("eval compression report: {}".format(compression_path))
    log_info("eval details: {}".format(details_path))
    return {
        "summary_path": summary_path,
        "compression_path": compression_path,
        "details_path": details_path,
        "summary_text": summary_text,
        "compression": compression_report,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-eval-summary", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--decode_merge_report", required=True)
    parser.add_argument("--ori_run_meta", required=True)
    parser.add_argument("--gen_run_meta", required=True)
    parser.add_argument("--evo_summary", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_eval_summary:
        raise SystemExit("eval summary helper requires --run-eval-summary")
    build_eval_summary(vars(args))


if __name__ == "__main__":
    main()
