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


def _as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_number(value, digits=2):
    if value is None:
        return "n/a"
    try:
        return "{:.{digits}f}".format(float(value), digits=int(digits))
    except Exception:
        return "n/a"


def _fmt_ratio(value):
    if value is None:
        return "n/a"
    try:
        return "{:.4f}".format(float(value))
    except Exception:
        return "n/a"


def _bytes_to_mib(value):
    try:
        if value is None:
            return None
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return None


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
        "raw_frame_count": int(len(frame_files)),
        "transmitted_frame_count": int(len(boundary_frame_paths)),
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
    for prefix, name in [("ori", "ape_ori_vs_gt"), ("gen", "ape_gen_vs_gt")]:
        item = _as_dict(evo_summary.get("{}_stats".format(prefix)))
        rows.append(
            {
                "comparison": name,
                "metric_source": "evo_ape",
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
    rpe = _as_dict(evo_summary.get("rpe"))
    for prefix, name in [("ori", "rpe_ori"), ("gen", "rpe_gen")]:
        track = _as_dict(rpe.get(prefix))
        for relation_label, suffix in [("trans", "trans"), ("rot", "rot")]:
            item = _as_dict(track.get(relation_label))
            stats = _as_dict(item.get("stats"))
            rows.append(
                {
                    "comparison": "{}_{}".format(name, suffix),
                    "metric_source": "evo_rpe",
                    "alignment": evo_summary.get("alignment", "sim3"),
                    "rmse": stats.get("rmse", ""),
                    "mean": stats.get("mean", ""),
                    "median": stats.get("median", ""),
                    "std": stats.get("std", ""),
                    "min": stats.get("min", ""),
                    "max": stats.get("max", ""),
                    "sse": stats.get("sse", ""),
                    "pose_pairs": item.get("pose_pairs", ""),
                    "result_zip": item.get("result_zip", ""),
                }
            )
    details_path = Path(out_dir).resolve() / "eval_details.csv"
    _write_csv(details_path, _detail_fieldnames(), rows)
    return details_path


def _decode_generate_sec(decode_report):
    return (
        _pick_float(decode_report, ["wall_generate_sec"])
        or _pick_float(decode_report, ["total_runtime_sec"])
        or _pick_float(decode_report, ["backend_result", "wall_generate_sec"])
        or _pick_float(decode_report, ["backend_result", "total_runtime_sec"])
    )


def _decode_frame_count(decode_report, merge_report):
    count = (
        _pick_int(merge_report, ["summary", "execution_frame_count"])
        or _pick_int(merge_report, ["summary", "merged_frame_count"])
        or _pick_int(merge_report, ["outputs", "frame_count"])
    )
    if count is not None:
        return count
    total = 0
    for item in list(decode_report.get("units") or []):
        if isinstance(item, dict):
            value = _pick_int(item, ["num_frames"])
            if value is not None:
                total += int(value)
    return total if total > 0 else None


def _row(lines, key, value):
    lines.append("{:<30} : {}".format(str(key), str(value)))


def _summary_lines(exp_dir, inputs, reports, compression_report, warnings, eval_runtime_sec=None):
    encode_result = reports["encode_result"]
    decode_report = reports["decode_report"]
    decode_merge_report = reports["decode_merge_report"]
    evo_summary = reports["evo_summary"]

    alignment = str(evo_summary.get("alignment") or "sim3").strip().lower()
    alignment_text = "Sim3 (-a -s)" if alignment == "sim3" else (alignment or "n/a")
    encode_profile = _as_dict(encode_result.get("profile"))
    encode_motion_profile = _as_dict(encode_profile.get("motion"))
    encode_sec = _pick_float(encode_result, ["profile", "total_sec"])
    decode_sec = _decode_generate_sec(decode_report)
    eval_sec = _as_float(eval_runtime_sec)
    total_sec = None
    if encode_sec is not None and decode_sec is not None and eval_sec is not None:
        total_sec = float(encode_sec) + float(decode_sec) + float(eval_sec)
    decode_frames = _decode_frame_count(decode_report, decode_merge_report)
    decode_avg_fps = None
    if decode_frames is not None and decode_sec is not None and float(decode_sec) > 0:
        decode_avg_fps = float(decode_frames) / float(decode_sec)
    all_warnings = []
    for warning in list(warnings) + list(evo_summary.get("warnings") or []):
        text = str(warning)
        if text not in all_warnings:
            all_warnings.append(text)

    lines = [
        "=== ExpHub Eval Summary ===",
        "created_at: {}".format(datetime.now().isoformat(timespec="seconds")),
        "workflow: slam_run -> evo -> summary_build",
        "",
        "[Time]",
    ]

    _row(lines, "total_sec", _fmt_number(total_sec))
    _row(lines, "encode_sec", _fmt_number(encode_sec))
    _row(lines, "decode_sec", _fmt_number(decode_sec))
    _row(lines, "eval_sec", _fmt_number(eval_sec))
    _row(lines, "encode.motion_segment_sec", _fmt_number(encode_profile.get("motion_segment_sec")))
    _row(lines, "encode.semantic_anchor_sec", _fmt_number(encode_profile.get("semantic_anchor_sec")))
    _row(lines, "encode.result_writer_sec", _fmt_number(encode_profile.get("result_writer_sec")))
    _row(lines, "motion.phase_correlation_sec", _fmt_number(encode_motion_profile.get("phase_correlation_sec")))
    _row(lines, "motion.orb_tracking_sec", _fmt_number(encode_motion_profile.get("orb_tracking_sec")))
    _row(lines, "motion.optical_flow_sec", _fmt_number(encode_motion_profile.get("optical_flow_sec")))
    _row(lines, "decode.generate_sec", _fmt_number(decode_sec))
    _row(lines, "decode.avg_fps", _fmt_number(decode_avg_fps))

    lines.extend(["", "[Quality: evo]"])
    _row(lines, "alignment", alignment_text)
    _row(lines, "gt_path_length_m", _fmt_number(evo_summary.get("gt_path_length_m")))
    _row(lines, "ape.ori_rmse_m", _fmt_number(evo_summary.get("ori_ape_rmse"), digits=3))
    _row(lines, "ape.gen_rmse_m", _fmt_number(evo_summary.get("gen_ape_rmse"), digits=3))
    _row(lines, "ape.delta_gen_minus_ori_m", _fmt_number(evo_summary.get("rmse_delta_gen_minus_ori"), digits=3))
    _row(lines, "rpe.ori_trans_rmse_m", _fmt_number(evo_summary.get("ori_rpe_trans_rmse"), digits=3))
    _row(lines, "rpe.gen_trans_rmse_m", _fmt_number(evo_summary.get("gen_rpe_trans_rmse"), digits=3))
    _row(lines, "rpe.delta_trans_m", _fmt_number(evo_summary.get("rpe_delta_trans"), digits=3))
    _row(lines, "rpe.ori_rot_rmse_deg", _fmt_number(evo_summary.get("ori_rpe_rot_rmse_deg"), digits=2))
    _row(lines, "rpe.gen_rot_rmse_deg", _fmt_number(evo_summary.get("gen_rpe_rot_rmse_deg"), digits=2))
    _row(lines, "rpe.delta_rot_deg", _fmt_number(evo_summary.get("rpe_delta_rot_deg"), digits=2))
    _row(lines, "eval_reliability", str(evo_summary.get("eval_reliability") or "n/a"))

    lines.extend(["", "[Compression]"])
    _row(lines, "raw_size_mib", _fmt_number(_bytes_to_mib(compression_report.get("orig_size_bytes"))))
    _row(lines, "hvm_size_mib", _fmt_number(_bytes_to_mib(compression_report.get("comp_size_bytes"))))
    _row(lines, "transmission_ratio", _fmt_ratio(compression_report.get("ratio")))
    _row(lines, "reduction_pct", _fmt_number(compression_report.get("reduction_pct")))
    _row(lines, "raw_frames", compression_report.get("raw_frame_count", "n/a"))
    _row(lines, "transmitted_frames", compression_report.get("transmitted_frame_count", "n/a"))

    lines.extend(["", "[Diagnostics]"])
    rpe_delta_frames = evo_summary.get("rpe_delta_frames")
    rpe_delta_seconds = evo_summary.get("rpe_delta_seconds_approx")
    if rpe_delta_frames is not None and rpe_delta_seconds is not None:
        _row(
            lines,
            "RPE delta",
            "{} frames (~{} s)".format(
                int(rpe_delta_frames),
                _fmt_number(rpe_delta_seconds),
            ),
        )
    _row(lines, "ape.ori_pose_pairs", evo_summary.get("ori_pose_pairs") if evo_summary.get("ori_pose_pairs") is not None else "n/a")
    _row(lines, "ape.gen_pose_pairs", evo_summary.get("gen_pose_pairs") if evo_summary.get("gen_pose_pairs") is not None else "n/a")
    _row(lines, "rpe.ori_trans_pose_pairs", evo_summary.get("ori_rpe_trans_pose_pairs") if evo_summary.get("ori_rpe_trans_pose_pairs") is not None else "n/a")
    _row(lines, "rpe.gen_trans_pose_pairs", evo_summary.get("gen_rpe_trans_pose_pairs") if evo_summary.get("gen_rpe_trans_pose_pairs") is not None else "n/a")
    _row(lines, "rpe.ori_rot_pose_pairs", evo_summary.get("ori_rpe_rot_pose_pairs") if evo_summary.get("ori_rpe_rot_pose_pairs") is not None else "n/a")
    _row(lines, "rpe.gen_rot_pose_pairs", evo_summary.get("gen_rpe_rot_pose_pairs") if evo_summary.get("gen_rpe_rot_pose_pairs") is not None else "n/a")
    _row(lines, "trajectory_plot_status", str(evo_summary.get("plot_status") or "skipped"))
    if all_warnings:
        lines.append("warnings:")
        for warning in all_warnings:
            lines.append("- {}".format(warning))
    else:
        _row(lines, "warnings", "none")
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
    summary_text = "\n".join(
        _summary_lines(
            exp_dir,
            inputs,
            reports,
            compression_report,
            warnings,
            eval_runtime_sec=_get_arg(config, "eval_runtime_sec"),
        )
    )
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
