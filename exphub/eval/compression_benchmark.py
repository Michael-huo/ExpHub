from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

from exphub.common.compression_benchmark import (
    DISPLAY_NAMES,
    METHOD_ORDER,
    TRAJECTORY_ROLES,
    as_dict,
    benchmark_method_order,
    method_enc_time_sec,
    relative_path,
    resolve_path,
    resolve_method_report,
    safe_float,
)
from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_prog, log_warn
from exphub.eval.evo_eval import run_evo_eval_single_track
from exphub.eval.slam_run import run_single_slam_track


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _format_clip(duration, frame_count, fps):
    parsed = safe_float(duration)
    if parsed is None:
        frames = safe_float(frame_count)
        fps_value = safe_float(fps)
        if frames is not None and fps_value is not None and fps_value > 0.0:
            parsed = frames / fps_value
    if parsed is None:
        return ""
    if abs(parsed - round(parsed)) < 1e-9:
        return "{}s".format(int(round(parsed)))
    return "{:.3f}".format(parsed).rstrip("0").rstrip(".") + "s"


def _norm_ape_pct(ape_rmse_m, path_m):
    ape = safe_float(ape_rmse_m)
    path = safe_float(path_m)
    if ape is None or path is None or path <= 0.0:
        return None
    return float(ape / path * 100.0)


def _image_size(path_obj):
    path = Path(path_obj).resolve()
    if path.suffix.lower() == ".png":
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                if width > 0 and height > 0:
                    return int(width), int(height)
        except Exception:
            pass
    try:
        from PIL import Image

        with Image.open(str(path)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise RuntimeError("cannot read image dimensions for {}: {}".format(path, exc)) from exc


def _validate_decoded_frames(method_key, prepared_frames_dir, decoded_frames_dir):
    prepared = list_frames_sorted(prepared_frames_dir)
    decoded = list_frames_sorted(decoded_frames_dir)
    if len(decoded) != len(prepared):
        raise RuntimeError(
            "{} decoded frame count mismatch: decoded={} prepared={}".format(
                method_key,
                int(len(decoded)),
                int(len(prepared)),
            )
        )
    expected_names = ["{:06d}.png".format(int(idx)) for idx in range(len(prepared))]
    actual_names = [Path(item).name for item in decoded]
    if actual_names != expected_names:
        raise RuntimeError(
            "{} decoded frames are not normalized to ExpHub order/names; first actual names={}".format(
                method_key,
                actual_names[:5],
            )
        )
    for prepared_path, decoded_path in zip(prepared, decoded):
        prepared_size = _image_size(prepared_path)
        decoded_size = _image_size(decoded_path)
        if decoded_size != prepared_size:
            raise RuntimeError(
                "{} decoded dimension mismatch at {}: decoded={} prepared={}".format(
                    method_key,
                    Path(decoded_path).name,
                    decoded_size,
                    prepared_size,
                )
            )
    return decoded


def _base_row(method_key, method_report, clip):
    item = as_dict(method_report)
    frames = item.get("frame_count")
    fps = item.get("fps")
    reduction = item.get("reduction_pct")
    if reduction is None:
        reduction = item.get("reduction_pct_vs_raw_frames")
    if reduction is None:
        reduction = item.get("reduction_pct_vs_zip")
    return {
        "clip": clip,
        "frames": frames,
        "fps": fps,
        "path_m": None,
        "method_key": method_key,
        "display_name": str(item.get("display_name") or DISPLAY_NAMES.get(method_key, method_key)),
        "trajectory_role": str(item.get("trajectory_role") or TRAJECTORY_ROLES.get(method_key, "")),
        "status": "skipped",
        "error_message": "",
        "payload_mib": item.get("payload_mib"),
        "payload_bytes": item.get("payload_bytes"),
        "reduction_pct": reduction,
        "reduction_pct_vs_zip": item.get("reduction_pct_vs_zip"),
        "reduction_pct_vs_raw_frames": item.get("reduction_pct_vs_raw_frames"),
        "enc_time_sec": method_enc_time_sec(item),
        "decode_time_sec": item.get("decode_time_sec"),
        "decoded_frame_generation_sec": item.get("decoded_frame_generation_sec"),
        "codec_wall_time_sec": item.get("codec_wall_time_sec"),
        "time_semantics": item.get("time_semantics"),
        "decoded_frames_dir": item.get("decoded_frames_dir"),
        "frame_source": "",
        "trajectory_path": "",
        "ape_summary_path": "",
        "ape_rmse_m": None,
        "norm_ape_pct": None,
    }


def _finalize_quality(row, ape_rmse_m, path_m):
    row["ape_rmse_m"] = ape_rmse_m
    row["path_m"] = path_m
    row["norm_ape_pct"] = _norm_ape_pct(ape_rmse_m, path_m)
    return row


def _mainline_row(exp_dir, method_key, method_report, track_key, trajectory_path, main_eval_summary, clip):
    row = _base_row(method_key, method_report, clip)
    track = str(track_key)
    vslam = as_dict(as_dict(main_eval_summary).get("vslam"))
    ape_key = "ori_ape_rmse_m" if track == "ori" else "rec_ape_rmse_m"
    row["status"] = "ok"
    row["frame_source"] = "existing_eval_{}".format(track)
    row["trajectory_path"] = relative_path(exp_dir, trajectory_path)
    row["ape_summary_path"] = relative_path(exp_dir, Path(exp_dir).resolve() / "eval" / "summary.json")
    return _finalize_quality(row, vslam.get(ape_key), vslam.get("gt_path_length_m"))


def _single_track_row(exp_dir, method_key, method_report, frames_dir, traj_result, evo_result, clip):
    row = _base_row(method_key, method_report, clip)
    ape_summary = as_dict(as_dict(evo_result).get("summary"))
    row["status"] = "ok"
    row["frame_source"] = relative_path(exp_dir, frames_dir)
    row["trajectory_path"] = as_dict(traj_result).get("trajectory_path") or ""
    row["ape_summary_path"] = relative_path(exp_dir, as_dict(evo_result).get("summary_path"))
    return _finalize_quality(row, ape_summary.get("ape_rmse"), ape_summary.get("gt_path_length_m"))


def _failure_row(method_key, method_report, message, clip, status="failed"):
    row = _base_row(method_key, method_report, clip)
    row["status"] = str(status)
    row["error_message"] = str(message or "")
    return row


def _slam_config(config, out_dir):
    return {
        "exp_dir": str(_get_arg(config, "exp_dir")),
        "out_dir": str(out_dir),
        "prepare_result": str(_get_arg(config, "prepare_result")),
        "prepare_frames_dir": str(_get_arg(config, "prepare_frames_dir")),
        "decode_frames_dir": str(_get_arg(config, "decode_frames_dir")),
        "decode_calib": str(_get_arg(config, "decode_calib")),
        "decode_timestamps": str(_get_arg(config, "decode_timestamps")),
        "decode_report": str(_get_arg(config, "decode_report")),
        "generation_units": str(_get_arg(config, "generation_units")),
        "encode_result": str(_get_arg(config, "encode_result")),
        "seq": "both",
        "droid_repo": str(_get_arg(config, "droid_repo")),
        "weights": str(_get_arg(config, "weights")),
        "fps": float(_get_arg(config, "fps", 0.0) or 0.0),
        "disable_vis": bool(_get_arg(config, "disable_vis", True)),
        "t0": 0,
        "stride": 1,
        "max_frames": 0,
        "undistort_mode": "auto",
        "resize_interp": "linear",
        "intr_scale_mode": "demo",
        "buffer": 512,
        "image_size": [240, 320],
        "beta": 0.3,
        "filter_thresh": 1.5,
        "warmup": 12,
        "keyframe_thresh": 2.0,
        "frontend_thresh": 12.0,
        "frontend_window": 25,
        "frontend_radius": 2,
        "frontend_nms": 1,
        "backend_thresh": 20.0,
        "backend_radius": 2,
        "backend_nms": 3,
        "upsample": False,
        "no_tqdm": False,
        "stereo": False,
    }


def _run_method_track(config, method_key, method_report, frames_dir, clip):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    gt_traj = ensure_file(_get_arg(config, "gt_traj"), "ground truth trajectory")
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    method_dir = out_dir / method_key
    method_dir.mkdir(parents=True, exist_ok=True)
    traj_result = run_single_slam_track(_slam_config(config, out_dir), method_key, frames_dir)
    traj_path = exp_dir / str(as_dict(traj_result).get("trajectory_path", ""))
    evo_result = run_evo_eval_single_track(
        {
            "out_dir": str(method_dir),
            "exp_dir": str(exp_dir),
            "gt_traj": str(gt_traj),
            "est_traj": str(traj_path),
            "method_key": method_key,
            "display_name": DISPLAY_NAMES.get(method_key, method_key),
            "t_max_diff": float(_get_arg(config, "t_max_diff", 0.03)),
            "fps": float(_get_arg(config, "fps", 0.0) or 0.0),
            "prepare_result": str(_get_arg(config, "prepare_result")),
            "decode_report": str(_get_arg(config, "decode_report")),
        }
    )
    return _single_track_row(exp_dir, method_key, method_report, frames_dir, traj_result, evo_result, clip)


def _write_csv(path_obj, rows):
    fieldnames = [
        "clip",
        "frames",
        "fps",
        "path_m",
        "method_key",
        "display_name",
        "payload_mib",
        "reduction_pct",
        "enc_time_sec",
        "ape_rmse_m",
        "norm_ape_pct",
        "status",
        "error_message",
    ]
    path = Path(path_obj).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(str(tmp_path), str(path))


def _write_method_manifest(exp_dir, method_dir, row, method_report, reused_mainline=False):
    payload = {
        "version": 1,
        "source": "exphub.eval.compression_benchmark.method",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_key": row.get("method_key"),
        "display_name": row.get("display_name"),
        "status": row.get("status"),
        "reused_mainline_eval": bool(reused_mainline),
        "row": dict(row),
        "method_report": dict(method_report),
    }
    path = Path(method_dir).resolve() / "method_summary.json"
    write_json_atomic(path, payload, indent=2)
    return relative_path(exp_dir, path)


def run_compression_benchmark_eval(config):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    decode_report_path = ensure_file(_get_arg(config, "decode_benchmark_report"), "compression benchmark decode report")
    prepare_frames_dir = ensure_dir(_get_arg(config, "prepare_frames_dir"), "prepare frames dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_report = read_json_dict(decode_report_path)
    if not decode_report:
        raise RuntimeError("invalid compression benchmark decode report: {}".format(decode_report_path))
    main_eval_summary = read_json_dict(Path(_get_arg(config, "main_eval_summary")).resolve())
    frame_count = decode_report.get("frame_count")
    fps = decode_report.get("fps") or _get_arg(config, "fps")
    clip = _format_clip(_get_arg(config, "clip_duration"), frame_count, fps)

    rows = []
    manifest_paths = {}
    for method_key in METHOD_ORDER:
        method_report = resolve_method_report(decode_report, method_key)
        method_dir = out_dir / method_key
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not method_report:
                row = _failure_row(method_key, {}, "decode report missing method {}".format(method_key), clip)
                rows.append(row)
                manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, {})
                continue

            method_status = str(method_report.get("status") or "skipped")
            if method_status != "ok":
                row = _failure_row(
                    method_key,
                    method_report,
                    method_report.get("error_message") or "decode method status={}".format(method_status),
                    clip,
                    status=method_status if method_status in ("skipped", "failed") else "failed",
                )
                rows.append(row)
                manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report)
                continue

            if method_key == "raw":
                traj_path = exp_dir / "eval" / "ori" / "traj_est.tum"
                if traj_path.is_file() and main_eval_summary:
                    row = _mainline_row(exp_dir, method_key, method_report, "ori", traj_path, main_eval_summary, clip)
                    rows.append(row)
                    manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report, reused_mainline=True)
                else:
                    frames_dir = resolve_path(exp_dir, method_report.get("decoded_frames_dir")) or prepare_frames_dir
                    _validate_decoded_frames(method_key, prepare_frames_dir, frames_dir)
                    row = _run_method_track(config, method_key, method_report, frames_dir, clip)
                    rows.append(row)
                    manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report)
                continue

            if method_key == "vlmem":
                traj_path = exp_dir / "eval" / "rec" / "traj_est.tum"
                if traj_path.is_file() and main_eval_summary:
                    row = _mainline_row(exp_dir, method_key, method_report, "rec", traj_path, main_eval_summary, clip)
                    rows.append(row)
                    manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report, reused_mainline=True)
                else:
                    frames_dir = resolve_path(exp_dir, method_report.get("decoded_frames_dir"))
                    if frames_dir is None:
                        raise RuntimeError("vlmem decoded_frames_dir missing")
                    _validate_decoded_frames(method_key, prepare_frames_dir, frames_dir)
                    row = _run_method_track(config, method_key, method_report, frames_dir, clip)
                    rows.append(row)
                    manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report)
                continue

            frames_dir = resolve_path(exp_dir, method_report.get("decoded_frames_dir"))
            if frames_dir is None:
                raise RuntimeError("{} decoded_frames_dir missing".format(method_key))
            _validate_decoded_frames(method_key, prepare_frames_dir, frames_dir)
            row = _run_method_track(config, method_key, method_report, frames_dir, clip)
            rows.append(row)
            manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report)
        except Exception as exc:
            log_warn("compression benchmark eval {} failed: {}".format(method_key, exc))
            row = _failure_row(method_key, method_report, str(exc), clip)
            rows.append(row)
            manifest_paths[method_key] = _write_method_manifest(exp_dir, method_dir, row, method_report)

    summary = {
        "version": 2,
        "source": "exphub.eval.compression_benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decode_benchmark_report": relative_path(exp_dir, decode_report_path),
        "clip": clip,
        "frame_count": int(frame_count or 0),
        "fps": fps,
        "reduction_reference": "raw_payload_bytes",
        "table_time_field": "enc_time_sec",
        "methods_order": list(METHOD_ORDER),
        "method_manifests": dict(manifest_paths),
        "methods": {row["method_key"]: dict(row) for row in rows},
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    write_json_atomic(summary_path, summary, indent=2)
    _write_csv(csv_path, rows)
    log_prog("compression benchmark eval summary generated")
    return {
        "summary_path": summary_path,
        "csv_path": csv_path,
        "summary": summary,
        "out_dir": out_dir,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub internal compression benchmark eval helper")
    parser.add_argument("--run-compression-benchmark-eval", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--decode_benchmark_report", required=True)
    parser.add_argument("--main_eval_summary", required=True)
    parser.add_argument("--prepare_result", required=True)
    parser.add_argument("--prepare_frames_dir", required=True)
    parser.add_argument("--generation_units", required=True)
    parser.add_argument("--encode_result", required=True)
    parser.add_argument("--decode_frames_dir", required=True)
    parser.add_argument("--decode_calib", required=True)
    parser.add_argument("--decode_timestamps", required=True)
    parser.add_argument("--decode_report", required=True)
    parser.add_argument("--gt_traj", required=True)
    parser.add_argument("--droid_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--clip_duration", default="")
    parser.add_argument("--t_max_diff", type=float, default=0.03)
    parser.add_argument("--disable_vis", action="store_true")
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run_compression_benchmark_eval:
        raise SystemExit("compression benchmark helper requires --run-compression-benchmark-eval")
    return run_compression_benchmark_eval(vars(args))


if __name__ == "__main__":
    main()
