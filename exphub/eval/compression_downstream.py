from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_warn
from exphub.eval.evo_eval import run_evo_eval_single_track
from exphub.eval.slam_run import run_single_slam_track


METHOD_ORDER = ["zip", "h265", "dcvc_fm_q21", "vlmem"]
DISPLAY_NAMES = {
    "zip": "ZIP/ORI",
    "h265": "H.265",
    "dcvc_fm_q21": "DCVC-FM q21",
    "vlmem": "VLMem/REC",
}
TRAJECTORY_ROLES = {
    "zip": "ORI",
    "h265": "codec_decoded",
    "dcvc_fm_q21": "codec_decoded",
    "vlmem": "REC",
}


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _format_mib_from_row(row):
    item = _as_dict(row)
    mib = _safe_float(item.get("payload_mib"))
    if mib is None:
        payload_bytes = _safe_float(item.get("payload_bytes"))
        if payload_bytes is not None:
            mib = payload_bytes / (1024.0 * 1024.0)
    if mib is None:
        return "n/a"
    return "{:.2f}MiB".format(mib)


def _format_ape(value):
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return "{:.4f}m".format(parsed)


def _short_reason(value, max_len=120):
    try:
        text = str(value or "").splitlines()[0].strip()
    except Exception:
        text = ""
    if not text:
        return ""
    if len(text) <= int(max_len):
        return text
    return text[: max(0, int(max_len) - 3)].rstrip() + "..."


def _log_downstream_method_summary(summary):
    try:
        methods = _as_dict(_as_dict(summary).get("methods"))
        log_info("[Compression Downstream] Method Summary:")
        for method_key in METHOD_ORDER:
            row = _as_dict(methods.get(method_key))
            display_name = str(row.get("display_name") or DISPLAY_NAMES.get(method_key, method_key))
            status = str(row.get("status") or ("missing" if not row else "n/a"))
            line = "  - {}: status={} APE={} payload={}".format(
                display_name,
                status,
                _format_ape(row.get("ape_rmse_m")),
                _format_mib_from_row(row),
            )
            if status in ("missing", "skipped", "failed"):
                reason = _short_reason(row.get("error_message"))
                if not reason and status == "missing":
                    reason = "method summary missing"
                if reason:
                    line += " reason={}".format(reason)
            log_info(line)
    except Exception as exc:
        log_warn("compression downstream method summary logging skipped: {}".format(exc))


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return target.relative_to(base).as_posix()
    except Exception:
        return str(target)


def _resolve_path(exp_dir, text):
    value = str(text or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(exp_dir).resolve() / path
    return path.resolve()


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


def _base_row(method_key, benchmark_method):
    item = _as_dict(benchmark_method)
    return {
        "method_key": method_key,
        "display_name": str(item.get("display_name") or DISPLAY_NAMES.get(method_key, method_key)),
        "trajectory_role": TRAJECTORY_ROLES.get(method_key, ""),
        "status": "skipped",
        "error_message": "",
        "frame_source": "",
        "trajectory_path": "",
        "evo_summary_path": "",
        "ape_rmse_m": None,
        "rpe_trans_rmse_m": None,
        "rpe_rot_rmse_deg": None,
        "payload_mib": item.get("payload_mib"),
        "payload_bytes": item.get("payload_bytes"),
        "encode_time_sec": item.get("encode_time_sec"),
        "decode_time_sec": item.get("decode_time_sec"),
        "codec_wall_time_sec": item.get("codec_wall_time_sec"),
        "codec_time_semantics": item.get("codec_time_semantics"),
        "reduction_pct_vs_zip": item.get("reduction_pct_vs_zip"),
        "reduction_pct_vs_raw_frames": item.get("reduction_pct_vs_raw_frames"),
    }


def _mainline_metrics_row(exp_dir, method_key, benchmark_method, track_key, trajectory_path, main_evo):
    row = _base_row(method_key, benchmark_method)
    track = str(track_key)
    row["status"] = "ok"
    row["frame_source"] = "existing_eval_{}".format(track)
    row["trajectory_path"] = _relative_path(exp_dir, trajectory_path)
    row["evo_summary_path"] = _relative_path(exp_dir, Path(exp_dir).resolve() / "eval" / "evo_summary.json")
    row["ape_rmse_m"] = main_evo.get("{}_ape_rmse".format(track))
    row["rpe_trans_rmse_m"] = main_evo.get("{}_rpe_trans_rmse".format(track))
    row["rpe_rot_rmse_deg"] = main_evo.get("{}_rpe_rot_rmse_deg".format(track))
    return row


def _single_track_row(exp_dir, method_key, benchmark_method, frames_dir, traj_result, evo_result):
    row = _base_row(method_key, benchmark_method)
    evo_summary = _as_dict(_as_dict(evo_result).get("summary"))
    row["status"] = "ok"
    row["frame_source"] = _relative_path(exp_dir, frames_dir)
    row["trajectory_path"] = _as_dict(traj_result).get("trajectory_path") or ""
    row["evo_summary_path"] = _relative_path(exp_dir, _as_dict(evo_result).get("summary_path"))
    row["ape_rmse_m"] = evo_summary.get("ape_rmse")
    row["rpe_trans_rmse_m"] = evo_summary.get("rpe_trans_rmse")
    row["rpe_rot_rmse_deg"] = evo_summary.get("rpe_rot_rmse_deg")
    return row


def _failure_row(method_key, benchmark_method, message, status="failed"):
    row = _base_row(method_key, benchmark_method)
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
        "decode_merge_report": str(_get_arg(config, "decode_merge_report")),
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


def _run_method_track(config, method_key, benchmark_method, frames_dir):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    gt_traj = ensure_file(_get_arg(config, "gt_traj"), "ground truth trajectory")
    downstream_dir = Path(_get_arg(config, "out_dir")).resolve()
    method_dir = downstream_dir / method_key
    method_dir.mkdir(parents=True, exist_ok=True)
    traj_result = run_single_slam_track(_slam_config(config, downstream_dir), method_key, frames_dir)
    traj_path = exp_dir / str(_as_dict(traj_result).get("trajectory_path", ""))
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
    return _single_track_row(exp_dir, method_key, benchmark_method, frames_dir, traj_result, evo_result)


def _write_csv(path_obj, rows):
    fieldnames = [
        "method_key",
        "display_name",
        "trajectory_role",
        "status",
        "ape_rmse_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "payload_mib",
        "encode_time_sec",
        "decode_time_sec",
        "codec_wall_time_sec",
        "codec_time_semantics",
        "reduction_pct_vs_zip",
        "reduction_pct_vs_raw_frames",
        "frame_source",
        "trajectory_path",
        "evo_summary_path",
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


def run_compression_downstream(config):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    benchmark_path = ensure_file(_get_arg(config, "benchmark_report"), "compression benchmark report")
    prepare_frames_dir = ensure_dir(_get_arg(config, "prepare_frames_dir"), "prepare frames dir")
    decode_frames_dir = ensure_dir(_get_arg(config, "decode_frames_dir"), "decode frames dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = read_json_dict(benchmark_path)
    methods = _as_dict(benchmark.get("methods"))
    main_evo = read_json_dict(Path(_get_arg(config, "main_evo_summary")).resolve())
    rows = []

    for method_key in METHOD_ORDER:
        benchmark_method = _as_dict(methods.get(method_key))
        try:
            if method_key == "zip":
                traj_path = exp_dir / "eval" / "ori" / "traj_est.tum"
                if traj_path.is_file() and main_evo:
                    rows.append(_mainline_metrics_row(exp_dir, method_key, benchmark_method, "ori", traj_path, main_evo))
                else:
                    rows.append(_run_method_track(config, method_key, benchmark_method, prepare_frames_dir))
                continue

            if method_key == "vlmem":
                traj_path = exp_dir / "eval" / "rec" / "traj_est.tum"
                if traj_path.is_file() and main_evo:
                    rows.append(_mainline_metrics_row(exp_dir, method_key, benchmark_method, "rec", traj_path, main_evo))
                else:
                    rows.append(_run_method_track(config, method_key, benchmark_method, decode_frames_dir))
                continue

            status = str(benchmark_method.get("status") or "skipped")
            if status != "ok":
                rows.append(
                    _failure_row(
                        method_key,
                        benchmark_method,
                        benchmark_method.get("error_message") or "benchmark method status={}".format(status),
                        status=status if status in ("skipped", "failed") else "failed",
                    )
                )
                continue

            decoded_dir = _resolve_path(exp_dir, benchmark_method.get("decoded_frames_dir"))
            if decoded_dir is None:
                rows.append(_failure_row(method_key, benchmark_method, "decoded_frames_dir missing"))
                continue
            _validate_decoded_frames(method_key, prepare_frames_dir, decoded_dir)
            rows.append(_run_method_track(config, method_key, benchmark_method, decoded_dir))
        except Exception as exc:
            log_warn("compression downstream {} failed: {}".format(method_key, exc))
            rows.append(_failure_row(method_key, benchmark_method, exc))

    summary = {
        "version": 1,
        "source": "exphub.eval.compression_downstream",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_report": _relative_path(exp_dir, benchmark_path),
        "methods_order": list(METHOD_ORDER),
        "methods": {row["method_key"]: dict(row) for row in rows},
        "rows": rows,
    }
    summary_path = out_dir / "downstream_summary.json"
    csv_path = out_dir / "downstream_summary.csv"
    write_json_atomic(summary_path, summary, indent=2)
    _write_csv(csv_path, rows)
    log_prog("compression downstream summary generated")
    log_info("compression downstream summary: {}".format(summary_path))
    _log_downstream_method_summary(summary)
    return {
        "summary_path": summary_path,
        "csv_path": csv_path,
        "summary": summary,
        "out_dir": out_dir,
    }
