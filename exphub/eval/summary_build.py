from __future__ import annotations

from datetime import datetime
from pathlib import Path

from exphub.common.io import read_json_dict, write_csv_atomic, write_json_atomic
from exphub.common.logging import log_prog, log_warn


_REQUIRED_PAYLOAD_FIELDS = (
    "raw_bytes",
    "payload_bytes",
    "reduction_pct",
    "transmitted_frame_count",
    "generation_unit_count",
)


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _number_delta(after, before):
    after_value = _as_float(after)
    before_value = _as_float(before)
    if after_value is None or before_value is None:
        return None
    return float(after_value) - float(before_value)


def _relative_path(base_dir, target_path):
    if not target_path:
        return ""
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _json_null_row(value):
    return "" if value is None else value


def _payload_null(warnings, reason):
    warning = "canonical encode payload unavailable: {}".format(reason)
    if warning not in warnings:
        warnings.append(warning)
        log_warn(warning)
    return {
        "raw_bytes": None,
        "payload_bytes": None,
        "ratio": None,
        "reduction_pct": None,
        "raw_frame_count": None,
        "transmitted_frame_count": None,
        "unit_count": None,
        "unit_boundary_count": None,
        "boundary_frame_bytes": None,
        "json_payload_bytes": None,
    }


def _payload_from_encode_result(encode_result_path, *, complete_main_chain, warnings):
    path_text = str(encode_result_path or "").strip()
    if not path_text:
        if complete_main_chain:
            raise RuntimeError("complete infer requires encode_result canonical payload")
        return _payload_null(warnings, "encode_result path not provided")
    path = Path(path_text).resolve()
    payload = read_json_dict(path)
    if not payload:
        if complete_main_chain:
            raise RuntimeError("complete infer requires valid encode_result canonical payload: {}".format(path))
        return _payload_null(warnings, "encode_result missing or invalid")
    missing = [field for field in _REQUIRED_PAYLOAD_FIELDS if field not in payload or payload.get(field) is None]
    if missing:
        if complete_main_chain:
            raise RuntimeError("complete infer encode_result missing canonical payload fields: {}".format(", ".join(missing)))
        return _payload_null(warnings, "missing fields {}".format(", ".join(missing)))
    raw_bytes = int(payload.get("raw_bytes") or 0)
    payload_bytes = int(payload.get("payload_bytes") or 0)
    ratio = payload.get("payload_ratio")
    if ratio is None and raw_bytes > 0:
        ratio = float(payload_bytes) / float(raw_bytes)
    return {
        "raw_bytes": raw_bytes,
        "payload_bytes": payload_bytes,
        "ratio": ratio,
        "reduction_pct": payload.get("reduction_pct"),
        "raw_frame_count": payload.get("raw_frame_count"),
        "transmitted_frame_count": payload.get("transmitted_frame_count"),
        "unit_count": payload.get("generation_unit_count"),
        "unit_boundary_count": payload.get("unit_boundary_count"),
        "boundary_frame_bytes": payload.get("boundary_frame_bytes"),
        "json_payload_bytes": payload.get("json_payload_bytes"),
    }


def _runtime_summary(config):
    complete = bool(_get_arg(config, "complete_main_chain", False))
    stage_times = _as_dict(_get_arg(config, "stage_times", {}))
    eval_sec = _as_float(stage_times.get("eval"))
    if eval_sec is None:
        eval_sec = _as_float(_get_arg(config, "eval_runtime_sec"))
    return {
        "prepare_time_s": _as_float(stage_times.get("prepare")) if complete else None,
        "encode_time_s": _as_float(stage_times.get("encode")) if complete else None,
        "decode_time_s": _as_float(stage_times.get("decode")) if complete else None,
        "eval_time_s": eval_sec,
        "main_pipeline_time_s": _as_float(_get_arg(config, "main_pipeline_wall_time_s")) if complete else None,
    }


def _vslam_summary(evo_result):
    evo = _as_dict(evo_result)
    ori_ape = evo.get("ori_ape_rmse")
    rec_ape = evo.get("rec_ape_rmse")
    return {
        "ori_ape_rmse_m": ori_ape,
        "rec_ape_rmse_m": rec_ape,
        "ape_delta_rec_minus_ori_m": _number_delta(rec_ape, ori_ape),
        "gt_path_length_m": evo.get("gt_path_length_m"),
    }


def _trajectory_summary(evo_result):
    evo = _as_dict(evo_result)
    return {
        "plot_status": evo.get("plot_status"),
        "trajectory_overlay_path": evo.get("trajectory_overlay_path"),
        "trajectory_interactive_path": evo.get("trajectory_interactive_path"),
        "trajectory_interactive_status": evo.get("trajectory_interactive_status"),
        "selected_plot_plane": evo.get("selected_plot_plane"),
    }


def _canonical_summary(exp_dir, config):
    root_meta = read_json_dict(Path(exp_dir).resolve() / "run_meta.json")
    warnings = []
    complete = bool(_get_arg(config, "complete_main_chain", False))
    payload = _payload_from_encode_result(
        _get_arg(config, "encode_result", ""),
        complete_main_chain=complete,
        warnings=warnings,
    )
    evo_result = _as_dict(_get_arg(config, "evo_result", {}))
    warnings.extend(str(item) for item in list(evo_result.get("warnings") or []) if str(item) not in warnings)
    return {
        "schema_version": 1,
        "source": "exphub.eval.summary_build",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "identity": {
            "dataset": root_meta.get("dataset"),
            "sequence": root_meta.get("sequence"),
            "tag": root_meta.get("tag"),
            "start": root_meta.get("start"),
            "dur": root_meta.get("dur"),
            "fps": root_meta.get("fps"),
            "seed": root_meta.get("seed"),
            "decode_profile": root_meta.get("decode_profile"),
        },
        "payload": payload,
        "vslam": _vslam_summary(evo_result),
        "trajectory": _trajectory_summary(evo_result),
        "runtime": _runtime_summary(config),
        "sources": {
            "prepare_result": _relative_path(exp_dir, _get_arg(config, "prepare_result", "")),
            "encode_result": _relative_path(exp_dir, _get_arg(config, "encode_result", "")),
            "decode_report": _relative_path(exp_dir, _get_arg(config, "decode_report", "")),
            "ori_run_meta": _relative_path(exp_dir, _get_arg(config, "ori_run_meta", "")),
            "rec_run_meta": _relative_path(exp_dir, _get_arg(config, "rec_run_meta", "")),
        },
        "warnings": warnings,
    }


def _write_canonical_summary(out_dir, summary):
    out_dir = Path(out_dir).resolve()
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    write_json_atomic(json_path, summary, indent=2)
    identity = _as_dict(summary.get("identity"))
    payload = _as_dict(summary.get("payload"))
    vslam = _as_dict(summary.get("vslam"))
    runtime = _as_dict(summary.get("runtime"))
    row = {
        "dataset": identity.get("dataset") or "",
        "sequence": identity.get("sequence") or "",
        "tag": identity.get("tag") or "",
        "start": identity.get("start") or "",
        "dur": identity.get("dur") or "",
        "fps": _json_null_row(identity.get("fps")),
        "seed": _json_null_row(identity.get("seed")),
        "decode_profile": identity.get("decode_profile") or "",
        "raw_bytes": _json_null_row(payload.get("raw_bytes")),
        "payload_bytes": _json_null_row(payload.get("payload_bytes")),
        "ratio": _json_null_row(payload.get("ratio")),
        "reduction_pct": _json_null_row(payload.get("reduction_pct")),
        "raw_frame_count": _json_null_row(payload.get("raw_frame_count")),
        "transmitted_frame_count": _json_null_row(payload.get("transmitted_frame_count")),
        "unit_count": _json_null_row(payload.get("unit_count")),
        "unit_boundary_count": _json_null_row(payload.get("unit_boundary_count")),
        "boundary_frame_bytes": _json_null_row(payload.get("boundary_frame_bytes")),
        "json_payload_bytes": _json_null_row(payload.get("json_payload_bytes")),
        "ori_ape_rmse_m": _json_null_row(vslam.get("ori_ape_rmse_m")),
        "rec_ape_rmse_m": _json_null_row(vslam.get("rec_ape_rmse_m")),
        "ape_delta_rec_minus_ori_m": _json_null_row(vslam.get("ape_delta_rec_minus_ori_m")),
        "prepare_time_s": _json_null_row(runtime.get("prepare_time_s")),
        "encode_time_s": _json_null_row(runtime.get("encode_time_s")),
        "decode_time_s": _json_null_row(runtime.get("decode_time_s")),
        "eval_time_s": _json_null_row(runtime.get("eval_time_s")),
        "main_pipeline_time_s": _json_null_row(runtime.get("main_pipeline_time_s")),
    }
    write_csv_atomic(csv_path, list(row.keys()), [row])
    return json_path, csv_path


def build_eval_summary(config):
    exp_dir = Path(_get_arg(config, "exp_dir")).resolve()
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_summary = _canonical_summary(exp_dir, config)
    canonical_json_path, canonical_csv_path = _write_canonical_summary(out_dir, canonical_summary)
    log_prog("eval summary generated")
    return {
        "canonical_summary_path": canonical_json_path,
        "canonical_summary_csv_path": canonical_csv_path,
        "canonical_summary": canonical_summary,
    }
