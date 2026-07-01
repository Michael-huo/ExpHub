from __future__ import annotations

import re
from pathlib import Path


RAW_METHOD_KEY = "raw"
METHOD_ORDER = (RAW_METHOD_KEY, "h265", "dcvc_fm_q21", "vlmem")
DISPLAY_NAMES = {
    "raw": "Raw",
    "h265": "H.265",
    "dcvc_fm_q21": "DCVC-FM q21",
    "vlmem": "VLMem/REC",
}
TRAJECTORY_ROLES = {
    "raw": "ORI",
    "h265": "codec_decoded",
    "dcvc_fm_q21": "codec_decoded",
    "vlmem": "REC",
}


def as_dict(value):
    return value if isinstance(value, dict) else {}


def safe_token(value):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return token or "value"


def relative_path(base_dir, target_path):
    if target_path is None:
        return None
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return target.relative_to(base).as_posix()
    except Exception:
        return str(target)


def resolve_path(exp_dir, text):
    value = str(text or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(exp_dir).resolve() / path
    return path.resolve()


def file_size(path_obj):
    try:
        path = Path(path_obj).resolve()
        if path.is_file():
            return int(path.stat().st_size)
    except Exception:
        pass
    return 0


def sum_file_sizes(paths):
    return int(sum(file_size(path) for path in list(paths or [])))


def bytes_to_mib(value):
    try:
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return None


def reduction_pct(reference_bytes, payload_bytes):
    try:
        ref = float(reference_bytes)
        val = float(payload_bytes)
    except Exception:
        return None
    if ref <= 0.0:
        return None
    return float((1.0 - val / ref) * 100.0)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def method_enc_time_sec(item):
    report = as_dict(item)
    for key in ["enc_time_sec", "encode_time_sec", "encoding_time_sec"]:
        if key not in report:
            continue
        parsed = safe_float(report.get(key))
        if parsed is not None:
            return float(parsed)
    return None


def format_seconds(value):
    parsed = safe_float(value)
    if parsed is None:
        return "N/A"
    return "{:.2f}s".format(parsed)


def format_mib_from_report(item):
    report = as_dict(item)
    mib = safe_float(report.get("payload_mib"))
    if mib is None:
        payload_bytes = safe_float(report.get("payload_bytes"))
        if payload_bytes is not None:
            mib = payload_bytes / (1024.0 * 1024.0)
    if mib is None:
        return "N/A"
    return "{:.2f}MiB".format(mib)


def short_reason(value, max_len=120):
    try:
        text = str(value or "").splitlines()[0].strip()
    except Exception:
        text = ""
    if not text:
        return ""
    if len(text) <= int(max_len):
        return text
    return text[: max(0, int(max_len) - 3)].rstrip() + "..."


def raw_payload_bytes_from_report(report):
    payload = as_dict(report)
    methods = as_dict(payload.get("methods"))
    raw_method = as_dict(methods.get(RAW_METHOD_KEY))
    candidates = [
        raw_method.get("payload_bytes"),
        payload.get("raw_frame_bytes"),
        payload.get("raw_payload_bytes"),
    ]
    for value in candidates:
        parsed = safe_int(value)
        if parsed is not None:
            return int(parsed)
    raise RuntimeError("compression benchmark report missing canonical raw payload bytes")


def resolve_method_report(report, method_key):
    payload = as_dict(report)
    methods = as_dict(payload.get("methods"))
    key = str(method_key)
    if key == RAW_METHOD_KEY:
        raw_method = as_dict(methods.get(RAW_METHOD_KEY))
        if raw_method:
            return raw_method
        raise RuntimeError("compression benchmark report missing canonical raw method")
    return as_dict(methods.get(key))


def benchmark_method_order(report):
    payload = as_dict(report)
    methods = as_dict(payload.get("methods"))
    source_order = list(payload.get("methods_order") or METHOD_ORDER)
    if not source_order:
        source_order = list(METHOD_ORDER)
    result = []
    for raw_key in source_order:
        key = str(raw_key)
        if key not in result:
            result.append(key)
    for key in METHOD_ORDER:
        if key not in result:
            result.append(key)
    return result


def canonical_method_report(
    exp_dir,
    method_key,
    status,
    source_frames_dir,
    fps,
    frame_count,
    display_name=None,
    error_message="",
    payload_bytes=None,
    raw_reference_bytes=None,
    zip_reference_bytes=None,
    enc_time_sec=None,
    decode_time_sec=None,
    codec_wall_time_sec=None,
    time_semantics="",
    encoded_artifact_path=None,
    encoded_artifact_dir=None,
    decoded_frames_dir=None,
    command=None,
    extra=None,
):
    method = str(method_key)
    reduction = reduction_pct(raw_reference_bytes, payload_bytes)
    report = {
        "method_key": method,
        "display_name": str(display_name or DISPLAY_NAMES.get(method, method)),
        "status": str(status),
        "error_message": str(error_message or ""),
        "payload_bytes": int(payload_bytes) if payload_bytes is not None else None,
        "payload_mib": bytes_to_mib(payload_bytes) if payload_bytes is not None else None,
        "reduction_pct": reduction,
        "reduction_pct_vs_raw_frames": reduction,
        "reduction_pct_vs_zip": reduction,
        "enc_time_sec": float(enc_time_sec) if enc_time_sec is not None else None,
        "decode_time_sec": float(decode_time_sec) if decode_time_sec is not None else None,
        "codec_wall_time_sec": float(codec_wall_time_sec) if codec_wall_time_sec is not None else None,
        "time_semantics": str(time_semantics or ""),
        "source_frames_dir": relative_path(exp_dir, source_frames_dir),
        "encoded_artifact_path": relative_path(exp_dir, encoded_artifact_path),
        "encoded_artifact_dir": relative_path(exp_dir, encoded_artifact_dir),
        "decoded_frames_dir": relative_path(exp_dir, decoded_frames_dir),
        "frame_count": int(frame_count),
        "fps": int(fps),
        "trajectory_role": TRAJECTORY_ROLES.get(method, ""),
        "command": list(command or []),
    }
    if extra:
        report.update(dict(extra))
    return report


def method_summary_lines(report):
    lines = []
    for method_key in benchmark_method_order(report):
        item = resolve_method_report(report, method_key)
        display_name = str(item.get("display_name") or DISPLAY_NAMES.get(method_key, method_key))
        status = str(item.get("status") or ("missing" if not item else "n/a"))
        line = "  - {}: status={} payload={} enc={} dec={}".format(
            display_name,
            status,
            format_mib_from_report(item),
            format_seconds(method_enc_time_sec(item)),
            format_seconds(item.get("decode_time_sec")),
        )
        if status in ("missing", "skipped", "failed"):
            reason = short_reason(item.get("error_message"))
            if not reason and status == "missing":
                reason = "method report missing"
            if reason:
                line += " reason={}".format(reason)
        lines.append(line)
    return lines
