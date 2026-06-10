from __future__ import annotations

import re
from pathlib import Path


METHOD_ORDER = ("zip", "h265", "dcvc_fm_q21", "vlmem")
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


def format_seconds(value):
    parsed = safe_float(value)
    if parsed is None:
        return "n/a"
    return "{:.2f}s".format(parsed)


def format_mib_from_report(item):
    report = as_dict(item)
    mib = safe_float(report.get("payload_mib"))
    if mib is None:
        payload_bytes = safe_float(report.get("payload_bytes"))
        if payload_bytes is not None:
            mib = payload_bytes / (1024.0 * 1024.0)
    if mib is None:
        return "n/a"
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
    report = {
        "method_key": method,
        "display_name": str(display_name or DISPLAY_NAMES.get(method, method)),
        "status": str(status),
        "error_message": str(error_message or ""),
        "payload_bytes": int(payload_bytes) if payload_bytes is not None else None,
        "payload_mib": bytes_to_mib(payload_bytes) if payload_bytes is not None else None,
        "reduction_pct_vs_zip": reduction_pct(zip_reference_bytes, payload_bytes),
        "reduction_pct_vs_raw_frames": reduction_pct(raw_reference_bytes, payload_bytes),
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
    methods = as_dict(as_dict(report).get("methods"))
    lines = []
    for method_key in METHOD_ORDER:
        item = as_dict(methods.get(method_key))
        display_name = str(item.get("display_name") or DISPLAY_NAMES.get(method_key, method_key))
        status = str(item.get("status") or ("missing" if not item else "n/a"))
        line = "  - {}: status={} payload={} enc={} dec={}".format(
            display_name,
            status,
            format_mib_from_report(item),
            format_seconds(item.get("enc_time_sec")),
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
