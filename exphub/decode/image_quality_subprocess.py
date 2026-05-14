from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from exphub.common.io import ensure_file, read_json_dict
from exphub.common.logging import log_warn
from exphub.config import get_phase_python_config


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _format_metric(value):
    try:
        return "{:.6f}".format(float(value))
    except Exception:
        return "nan"


def _python_exists(cmd):
    text = str(cmd or "").strip()
    if not text:
        return False
    if os.path.isabs(text) or os.sep in text:
        path = Path(text).expanduser()
        return path.is_file() and os.access(str(path), os.X_OK)
    return shutil.which(text) is not None


def _python_display(cmd):
    text = str(cmd or "").strip()
    if not text:
        return ""
    if os.path.isabs(text) or os.sep in text:
        return str(Path(text).expanduser())
    resolved = shutil.which(text)
    return str(Path(resolved).resolve()) if resolved else text


def _resolve_decode_python(runtime):
    configured = get_phase_python_config("decode", exphub_root=runtime.exphub_root)
    if configured and _python_exists(configured):
        return _python_display(configured)
    if configured:
        log_warn(
            "decode image quality python is configured but not executable: {}; "
            "falling back to current interpreter.".format(configured)
        )
    else:
        log_warn("decode image quality python is not configured; falling back to current interpreter.")
    return sys.executable


def _print_terminal_summary(report, python_bin, exp_dir):
    outputs = _as_dict(report.get("outputs"))
    report_path = outputs.get("report") or "decode/image_quality_report.json"
    print("[Image Quality]")
    print("python            : {}".format(str(python_bin)))
    print("matched_frames    : {}".format(int(report.get("frame_count_matched", 0) or 0)))
    print("evaluated_frames  : {}".format(int(report.get("frame_count_evaluated", 0) or 0)))
    print("lpips_mean        : {}".format(_format_metric(_as_dict(report.get("lpips")).get("mean"))))
    print("ssim_mean         : {}".format(_format_metric(_as_dict(report.get("ssim")).get("mean"))))
    print("fid               : {}".format(_format_metric(report.get("fid"))))
    print("report            : {}".format(str(report_path)))


def _subprocess_env(exphub_root):
    env = dict(os.environ)
    root = str(Path(exphub_root).resolve())
    current = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = root if not current else root + os.pathsep + current
    return env


def run_decode_image_quality_subprocess(runtime):
    if not bool(getattr(runtime.args, "decode_image_quality", False)):
        return None

    python_bin = _resolve_decode_python(runtime)
    cmd = [
        python_bin,
        "-m",
        "exphub.decode.image_quality_cli",
        "--run-root",
        str(runtime.paths.exp_dir),
        "--prepare-result",
        str(runtime.paths.prepare_result_path),
        "--decode-merge-report",
        str(runtime.paths.decode_merge_report_path),
        "--output-report",
        str(runtime.paths.decode_image_quality_report_path),
        "--output-summary",
        str(runtime.paths.decode_image_quality_summary_path),
        "--output-details-csv",
        str(runtime.paths.decode_image_quality_details_path),
        "--stride",
        str(int(getattr(runtime.args, "decode_image_quality_stride", 1))),
        "--max-frames",
        str(int(getattr(runtime.args, "decode_image_quality_max_frames", 0))),
        "--device",
        str(getattr(runtime.args, "decode_image_quality_device", "auto") or "auto"),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(Path(runtime.exphub_root).resolve()),
        env=_subprocess_env(runtime.exphub_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stdout = str(proc.stdout or "").strip()
        stderr = str(proc.stderr or "").strip()
        message = [
            "decode image quality subprocess failed with exit code {}".format(int(proc.returncode)),
            "",
            "decode image quality was executed with:",
            str(python_bin),
            "",
            "command:",
            " ".join(shlex.quote(str(item)) for item in cmd),
        ]
        if stdout:
            message.extend(["", "stdout:", stdout])
        if stderr:
            message.extend(["", "stderr:", stderr])
        message.extend(
            [
                "",
                "Install missing optional packages in the decode Python environment, or configure "
                "environments.phases.decode.python correctly. This optional evaluation is only required "
                "when --decode_image_quality is enabled.",
            ]
        )
        raise RuntimeError("\n".join(message))

    report_path = ensure_file(runtime.paths.decode_image_quality_report_path, "decode image quality report")
    report = read_json_dict(report_path)
    if not report:
        raise RuntimeError("invalid decode image quality report: {}".format(report_path))
    _print_terminal_summary(report, python_bin, runtime.paths.exp_dir)
    return report
