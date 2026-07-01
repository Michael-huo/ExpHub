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


def _subprocess_env(exphub_root):
    env = dict(os.environ)
    root = str(Path(exphub_root).resolve())
    current = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = root if not current else root + os.pathsep + current
    return env


def run_decode_image_quality_subprocess(runtime, *, stride: int = 1, max_frames: int = 0, device: str = "auto"):
    python_bin = _resolve_decode_python(runtime)
    cmd = [
        python_bin,
        "-m",
        "exphub.decode.image_quality_cli",
        "--run-root",
        str(runtime.paths.exp_dir),
        "--prepare-result",
        str(runtime.paths.prepare_result_path),
        "--decode-report",
        str(runtime.paths.decode_report_path),
        "--output-report",
        str(runtime.paths.decode_image_quality_report_path),
        "--output-details-csv",
        str(runtime.paths.decode_image_quality_details_path),
        "--output-summary-json",
        str(runtime.paths.decode_image_quality_canonical_json_path),
        "--output-summary-csv",
        str(runtime.paths.decode_image_quality_canonical_csv_path),
        "--dataset",
        str(runtime.config.dataset),
        "--sequence",
        str(runtime.config.sequence),
        "--tag",
        str(runtime.config.tag),
        "--decode-profile",
        str(runtime.config.decode_profile),
        "--stride",
        str(int(stride)),
        "--max-frames",
        str(int(max_frames)),
        "--device",
        str(device or "auto"),
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
                "when --experiments image-quality is requested.",
            ]
        )
        raise RuntimeError("\n".join(message))

    report_path = ensure_file(runtime.paths.decode_image_quality_report_path, "decode image quality report")
    ensure_file(runtime.paths.decode_image_quality_details_path, "decode image quality details")
    ensure_file(runtime.paths.decode_image_quality_canonical_json_path, "decode image quality canonical summary")
    ensure_file(runtime.paths.decode_image_quality_canonical_csv_path, "decode image quality canonical summary csv")
    report = read_json_dict(report_path)
    if not report:
        raise RuntimeError("invalid decode image quality report: {}".format(report_path))
    return report
