from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from exphub.config import get_phase_python_config


_ANSI_RESET = "\033[0m"
_ANSI_STEP = "\033[1;36m"
_ANSI_ERR = "\033[31m"
_ANSI_WARN = "\033[33m"


def _which(cmd):
    from shutil import which

    return which(cmd)


def _python_cmd_exists(cmd) -> bool:
    if not cmd:
        return False
    text = str(cmd).strip()
    if not text:
        return False
    if os.path.isabs(text) or os.sep in text:
        path = Path(text).expanduser()
        return path.is_file() and os.access(str(path), os.X_OK)
    return bool(_which(text))


def resolve_phase_python(phase_name, exphub_root=None) -> str:
    python_bin = get_phase_python_config(phase_name, exphub_root=exphub_root)
    if not python_bin:
        raise RuntimeError("Missing 'environments.phases.{}.python' in config/platform.yaml.".format(phase_name))
    if not _python_cmd_exists(python_bin):
        raise RuntimeError("Configured python for phase '{}' not found or not executable: {}".format(phase_name, python_bin))
    return python_bin


class RunError(RuntimeError):
    def __init__(self, message, returncode=None, cmd=None, log_path=None, tail_lines=None):
        super().__init__(message)
        self.returncode = returncode
        self.cmd = list(cmd) if cmd is not None else None
        self.log_path = log_path
        self.tail_lines = list(tail_lines) if tail_lines else []


@dataclass
class RunnerConfig:
    auto_conda: bool
    conda_base: Optional[Path]
    ros_setup: Optional[Path]


class StepRunner:
    def __init__(self, logs_dir, log_level, runner_cfg, pass_prefixes=("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]"), fail_tail_lines=30):
        self.logs_dir = logs_dir
        self.log_level = log_level
        self.runner_cfg = runner_cfg
        self.pass_prefixes = tuple(pass_prefixes)
        self.fail_tail_lines = int(fail_tail_lines) if int(fail_tail_lines) > 0 else 30
        self._log_opened: Dict[str, bool] = {}

    def _cmd_log_kwargs(self, log_name):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / log_name
        append = bool(self._log_opened.get(log_name, False))
        self._log_opened[log_name] = True
        return {
            "log_path": log_path,
            "log_level": self.log_level,
            "pass_prefixes": self.pass_prefixes,
            "fail_tail_lines": self.fail_tail_lines,
            "log_append": append,
            "stderr": subprocess.STDOUT,
        }

    def run_env_python(self, cmd, phase_name, log_name, cwd=None, check=True, extra_env=None, stream_mode="filtered"):
        new_cmd = build_env_python_cmd(cmd, phase_name)
        return run_cmd(
            new_cmd,
            cwd=cwd,
            env=extra_env,
            check=check,
            display_phase_name=phase_name,
            stream_mode=stream_mode,
            **self._cmd_log_kwargs(log_name)
        )

    def run_ros(self, cmd, log_name, cwd, check=True):
        return ros_exec(cmd, cfg=self.runner_cfg, cwd=cwd, check=check, **self._cmd_log_kwargs(log_name))


def _looks_like_python_cmd(cmd) -> bool:
    if cmd is None:
        return False
    text = str(cmd).strip()
    if not text:
        return False
    return os.path.basename(text).startswith("python")


def _python_cmd_identity(cmd) -> str:
    text = str(cmd).strip()
    if not text:
        return ""
    if os.path.isabs(text) or os.sep in text:
        return os.path.abspath(os.path.expanduser(text))
    return text


def build_env_python_cmd(cmd, phase_name):
    python_bin = resolve_phase_python(phase_name)
    new_cmd = list(cmd or [])
    if not new_cmd:
        raise RuntimeError("run_env_python requires a non-empty command argv")
    if len(new_cmd) >= 2 and _looks_like_python_cmd(new_cmd[0]) and _looks_like_python_cmd(new_cmd[1]):
        first_id = _python_cmd_identity(new_cmd[0])
        second_id = _python_cmd_identity(new_cmd[1])
        if first_id and second_id and first_id == second_id:
            raise RuntimeError("duplicate python executable in argv for phase '{}': {}".format(phase_name, first_id))
    if _looks_like_python_cmd(new_cmd[0]):
        new_cmd[0] = python_bin
    else:
        new_cmd.insert(0, python_bin)
    return new_cmd


def detect_conda_base():
    if not _which("conda"):
        return None
    try:
        output = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        if output:
            return Path(output).resolve()
    except Exception:
        return None
    return None


def _phase_name_from_log_path(log_path) -> str:
    if log_path is None:
        return ""
    stem = str(log_path.stem or "").strip().lower()
    if stem.startswith("slam_"):
        return "slam"
    return stem


def _normalize_display_phase_name(phase_name) -> str:
    text = str(phase_name or "").strip().lower()
    if not text:
        return ""
    if text.startswith("infer"):
        return "infer"
    if text.startswith("prompt"):
        return "prompt"
    if text.startswith("slam"):
        return "slam"
    return text


def _strip_log_prefix(text: str) -> str:
    stripped = text.lstrip()
    for prefix in ("[INFO]", "[PROG]", "[WARN]", "[ERR]", "[STEP]", "[BAR]", "[PROMPT]"):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def _should_emit_bar_terminal_line(phase_name: str) -> bool:
    return phase_name in ("segment", "infer", "slam")


def _should_emit_prompt_terminal_line(phase_name: str, payload: str) -> bool:
    if phase_name != "infer":
        return False
    return "source=" in payload


def _should_emit_prog_terminal_line(payload: str) -> bool:
    return payload.startswith(
        (
            "scene split summary:",
            "prompt runtime plan assembled from",
            "image gen config:",
            "slam summary:",
            "merge summary:",
        )
    )


def _should_emit_info_terminal_line(phase_name: str, payload: str) -> bool:
    if phase_name == "prompt":
        return payload.startswith("processor loaded in") or payload.startswith("model weights loaded in")
    if phase_name == "infer":
        return payload.startswith(
            (
                "image gen detail:",
                "initializing model pipeline and loading weights from disk",
                "starting float8 quantization",
                "quantizing transformer 1/2",
                "quantizing transformer 2/2",
                "transformer 1/2 quantized in",
                "transformer 2/2 quantized in",
                "initialization completed in",
                "report written:",
                "done: segments=",
            )
        ) or (payload.startswith("seg ") and " infer=" in payload)
    if phase_name == "eval":
        return payload.startswith("eval summary:")
    if phase_name == "slam":
        return payload.startswith("slam tracking start:")
    return False


def _should_emit_terminal_line(phase_name: str, stripped: str) -> bool:
    payload = _strip_log_prefix(stripped).lower()
    if stripped.startswith("[STEP]") or stripped.startswith("[WARN]") or stripped.startswith("[ERR]"):
        return True
    if stripped.startswith("[BAR]"):
        return _should_emit_bar_terminal_line(phase_name)
    if stripped.startswith("[PROMPT]"):
        return _should_emit_prompt_terminal_line(phase_name, payload)
    if stripped.startswith("[PROG]"):
        return _should_emit_prog_terminal_line(payload)
    if not stripped.startswith("[INFO]"):
        return False
    return _should_emit_info_terminal_line(phase_name, payload)


def run_cmd(
    argv,
    cwd=None,
    env=None,
    check=True,
    log_path=None,
    log_level="info",
    pass_prefixes=None,
    fail_tail_lines=30,
    log_append=False,
    stderr=None,
    display_phase_name=None,
    stream_mode="filtered",
):
    level = (log_level or "info").strip().lower()
    if level not in ("info", "quiet"):
        level = "info"
    stream_mode_norm = str(stream_mode or "filtered").strip().lower()
    if stream_mode_norm not in ("filtered", "tee"):
        stream_mode_norm = "filtered"
    prefixes = tuple(pass_prefixes) if pass_prefixes else ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]")
    tail_cap = int(fail_tail_lines) if fail_tail_lines and int(fail_tail_lines) > 0 else 30
    tail = deque(maxlen=tail_cap)
    return_code = -1
    bar_line_active = False
    phase_name = _normalize_display_phase_name(display_phase_name or _phase_name_from_log_path(log_path))
    last_bar_display = ""
    last_bar_completed = False

    def _colorize_terminal_line(line):
        stripped = line.lstrip()
        if stripped.startswith("[STEP]"):
            return "{}{}{}".format(_ANSI_STEP, line, _ANSI_RESET)
        if stripped.startswith("[ERR]"):
            return "{}{}{}".format(_ANSI_ERR, line, _ANSI_RESET)
        if stripped.startswith("[WARN]"):
            return "{}{}{}".format(_ANSI_WARN, line, _ANSI_RESET)
        return line

    def _flush_bar_line():
        nonlocal bar_line_active, last_bar_display, last_bar_completed
        if not bar_line_active:
            last_bar_display = ""
            last_bar_completed = False
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        bar_line_active = False
        last_bar_display = ""
        last_bar_completed = False

    def _emit_bar_line(line):
        nonlocal bar_line_active, last_bar_display, last_bar_completed
        display = line.lstrip()
        if not display:
            return
        is_complete = "100%" in display
        if is_complete and last_bar_completed:
            return
        sys.stdout.write("\r\033[K{}".format(display))
        last_bar_display = display
        last_bar_completed = is_complete
        if is_complete:
            sys.stdout.write("\n")
            bar_line_active = False
        else:
            bar_line_active = True
        sys.stdout.flush()

    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(str(log_path), "a" if log_append else "w", encoding="utf-8")

        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if stderr is None else stderr,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if log_handle is not None:
                log_handle.write(raw_line)
                log_handle.flush()
            if level == "quiet":
                continue
            if stream_mode_norm == "tee":
                sys.stdout.write(raw_line)
                sys.stdout.flush()
                if any(line.lstrip().startswith(prefix) for prefix in prefixes):
                    tail.append(line)
                continue
            stripped = line.lstrip()
            if any(stripped.startswith(prefix) for prefix in prefixes):
                tail.append(line)
            if not _should_emit_terminal_line(phase_name, stripped):
                continue
            if stripped.startswith("[BAR]"):
                _emit_bar_line(stripped)
                continue
            _flush_bar_line()
            print(_colorize_terminal_line(line))

        return_code = process.wait()
    finally:
        if log_handle is not None:
            log_handle.flush()
        if log_handle is not None:
            log_handle.close()
        if stream_mode_norm == "filtered":
            _flush_bar_line()

    if check and return_code != 0:
        raise RunError(
            "command failed with exit code {}".format(return_code),
            returncode=return_code,
            cmd=argv,
            log_path=log_path,
            tail_lines=list(tail),
        )
    return return_code


def run_in_bash_login(cmd_text, cwd=None, env=None, check=True, **kwargs):
    argv = ["bash", "-lc", str(cmd_text)]
    return run_cmd(argv, cwd=cwd, env=env, check=check, **kwargs)


def conda_exec(cmd, env_name, cfg, cwd=None, check=True, **kwargs):
    conda_base = cfg.conda_base if cfg is not None else None
    if not conda_base:
        raise RuntimeError("conda base not available")
    conda_hook = Path(conda_base).resolve() / "etc" / "profile.d" / "conda.sh"
    cmd_text = "source {} && conda activate {} && {}".format(
        shlex.quote(str(conda_hook)),
        shlex.quote(str(env_name)),
        " ".join(shlex.quote(str(item)) for item in cmd),
    )
    return run_in_bash_login(cmd_text, cwd=cwd, check=check, **kwargs)


def ros_exec(cmd, cfg, cwd=None, check=True, **kwargs):
    ros_setup = cfg.ros_setup if cfg is not None else None
    command_text = " ".join(shlex.quote(str(item)) for item in cmd)
    if ros_setup:
        command_text = "source {} && {}".format(shlex.quote(str(ros_setup)), command_text)
    return run_in_bash_login(command_text, cwd=cwd, check=check, **kwargs)


__all__ = [
    "RunError",
    "RunnerConfig",
    "StepRunner",
    "build_env_python_cmd",
    "conda_exec",
    "detect_conda_base",
    "resolve_phase_python",
    "ros_exec",
    "run_cmd",
    "run_in_bash_login",
]
