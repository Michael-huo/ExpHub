from __future__ import annotations

import os
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import get_phase_python_config


_ANSI_RESET = "\033[0m"
_ANSI_STEP = "\033[1;36m"
_ANSI_ERR = "\033[31m"
_ANSI_WARN = "\033[33m"


def _python_cmd_exists(cmd):
    if not cmd:
        return False

    text = str(cmd).strip()
    if not text:
        return False

    if os.path.isabs(text) or os.sep in text:
        path = Path(text).expanduser()
        return path.is_file() and os.access(str(path), os.X_OK)

    return bool(_which(text))


def resolve_phase_python(phase_name, exphub_root=None):
    python_bin = get_phase_python_config(phase_name, exphub_root=exphub_root)
    if not python_bin:
        raise RuntimeError(
            "Missing 'environments.phases.{}.python' in config/platform.yaml.".format(phase_name)
        )
    if not _python_cmd_exists(python_bin):
        raise RuntimeError(
            "Configured python for phase '{}' not found or not executable: {}".format(
                phase_name, python_bin
            )
        )
    return python_bin


class RunError(RuntimeError):
    def __init__(
        self,
        message,
        returncode=None,
        cmd=None,
        log_path=None,
        tail_lines=None,
    ):
        RuntimeError.__init__(self, message)
        self.returncode = returncode
        self.cmd = list(cmd) if cmd is not None else None
        self.log_path = log_path
        self.tail_lines = list(tail_lines) if tail_lines else []


@dataclass
class EnvSpec:
    name: str


@dataclass
class RunnerConfig:
    auto_conda: bool
    conda_base: Optional[Path]
    ros_setup: Optional[Path]


class StepRunner:
    def __init__(
        self,
        logs_dir,
        log_level,
        runner_cfg,
        pass_prefixes=("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]"),
        fail_tail_lines=30,
    ):
        self.logs_dir = logs_dir
        self.log_level = log_level
        self.runner_cfg = runner_cfg
        self.pass_prefixes = tuple(pass_prefixes)
        self.fail_tail_lines = int(fail_tail_lines) if int(fail_tail_lines) > 0 else 30
        self._log_opened = {}  # type: Dict[str, bool]

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

    def run_conda(self, cmd, env_name, log_name, cwd, check=True):
        return conda_exec(
            cmd,
            env_name=env_name,
            cfg=self.runner_cfg,
            cwd=cwd,
            check=check,
            **self._cmd_log_kwargs(log_name)
        )

    def run_env_python(self, cmd, phase_name, log_name, cwd=None, check=True, extra_env=None):
        python_bin = resolve_phase_python(phase_name)

        new_cmd = list(cmd)
        if new_cmd and new_cmd[0] in ("python", "python3"):
            new_cmd[0] = python_bin
        elif new_cmd:
            new_cmd.insert(0, python_bin)

        return run_cmd(
            new_cmd,
            cwd=cwd,
            env=extra_env,
            check=check,
            **self._cmd_log_kwargs(log_name)
        )

    def run_ros(self, cmd, log_name, cwd, check=True):
        return ros_exec(
            cmd,
            cfg=self.runner_cfg,
            cwd=cwd,
            check=check,
            **self._cmd_log_kwargs(log_name)
        )


def _which(cmd):
    from shutil import which

    return which(cmd)


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


def _phase_name_from_log_path(log_path):
    if log_path is None:
        return ""
    stem = str(log_path.stem or "").strip().lower()
    if stem.startswith("slam_"):
        return "slam"
    return stem


def _strip_log_prefix(text):
    stripped = text.lstrip()
    for prefix in ("[INFO]", "[PROG]", "[WARN]", "[ERR]", "[STEP]", "[BAR]", "[PROMPT]"):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def _should_emit_info_terminal_line(phase_name, stripped):
    payload = _strip_log_prefix(stripped)
    payload_lower = payload.lower()

    if stripped.startswith("[STEP]") or stripped.startswith("[WARN]") or stripped.startswith("[ERR]"):
        return True

    if stripped.startswith("[PROG]"):
        if phase_name == "segment":
            return payload_lower.startswith("segment summary:")
        if phase_name == "prompt":
            return payload_lower.startswith("prompt profile generated from")
        if phase_name == "infer":
            return payload_lower.startswith("infer config:")
        if phase_name == "merge":
            return payload_lower.startswith("merge summary:")
        if phase_name == "slam":
            return payload_lower.startswith("slam summary:")
        if phase_name == "stats":
            return payload_lower.startswith("stats summary:")
        return False

    if not stripped.startswith("[INFO]"):
        return False

    if phase_name == "prompt":
        return payload_lower.startswith("processor loaded in") or payload_lower.startswith("model weights loaded in")

    if phase_name == "infer":
        return (
            payload_lower.startswith("initializing model pipeline and loading weights from disk")
            or payload_lower.startswith("starting float8 quantization")
            or payload_lower.startswith("quantizing transformer 1/2")
            or payload_lower.startswith("quantizing transformer 2/2")
            or payload_lower.startswith("transformer 1/2 quantized in")
            or payload_lower.startswith("transformer 2/2 quantized in")
            or payload_lower.startswith("initialization completed in")
            or (payload_lower.startswith("seg ") and " infer=" in payload_lower)
            or payload_lower.startswith("done: segments=")
        )

    if phase_name == "slam":
        return payload_lower.startswith("slam tracking start:")

    if phase_name == "eval":
        return payload_lower.startswith("eval summary:")

    return False


def run_cmd(
    argv,
    cwd=None,
    env=None,
    check=True,
    log_path=None,
    log_level="debug",
    pass_prefixes=None,
    fail_tail_lines=30,
    log_append=False,
    stderr=None,
):
    level = (log_level or "debug").strip().lower()
    if level not in ("debug", "info", "quiet"):
        level = "debug"

    prefixes = tuple(pass_prefixes) if pass_prefixes else ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]")
    tail_cap = int(fail_tail_lines) if fail_tail_lines and int(fail_tail_lines) > 0 else 30
    tail = deque(maxlen=tail_cap)
    return_code = -1
    bar_line_active = False
    phase_name = _phase_name_from_log_path(log_path)

    def _colorize_terminal_line(line):
        stripped = line.lstrip()
        if stripped.startswith("[STEP]"):
            return "{}{}{}".format(_ANSI_STEP, line, _ANSI_RESET)
        if stripped.startswith("[ERR]"):
            return "{}{}{}".format(_ANSI_ERR, line, _ANSI_RESET)
        if stripped.startswith("[WARN]"):
            return "{}{}{}".format(_ANSI_WARN, line, _ANSI_RESET)
        return line

    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if log_append else "w"
            log_handle = open(str(log_path), mode, encoding="utf-8")

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
            tail.append(line)
            stripped = line.lstrip()
            is_bar = ("[BAR]" in raw_line) or stripped.startswith("[BAR]")

            if log_handle is not None and not is_bar:
                log_handle.write(raw_line)

            if level == "quiet":
                if stripped.startswith("[STEP]") or stripped.startswith("[WARN]") or stripped.startswith("[ERR]"):
                    print(_colorize_terminal_line(line))
                continue
            if is_bar:
                print("\r{}".format(line), end="", flush=True)
                bar_line_active = True
                continue
            if bar_line_active:
                print("")
                bar_line_active = False
            if level == "debug":
                print(_colorize_terminal_line(line))
                continue

            if prefixes and _should_emit_info_terminal_line(phase_name, stripped):
                print(_colorize_terminal_line(line))

        if bar_line_active and level != "quiet":
            print("")
        return_code = process.wait()
        if log_handle is not None:
            log_handle.flush()
    finally:
        if log_handle is not None:
            log_handle.close()

    if check and return_code != 0:
        raise RunError(
            "Command failed ({}): {}".format(return_code, " ".join(map(shlex.quote, argv))),
            returncode=return_code,
            cmd=argv,
            log_path=log_path,
            tail_lines=list(tail),
        )
    return return_code


def run_in_bash_login(
    script,
    check=True,
    cwd=None,
    env=None,
    log_path=None,
    log_level="debug",
    pass_prefixes=None,
    fail_tail_lines=30,
    log_append=False,
    stderr=None,
):
    return run_cmd(
        ["bash", "-lc", script],
        cwd=cwd,
        env=env,
        check=check,
        log_path=log_path,
        log_level=log_level,
        pass_prefixes=pass_prefixes,
        fail_tail_lines=fail_tail_lines,
        log_append=log_append,
        stderr=stderr,
    )


def conda_exec(
    cmd,
    env_name,
    cfg,
    cwd=None,
    extra_env=None,
    check=True,
    log_path=None,
    log_level="debug",
    pass_prefixes=None,
    fail_tail_lines=30,
    log_append=False,
    stderr=None,
):
    if not cfg.auto_conda:
        return run_cmd(
            cmd,
            cwd=cwd,
            env=extra_env,
            check=check,
            log_path=log_path,
            log_level=log_level,
            pass_prefixes=pass_prefixes,
            fail_tail_lines=fail_tail_lines,
            log_append=log_append,
            stderr=stderr,
        )

    if cfg.conda_base is None:
        raise RunError("auto_conda enabled but conda base not found. Ensure `conda` in PATH.")

    conda_sh = cfg.conda_base / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise RunError("conda.sh not found: {}".format(conda_sh))

    cmd_text = " ".join(shlex.quote(str(item)) for item in cmd)
    exports = ""
    if extra_env:
        exports = " ".join("{}={}".format(key, shlex.quote(value)) for key, value in extra_env.items()) + " "

    script = "source {} && conda activate {} && {}{}".format(
        shlex.quote(str(conda_sh)),
        shlex.quote(env_name),
        exports,
        cmd_text,
    )
    return run_in_bash_login(
        script,
        check=check,
        cwd=cwd,
        log_path=log_path,
        log_level=log_level,
        pass_prefixes=pass_prefixes,
        fail_tail_lines=fail_tail_lines,
        log_append=log_append,
        stderr=stderr,
    )


def ros_exec(
    cmd,
    cfg,
    cwd=None,
    extra_env=None,
    check=True,
    log_path=None,
    log_level="debug",
    pass_prefixes=None,
    fail_tail_lines=30,
    log_append=False,
    stderr=None,
):
    cmd_text = " ".join(shlex.quote(str(item)) for item in cmd)
    exports = ""
    if extra_env:
        exports = " ".join("{}={}".format(key, shlex.quote(value)) for key, value in extra_env.items()) + " "

    if cfg.ros_setup and cfg.ros_setup.exists():
        script = "source {} && {}{}".format(shlex.quote(str(cfg.ros_setup)), exports, cmd_text)
    else:
        script = "{}{}".format(exports, cmd_text)
    return run_in_bash_login(
        script,
        check=check,
        cwd=cwd,
        log_path=log_path,
        log_level=log_level,
        pass_prefixes=pass_prefixes,
        fail_tail_lines=fail_tail_lines,
        log_append=log_append,
        stderr=stderr,
    )
