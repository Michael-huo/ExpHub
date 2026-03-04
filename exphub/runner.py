from __future__ import annotations

import os
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_ANSI_RESET = "\033[0m"
_ANSI_STEP = "\033[1;36m"
_ANSI_ERR = "\033[31m"
_ANSI_WARN = "\033[33m"


class RunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        returncode: Optional[int] = None,
        cmd: Optional[Sequence[str]] = None,
        log_path: Optional[Path] = None,
        tail_lines: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
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
        logs_dir: Path,
        log_level: str,
        runner_cfg: RunnerConfig,
        pass_prefixes: Tuple[str, ...] = ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]"),
        fail_tail_lines: int = 30,
    ) -> None:
        self.logs_dir = logs_dir
        self.log_level = log_level
        self.runner_cfg = runner_cfg
        self.pass_prefixes = tuple(pass_prefixes)
        self.fail_tail_lines = int(fail_tail_lines) if int(fail_tail_lines) > 0 else 30
        self._log_opened = {}  # type: Dict[str, bool]

    def _cmd_log_kwargs(self, log_name: str) -> Dict[str, object]:
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

    def run_conda(
        self,
        cmd: Sequence[str],
        env_name: str,
        log_name: str,
        cwd: Optional[Path],
        check: bool = True,
    ) -> int:
        return conda_exec(
            cmd,
            env_name=env_name,
            cfg=self.runner_cfg,
            cwd=cwd,
            check=check,
            **self._cmd_log_kwargs(log_name),
        )

    def run_ros(
        self,
        cmd: Sequence[str],
        log_name: str,
        cwd: Optional[Path],
        check: bool = True,
    ) -> int:
        return ros_exec(
            cmd,
            cfg=self.runner_cfg,
            cwd=cwd,
            check=check,
            **self._cmd_log_kwargs(log_name),
        )


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def detect_conda_base() -> Optional[Path]:
    if not _which("conda"):
        return None
    try:
        out = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        if out:
            return Path(out).resolve()
    except Exception:
        return None
    return None


def run_cmd(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    log_path: Optional[Path] = None,
    log_level: str = "debug",
    pass_prefixes: Optional[Sequence[str]] = None,
    fail_tail_lines: int = 30,
    log_append: bool = False,
    stderr: Optional[int] = None,
) -> int:
    lvl = (log_level or "debug").strip().lower()
    if lvl not in ("debug", "info", "quiet"):
        lvl = "debug"

    prefixes = tuple(pass_prefixes) if pass_prefixes else ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]")
    tail_cap = int(fail_tail_lines) if fail_tail_lines and int(fail_tail_lines) > 0 else 30
    tail = deque(maxlen=tail_cap)
    rc = -1
    bar_line_active = False

    def _colorize_terminal_line(line: str) -> str:
        stripped = line.lstrip()
        if stripped.startswith("[STEP]"):
            return "{}{}{}".format(_ANSI_STEP, line, _ANSI_RESET)
        if stripped.startswith("[ERR]"):
            return "{}{}{}".format(_ANSI_ERR, line, _ANSI_RESET)
        if stripped.startswith("[WARN]"):
            return "{}{}{}".format(_ANSI_WARN, line, _ANSI_RESET)
        return line

    lf = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if log_append else "w"
            lf = open(str(log_path), mode, encoding="utf-8")

        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if stderr is None else stderr,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None

        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            tail.append(line)
            stripped = line.lstrip()
            is_bar = ("[BAR]" in raw_line) or stripped.startswith("[BAR]")

            if lf is not None and not is_bar:
                lf.write(raw_line)

            if lvl == "quiet":
                continue
            if is_bar:
                print("\r{}".format(line), end="", flush=True)
                bar_line_active = True
                continue
            if bar_line_active:
                print("")
                bar_line_active = False
            if lvl == "debug":
                print(_colorize_terminal_line(line))
                continue

            # info: only pass through key prefixed lines
            for p in prefixes:
                if stripped.startswith(p):
                    print(_colorize_terminal_line(line))
                    break

        if bar_line_active and lvl != "quiet":
            print("")
        rc = proc.wait()
        if lf is not None:
            lf.flush()
    finally:
        if lf is not None:
            lf.close()

    if check and rc != 0:
        raise RunError(
            "Command failed ({}): {}".format(rc, " ".join(map(shlex.quote, argv))),
            returncode=rc,
            cmd=argv,
            log_path=log_path,
            tail_lines=list(tail),
        )
    return rc


def run_in_bash_login(
    script: str,
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    log_level: str = "debug",
    pass_prefixes: Optional[Sequence[str]] = None,
    fail_tail_lines: int = 30,
    log_append: bool = False,
    stderr: Optional[int] = None,
) -> int:
    """Run a command string in `bash -lc`."""
    argv = ["bash", "-lc", script]
    return run_cmd(
        argv,
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
    cmd: Sequence[str],
    *,
    env_name: str,
    cfg: RunnerConfig,
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
    log_path: Optional[Path] = None,
    log_level: str = "debug",
    pass_prefixes: Optional[Sequence[str]] = None,
    fail_tail_lines: int = 30,
    log_append: bool = False,
    stderr: Optional[int] = None,
) -> int:
    """Execute cmd within conda env using bash -lc + conda activate.

    This mimics manual activation (more compatible than `conda run` on some setups).
    """

    if not cfg.auto_conda:
        # Assume user has activated correct env already.
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
        raise RunError(f"conda.sh not found: {conda_sh}")

    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    exports = ""
    if extra_env:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in extra_env.items()) + " "

    script = f"source {shlex.quote(str(conda_sh))} && conda activate {shlex.quote(env_name)} && {exports}{cmd_str}"
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
    cmd: Sequence[str],
    *,
    cfg: RunnerConfig,
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
    log_path: Optional[Path] = None,
    log_level: str = "debug",
    pass_prefixes: Optional[Sequence[str]] = None,
    fail_tail_lines: int = 30,
    log_append: bool = False,
    stderr: Optional[int] = None,
) -> int:
    """Execute cmd with ROS setup sourced (bash -lc)."""

    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    exports = ""
    if extra_env:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in extra_env.items()) + " "

    if cfg.ros_setup and cfg.ros_setup.exists():
        script = f"source {shlex.quote(str(cfg.ros_setup))} && {exports}{cmd_str}"
    else:
        script = f"{exports}{cmd_str}"
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
