from __future__ import annotations

import os
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
) -> int:
    lvl = (log_level or "debug").strip().lower()
    if lvl not in ("debug", "info", "quiet"):
        lvl = "debug"

    prefixes = tuple(pass_prefixes) if pass_prefixes else ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]")
    tail_cap = int(fail_tail_lines) if fail_tail_lines and int(fail_tail_lines) > 0 else 30
    tail = deque(maxlen=tail_cap)
    rc = -1

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
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None

        for raw_line in proc.stdout:
            if lf is not None:
                lf.write(raw_line)
            line = raw_line.rstrip("\n")
            tail.append(line)

            if lvl == "quiet":
                continue
            if lvl == "debug":
                print(line)
                continue

            # info: only pass through key prefixed lines
            stripped = line.lstrip()
            for p in prefixes:
                if stripped.startswith(p):
                    print(line)
                    break

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
    )
