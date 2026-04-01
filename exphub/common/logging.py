from __future__ import annotations

import sys


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_STEP = "\033[1;36m"
_STEP_SEPARATOR = "=" * 70
_CLI_LOG_LEVEL = "info"


def set_cli_log_level(level):
    global _CLI_LOG_LEVEL
    value = str(level or "info").strip().lower()
    if value not in ("info", "debug", "quiet"):
        value = "info"
    _CLI_LOG_LEVEL = value


def get_cli_log_level():
    return _CLI_LOG_LEVEL


def log_info(msg):
    print("[INFO] {}".format(msg))


def runtime_info(msg):
    if _CLI_LOG_LEVEL != "quiet":
        log_info(msg)


def debug_info(msg):
    if _CLI_LOG_LEVEL == "debug":
        log_info(msg)


def log_run(msg):
    print("[RUN] {}".format(msg))


def log_prog(msg):
    print("[PROG] {}".format(msg))


def log_prompt(msg):
    print("[PROMPT] {}".format(msg))


def log_warn(msg):
    print("[WARN] {}".format(msg))


def log_err(msg):
    print("[ERR] {}".format(msg), file=sys.stderr)


def log_step(msg):
    line = "{}[STEP] {}{}".format(_ANSI_STEP, msg, _ANSI_RESET)
    sep = "{}{}{}".format(_ANSI_BOLD, _STEP_SEPARATOR, _ANSI_RESET)
    lower_msg = str(msg).strip().lower()
    is_start = (" start " in lower_msg) or lower_msg.endswith(" start")
    is_done = (" done " in lower_msg) or lower_msg.endswith(" done")
    is_fail = (" fail " in lower_msg) or lower_msg.endswith(" fail")

    if is_start:
        print(sep)
        print(line)
        print(sep)
        return

    print(line)
    if is_done or is_fail:
        print(sep)


def die(msg):
    raise SystemExit("[ERR] {}".format(msg))
