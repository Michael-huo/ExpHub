from __future__ import annotations


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_STEP = "\033[1;36m"
_STEP_SEPARATOR = "=" * 70
_CLI_LOG_LEVEL = "info"


def set_cli_log_level(level):
    global _CLI_LOG_LEVEL
    value = str(level or "info").strip().lower()
    if value not in ("info", "quiet"):
        value = "info"
    _CLI_LOG_LEVEL = value


def log_info(msg):
    print("[INFO] {}".format(msg))


def runtime_info(msg):
    if _CLI_LOG_LEVEL != "quiet":
        log_info(msg)


def log_prog(msg):
    print("[PROG] {}".format(msg))


def log_warn(msg):
    print("[WARN] {}".format(msg))


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
