#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

from scripts._common import get_platform_config

TASK_SEGMENT = "segment"
TASK_SIGNAL_EXTRACT = "signal_extract"
TASK_STATE_SEGMENT = "state_segment"
TASK_ANALYZE = "analyze"
TASK_NAMES = (
    TASK_SEGMENT,
    TASK_SIGNAL_EXTRACT,
    TASK_STATE_SEGMENT,
    TASK_ANALYZE,
)


def _segment_phase_python():
    cfg = get_platform_config()
    phases_cfg = cfg.get("environments", {}).get("phases", {})
    if not isinstance(phases_cfg, dict):
        return ""
    phase_cfg = phases_cfg.get("segment", {})
    if not isinstance(phase_cfg, dict):
        return ""
    return str(phase_cfg.get("python", "") or "").strip()


def _same_python(current_python, target_python):
    try:
        current_path = Path(current_python).expanduser().resolve()
        target_path = Path(target_python).expanduser().resolve()
        return str(current_path) == str(target_path)
    except Exception:
        return str(current_python) == str(target_python)


def _maybe_reexec_in_segment_phase():
    target_python = _segment_phase_python()
    if not target_python:
        return
    if _same_python(sys.executable, target_python):
        return
    if not Path(target_python).expanduser().is_file():
        return
    os.execv(str(target_python), [str(target_python), str(sys.argv[0])] + list(sys.argv[1:]))


def _load_task_runner(task_name):
    if task_name == TASK_SEGMENT:
        from .make import run_segment_make_cli

        return run_segment_make_cli
    if task_name == TASK_SIGNAL_EXTRACT:
        from .signal_extraction.app import run_signal_extraction

        return run_signal_extraction
    if task_name == TASK_STATE_SEGMENT:
        from .state_segmentation.app import run_state_segmentation_cli

        return run_state_segmentation_cli
    from .analysis.app import run_segment_analyze

    return run_segment_analyze


def _build_task_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task",
        default=TASK_SEGMENT,
        choices=list(TASK_NAMES),
        help="segment internal task entry: segment | signal_extract(sidecar) | state_segment | analyze(sidecar)",
    )
    return parser


def run_cli(argv=None):
    _maybe_reexec_in_segment_phase()
    argv = list(sys.argv[1:] if argv is None else argv)
    task_args, remaining = _build_task_parser().parse_known_args(argv)
    task_name = str(task_args.task or TASK_SEGMENT)
    return _load_task_runner(task_name)(remaining)


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    main()
