#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from _eval.traj_eval import add_traj_eval_args, run_traj_eval


def _parse_args():
    parser = argparse.ArgumentParser()
    add_traj_eval_args(parser)
    return parser.parse_args()


def main():
    run_traj_eval(_parse_args(), emit_terminal_summary=True)


if __name__ == "__main__":
    main()
