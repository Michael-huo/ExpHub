#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deprecated compatibility shim for the legacy Wan A14B infer script.

Real infer backend logic now lives in scripts/_infer/backends/wan_fun_a14b_inp_backend.py.
This file is retained only so historical entrypoints can continue to work.
"""

from __future__ import annotations

from _infer.backends.wan_fun_a14b_inp_backend import run_compat_cli


def main():
    # type: () -> None
    run_compat_cli()


if __name__ == "__main__":
    main()
