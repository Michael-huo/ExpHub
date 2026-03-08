#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable entrypoint for segment extraction."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._segment.api import run_segment_make


if __name__ == "__main__":
    run_segment_make()
