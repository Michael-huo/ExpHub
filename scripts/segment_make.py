#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from _segment.app import main


if __name__ == "__main__":
    main()
