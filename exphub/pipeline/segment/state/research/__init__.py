#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


_EXPORTS = {
    "compute_motion_rows": ".motion_energy",
    "compute_semantic_rows": ".semantic_openclip",
    "compute_frame_signal_rows": ".signals",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
