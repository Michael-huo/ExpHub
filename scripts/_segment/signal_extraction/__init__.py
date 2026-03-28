#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


_EXPORTS = {
    "FORMAL_STATE_INPUT_COLUMNS": ".extract",
    "FORMAL_STATE_INPUT_PREPROCESSING": ".extract",
    "FORMAL_STATE_INPUT_SIGNALS": ".extract",
    "build_formal_state_input_rows": ".extract",
    "extract_signal_timeseries": ".extract",
    "extract_signal_timeseries_from_frames": ".extract",
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
