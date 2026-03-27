#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


_EXPORTS = {
    "DEFAULT_PLOT_SMOOTH_WINDOW": ".extract",
    "REPRESENTATIVE_SIGNALS": ".extract",
    "REPORT_SCHEMA_VERSION": ".extract",
    "SELECTED_SIGNALS": ".extract",
    "SIGNAL_FAMILIES": ".extract",
    "build_signal_report": ".extract",
    "build_signal_extraction_meta": ".extract",
    "extract_signal_timeseries": ".extract",
    "extract_signal_timeseries_from_frames": ".extract",
    "materialize_signal_extraction_outputs": ".extract",
    "write_signal_report": ".extract",
    "write_signal_extraction_meta": ".extract",
    "save_signal_overview": ".visualize",
    "save_signal_plots": ".visualize",
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
