#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .peaks import DEFAULT_PEAK_CONFIG, annotate_peaks
from .scorer import DEFAULT_SCORE_WEIGHTS, apply_scores
from .signals import compute_frame_signal_rows
from .visualize import save_peaks_preview, save_score_curve, save_score_curve_with_keyframes

__all__ = [
    "DEFAULT_PEAK_CONFIG",
    "DEFAULT_SCORE_WEIGHTS",
    "annotate_peaks",
    "apply_scores",
    "compute_frame_signal_rows",
    "save_peaks_preview",
    "save_score_curve",
    "save_score_curve_with_keyframes",
]
