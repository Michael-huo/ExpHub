#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._segment.research.kinematics import minmax_normalize, moving_average

from .extract import DEFAULT_PLOT_SMOOTH_WINDOW, REPRESENTATIVE_SIGNALS, SIGNAL_FAMILIES


_FAMILY_TITLES = {
    "image": "Signal Extraction: Image Family",
    "motion": "Signal Extraction: Motion Family",
    "semantic": "Signal Extraction: Semantic Family",
}

_FAMILY_OUTPUT_NAMES = {
    "image": "signal_image_family.png",
    "motion": "signal_motion_family.png",
    "semantic": "signal_semantic_family.png",
}

_SIGNAL_LABELS = {
    "feature_motion": "feature_motion",
    "appearance_delta": "appearance_delta",
    "brightness_jump": "brightness_jump",
    "blur_score": "blur_score",
    "motion_displacement": "motion_displacement",
    "motion_velocity": "motion_velocity",
    "motion_acceleration": "motion_acceleration",
    "semantic_delta": "semantic_delta",
    "semantic_velocity": "semantic_velocity",
    "semantic_acceleration": "semantic_acceleration",
}

_SIGNAL_COLORS = {
    "feature_motion": "#1f77b4",
    "appearance_delta": "#ff7f0e",
    "brightness_jump": "#2ca02c",
    "blur_score": "#d62728",
    "motion_displacement": "#9467bd",
    "motion_velocity": "#8c564b",
    "motion_acceleration": "#e377c2",
    "semantic_delta": "#17becf",
    "semantic_velocity": "#bcbd22",
    "semantic_acceleration": "#7f7f7f",
}


def _series(rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _frame_indices(rows):
    return [int(row.get("frame_idx", 0)) for row in rows]


def _plot_values(values, smooth_window):
    smooth_values, actual_window = moving_average(values, smooth_window)
    norm_values = minmax_normalize(smooth_values)
    return norm_values, actual_window


def _save_signal_group_plot(rows, signal_names, output_path, title, smooth_window):
    x = _frame_indices(rows)
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    actual_window = 1

    for signal_name in signal_names:
        values = _series(rows, signal_name)
        plot_values, actual_window = _plot_values(values, smooth_window)
        ax.plot(
            x,
            plot_values,
            linewidth=1.9,
            label=_SIGNAL_LABELS.get(signal_name, signal_name),
            color=_SIGNAL_COLORS.get(signal_name),
        )

    ax.set_title(title)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("normalized magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    return actual_window


def save_signal_plots(output_dir, rows, smooth_window=DEFAULT_PLOT_SMOOTH_WINDOW):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_window = 1
    for family_name in ("image", "motion", "semantic"):
        actual_window = _save_signal_group_plot(
            rows=rows,
            signal_names=SIGNAL_FAMILIES[family_name],
            output_path=output_dir / _FAMILY_OUTPUT_NAMES[family_name],
            title=_FAMILY_TITLES[family_name],
            smooth_window=smooth_window,
        )

    representatives_window = _save_signal_group_plot(
        rows=rows,
        signal_names=REPRESENTATIVE_SIGNALS,
        output_path=output_dir / "signal_representatives.png",
        title="Signal Extraction: Representative Signals",
        smooth_window=smooth_window,
    )
    actual_window = max(int(actual_window), int(representatives_window))

    return {
        "smoothing_used_for_plot": {
            "enabled": True,
            "method": "moving_average",
            "window_size": int(actual_window),
        },
        "normalization_used_for_plot": {
            "enabled": True,
            "method": "minmax_per_signal",
            "scope": "each plotted signal independently",
        },
    }
