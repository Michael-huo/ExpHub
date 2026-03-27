#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._segment.research.kinematics import minmax_normalize, moving_average

from .extract import DEFAULT_PLOT_SMOOTH_WINDOW, REPRESENTATIVE_SIGNALS, SIGNAL_FAMILIES


_FAMILY_TITLES = {
    "image": "Image Family",
    "motion": "Motion Family",
    "semantic": "Semantic Family",
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


def _plot_group(ax, rows, signal_names, title, smooth_window):
    x = _frame_indices(rows)
    actual_window = 1

    for signal_name in signal_names:
        values = _series(rows, signal_name)
        plot_values, actual_window = _plot_values(values, smooth_window)
        ax.plot(
            x,
            plot_values,
            linewidth=1.8,
            label=_SIGNAL_LABELS.get(signal_name, signal_name),
            color=_SIGNAL_COLORS.get(signal_name),
        )

    ax.set_title(title)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("normalized magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return int(actual_window)


def save_signal_overview(output_dir, rows, smooth_window=DEFAULT_PLOT_SMOOTH_WINDOW):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "signal_overview.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150, sharex=True)
    actual_window = 1
    panel_order = [
        ("image", SIGNAL_FAMILIES["image"], _FAMILY_TITLES["image"]),
        ("motion", SIGNAL_FAMILIES["motion"], _FAMILY_TITLES["motion"]),
        ("semantic", SIGNAL_FAMILIES["semantic"], _FAMILY_TITLES["semantic"]),
        ("representatives", REPRESENTATIVE_SIGNALS, "Representative Signals"),
    ]

    for idx, panel in enumerate(panel_order):
        family_name, signal_names, title = panel
        row_idx = int(idx / 2)
        col_idx = int(idx % 2)
        actual_window = max(
            int(actual_window),
            int(_plot_group(axes[row_idx][col_idx], rows, signal_names, title, smooth_window)),
        )

    fig.suptitle("Signal Extraction Overview", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(str(overview_path))
    plt.close(fig)
    return {
        "overview_path": overview_path,
        "panels": [
            {
                "panel_name": str(panel[0]),
                "signals": list(panel[1]),
            }
            for panel in panel_order
        ],
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


def save_signal_plots(output_dir, rows, smooth_window=DEFAULT_PLOT_SMOOTH_WINDOW):
    return save_signal_overview(output_dir=output_dir, rows=rows, smooth_window=smooth_window)
