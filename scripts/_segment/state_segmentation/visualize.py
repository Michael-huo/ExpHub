#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STATE_COLORS = {
    "low_state": "#dfe7ec",
    "high_state": "#ffcc80",
}

_SIGNAL_COLORS = {
    "motion_velocity_state_signal": "#1f77b4",
    "feature_motion_state_signal": "#2ca02c",
    "state_score": "#d62728",
}


def _frame_indices(frame_rows):
    return [int(row.get("frame_idx", 0) or 0) for row in frame_rows]


def _values(frame_rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in frame_rows]


def _state_level(frame_rows):
    values = []
    for row in frame_rows:
        values.append(1.0 if str(row.get("state_label", "")) == "high_state" else 0.0)
    return values


def _shade_segments(ax, segments):
    for segment in segments:
        color = STATE_COLORS.get(segment.get("state_label"), "#dfe7ec")
        ax.axvspan(
            float(segment.get("start_frame", 0)),
            float(segment.get("end_frame", 0)),
            color=color,
            alpha=0.4,
            linewidth=0.0,
        )


def save_state_segmentation_overview(output_path, frame_rows, segments):
    x = _frame_indices(frame_rows)
    motion_values = _values(frame_rows, "motion_velocity_state_signal")
    feature_values = _values(frame_rows, "feature_motion_state_signal")
    state_scores = _values(frame_rows, "state_score")
    state_level = _state_level(frame_rows)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3.0, 1.2]})
    ax_top = axes[0]
    ax_bottom = axes[1]

    ax_top.plot(x, motion_values, color=_SIGNAL_COLORS["motion_velocity_state_signal"], linewidth=1.8, label="motion_velocity (normalized + smoothed)")
    ax_top.plot(x, feature_values, color=_SIGNAL_COLORS["feature_motion_state_signal"], linewidth=1.8, label="feature_motion (normalized + smoothed)")
    ax_top.plot(x, state_scores, color=_SIGNAL_COLORS["state_score"], linewidth=2.2, label="state_score")
    ax_top.set_ylabel("score")
    ax_top.set_title("State Segmentation Overview")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=9)

    _shade_segments(ax_bottom, segments)
    ax_bottom.step(x, state_level, where="mid", color="#37474f", linewidth=1.6)
    ax_bottom.set_ylabel("state")
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_yticklabels(["low", "high"])
    ax_bottom.set_xlabel("frame_idx")
    ax_bottom.set_ylim(-0.15, 1.15)
    ax_bottom.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_state_signal_overlay(output_path, frame_rows, segments, enter_th, exit_th):
    x = _frame_indices(frame_rows)
    state_scores = _values(frame_rows, "state_score")

    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=150)
    _shade_segments(ax, segments)
    ax.plot(x, state_scores, color=_SIGNAL_COLORS["state_score"], linewidth=2.3, label="state_score")
    ax.axhline(float(enter_th), color="#ef6c00", linestyle="--", linewidth=1.2, label="enter_th")
    ax.axhline(float(exit_th), color="#546e7a", linestyle="--", linewidth=1.2, label="exit_th")

    ymin = float(np.min(np.asarray(state_scores, dtype=np.float32))) if state_scores else -1.0
    ymax = float(np.max(np.asarray(state_scores, dtype=np.float32))) if state_scores else 1.0
    padding = max(0.2, float((ymax - ymin) * 0.12))
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("state_score")
    ax.set_title("State Score Overlay")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_state_segmentation_plots(output_dir, frame_rows, segments, enter_th, exit_th):
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "state_segmentation_overview.png"
    overlay_path = output_dir / "state_signal_overlay.png"
    save_state_segmentation_overview(overview_path, frame_rows, segments)
    save_state_signal_overlay(overlay_path, frame_rows, segments, enter_th, exit_th)
    return {
        "overview_path": overview_path,
        "overlay_path": overlay_path,
    }
