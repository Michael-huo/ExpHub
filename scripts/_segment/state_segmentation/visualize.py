#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_COLORS = {
    "low_state": "#dfe7ec",
    "high_state": "#ffcc80",
}

SIGNAL_COLORS = {
    "motion_velocity_state_signal": "#1f77b4",
    "semantic_velocity_state_signal": "#2a9d8f",
    "state_score": "#ef6c00",
    "detector_score": "#d62828",
}


def _frame_indices(frame_rows):
    return [int(row.get("frame_idx", 0) or 0) for row in list(frame_rows or [])]


def _values(frame_rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in list(frame_rows or [])]


def _high_segments(segments):
    return [item for item in list(segments or []) if str(item.get("state_label", "")) == "high_state"]


def _shade_high_segments(ax, segments, alpha):
    for segment in _high_segments(segments):
        ax.axvspan(
            float(segment.get("start_frame", 0) or 0),
            float(segment.get("end_frame", 0) or 0),
            color=STATE_COLORS["high_state"],
            alpha=float(alpha),
            linewidth=0.0,
        )


def _segment_label(segment):
    state_label = str(segment.get("state_label", "low_state") or "low_state")
    if state_label == "high_state":
        return "high_state"
    return "low_state"


def _draw_segment_sequence(ax, segments, y_value, height):
    for segment in list(segments or []):
        start_frame = int(segment.get("start_frame", 0) or 0)
        end_frame = int(segment.get("end_frame", 0) or 0)
        width = max(1, int(end_frame - start_frame + 1))
        state_label = str(segment.get("state_label", "low_state") or "low_state")
        ax.broken_barh(
            [(start_frame, width)],
            (float(y_value), float(height)),
            facecolors=STATE_COLORS.get(state_label, "#dfe7ec"),
            edgecolors="#546e7a",
            linewidth=0.8,
            alpha=0.94,
        )
        ax.text(
            float(start_frame + width / 2.0),
            float(y_value + height / 2.0),
            _segment_label(segment),
            ha="center",
            va="center",
            fontsize=8,
            color="#263238",
        )


def save_state_overview(
    output_path,
    frame_rows,
    segments,
    enter_th=None,
    exit_th=None,
    density_rows=None,
    schedule_runs=None,
    final_indices=None,
    uniform_indices=None,
    signal_rows=None,
):
    del enter_th
    del exit_th
    del density_rows
    del schedule_runs
    del uniform_indices
    del signal_rows

    frame_rows = list(frame_rows or [])
    segments = list(segments or [])
    final_indices = [int(idx) for idx in list(final_indices or [])]
    if not frame_rows:
        raise ValueError("state overview requires non-empty frame rows")

    x = _frame_indices(frame_rows)
    motion_values = _values(frame_rows, "motion_velocity_state_signal")
    semantic_values = _values(frame_rows, "semantic_velocity_state_signal")
    state_scores = _values(frame_rows, "state_score")
    detector_scores = _values(frame_rows, "detector_score")
    high_segments = _high_segments(segments)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 8.8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.7, 1.5]},
    )
    signal_ax = axes[0]
    score_ax = axes[1]
    stage_ax = axes[2]

    _shade_high_segments(signal_ax, segments, 0.22)
    signal_ax.plot(x, motion_values, color=SIGNAL_COLORS["motion_velocity_state_signal"], linewidth=1.8, label="motion_velocity (processed)")
    signal_ax.plot(x, semantic_values, color=SIGNAL_COLORS["semantic_velocity_state_signal"], linewidth=1.8, label="semantic_velocity (processed)")
    signal_ax.plot(x, state_scores, color=SIGNAL_COLORS["state_score"], linewidth=2.2, label="state_score")
    signal_ax.set_ylabel("value")
    signal_ax.set_title("State Segmentation Overview: formal inputs and official state score")
    signal_ax.grid(True, alpha=0.22)
    signal_ax.legend(loc="upper right", fontsize=9)

    _shade_high_segments(score_ax, segments, 0.26)
    score_ax.plot(x, state_scores, color=SIGNAL_COLORS["state_score"], linewidth=2.1, label="state_score")
    score_ax.plot(x, detector_scores, color=SIGNAL_COLORS["detector_score"], linewidth=1.8, label="detector_score")
    for segment in list(high_segments or []):
        start_frame = int(segment.get("start_frame", 0) or 0)
        end_frame = int(segment.get("end_frame", 0) or 0)
        score_ax.axvline(float(start_frame), color="#d62828", linestyle="--", linewidth=1.1)
        score_ax.axvline(float(end_frame), color="#264653", linestyle="--", linewidth=1.1)
    if final_indices:
        y_map = dict((int(frame_idx), float(detector_scores[idx])) for idx, frame_idx in enumerate(x))
        score_ax.scatter(
            final_indices,
            [float(y_map.get(int(frame_idx), 0.0)) for frame_idx in final_indices],
            color="#111111",
            marker="o",
            s=18,
            label="final keyframes",
            zorder=5,
        )
    score_ax.set_ylabel("score")
    score_ax.set_title("Regime-shift detector score and final high-risk interval boundaries")
    score_ax.grid(True, alpha=0.22)
    score_ax.legend(loc="upper right", fontsize=9)

    _draw_segment_sequence(stage_ax, segments, 0.30, 0.55)
    if final_indices:
        for frame_idx in list(final_indices or []):
            stage_ax.axvline(float(frame_idx), color="#111111", linewidth=0.8, alpha=0.45)
    stage_ax.set_yticks([0.575])
    stage_ax.set_yticklabels(["state"])
    stage_ax.set_ylim(0.0, 1.15)
    stage_ax.set_ylabel("interval")
    stage_ax.set_xlabel("frame_idx")
    stage_ax.set_title("Final low/high-risk interval sequence with keyframe positions")
    stage_ax.grid(True, axis="x", alpha=0.18)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    return output_path


def save_state_segmentation_plots(
    output_dir,
    frame_rows,
    segments,
    enter_th,
    exit_th,
    density_rows=None,
    schedule_runs=None,
    final_indices=None,
    uniform_indices=None,
    signal_rows=None,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "state_overview.png"
    legacy_candidate_compare = output_dir / "state_signal_candidate_compare.png"
    if legacy_candidate_compare.is_file():
        try:
            legacy_candidate_compare.unlink()
        except Exception:
            pass

    save_state_overview(
        output_path=overview_path,
        frame_rows=frame_rows,
        segments=segments,
        enter_th=enter_th,
        exit_th=exit_th,
        density_rows=density_rows,
        schedule_runs=schedule_runs,
        final_indices=final_indices,
        uniform_indices=uniform_indices,
        signal_rows=signal_rows,
    )
    return {
        "overview_path": overview_path,
    }
