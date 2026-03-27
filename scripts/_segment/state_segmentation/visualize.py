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

ZONE_LEVELS = {
    "low": 0.0,
    "transition": 1.0,
    "high": 2.0,
}

ZONE_COLORS = {
    "low": "#dfe7ec",
    "transition": "#bde0fe",
    "high": "#ffcc80",
}

SIGNAL_COLORS = {
    "motion_velocity_state_signal": "#1f77b4",
    "feature_motion_state_signal": "#2ca02c",
    "state_score": "#d62728",
    "appearance_delta": "#1f3c88",
    "motion_velocity": "#2a9d8f",
    "semantic_velocity": "#bcbd22",
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


def _normalize(values):
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    arr = (arr - vmin) / float(vmax - vmin)
    return [float(x) for x in arr.tolist()]


def _shade_segments(ax, segments, alpha):
    for segment in list(segments or []):
        color = STATE_COLORS.get(segment.get("state_label"), "#dfe7ec")
        ax.axvspan(
            float(segment.get("start_frame", 0)),
            float(segment.get("end_frame", 0)),
            color=color,
            alpha=float(alpha),
            linewidth=0.0,
        )


def _shade_schedule_runs(ax, schedule_runs, alpha):
    for run in list(schedule_runs or []):
        zone_name = str(run.get("schedule_zone", "low") or "low")
        ax.axvspan(
            float(run.get("start_frame", 0) or 0),
            float(run.get("end_frame", 0) or 0),
            color=ZONE_COLORS.get(zone_name, "#dfe7ec"),
            alpha=float(alpha),
            linewidth=0.0,
        )


def _signal_row_y_map(signal_rows, key):
    out = {}
    for row in list(signal_rows or []):
        out[int(row.get("frame_idx", 0) or 0)] = float(row.get(key, 0.0) or 0.0)
    return out


def save_state_overview(
    output_path,
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
    frame_rows = list(frame_rows or [])
    segments = list(segments or [])
    density_rows = list(density_rows or [])
    schedule_runs = list(schedule_runs or [])
    final_indices = [int(idx) for idx in list(final_indices or [])]
    uniform_indices = [int(idx) for idx in list(uniform_indices or [])]
    signal_rows = list(signal_rows or [])

    x = _frame_indices(frame_rows)
    motion_values = _values(frame_rows, "motion_velocity_state_signal")
    feature_values = _values(frame_rows, "feature_motion_state_signal")
    state_scores = _values(frame_rows, "state_score")
    state_level = _state_level(frame_rows)

    panel_count = 4 if signal_rows else 3
    height_ratios = [3.0, 2.4, 1.5]
    if signal_rows:
        height_ratios.append(2.2)

    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(13.5, 10.5 if signal_rows else 8.8),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if panel_count == 1:
        axes = [axes]

    signal_ax = axes[0]
    overlay_ax = axes[1]
    timeline_ax = axes[2]

    signal_ax.plot(x, motion_values, color=SIGNAL_COLORS["motion_velocity_state_signal"], linewidth=1.8, label="motion_velocity (normalized + smoothed)")
    signal_ax.plot(x, feature_values, color=SIGNAL_COLORS["feature_motion_state_signal"], linewidth=1.8, label="feature_motion (normalized + smoothed)")
    signal_ax.plot(x, state_scores, color=SIGNAL_COLORS["state_score"], linewidth=2.1, label="state_score")
    signal_ax.set_ylabel("score")
    signal_ax.set_title("State Overview")
    signal_ax.grid(True, alpha=0.25)
    signal_ax.legend(loc="upper right", fontsize=9)

    _shade_segments(overlay_ax, segments, 0.30)
    overlay_ax.plot(x, state_scores, color=SIGNAL_COLORS["state_score"], linewidth=2.1, label="state_score")
    overlay_ax.axhline(float(enter_th), color="#ef6c00", linestyle="--", linewidth=1.2, label="enter_th")
    overlay_ax.axhline(float(exit_th), color="#546e7a", linestyle="--", linewidth=1.2, label="exit_th")
    ymin = float(np.min(np.asarray(state_scores, dtype=np.float32))) if state_scores else -1.0
    ymax = float(np.max(np.asarray(state_scores, dtype=np.float32))) if state_scores else 1.0
    padding = max(0.2, float((ymax - ymin) * 0.12))
    overlay_ax.set_ylim(ymin - padding, ymax + padding)
    overlay_ax.set_ylabel("state_score")
    overlay_ax.grid(True, alpha=0.25)
    overlay_ax.legend(loc="upper right", fontsize=9)

    if density_rows:
        schedule_levels = [float(ZONE_LEVELS.get(str(row.get("schedule_zone", "low") or "low"), 0.0)) for row in density_rows]
        if schedule_runs:
            _shade_schedule_runs(timeline_ax, schedule_runs, 0.45)
        _shade_segments(timeline_ax, segments, 0.20)
        timeline_ax.step(x, state_level, where="mid", color="#37474f", linewidth=1.1, linestyle="--", label="state label")
        timeline_ax.step(x, schedule_levels, where="mid", color="#1d3557", linewidth=1.8, label="schedule zone")
        if final_indices:
            y_values = []
            density_map = dict((int(row.get("frame_idx", 0) or 0), float(ZONE_LEVELS.get(str(row.get("schedule_zone", "low") or "low"), 0.0))) for row in density_rows)
            for frame_idx in final_indices:
                y_values.append(float(density_map.get(int(frame_idx), 0.0)))
            timeline_ax.scatter(final_indices, y_values, color="#111111", marker="|", s=150, label="final keyframes", zorder=5)
        timeline_ax.set_yticks([0.0, 1.0, 2.0])
        timeline_ax.set_yticklabels(["low", "transition", "high"])
        timeline_ax.set_ylim(-0.25, 2.25)
        timeline_ax.set_ylabel("zone")
    else:
        _shade_segments(timeline_ax, segments, 0.35)
        timeline_ax.step(x, state_level, where="mid", color="#37474f", linewidth=1.5, label="state label")
        timeline_ax.set_yticks([0.0, 1.0])
        timeline_ax.set_yticklabels(["low", "high"])
        timeline_ax.set_ylim(-0.15, 1.15)
        timeline_ax.set_ylabel("state")
    timeline_ax.grid(True, axis="x", alpha=0.2)
    timeline_ax.legend(loc="upper right", fontsize=9, ncol=2)

    if signal_rows:
        detail_ax = axes[3]
        signal_x = [int(row.get("frame_idx", 0) or 0) for row in signal_rows]
        appearance = _normalize(_values(signal_rows, "appearance_delta"))
        motion_velocity = _normalize(_values(signal_rows, "motion_velocity"))
        semantic_velocity = _normalize(_values(signal_rows, "semantic_velocity"))
        detail_ax.plot(signal_x, appearance, color=SIGNAL_COLORS["appearance_delta"], linewidth=1.6, label="appearance_delta")
        detail_ax.plot(signal_x, motion_velocity, color=SIGNAL_COLORS["motion_velocity"], linewidth=1.6, label="motion_velocity")
        detail_ax.plot(signal_x, semantic_velocity, color=SIGNAL_COLORS["semantic_velocity"], linewidth=1.6, label="semantic_velocity")

        motion_y_map = _signal_row_y_map(
            [{"frame_idx": signal_x[idx], "marker_value": motion_velocity[idx]} for idx in range(len(signal_x))],
            "marker_value",
        )
        if uniform_indices:
            detail_ax.scatter(
                uniform_indices,
                [float(motion_y_map.get(int(frame_idx), 0.0)) for frame_idx in uniform_indices],
                color="#7f7f7f",
                marker="|",
                s=120,
                label="uniform anchors",
                zorder=5,
            )
        if final_indices:
            detail_ax.scatter(
                final_indices,
                [float(motion_y_map.get(int(frame_idx), 0.0)) for frame_idx in final_indices],
                color="#111111",
                marker="o",
                s=22,
                label="final keyframes",
                zorder=5,
            )
        detail_ax.set_ylabel("norm signal")
        detail_ax.grid(True, alpha=0.22)
        detail_ax.legend(loc="upper right", fontsize=8, ncol=3)
        detail_ax.set_xlabel("frame_idx")
    else:
        timeline_ax.set_xlabel("frame_idx")

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "state_overview.png"
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
