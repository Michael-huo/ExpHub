#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..policies.naming import normalize_policy_name, policy_display_name


_ZONE_COLORS = {
    "low": "#dfe7ec",
    "transition": "#bde0fe",
    "high": "#ffcc80",
}


def _series(rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


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


def _x_values(rows):
    return [int(row.get("frame_idx", 0) or 0) for row in rows]


def _marker_y_map(rows, key):
    return {int(row.get("frame_idx", 0) or 0): float(row.get(key, 0.0) or 0.0) for row in rows}


def _scatter_candidates(ax, frame_indices, y_map, color, marker, label, size=36, alpha=1.0):
    if not frame_indices:
        return
    xs = [int(frame_idx) for frame_idx in frame_indices]
    ys = [float(y_map.get(int(frame_idx), 0.0)) for frame_idx in frame_indices]
    ax.scatter(xs, ys, s=size, color=color, marker=marker, alpha=alpha, linewidths=1.0, zorder=4, label=label)


def _keyframe_lines(ax, keyframe_indices, color="#b0b0b0", linewidth=0.8, alpha=0.35):
    for idx in list(keyframe_indices or []):
        ax.axvline(int(idx), color=color, linewidth=linewidth, alpha=alpha)


def _shade_schedule_zones(ax, rows):
    if not rows:
        return
    start_idx = None
    current_zone = None
    for row in rows:
        zone_name = str(row.get("schedule_zone", "") or "")
        frame_idx = int(row.get("frame_idx", 0) or 0)
        if not zone_name:
            zone_name = ""
        if current_zone is None:
            current_zone = zone_name
            start_idx = frame_idx
            continue
        if zone_name == current_zone:
            continue
        if current_zone in _ZONE_COLORS:
            ax.axvspan(int(start_idx), int(frame_idx), color=_ZONE_COLORS[current_zone], alpha=0.22, linewidth=0.0)
        current_zone = zone_name
        start_idx = frame_idx
    if current_zone in _ZONE_COLORS and start_idx is not None:
        ax.axvspan(int(start_idx), int(rows[-1].get("frame_idx", 0) or 0), color=_ZONE_COLORS[current_zone], alpha=0.22, linewidth=0.0)


def _title(policy_name, suffix):
    return "{} {}".format(policy_display_name(normalize_policy_name(policy_name)), suffix)


def save_comparison_overview(
    rows,
    output_path,
    keyframe_indices,
    policy_name="",
    uniform_indices=None,
    keyframe_items=None,
    comparison_payload=None,
):
    rows = list(rows or [])
    x = _x_values(rows)
    uniform_indices = [int(idx) for idx in list(uniform_indices or [])]
    keyframe_indices = [int(idx) for idx in list(keyframe_indices or [])]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=145, sharex=True)
    top_ax = axes[0]
    bottom_ax = axes[1]

    appearance = _normalize(_series(rows, "appearance_delta"))
    feature_motion = _normalize(_series(rows, "feature_motion"))
    motion_velocity = _normalize(_series(rows, "motion_velocity"))
    semantic_velocity = _normalize(_series(rows, "semantic_velocity"))
    state_score = _normalize(_series(rows, "state_score"))

    if rows:
        top_ax.plot(x, appearance, color="#1f3c88", linewidth=1.8, label="appearance_delta")
        top_ax.plot(x, feature_motion, color="#8c564b", linewidth=1.6, label="feature_motion")
        top_ax.plot(x, motion_velocity, color="#2a9d8f", linewidth=1.6, label="motion_velocity")
        top_ax.plot(x, semantic_velocity, color="#bcbd22", linewidth=1.5, label="semantic_velocity")
        if any(abs(float(v)) > 1e-12 for v in state_score):
            top_ax.plot(x, state_score, color="#d62728", linewidth=1.8, label="state_score")

    _shade_schedule_zones(top_ax, rows)
    y_map = _marker_y_map(rows, "feature_motion")
    _scatter_candidates(top_ax, uniform_indices, y_map, "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.85)
    _scatter_candidates(top_ax, keyframe_indices, y_map, "#111111", "o", "final keyframes", size=28, alpha=0.95)
    _keyframe_lines(top_ax, keyframe_indices)
    top_ax.set_ylabel("normalized magnitude")
    top_ax.grid(True, alpha=0.24)
    top_ax.legend(loc="upper right", fontsize=8, ncol=3)

    if uniform_indices:
        bottom_ax.scatter(uniform_indices, [1.0 for _ in uniform_indices], color="#7f7f7f", marker="|", s=120, label="uniform anchors")
    if keyframe_indices:
        bottom_ax.scatter(keyframe_indices, [0.65 for _ in keyframe_indices], color="#111111", marker="o", s=22, label="final keyframes")

    if rows and any(str(row.get("schedule_zone", "") or "") for row in rows):
        zone_values = []
        for row in rows:
            zone_name = str(row.get("schedule_zone", "low") or "low")
            if zone_name == "high":
                zone_values.append(0.2)
            elif zone_name == "transition":
                zone_values.append(0.4)
            else:
                zone_values.append(0.85)
        bottom_ax.plot(x, zone_values, color="#4a4a4a", linewidth=1.0, alpha=0.6, label="schedule zone")
        bottom_ax.set_yticks([0.2, 0.4, 0.65, 0.85, 1.0])
        bottom_ax.set_yticklabels(["high", "trans", "final", "low", "uniform"])
    else:
        bottom_ax.set_yticks([0.65, 1.0])
        bottom_ax.set_yticklabels(["final", "uniform"])

    _shade_schedule_zones(bottom_ax, rows)
    _keyframe_lines(bottom_ax, keyframe_indices)
    bottom_ax.set_ylim(0.05, 1.08)
    bottom_ax.set_xlabel("frame_idx")
    bottom_ax.set_ylabel("allocation")
    bottom_ax.grid(True, alpha=0.2)
    bottom_ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(_title(policy_name, "Comparison Overview"), fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_allocation_overview(rows, output_path, keyframe_indices, policy_name="", uniform_indices=None, keyframe_items=None):
    rows = list(rows or [])
    x = _x_values(rows)
    uniform_indices = [int(idx) for idx in list(uniform_indices or [])]
    keyframe_indices = [int(idx) for idx in list(keyframe_indices or [])]
    relocated_indices = []
    for item in list(keyframe_items or []):
        if bool(item.get("is_inserted", False)) or bool(item.get("is_relocated", False)):
            relocated_indices.append(int(item.get("frame_idx", 0) or 0))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), dpi=145, sharex=True)
    top_ax = axes[0]
    bottom_ax = axes[1]

    feature_motion = _normalize(_series(rows, "feature_motion"))
    motion_velocity = _normalize(_series(rows, "motion_velocity"))
    semantic_velocity = _normalize(_series(rows, "semantic_velocity"))

    if rows:
        top_ax.plot(x, feature_motion, color="#8c564b", linewidth=1.8, label="feature_motion")
        top_ax.plot(x, motion_velocity, color="#2a9d8f", linewidth=1.6, label="motion_velocity")
        top_ax.plot(x, semantic_velocity, color="#bcbd22", linewidth=1.5, label="semantic_velocity")
    _shade_schedule_zones(top_ax, rows)
    y_map = _marker_y_map(rows, "feature_motion")
    _scatter_candidates(top_ax, uniform_indices, y_map, "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.85)
    _scatter_candidates(top_ax, keyframe_indices, y_map, "#111111", "o", "final keyframes", size=28, alpha=0.95)
    _scatter_candidates(top_ax, relocated_indices, y_map, "#d62728", "D", "inserted / relocated", size=28, alpha=0.9)
    _keyframe_lines(top_ax, keyframe_indices)
    top_ax.set_ylabel("normalized magnitude")
    top_ax.grid(True, alpha=0.24)
    top_ax.legend(loc="upper right", fontsize=8, ncol=3)

    if uniform_indices:
        bottom_ax.scatter(uniform_indices, [1.0 for _ in uniform_indices], color="#7f7f7f", marker="|", s=120, label="uniform anchors")
    if keyframe_indices:
        bottom_ax.scatter(keyframe_indices, [0.65 for _ in keyframe_indices], color="#111111", marker="o", s=22, label="final keyframes")
    if relocated_indices:
        bottom_ax.scatter(relocated_indices, [0.3 for _ in relocated_indices], color="#d62728", marker="D", s=24, label="inserted / relocated")

    _shade_schedule_zones(bottom_ax, rows)
    _keyframe_lines(bottom_ax, keyframe_indices)
    if relocated_indices:
        bottom_ax.set_yticks([0.3, 0.65, 1.0])
        bottom_ax.set_yticklabels(["delta", "final", "uniform"])
    else:
        bottom_ax.set_yticks([0.65, 1.0])
        bottom_ax.set_yticklabels(["final", "uniform"])
    bottom_ax.set_ylim(0.05, 1.08)
    bottom_ax.set_xlabel("frame_idx")
    bottom_ax.set_ylabel("allocation")
    bottom_ax.grid(True, alpha=0.2)
    bottom_ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(_title(policy_name, "Allocation Overview"), fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_kinematics_overview(rows, output_path, keyframe_indices, keyframe_items=None, policy_name="", uniform_indices=None):
    rows = list(rows or [])
    x = _x_values(rows)
    uniform_indices = [int(idx) for idx in list(uniform_indices or [])]
    keyframe_indices = [int(idx) for idx in list(keyframe_indices or [])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), dpi=145, sharex=True)

    appearance = _series(rows, "appearance_delta")
    blur_score = _normalize(_series(rows, "blur_score"))
    motion_velocity = _series(rows, "motion_velocity")
    motion_acceleration = _series(rows, "motion_acceleration")
    semantic_velocity = _series(rows, "semantic_velocity")
    semantic_acceleration = _series(rows, "semantic_acceleration")
    state_score = _series(rows, "state_score")

    y_maps = [
        _marker_y_map(rows, "appearance_delta"),
        _marker_y_map(rows, "motion_velocity"),
        _marker_y_map(rows, "semantic_velocity"),
    ]

    axes[0].plot(x, appearance, color="#1f3c88", linewidth=1.8, label="appearance_delta")
    axes[0].plot(x, blur_score, color="#6c757d", linewidth=1.4, label="blur_score (norm)")
    axes[1].plot(x, motion_velocity, color="#2a9d8f", linewidth=1.8, label="motion_velocity")
    axes[1].plot(x, motion_acceleration, color="#457b9d", linewidth=1.4, label="motion_acceleration")
    axes[2].plot(x, semantic_velocity, color="#bcbd22", linewidth=1.8, label="semantic_velocity")
    axes[2].plot(x, semantic_acceleration, color="#8d99ae", linewidth=1.4, label="semantic_acceleration")
    if any(abs(float(v)) > 1e-12 for v in state_score):
        axes[2].plot(x, state_score, color="#d62728", linewidth=1.5, label="state_score")

    for idx, ax in enumerate(axes):
        _shade_schedule_zones(ax, rows)
        _scatter_candidates(ax, uniform_indices, y_maps[idx], "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.8)
        _scatter_candidates(ax, keyframe_indices, y_maps[idx], "#111111", "o", "final keyframes", size=24, alpha=0.95)
        _keyframe_lines(ax, keyframe_indices)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[0].set_ylabel("image")
    axes[1].set_ylabel("motion")
    axes[2].set_ylabel("semantic")
    axes[2].set_xlabel("frame_idx")

    fig.suptitle(_title(policy_name, "Kinematics Overview"), fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_projection_overview(output_path, policy_name, raw_keyframe_indices, deploy_keyframe_indices, segments, uniform_indices=None):
    policy_name = normalize_policy_name(policy_name)
    raw_keyframe_indices = [int(idx) for idx in list(raw_keyframe_indices or [])]
    deploy_keyframe_indices = [int(idx) for idx in list(deploy_keyframe_indices or [])]
    uniform_indices = [int(idx) for idx in list(uniform_indices or [])]
    segments = list(segments or [])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=150)
    top_ax = axes[0]
    bottom_ax = axes[1]

    if uniform_indices:
        top_ax.scatter(uniform_indices, [0.2 for _ in uniform_indices], color="#7f7f7f", marker="|", s=120, label="uniform anchors")
    if raw_keyframe_indices:
        top_ax.scatter(raw_keyframe_indices, [1.0 for _ in raw_keyframe_indices], color="#1f3c88", marker="o", s=24, label="raw keyframes")
    if deploy_keyframe_indices:
        top_ax.scatter(deploy_keyframe_indices, [0.6 for _ in deploy_keyframe_indices], color="#e76f51", marker="D", s=24, label="deploy keyframes")

    for idx in range(min(len(raw_keyframe_indices), len(deploy_keyframe_indices))):
        raw_idx = int(raw_keyframe_indices[idx])
        deploy_idx = int(deploy_keyframe_indices[idx])
        if raw_idx == deploy_idx:
            continue
        top_ax.annotate("", xy=(deploy_idx, 0.62), xytext=(raw_idx, 0.98), arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.0, alpha=0.8))

    top_ax.set_yticks([0.2, 0.6, 1.0])
    top_ax.set_yticklabels(["uniform", "deploy", "raw"])
    top_ax.set_ylabel("schedule")
    top_ax.grid(True, alpha=0.2)
    handles, labels = top_ax.get_legend_handles_labels()
    if handles:
        top_ax.legend(loc="upper right", fontsize=9, ncol=2)

    centers = []
    raw_gaps = []
    deploy_gaps = []
    gap_errors = []
    for item in segments:
        raw_start_idx = int(item.get("raw_start_idx", 0) or 0)
        raw_end_idx = int(item.get("raw_end_idx", raw_start_idx) or raw_start_idx)
        centers.append(float(raw_start_idx + raw_end_idx) / 2.0)
        raw_gaps.append(float(item.get("raw_gap", 0) or 0))
        deploy_gaps.append(float(item.get("deploy_gap", 0) or 0))
        gap_errors.append(float(item.get("gap_error", 0) or 0))

    if centers:
        bottom_ax.plot(centers, raw_gaps, color="#1f3c88", linewidth=2.0, marker="o", label="raw_gap")
        bottom_ax.plot(centers, deploy_gaps, color="#e76f51", linewidth=1.8, marker="D", label="deploy_gap")
        bottom_ax.bar(centers, gap_errors, width=1.8, color="#2a9d8f", alpha=0.28, label="gap_error")

    bottom_ax.set_xlabel("frame_idx / segment center")
    bottom_ax.set_ylabel("gap")
    bottom_ax.grid(True, alpha=0.2)
    handles, labels = bottom_ax.get_legend_handles_labels()
    if handles:
        bottom_ax.legend(loc="upper right", fontsize=9, ncol=3)

    fig.suptitle("{} Projection Overview".format(policy_display_name(policy_name)), fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
