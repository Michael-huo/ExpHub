#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def _keyframe_lines(ax, keyframe_indices, color="#b0b0b0", linewidth=0.8, alpha=0.35):
    if not keyframe_indices:
        return
    for idx in keyframe_indices:
        ax.axvline(int(idx), color=color, linewidth=linewidth, alpha=alpha)


def _scatter_candidates(ax, items, y_map, color, marker, label, size=42, alpha=1.0):
    if not items:
        return
    xs = [int(item["frame_idx"]) for item in items]
    ys = [float(y_map.get(int(item["frame_idx"]), 0.0)) for item in items]
    ax.scatter(
        xs,
        ys,
        s=size,
        color=color,
        marker=marker,
        linewidths=1.0,
        alpha=alpha,
        zorder=4,
        label=label,
    )


def _policy_signal_prefix(policy_name):
    if policy_name == "sks_v1":
        return "semantic"
    if policy_name == "motion_energy_v1":
        return "motion"
    return ""


def _official_points(keyframe_items, uniform_indices, keyframe_indices):
    uniform_points = [{"frame_idx": int(idx)} for idx in uniform_indices or []]
    final_points = [{"frame_idx": int(idx)} for idx in keyframe_indices or []]
    relocated_points = []
    for item in keyframe_items or []:
        if bool(item.get("is_relocated", False)):
            relocated_points.append({"frame_idx": int(item.get("frame_idx", 0))})
    return uniform_points, final_points, relocated_points


def _official_y_map(rows, key):
    return {int(row["frame_idx"]): float(row.get(key, 0.0) or 0.0) for row in rows}


def _values_map(rows, values):
    return {int(row["frame_idx"]): float(values[pos]) for pos, row in enumerate(rows)}


def _policy_display_name(policy_name):
    if policy_name == "sks_v1":
        return "sks_v1 (semantic)"
    if policy_name == "motion_energy_v1":
        return "motion_energy_v1 (motion)"
    return str(policy_name)


def _save_official_score_overview(
    rows,
    output_path,
    keyframe_indices,
    policy_name,
    uniform_indices=None,
    keyframe_items=None,
    comparison_payload=None,
):
    x = [int(row["frame_idx"]) for row in rows]
    comparison_payload = dict(comparison_payload or {})
    observer_map = dict(comparison_payload.get("observers", {}) or {})
    uniform_points = [{"frame_idx": int(idx)} for idx in uniform_indices or []]

    if not observer_map:
        prefix = _policy_signal_prefix(policy_name)
        appearance = _normalize(_series(rows, "appearance_delta"))
        motion = _normalize(_series(rows, "feature_motion"))
        fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
        ax.plot(x, appearance, label="appearance_delta (norm)", linewidth=1.8, color="#1f3c88")
        ax.plot(x, motion, label="feature_motion (norm)", linewidth=1.5, color="#8c564b", alpha=0.9)
        y_key = "feature_motion"
        if prefix:
            density_key = "{}_density".format(prefix)
            density = _normalize(_series(rows, density_key))
            action = _normalize(_series(rows, "{}_action".format(prefix)))
            ax.fill_between(x, density, color="#dcefd9", alpha=0.45, label="{}_density (norm)".format(prefix))
            ax.plot(x, action, linewidth=1.2, color="#2a9d8f", alpha=0.95, label="{}_action (norm)".format(prefix))
            y_key = density_key
        y_map = _official_y_map(rows, y_key)
        _scatter_candidates(ax, uniform_points, y_map, "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.9)
        _scatter_candidates(ax, [{"frame_idx": int(idx)} for idx in keyframe_indices], y_map, "#111111", "o", "final keyframes", size=34)
        _keyframe_lines(ax, keyframe_indices)
        ax.set_title("Comparison Overview")
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("normalized magnitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=145, sharex=True)
    density_ax = axes[0]
    action_ax = axes[1]
    alloc_ax = axes[2]

    density_y_maps = {}

    if policy_name in ("sks_v1", "motion_energy_v1"):
        active_prefix = _policy_signal_prefix(policy_name)
        observer_policy = sorted(observer_map.keys())[0]
        observer_prefix = _policy_signal_prefix(observer_policy)

        active_density = _normalize(_series(rows, "{}_density".format(active_prefix)))
        observer_density = _normalize(_series(rows, "{}_density".format(observer_prefix)))
        active_action = _normalize(_series(rows, "{}_action".format(active_prefix)))
        observer_action = _normalize(_series(rows, "{}_action".format(observer_prefix)))

        density_ax.plot(x, active_density, color="#1f3c88", linewidth=2.0, label="active {} density (norm)".format(_policy_display_name(policy_name)))
        density_ax.plot(x, observer_density, color="#e76f51", linewidth=1.8, label="observer {} density (norm)".format(_policy_display_name(observer_policy)))
        action_ax.plot(x, active_action, color="#2a9d8f", linewidth=2.0, label="active {} action (norm)".format(_policy_display_name(policy_name)))
        action_ax.plot(x, observer_action, color="#b56576", linewidth=1.8, label="observer {} action (norm)".format(_policy_display_name(observer_policy)))

        density_y_maps[policy_name] = _values_map(rows, active_density)
        density_y_maps[observer_policy] = _values_map(rows, observer_density)

        active_points = [{"frame_idx": int(idx)} for idx in keyframe_indices]
        observer_points = [{"frame_idx": int(idx)} for idx in observer_map[observer_policy].get("keyframes", [])]
        _scatter_candidates(density_ax, active_points, density_y_maps[policy_name], "#111111", "o", "active final keyframes", size=26)
        _scatter_candidates(density_ax, observer_points, density_y_maps[observer_policy], "#d62728", "D", "observer final keyframes", size=28)
        _scatter_candidates(density_ax, uniform_points, density_y_maps[policy_name], "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.7)
    else:
        sks_points = observer_map.get("sks_v1", {}).get("keyframes", [])
        motion_points = observer_map.get("motion_energy_v1", {}).get("keyframes", [])
        semantic_density = _normalize(_series(rows, "semantic_density"))
        motion_density = _normalize(_series(rows, "motion_density"))
        semantic_action = _normalize(_series(rows, "semantic_action"))
        motion_action = _normalize(_series(rows, "motion_action"))

        density_ax.plot(x, semantic_density, color="#1f3c88", linewidth=2.0, label="observer sks_v1 density (norm)")
        density_ax.plot(x, motion_density, color="#e76f51", linewidth=1.8, label="observer motion_energy_v1 density (norm)")
        action_ax.plot(x, semantic_action, color="#2a9d8f", linewidth=2.0, label="observer sks_v1 action (norm)")
        action_ax.plot(x, motion_action, color="#b56576", linewidth=1.8, label="observer motion_energy_v1 action (norm)")

        density_y_maps["sks_v1"] = _values_map(rows, semantic_density)
        density_y_maps["motion_energy_v1"] = _values_map(rows, motion_density)

        _scatter_candidates(density_ax, [{"frame_idx": int(idx)} for idx in sks_points], density_y_maps["sks_v1"], "#111111", "o", "observer sks_v1 keyframes", size=26)
        _scatter_candidates(density_ax, [{"frame_idx": int(idx)} for idx in motion_points], density_y_maps["motion_energy_v1"], "#d62728", "D", "observer motion keyframes", size=28)
        _scatter_candidates(density_ax, uniform_points, density_y_maps["sks_v1"], "#7f7f7f", "|", "active uniform anchors", size=120, alpha=0.7)

    density_ax.set_ylabel("density (norm)")
    density_ax.grid(True, alpha=0.25)
    density_ax.legend(loc="upper right", fontsize=8, ncol=2)

    action_ax.set_ylabel("action (norm)")
    action_ax.grid(True, alpha=0.25)
    action_ax.legend(loc="upper right", fontsize=8, ncol=2)

    if uniform_points:
        alloc_ax.scatter(
            [int(item["frame_idx"]) for item in uniform_points],
            [1.0 for _ in uniform_points],
            color="#7f7f7f",
            marker="|",
            s=120,
            label="uniform anchors",
        )

    if policy_name in ("sks_v1", "motion_energy_v1"):
        alloc_ax.scatter(
            [int(idx) for idx in keyframe_indices],
            [0.68 for _ in keyframe_indices],
            color="#111111",
            marker="o",
            s=22,
            label="active final keyframes",
        )
        observer_policy = sorted(observer_map.keys())[0]
        observer_points = observer_map[observer_policy].get("keyframes", [])
        alloc_ax.scatter(
            [int(idx) for idx in observer_points],
            [0.32 for _ in observer_points],
            color="#d62728",
            marker="D",
            s=24,
            label="observer final keyframes",
        )
        alloc_ax.set_yticks([0.32, 0.68, 1.0])
        alloc_ax.set_yticklabels(["observer", "active", "uniform"])
    else:
        sks_points = observer_map.get("sks_v1", {}).get("keyframes", [])
        motion_points = observer_map.get("motion_energy_v1", {}).get("keyframes", [])
        alloc_ax.scatter(
            [int(idx) for idx in keyframe_indices],
            [0.82 for _ in keyframe_indices],
            color="#111111",
            marker="o",
            s=20,
            label="active uniform final",
        )
        alloc_ax.scatter(
            [int(idx) for idx in sks_points],
            [0.52 for _ in sks_points],
            color="#1f3c88",
            marker="D",
            s=24,
            label="observer sks_v1 final",
        )
        alloc_ax.scatter(
            [int(idx) for idx in motion_points],
            [0.22 for _ in motion_points],
            color="#e76f51",
            marker="s",
            s=22,
            label="observer motion final",
        )
        alloc_ax.set_yticks([0.22, 0.52, 0.82, 1.0])
        alloc_ax.set_yticklabels(["motion", "sks_v1", "active", "uniform"])

    alloc_ax.set_ylim(0.05, 1.1)
    alloc_ax.set_xlabel("frame_idx")
    alloc_ax.set_ylabel("allocation")
    alloc_ax.grid(True, alpha=0.2)
    alloc_ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("Comparison Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def _save_official_roles_overview(rows, output_path, keyframe_indices, policy_name, uniform_indices=None, keyframe_items=None):
    x = [int(row["frame_idx"]) for row in rows]
    prefix = _policy_signal_prefix(policy_name)
    uniform_points, final_points, relocated_points = _official_points(keyframe_items, uniform_indices, keyframe_indices)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True)
    ax_top = axes[0]
    ax_bottom = axes[1]

    if prefix:
        density_key = "{}_density".format(prefix)
        action_key = "{}_action".format(prefix)
        density = _normalize(_series(rows, density_key))
        action = _normalize(_series(rows, action_key))
        y_map = _official_y_map(rows, density_key)
        ax_top.plot(x, density, color="#2a9d8f", linewidth=2.0, label="{}_density (norm)".format(prefix))
        ax_top.plot(x, action, color="#e76f51", linewidth=1.5, label="{}_action (norm)".format(prefix))
        _scatter_candidates(ax_top, uniform_points, y_map, "#9a9a9a", "|", "uniform anchors", size=120, alpha=0.8)
        _scatter_candidates(ax_top, final_points, y_map, "#111111", "o", "final keyframes", size=32)
        _scatter_candidates(ax_top, relocated_points, y_map, "#ff7f0e", "D", "relocated keyframes", size=34)
        _keyframe_lines(ax_top, keyframe_indices)
        ax_top.set_ylabel("normalized magnitude")
        ax_top.legend(loc="upper right", fontsize=9, ncol=2)
    else:
        appearance = _normalize(_series(rows, "appearance_delta"))
        motion = _normalize(_series(rows, "feature_motion"))
        y_map = _official_y_map(rows, "feature_motion")
        ax_top.plot(x, appearance, color="#1f3c88", linewidth=1.8, label="appearance_delta (norm)")
        ax_top.plot(x, motion, color="#8c564b", linewidth=1.5, label="feature_motion (norm)")
        _scatter_candidates(ax_top, uniform_points, y_map, "#9a9a9a", "|", "uniform anchors", size=120, alpha=0.8)
        _scatter_candidates(ax_top, final_points, y_map, "#111111", "o", "final keyframes", size=32)
        _keyframe_lines(ax_top, keyframe_indices)
        ax_top.set_ylabel("normalized magnitude")
        ax_top.legend(loc="upper right", fontsize=9)

    if uniform_points:
        ax_bottom.scatter(
            [int(item["frame_idx"]) for item in uniform_points],
            [1.0 for _ in uniform_points],
            color="#7f7f7f",
            marker="|",
            s=120,
            label="uniform anchors",
        )
    if final_points:
        ax_bottom.scatter(
            [int(item["frame_idx"]) for item in final_points],
            [0.65 for _ in final_points],
            color="#111111",
            marker="o",
            s=24,
            label="final keyframes",
        )
    if relocated_points:
        ax_bottom.scatter(
            [int(item["frame_idx"]) for item in relocated_points],
            [0.3 for _ in relocated_points],
            color="#ff7f0e",
            marker="D",
            s=28,
            label="relocated keyframes",
        )
    _keyframe_lines(ax_bottom, keyframe_indices)
    ax_bottom.set_ylim(0.0, 1.2)
    ax_bottom.set_yticks([0.3, 0.65, 1.0])
    ax_bottom.set_yticklabels(["relocated", "final", "uniform"])
    ax_bottom.set_xlabel("frame_idx")
    ax_bottom.set_ylabel("allocation")
    ax_bottom.grid(True, alpha=0.2)
    ax_bottom.legend(loc="upper right", fontsize=9)

    fig.suptitle("Allocation Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def _official_marker_map(rows, key):
    return {int(row["frame_idx"]): float(row.get(key, 0.0) or 0.0) for row in rows}


def _save_official_semantic_overview(rows, output_path, keyframe_indices, policy_name, uniform_indices=None, keyframe_items=None):
    x = [int(row["frame_idx"]) for row in rows]
    prefix = _policy_signal_prefix(policy_name)
    uniform_points, final_points, relocated_points = _official_points(keyframe_items, uniform_indices, keyframe_indices)

    if not prefix:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        appearance = _normalize(_series(rows, "appearance_delta"))
        motion = _normalize(_series(rows, "feature_motion"))
        ax.plot(x, appearance, color="#1f3c88", linewidth=1.8, label="appearance_delta (norm)")
        ax.plot(x, motion, color="#8c564b", linewidth=1.5, label="feature_motion (norm)")
        y_map = _official_marker_map(rows, "feature_motion")
        _scatter_candidates(ax, uniform_points, y_map, "#7f7f7f", "|", "uniform anchors", size=120, alpha=0.85)
        _scatter_candidates(ax, final_points, y_map, "#111111", "o", "final keyframes", size=34)
        _keyframe_lines(ax, keyframe_indices)
        ax.set_title("Uniform Overview")
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("normalized magnitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(output_path))
        plt.close(fig)
        return

    displacement_key = "{}_displacement".format(prefix)
    velocity_key = "{}_velocity".format(prefix)
    velocity_smooth_key = "{}_velocity_smooth".format(prefix)
    acceleration_key = "{}_acceleration".format(prefix)
    acceleration_smooth_key = "{}_acceleration_smooth".format(prefix)
    density_key = "{}_density".format(prefix)
    action_key = "{}_action".format(prefix)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), dpi=150, sharex=True)

    displacement_map = _official_marker_map(rows, displacement_key)
    density_map = _official_marker_map(rows, density_key)
    action_map = _official_marker_map(rows, action_key)

    axes[0].plot(x, _series(rows, displacement_key), color="#6c757d", linewidth=1.3, label="{}_displacement".format(prefix))
    _scatter_candidates(axes[0], uniform_points, displacement_map, "#b0b0b0", "|", "uniform anchors", size=120, alpha=0.8)
    _scatter_candidates(axes[0], final_points, displacement_map, "#111111", "o", "final keyframes", size=28)
    _scatter_candidates(axes[0], relocated_points, displacement_map, "#ff7f0e", "D", "relocated keyframes", size=30)
    _keyframe_lines(axes[0], keyframe_indices)
    axes[0].set_ylabel("disp")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    axes[1].plot(x, _series(rows, velocity_key), color="#c44e52", linewidth=1.0, alpha=0.6, label="{}_velocity".format(prefix))
    axes[1].plot(x, _series(rows, velocity_smooth_key), color="#e76f51", linewidth=1.7, label="{}_velocity_smooth".format(prefix))
    _keyframe_lines(axes[1], keyframe_indices)
    axes[1].set_ylabel("vel")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(x, _series(rows, acceleration_key), color="#457b9d", linewidth=1.0, alpha=0.6, label="{}_acceleration".format(prefix))
    axes[2].plot(x, _series(rows, acceleration_smooth_key), color="#1d3557", linewidth=1.7, label="{}_acceleration_smooth".format(prefix))
    _keyframe_lines(axes[2], keyframe_indices)
    axes[2].set_ylabel("acc")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(x, _series(rows, density_key), color="#2a9d8f", linewidth=2.0, label="{}_density".format(prefix))
    _scatter_candidates(axes[3], uniform_points, density_map, "#b0b0b0", "|", "uniform anchors", size=120, alpha=0.8)
    _scatter_candidates(axes[3], final_points, density_map, "#111111", "o", "final keyframes", size=28)
    _scatter_candidates(axes[3], relocated_points, density_map, "#ff7f0e", "D", "relocated keyframes", size=30)
    _keyframe_lines(axes[3], keyframe_indices)
    axes[3].set_ylabel("density")
    axes[3].legend(loc="upper right", fontsize=8, ncol=2)

    axes[4].plot(x, _series(rows, action_key), color="#264653", linewidth=2.0, label="{}_action".format(prefix))
    _scatter_candidates(axes[4], uniform_points, action_map, "#b0b0b0", "|", "uniform anchors", size=120, alpha=0.8)
    _scatter_candidates(axes[4], final_points, action_map, "#111111", "o", "final keyframes", size=28)
    _scatter_candidates(axes[4], relocated_points, action_map, "#ff7f0e", "D", "relocated keyframes", size=30)
    _keyframe_lines(axes[4], keyframe_indices)
    axes[4].set_xlabel("frame_idx")
    axes[4].set_ylabel("action")
    axes[4].legend(loc="upper right", fontsize=8, ncol=2)

    title = "Semantic Kinematics Overview" if prefix == "semantic" else "Motion Kinematics Overview"
    for ax in axes:
        ax.grid(True, alpha=0.22)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_score_overview(
    rows,
    output_path,
    keyframe_indices,
    selected_candidates=None,
    policy_name="",
    uniform_indices=None,
    keyframe_items=None,
    comparison_payload=None,
):
    _save_official_score_overview(
        rows,
        output_path,
        keyframe_indices,
        policy_name,
        uniform_indices,
        keyframe_items,
        comparison_payload,
    )


def save_roles_overview(rows, output_path, keyframe_indices, candidate_roles_summary=None, policy_name="", uniform_indices=None, keyframe_items=None):
    _save_official_roles_overview(rows, output_path, keyframe_indices, policy_name, uniform_indices, keyframe_items)


def save_semantic_overview(rows, output_path, keyframe_indices, selected_candidates=None, keyframe_items=None, policy_name="", uniform_indices=None):
    _save_official_semantic_overview(rows, output_path, keyframe_indices, policy_name, uniform_indices, keyframe_items)
