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


def _keyframe_lines(ax, keyframe_indices):
    if not keyframe_indices:
        return
    for idx in keyframe_indices:
        ax.axvline(int(idx), color="#b0b0b0", linewidth=0.8, alpha=0.4)


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


def save_score_overview(rows, output_path, keyframe_indices, selected_candidates):
    x = [int(row["frame_idx"]) for row in rows]
    score = _series(rows, "score_smooth")
    semantic = _normalize(_series(rows, "semantic_smooth"))
    score_map = {int(row["frame_idx"]): float(row.get("score_smooth", 0.0)) for row in rows}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(x, score, label="score_smooth", linewidth=2.3, color="#1f3c88")
    ax.fill_between(x, semantic, color="#e4edf8", alpha=0.35, label="semantic_smooth (norm)")
    _scatter_candidates(
        ax,
        selected_candidates,
        score_map,
        color="#d62728",
        marker="o",
        label="selected candidates",
        size=46,
    )
    _keyframe_lines(ax, keyframe_indices)

    ax.set_title("Score Overview")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("score_smooth")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_roles_overview(rows, output_path, keyframe_indices, candidate_roles_summary):
    x = [int(row["frame_idx"]) for row in rows]
    score = _normalize(_series(rows, "score_smooth"))
    semantic = _normalize(_series(rows, "semantic_smooth"))
    score_map = {int(row["frame_idx"]): score[pos] for pos, row in enumerate(rows)}
    semantic_map = {int(row["frame_idx"]): semantic[pos] for pos, row in enumerate(rows)}

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(x, score, color="#1f3c88", linewidth=2.0, label="score_smooth")
    ax.plot(x, semantic, color="#c44e52", linewidth=1.8, alpha=0.9, label="semantic_smooth")
    _keyframe_lines(ax, keyframe_indices)
    _scatter_candidates(ax, candidate_roles_summary.get("boundary_candidates", []), score_map, "#d62728", "o", "boundary")
    _scatter_candidates(ax, candidate_roles_summary.get("support_candidates", []), score_map, "#2ca02c", "s", "support")
    _scatter_candidates(
        ax,
        candidate_roles_summary.get("promoted_candidates", []),
        score_map,
        "#9467bd",
        "D",
        "promoted",
    )
    _scatter_candidates(
        ax,
        candidate_roles_summary.get("suppressed_candidates", []),
        semantic_map,
        "#7f7f7f",
        "x",
        "suppressed",
        alpha=0.8,
    )

    ax.set_title("Roles Overview")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("normalized magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)


def save_semantic_overview(rows, output_path, keyframe_indices, selected_candidates, keyframe_items=None):
    x = [int(row["frame_idx"]) for row in rows]
    semantic_displacement = _series(rows, "semantic_displacement")
    semantic_velocity = _series(rows, "semantic_velocity_smooth")
    semantic_acceleration = _series(rows, "semantic_acceleration_smooth")
    semantic_density = _series(rows, "semantic_density")
    displacement_map = {int(row["frame_idx"]): float(row.get("semantic_displacement", 0.0)) for row in rows}
    density_map = {int(row["frame_idx"]): float(row.get("semantic_density", 0.0)) for row in rows}
    relocated_items = []
    for item in keyframe_items or []:
        if bool(item.get("is_relocated", False)):
            relocated_items.append({"frame_idx": int(item.get("frame_idx", 0))})

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=150, sharex=True)
    ax_top = axes[0]
    ax_bottom = axes[1]

    ax_top.plot(x, semantic_displacement, color="#7f7f7f", linewidth=1.3, label="semantic_displacement")
    ax_top.plot(x, _normalize(semantic_velocity), color="#c44e52", linewidth=1.8, label="semantic_velocity_smooth (norm)")
    _scatter_candidates(
        ax_top,
        selected_candidates,
        displacement_map,
        color="#111111",
        marker="o",
        label="selected candidates",
        size=34,
    )
    _keyframe_lines(ax_top, keyframe_indices)
    ax_top.set_title("Semantic Overview")
    ax_top.set_ylabel("displacement / velocity")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=9)

    ax_bottom.plot(x, _normalize(semantic_acceleration), color="#1f77b4", linewidth=1.7, label="semantic_acceleration_smooth (norm)")
    ax_bottom.plot(x, _normalize(semantic_density), color="#2ca02c", linewidth=2.0, label="semantic_density (norm)")
    _scatter_candidates(
        ax_bottom,
        selected_candidates,
        density_map,
        color="#111111",
        marker="o",
        label="selected candidates",
        size=34,
    )
    _scatter_candidates(
        ax_bottom,
        relocated_items,
        density_map,
        color="#ff7f0e",
        marker="D",
        label="relocated keyframes",
        size=36,
    )
    _keyframe_lines(ax_bottom, keyframe_indices)
    ax_bottom.set_xlabel("frame_idx")
    ax_bottom.set_ylabel("acceleration / density")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
