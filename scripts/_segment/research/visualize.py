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


def _build_plot(rows, title, keyframe_indices=None, peak_indices=None):
    x = [int(row["frame_idx"]) for row in rows]
    appearance = _normalize(_series(rows, "appearance_delta"))
    brightness = _normalize(_series(rows, "brightness_jump"))
    blur = _normalize(_series(rows, "blur_score"))
    motion = _normalize(_series(rows, "feature_motion"))
    score = _normalize(_series(rows, "score_smooth"))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.plot(x, appearance, label="appearance_delta", linewidth=1.4)
    ax.plot(x, brightness, label="brightness_jump", linewidth=1.2)
    ax.plot(x, blur, label="blur_score", linewidth=1.2)
    ax.plot(x, motion, label="feature_motion", linewidth=1.2)
    ax.plot(x, score, label="score_smooth", linewidth=2.2, color="#d62728")

    if keyframe_indices:
        for idx in keyframe_indices:
            ax.axvline(int(idx), color="#7f7f7f", linewidth=0.8, alpha=0.45)

    if peak_indices:
        yvals = []
        score_map = {int(row["frame_idx"]): score[pos] for pos, row in enumerate(rows)}
        for idx in peak_indices:
            yvals.append(float(score_map.get(int(idx), 0.0)))
        ax.scatter(list(peak_indices), yvals, s=28, color="#111111", label="peaks", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("normalized magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig



def save_score_curve(rows, output_path):
    fig = _build_plot(rows, "Segment Analysis Score Curve")
    fig.savefig(str(output_path))
    plt.close(fig)



def save_score_curve_with_keyframes(rows, output_path, keyframe_indices):
    fig = _build_plot(rows, "Segment Analysis Score Curve with Uniform Keyframes", keyframe_indices=keyframe_indices)
    fig.savefig(str(output_path))
    plt.close(fig)



def save_peaks_preview(rows, output_path):
    peak_indices = [int(row["frame_idx"]) for row in rows if bool(row.get("is_peak", False))]
    fig = _build_plot(rows, "Segment Analysis Peaks Preview", peak_indices=peak_indices)
    fig.savefig(str(output_path))
    plt.close(fig)



def save_candidate_points_overview(rows, output_path, keyframe_indices, candidate_points):
    x = [int(row["frame_idx"]) for row in rows]
    score = _series(rows, "score_smooth")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(x, score, color="#1f3c88", linewidth=2.0, label="score_smooth")

    for idx in keyframe_indices:
        ax.axvline(int(idx), color="#b0b0b0", linewidth=0.8, alpha=0.45)

    if candidate_points:
        cand_x = [int(item["frame_idx"]) for item in candidate_points]
        cand_y = [float(item["score_smooth"]) for item in candidate_points]
        ax.scatter(cand_x, cand_y, color="#d62728", s=38, zorder=4, label="candidate peaks")
        for item in candidate_points:
            ax.annotate(
                "#{}".format(int(item.get("peak_rank", 0))),
                (int(item["frame_idx"]), float(item["score_smooth"])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#222222",
            )

    ax.set_title("Candidate Points Overview")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("score_smooth")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
