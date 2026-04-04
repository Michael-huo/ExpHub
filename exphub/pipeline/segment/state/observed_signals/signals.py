#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def _semantic_row_map(semantic_rows):
    if not semantic_rows:
        return {}
    return {int(row.get("frame_idx", 0)): row for row in semantic_rows}


def _motion_row_map(motion_rows):
    if not motion_rows:
        return {}
    return {int(row.get("frame_idx", 0)): row for row in motion_rows}


def compute_frame_signal_rows(frame_paths, timestamps, semantic_rows=None, motion_rows=None):
    rows = []
    semantic_map = _semantic_row_map(semantic_rows)
    motion_map = _motion_row_map(motion_rows)

    for idx, frame_path in enumerate(frame_paths):
        semantic_row = semantic_map.get(int(idx), {})
        motion_row = motion_map.get(int(idx), {})
        semantic_displacement = float(
            semantic_row.get(
                "semantic_displacement",
                semantic_row.get("semantic_delta", 0.0),
            )
            or 0.0
        )
        rows.append(
            {
                "frame_idx": int(idx),
                "ts_sec": float(timestamps[idx]),
                "file_name": Path(frame_path).name,
                "motion_displacement": float(motion_row.get("motion_displacement", 0.0) or 0.0),
                "motion_velocity": float(motion_row.get("motion_velocity", 0.0) or 0.0),
                "semantic_delta": float(semantic_row.get("semantic_delta", semantic_displacement) or 0.0),
                "semantic_displacement": float(semantic_displacement),
                "semantic_velocity": float(semantic_row.get("semantic_velocity", 0.0) or 0.0),
            }
        )

    meta = {
        "enabled_signals": [
            "motion_displacement",
            "motion_velocity",
            "semantic_delta",
            "semantic_displacement",
            "semantic_velocity",
        ],
        "semantic_enabled": bool(semantic_rows),
        "motion_enabled": bool(motion_rows),
        "semantic_delta_method": "1 - cosine_similarity(e_t, e_{t-1}) using OpenCLIP image embeddings",
        "semantic_velocity_method": "semantic_displacement / dt",
        "motion_displacement_method": "mean absolute grayscale frame difference after Gaussian blur / 255",
        "motion_velocity_method": "motion_displacement / dt",
    }
    return rows, meta
