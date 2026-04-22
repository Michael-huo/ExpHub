from __future__ import annotations

import os
import shutil
from pathlib import Path

KeepLevel = str  # "max" | "min"

_KEEP_LEVELS = ("max", "min")


def normalize_keep_level(keep: KeepLevel) -> KeepLevel:
    k = str(keep or "").strip().lower()
    if k == _KEEP_LEVELS[1]:
        return k
    return _KEEP_LEVELS[0]


def _is_within(root: Path, target: Path) -> bool:
    root_abs = Path(os.path.abspath(str(root)))
    target_abs = Path(os.path.abspath(str(target)))
    try:
        target_abs.relative_to(root_abs)
        return True
    except ValueError:
        return False


def _rm_if_exists(root: Path, p: Path) -> None:
    if not _is_within(root, p):
        return
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(str(p), ignore_errors=True)
    except FileNotFoundError:
        return
    except Exception:
        return


def _prune_dir_keep_names(root: Path, d: Path, keep_names) -> None:
    if not d.is_dir():
        return
    for child in list(d.iterdir()):
        if child.name in keep_names:
            continue
        _rm_if_exists(root, child)


def _cleanup_min(exp_dir: Path) -> None:
    root_keep = {"prepare", "encode", "decode", "eval", "trainset", "logs"}
    if exp_dir.is_dir():
        for child in list(exp_dir.iterdir()):
            if child.name in root_keep or child.name == "run_meta.json":
                continue
            _rm_if_exists(exp_dir, child)

    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "prepare",
        {
            "prepare_result.json",
            "dataset_prepare_index.json",
            "frames",
            "sequences",
        },
    )

    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "encode",
        {
            "motion_segments.json",
            "semantic_anchors.json",
            "generation_units.json",
            "prompts.json",
            "encode_result.json",
            "encode_overview.png",
            "dataset_encode_index.json",
            "sequences",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "decode",
        {
            "decode_merge_report.json",
            "decode_report.json",
            "frames",
            "runs",
            "timestamps.txt",
            "calib.txt",
            "preview.mp4",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "eval",
        {
            "eval_slam_report.json",
            "eval_traj_report.json",
            "eval_compression_report.json",
            "eval_summary.txt",
            "eval_details.csv",
            "eval_traj_xy.png",
            "eval_metrics_overview.png",
            "ori",
            "gen",
        },
    )


def apply_keep_level(exp_dir: Path, keep: KeepLevel) -> None:
    keep_norm = normalize_keep_level(keep)
    if keep_norm == "max":
        return
    if not exp_dir.exists():
        return
    _cleanup_min(exp_dir)
