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
    heavy_dirs = [
        exp_dir / "prepare" / "frames",
        exp_dir / "input" / "frames",
        exp_dir / "decode" / "runs",
        exp_dir / "decode" / "frames",
        exp_dir / "export" / "clips",
    ]
    for d in heavy_dirs:
        _rm_if_exists(exp_dir, d)

    root_keep = {"prepare", "input", "encode", "decode", "eval", "export", "logs"}
    if exp_dir.is_dir():
        for child in list(exp_dir.iterdir()):
            if child.name in root_keep or child.name in ("run_meta.json", "exp_meta.json"):
                continue
            _rm_if_exists(exp_dir, child)

    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "prepare",
        {
            "prepare_result.json",
            "frames",
        },
    )

    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "input",
        {
            "input_report.json",
            "frames",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "encode",
        {
            "encode_plan.json",
            "legacy_segment_manifest.json",
            "segment_manifest.json",
            "prompt_spans.json",
            "encode_report.json",
            "motion_segments.json",
            "semantic_anchors.json",
            "generation_units.json",
            "prompts.json",
            "encode_result.json",
            "encode_overview.png",
            "encode_segmentation_overview.png",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "decode",
        {
            "decode_plan.json",
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
            "eval_report.json",
            "eval_slam_report.json",
            "eval_traj_report.json",
            "eval_compression_report.json",
            "eval_summary.txt",
            "eval_details.csv",
            "eval_traj_xy.png",
            "eval_metrics_overview.png",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "export",
        {
            "export_report.json",
            "export_dataset_report.json",
            "clips",
            "metadata",
            "clip_manifests",
        },
    )


def apply_keep_level(exp_dir: Path, keep: KeepLevel) -> None:
    keep_norm = normalize_keep_level(keep)
    if keep_norm == "max":
        return
    if not exp_dir.exists():
        return
    _cleanup_min(exp_dir)
