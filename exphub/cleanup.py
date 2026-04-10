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
    # `keep=min` is a retention policy, not a contract definition. These lists
    # keep the current lightweight outputs we still want after the mainline run
    # while pruning heavy intermediates that are no longer needed.
    heavy_dirs = [
        exp_dir / "segment" / "frames",
        exp_dir / "infer" / "runs",
        exp_dir / "infer" / "generation_units" / "runs",
        exp_dir / "merge" / "frames",
        exp_dir / "merge" / "generation_units" / "frames",
    ]
    for d in heavy_dirs:
        _rm_if_exists(exp_dir, d)

    # Keep stage roots required by directory contract.
    root_keep = {"segment", "prompt", "infer", "merge", "slam", "eval", "stats", "logs"}
    if exp_dir.is_dir():
        for child in list(exp_dir.iterdir()):
            if child.name in root_keep:
                continue
            _rm_if_exists(exp_dir, child)

    # Keep lightweight, reproducibility-critical metadata.
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "segment",
        {
            "calib.txt",
            "timestamps.txt",
            "keyframes",
            "aligned_segment_plan.json",
            "motion_score.json",
            "semantic_shift.json",
            "generation_risk.json",
            "candidate_boundaries.json",
            "generation_units.json",
            "segment_manifest.json",
            "report.json",
            "visuals",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "prompt",
        {
            "prompt_manifest.json",
            "prompt_spans.json",
            "report.json",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "infer",
        {"report.json", "runs_plan.json", "generation_units"},
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "merge",
        {"merge_manifest.json", "report.json", "calib.txt", "timestamps.txt", "generation_units"},
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "infer" / "generation_units",
        {"report.json", "runs_plan.json"},
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "merge" / "generation_units",
        {"merge_manifest.json", "report.json", "calib.txt", "timestamps.txt"},
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "eval",
        {
            "compression.json",
            "details.csv",
            "generation_units",
            "metrics",
            "plots",
            "report.json",
            "slam",
            "summary.txt",
        },
    )
    _prune_dir_keep_names(
        exp_dir,
        exp_dir / "eval" / "generation_units",
        {
            "compression.json",
            "details.csv",
            "metrics",
            "plots",
            "report.json",
            "slam",
            "summary.txt",
        },
    )


def apply_keep_level(exp_dir: Path, keep: KeepLevel) -> None:
    """Remove non-essential artifacts according to keep level.

    Canonical levels:
    - max: keep everything.
    - min: keep lightweight metadata + final outputs; prune heavy intermediates.
    """

    keep_norm = normalize_keep_level(keep)
    if keep_norm == "max":
        return
    if not exp_dir.exists():
        return

    _cleanup_min(exp_dir)
