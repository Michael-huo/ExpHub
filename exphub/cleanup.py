from __future__ import annotations

import os
import shutil
from pathlib import Path

# NOTE: Python 3.7 compatibility:
# - typing.Literal is not available in stdlib typing
# - Path.unlink(missing_ok=...) is 3.8+
KeepLevel = str  # "all" | "repro" | "min"

_KEEP_LEVEL_ALIASES = {
    "clean": "repro",  # legacy alias
    "debug": "all",    # legacy alias
}

_KEEP_LEVELS = ("all", "repro", "min")


def normalize_keep_level(keep: KeepLevel) -> KeepLevel:
    k = str(keep or "").strip().lower()
    k = _KEEP_LEVEL_ALIASES.get(k, k)
    if k in _KEEP_LEVELS:
        return k
    return "repro"


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


def _cleanup_repro(exp_dir: Path) -> None:
    # Keep reproducibility-critical metadata and final artifacts,
    # remove heavy image/video intermediates.
    heavy_dirs = [
        exp_dir / "segment" / "frames",
        exp_dir / "segment" / "keyframes",
        exp_dir / "infer" / "runs",
        exp_dir / "merge" / "frames",
    ]
    for d in heavy_dirs:
        _rm_if_exists(exp_dir, d)

    # Keep TUM + run_meta; npz is optional and can be large.
    slam_dir = exp_dir / "slam"
    if slam_dir.is_dir():
        for track_dir in list(slam_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            _rm_if_exists(exp_dir, track_dir / "traj_est.npz")


def _cleanup_min(exp_dir: Path) -> None:
    # Keep only final outputs: trajectory/eval/stats.
    root_keep = {"slam", "eval", "stats"}
    if exp_dir.is_dir():
        for child in list(exp_dir.iterdir()):
            if child.name in root_keep:
                continue
            _rm_if_exists(exp_dir, child)

    # Keep only final stats payloads.
    _prune_dir_keep_names(exp_dir, exp_dir / "stats", {"report.json", "compression.json"})

    # Keep only trajectory essentials under each SLAM track.
    slam_dir = exp_dir / "slam"
    if slam_dir.is_dir():
        for child in list(slam_dir.iterdir()):
            if not child.is_dir():
                _rm_if_exists(exp_dir, child)
                continue
            _prune_dir_keep_names(exp_dir, child, {"traj_est.tum", "run_meta.json"})


def apply_keep_level(exp_dir: Path, keep: KeepLevel) -> None:
    """Remove non-essential artifacts according to keep level.

    Canonical levels:
    - all: keep everything.
    - repro: keep reproducibility-critical artifacts; prune heavy intermediates.
    - min: keep only final stats + trajectory/eval outputs.

    Backward compatible aliases:
    - clean -> repro
    - debug -> all
    """

    keep_norm = normalize_keep_level(keep)
    if keep_norm == "all":
        return
    if not exp_dir.exists():
        return

    if keep_norm == "repro":
        _cleanup_repro(exp_dir)
        return
    _cleanup_min(exp_dir)
