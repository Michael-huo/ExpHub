from __future__ import annotations

from pathlib import Path

# NOTE: Python 3.7 compatibility:
# - typing.Literal is not available in stdlib typing
# - Path.unlink(missing_ok=...) is 3.8+
KeepLevel = str  # "clean" | "repro" | "debug"


def rm_if_exists(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            try:
                p.unlink()
            except FileNotFoundError:
                return
        elif p.is_dir():
            import shutil

            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def apply_keep_level(exp_dir: Path, keep: KeepLevel) -> None:
    """Remove non-essential artifacts according to keep level.

    - clean: keep only essentials for result + recommended reproducibility.
    - repro: similar to clean, but keep small meta useful for reproduction.
    - debug: keep everything.
    """

    if keep == "debug":
        return

    # Prompt debug outputs (new + legacy dir names).
    rm_if_exists(exp_dir / "prompt" / "resolved.json")
    rm_if_exists(exp_dir / "prompt" / "digest.txt")
    rm_if_exists(exp_dir / "prompts" / "resolved.json")
    rm_if_exists(exp_dir / "prompts" / "digest.txt")

    # Prompt per-clip file is optional; keep_level repro/clean removes it.
    rm_if_exists(exp_dir / "segment" / "clip_prompts.json")

    # Pointer file from old pipeline.
    rm_if_exists(exp_dir / "segment.txt")

    # Keep only compression.json.
    rm_if_exists(exp_dir / "stats" / "compression.txt")

    # SLAM: keep tum + run_meta.json, remove npz.
    for d in [
        exp_dir / "slam" / "ori",
        exp_dir / "slam" / "gen",
        exp_dir / "slam_ori",  # legacy
        exp_dir / "slam_gen",  # legacy
        exp_dir / "slam",      # temp/legacy
    ]:
        rm_if_exists(d / "traj_est.npz")

    # Segment old debug files (in case old generator still produces them).
    seg = exp_dir / "segment"
    rm_if_exists(seg / "args.json")
    rm_if_exists(seg / "frame_map.csv")
    rm_if_exists(seg / "README.md")
