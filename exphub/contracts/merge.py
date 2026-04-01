from __future__ import annotations

from .common import StageContract


MERGE_MANIFEST = "merge_manifest"


def build_contract(paths):
    return StageContract(
        stage="merge",
        root=paths.merge_dir,
        artifacts={
            "frames_dir": paths.merge_frames_dir,
            "timestamps": paths.merge_timestamps_path,
            "calib": paths.merge_calib_path,
            "merge_meta": paths.merge_dir / "merge_meta.json",
            "step_meta": paths.merge_dir / "step_meta.json",
        },
    )
