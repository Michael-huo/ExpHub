from __future__ import annotations

from .common import StageContract

FRAMES_DIR = "frames_dir"
MERGE_MANIFEST = "merge_manifest"
REPORT = "report"
TIMESTAMPS = "timestamps"
CALIB = "calib"


def build_contract(paths):
    return StageContract(
        stage="merge",
        root=paths.merge_dir,
        artifacts={
            FRAMES_DIR: paths.merge_frames_dir,
            MERGE_MANIFEST: paths.merge_manifest_path,
            REPORT: paths.merge_report_path,
            TIMESTAMPS: paths.merge_timestamps_path,
            CALIB: paths.merge_calib_path,
        },
    )
