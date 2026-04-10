from __future__ import annotations

from .common import StageContract

FRAMES_DIR = "frames_dir"
MERGE_MANIFEST = "merge_manifest"
REPORT = "report"
TIMESTAMPS = "timestamps"
CALIB = "calib"


def build_contract(paths, decode_source="aligned"):
    return StageContract(
        stage="merge",
        root=paths.merge_source_dir(decode_source),
        artifacts={
            FRAMES_DIR: paths.merge_frames_source_dir(decode_source),
            MERGE_MANIFEST: paths.merge_manifest_source_path(decode_source),
            REPORT: paths.merge_report_source_path(decode_source),
            TIMESTAMPS: paths.merge_timestamps_source_path(decode_source),
            CALIB: paths.merge_calib_source_path(decode_source),
        },
    )
