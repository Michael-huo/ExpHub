from __future__ import annotations

from .common import StageContract


RUNS_PLAN = "runs_plan"
RUNS_DIR = "runs_dir"
FRAMES_DIR = "frames_dir"
MERGE_MANIFEST = "merge_manifest"
REPORT = "report"
TIMESTAMPS = "timestamps"
CALIB = "calib"

FORMAL_MERGE_ARTIFACT_KEYS = (
    FRAMES_DIR,
    MERGE_MANIFEST,
    REPORT,
    TIMESTAMPS,
    CALIB,
)


def build_contract(paths):
    return StageContract(
        stage="merge",
        root=paths.merge_dir,
        artifacts={
            RUNS_PLAN: paths.infer_runs_plan_path,
            RUNS_DIR: paths.infer_runs_dir,
            FRAMES_DIR: paths.merge_frames_dir,
            MERGE_MANIFEST: paths.merge_manifest_path,
            REPORT: paths.merge_report_path,
            TIMESTAMPS: paths.merge_timestamps_path,
            CALIB: paths.merge_calib_path,
        },
    )
