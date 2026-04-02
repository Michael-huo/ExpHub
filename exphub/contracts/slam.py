from __future__ import annotations

from .common import StageContract


INPUT_MERGE_MANIFEST = "merge_manifest"
REPORT = "report"
PRIMARY_TRAJECTORY = "traj_est"
ORI_TUM = "ori_tum"
ORI_META = "ori_meta"
GEN_TUM = "gen_tum"
GEN_META = "gen_meta"

FORMAL_SLAM_ARTIFACT_KEYS = (
    REPORT,
    PRIMARY_TRAJECTORY,
    ORI_TUM,
    GEN_TUM,
)


def build_contract(paths):
    return StageContract(
        stage="slam",
        root=paths.slam_dir,
        artifacts={
            INPUT_MERGE_MANIFEST: paths.merge_manifest_path,
            REPORT: paths.slam_report_path,
            PRIMARY_TRAJECTORY: paths.slam_primary_traj_path,
            ORI_TUM: paths.slam_traj_path("ori"),
            ORI_META: paths.slam_run_meta_path("ori"),
            GEN_TUM: paths.slam_traj_path("gen"),
            GEN_META: paths.slam_run_meta_path("gen"),
        },
    )
