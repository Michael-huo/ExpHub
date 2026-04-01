from __future__ import annotations

from .common import StageContract


SLAM_OUTPUTS = "slam_outputs"


def build_contract(paths):
    return StageContract(
        stage="slam",
        root=paths.slam_dir,
        artifacts={
            "ori_tum": paths.slam_traj_path("ori"),
            "ori_meta": paths.slam_run_meta_path("ori"),
            "gen_tum": paths.slam_traj_path("gen"),
            "gen_meta": paths.slam_run_meta_path("gen"),
        },
    )
