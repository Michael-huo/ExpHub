from __future__ import annotations

from .common import StageContract


SEGMENT_MANIFEST = "segment_manifest"


def build_contract(paths):
    return StageContract(
        stage="segment",
        root=paths.segment_dir,
        artifacts={
            "frames_dir": paths.segment_frames_dir,
            "keyframes_meta": paths.segment_keyframes_dir / "keyframes_meta.json",
            "deploy_schedule": paths.segment_dir / "deploy_schedule.json",
            "state_segments": paths.segment_dir / "state_segmentation" / "state_segments.json",
            "step_meta": paths.segment_dir / "step_meta.json",
            "calib": paths.segment_calib_path,
            "timestamps": paths.segment_timestamps_path,
            "preprocess_meta": paths.segment_preprocess_meta_path,
        },
    )
