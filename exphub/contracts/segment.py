from __future__ import annotations

from dataclasses import asdict, dataclass

from .common import StageContract


FORMAL_SEGMENT_POLICY = "state"
FORMAL_SEGMENT_POLICIES = (FORMAL_SEGMENT_POLICY,)

SEGMENT_MANIFEST_NAME = "segment_manifest.json"
SEGMENT_REPORT_NAME = "report.json"
SEGMENT_VISUALS_DIRNAME = "visuals"
SEGMENT_OVERVIEW_NAME = "state_overview.png"


@dataclass(frozen=True)
class SegmentStateInterval:
    segment_id: int
    state_label: str
    start_frame: int
    end_frame: int
    duration_frames: int

    def to_dict(self):
        return asdict(self)

def normalize_formal_segment_policy(policy_name):
    name = str(policy_name or FORMAL_SEGMENT_POLICY).strip().lower()
    if not name:
        name = FORMAL_SEGMENT_POLICY
    return name


def require_formal_segment_policy(policy_name):
    name = normalize_formal_segment_policy(policy_name)
    if name != FORMAL_SEGMENT_POLICY:
        raise ValueError(
            "formal segment workflow only supports policy '{}' in the current mainline (got '{}')".format(
                FORMAL_SEGMENT_POLICY,
                str(policy_name or "").strip() or "<empty>",
            )
        )
    return name


def build_contract(paths):
    return StageContract(
        stage="segment",
        root=paths.segment_dir,
        artifacts={
            "frames_dir": paths.segment_frames_dir,
            "keyframes_dir": paths.segment_keyframes_dir,
            "manifest": paths.segment_manifest_path,
            "report": paths.segment_report_path,
            "visuals_dir": paths.segment_visuals_dir,
            "overview": paths.segment_state_overview_path,
            "calib": paths.segment_calib_path,
            "timestamps": paths.segment_timestamps_path,
        },
    )
