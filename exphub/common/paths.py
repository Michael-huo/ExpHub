from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .types import ExperimentSpec, canon_num_str, dot_to_p


@dataclass(frozen=True)
class ExperimentPaths:
    exphub_root: Path
    dataset: str
    sequence: str
    exp_name: str
    exp_root_override: Optional[Path] = None

    @classmethod
    def from_spec(cls, spec):
        return cls(
            exphub_root=Path(spec.exphub_root).resolve(),
            dataset=str(spec.dataset),
            sequence=str(spec.sequence),
            exp_name=str(spec.exp_name),
            exp_root_override=Path(spec.exp_root_override).resolve() if spec.exp_root_override is not None else None,
        )

    @property
    def exp_root(self):
        if self.exp_root_override is not None:
            return self.exp_root_override
        return self.exphub_root / "experiments" / self.dataset / self.sequence

    @property
    def exp_dir(self):
        return (self.exp_root / self.exp_name).resolve()

    @property
    def segment_dir(self):
        return self.exp_dir / "segment"

    @property
    def segment_manifest_path(self):
        return self.segment_dir / "segment_manifest.json"

    @property
    def segment_report_path(self):
        return self.segment_dir / "report.json"

    @property
    def segment_visuals_dir(self):
        return self.segment_dir / "visuals"

    @property
    def segment_state_overview_path(self):
        return self.segment_visuals_dir / "state_overview.png"

    @property
    def segment_state_dir(self):
        return self.segment_dir / "state_segmentation"

    @property
    def segment_state_segments_path(self):
        return self.segment_state_dir / "state_segments.json"

    @property
    def segment_state_report_path(self):
        return self.segment_state_dir / "state_report.json"

    @property
    def segment_state_overview_legacy_path(self):
        return self.segment_state_dir / "state_overview.png"

    @property
    def prompt_dir(self):
        return self.exp_dir / "prompt"

    @property
    def infer_dir(self):
        return self.exp_dir / "infer"

    @property
    def merge_dir(self):
        return self.exp_dir / "merge"

    @property
    def slam_dir(self):
        return self.exp_dir / "slam"

    @property
    def eval_dir(self):
        return self.exp_dir / "eval"

    @property
    def stats_dir(self):
        return self.exp_dir / "stats"

    @property
    def logs_dir(self):
        return self.exp_dir / "logs"

    @property
    def exp_meta_path(self):
        return self.exp_dir / "exp_meta.json"

    @property
    def segment_frames_dir(self):
        return self.segment_dir / "frames"

    @property
    def segment_keyframes_dir(self):
        return self.segment_dir / "keyframes"

    @property
    def segment_calib_path(self):
        return self.segment_dir / "calib.txt"

    @property
    def segment_timestamps_path(self):
        return self.segment_dir / "timestamps.txt"

    @property
    def segment_preprocess_meta_path(self):
        return self.segment_dir / "preprocess_meta.json"

    @property
    def segment_clip_prompts_path(self):
        return self.segment_dir / "clip_prompts.json"

    @property
    def prompt_profile_path(self):
        return self.prompt_dir / "profile.json"

    @property
    def prompt_report_path(self):
        return self.prompt_dir / "report.json"

    @property
    def prompt_base_path(self):
        return self.prompt_dir / "base_prompt.json"

    @property
    def prompt_runtime_plan_path(self):
        return self.prompt_dir / "runtime_prompt_plan.json"

    @property
    def infer_runs_dir(self):
        return self.infer_dir / "runs"

    @property
    def infer_runs_plan_path(self):
        return self.infer_dir / "runs_plan.json"

    @property
    def infer_report_path(self):
        return self.infer_dir / "report.json"

    @property
    def merge_frames_dir(self):
        return self.merge_dir / "frames"

    @property
    def merge_calib_path(self):
        return self.merge_dir / "calib.txt"

    @property
    def merge_timestamps_path(self):
        return self.merge_dir / "timestamps.txt"

    @property
    def stats_report_path(self):
        return self.stats_dir / "report.json"

    @property
    def stats_compression_path(self):
        return self.stats_dir / "compression.json"

    def slam_track_dir(self, track):
        return self.slam_dir / str(track)

    def slam_traj_path(self, track):
        return self.slam_track_dir(track) / "traj_est.tum"

    def slam_npz_path(self, track):
        return self.slam_track_dir(track) / "traj_est.npz"

    def slam_run_meta_path(self, track):
        return self.slam_track_dir(track) / "run_meta.json"

    def eval_artifact_path(self, name):
        return self.eval_dir / str(name)


@dataclass(frozen=True)
class ExperimentContext:
    exphub_root: Path
    dataset: str
    sequence: str
    tag: str
    w: int
    h: int
    start_sec: str
    dur: str
    fps: float
    kf_gap_input: int
    exp_root_override: Optional[Path] = None

    @property
    def spec(self):
        return ExperimentSpec(
            exphub_root=self.exphub_root,
            dataset=self.dataset,
            sequence=self.sequence,
            tag=self.tag,
            w=self.w,
            h=self.h,
            start_sec=self.start_sec,
            dur=self.dur,
            fps=self.fps,
            kf_gap_input=self.kf_gap_input,
            exp_root_override=self.exp_root_override,
        )

    @property
    def paths(self):
        return ExperimentPaths.from_spec(self.spec)

    @staticmethod
    def _canon_num_str(value):
        return canon_num_str(value)

    @staticmethod
    def _dot_to_p(value):
        return dot_to_p(value)

    @staticmethod
    def resolve_kf_gap(fps, kf_gap):
        return ExperimentSpec.resolve_kf_gap(fps, kf_gap)

    @staticmethod
    def build_exp_name(tag, w, h, start_sec, dur, fps, kf_gap):
        return ExperimentSpec.build_exp_name(tag, w, h, start_sec, dur, fps, kf_gap)

    @staticmethod
    def compute_segment_count(frames_avail, base_idx, kf_gap, requested_segments=0):
        return ExperimentSpec.compute_segment_count(frames_avail, base_idx, kf_gap, requested_segments)

    @property
    def kf_gap(self):
        return self.spec.kf_gap

    @property
    def exp_name(self):
        return self.spec.exp_name

    @property
    def exp_root(self):
        return self.paths.exp_root

    @property
    def exp_dir(self):
        return self.paths.exp_dir

    @property
    def segment_dir(self):
        return self.paths.segment_dir

    @property
    def prompt_dir(self):
        return self.paths.prompt_dir

    @property
    def infer_dir(self):
        return self.paths.infer_dir

    @property
    def merge_dir(self):
        return self.paths.merge_dir

    @property
    def slam_dir(self):
        return self.paths.slam_dir

    @property
    def eval_dir(self):
        return self.paths.eval_dir

    @property
    def stats_dir(self):
        return self.paths.stats_dir

    @property
    def logs_dir(self):
        return self.paths.logs_dir

    @property
    def exp_meta_path(self):
        return self.paths.exp_meta_path

    @property
    def segment_frames_dir(self):
        return self.paths.segment_frames_dir

    @property
    def segment_keyframes_dir(self):
        return self.paths.segment_keyframes_dir

    @property
    def segment_calib_path(self):
        return self.paths.segment_calib_path

    @property
    def segment_timestamps_path(self):
        return self.paths.segment_timestamps_path

    @property
    def segment_preprocess_meta_path(self):
        return self.paths.segment_preprocess_meta_path

    @property
    def segment_clip_prompts_path(self):
        return self.paths.segment_clip_prompts_path

    @property
    def prompt_profile_path(self):
        return self.paths.prompt_profile_path

    @property
    def prompt_report_path(self):
        return self.paths.prompt_report_path

    @property
    def prompt_base_path(self):
        return self.paths.prompt_base_path

    @property
    def prompt_runtime_plan_path(self):
        return self.paths.prompt_runtime_plan_path

    @property
    def infer_runs_dir(self):
        return self.paths.infer_runs_dir

    @property
    def infer_runs_plan_path(self):
        return self.paths.infer_runs_plan_path

    @property
    def infer_report_path(self):
        return self.paths.infer_report_path

    @property
    def merge_frames_dir(self):
        return self.paths.merge_frames_dir

    @property
    def merge_calib_path(self):
        return self.paths.merge_calib_path

    @property
    def merge_timestamps_path(self):
        return self.paths.merge_timestamps_path

    @property
    def stats_report_path(self):
        return self.paths.stats_report_path

    @property
    def stats_compression_path(self):
        return self.paths.stats_compression_path

    def slam_track_dir(self, track):
        return self.paths.slam_track_dir(track)

    def slam_traj_path(self, track):
        return self.paths.slam_traj_path(track)

    def slam_npz_path(self, track):
        return self.paths.slam_npz_path(track)

    def slam_run_meta_path(self, track):
        return self.paths.slam_run_meta_path(track)

    def eval_artifact_path(self, name):
        return self.paths.eval_artifact_path(name)

    def frames_available(self):
        frames_dir = self.segment_frames_dir
        if not frames_dir.is_dir():
            return 0
        return len([path for path in frames_dir.glob("*.png") if path.is_file()])

    def segment_count(self, base_idx, requested_segments=0):
        return self.compute_segment_count(
            frames_avail=self.frames_available(),
            base_idx=base_idx,
            kf_gap=self.kf_gap,
            requested_segments=requested_segments,
        )
