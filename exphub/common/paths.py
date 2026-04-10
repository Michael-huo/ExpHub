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
    def segment_aligned_plan_path(self):
        return self.segment_dir / "aligned_segment_plan.json"

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
    def prompt_dir(self):
        return self.exp_dir / "prompt"

    @property
    def infer_dir(self):
        return self.exp_dir / "infer"

    @property
    def merge_dir(self):
        return self.exp_dir / "merge"

    @property
    def eval_dir(self):
        return self.exp_dir / "eval"

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
    def prompt_report_path(self):
        return self.prompt_dir / "report.json"

    @property
    def prompt_manifest_path(self):
        return self.prompt_dir / "prompt_manifest.json"

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
    def merge_manifest_path(self):
        return self.merge_dir / "merge_manifest.json"

    @property
    def merge_report_path(self):
        return self.merge_dir / "report.json"

    @property
    def merge_calib_path(self):
        return self.merge_dir / "calib.txt"

    @property
    def merge_timestamps_path(self):
        return self.merge_dir / "timestamps.txt"

    @property
    def eval_slam_dir(self):
        return self.eval_dir / "slam"

    @property
    def eval_slam_report_path(self):
        return self.eval_slam_dir / "report.json"

    @property
    def eval_slam_primary_traj_path(self):
        return self.eval_slam_dir / "traj_est.txt"

    @property
    def eval_report_path(self):
        return self.eval_dir / "report.json"

    @property
    def eval_compression_path(self):
        return self.eval_dir / "compression.json"

    @property
    def eval_summary_path(self):
        return self.eval_dir / "summary.txt"

    @property
    def eval_metrics_dir(self):
        return self.eval_dir / "metrics"

    @property
    def eval_plots_dir(self):
        return self.eval_dir / "plots"

    @property
    def eval_details_path(self):
        return self.eval_dir / "details.csv"

    @property
    def eval_traj_metrics_path(self):
        return self.eval_metrics_dir / "traj_eval.json"

    @property
    def eval_traj_plot_path(self):
        return self.eval_plots_dir / "traj_xy.png"

    @property
    def eval_metrics_overview_path(self):
        return self.eval_plots_dir / "metrics_overview.png"


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
    def eval_dir(self):
        return self.paths.eval_dir

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
    def segment_aligned_plan_path(self):
        return self.paths.segment_aligned_plan_path

    @property
    def prompt_report_path(self):
        return self.paths.prompt_report_path

    @property
    def prompt_manifest_path(self):
        return self.paths.prompt_manifest_path

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
    def eval_report_path(self):
        return self.paths.eval_report_path

    @property
    def eval_compression_path(self):
        return self.paths.eval_compression_path

    @property
    def eval_summary_path(self):
        return self.paths.eval_summary_path

    @property
    def eval_details_path(self):
        return self.paths.eval_details_path

    @property
    def eval_traj_metrics_path(self):
        return self.paths.eval_traj_metrics_path

    @property
    def eval_traj_plot_path(self):
        return self.paths.eval_traj_plot_path

    @property
    def eval_metrics_overview_path(self):
        return self.paths.eval_metrics_overview_path

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
