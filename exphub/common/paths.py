from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from exphub.meta import ExperimentSpec


@dataclass(frozen=True)
class ExperimentPaths:
    exphub_root: Path
    dataset: str
    sequence: str
    exp_name: str
    exp_root_override: Optional[Path] = None

    @classmethod
    def from_spec(cls, spec: ExperimentSpec):
        return cls(
            exphub_root=Path(spec.exphub_root).resolve(),
            dataset=str(spec.dataset),
            sequence=str(spec.sequence),
            exp_name=str(spec.exp_name),
            exp_root_override=Path(spec.exp_root_override).resolve() if spec.exp_root_override is not None else None,
        )

    @property
    def exp_root(self) -> Path:
        if self.exp_root_override is not None:
            return self.exp_root_override
        return self.exphub_root / "experiments" / self.dataset / self.sequence

    @property
    def exp_dir(self) -> Path:
        return (self.exp_root / self.exp_name).resolve()

    @property
    def segment_dir(self) -> Path:
        return self.exp_dir / "segment"

    @property
    def prompt_dir(self) -> Path:
        return self.exp_dir / "prompt"

    @property
    def infer_dir(self) -> Path:
        return self.exp_dir / "infer"

    @property
    def merge_dir(self) -> Path:
        return self.exp_dir / "merge"

    @property
    def eval_dir(self) -> Path:
        return self.exp_dir / "eval"

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs"

    @property
    def default_export_root(self) -> Path:
        return (self.exphub_root / "exports").resolve()

    @property
    def exp_meta_path(self) -> Path:
        return self.exp_dir / "exp_meta.json"

    @property
    def segment_manifest_path(self) -> Path:
        return self.segment_dir / "segment_manifest.json"

    @property
    def segment_report_path(self) -> Path:
        return self.segment_dir / "report.json"

    @property
    def segment_visuals_dir(self) -> Path:
        return self.segment_dir / "visuals"

    @property
    def segment_motion_score_path(self) -> Path:
        return self.segment_dir / "motion_score.json"

    @property
    def segment_semantic_shift_path(self) -> Path:
        return self.segment_dir / "semantic_shift.json"

    @property
    def segment_generation_risk_path(self) -> Path:
        return self.segment_dir / "generation_risk.json"

    @property
    def segment_candidate_boundaries_path(self) -> Path:
        return self.segment_dir / "candidate_boundaries.json"

    @property
    def segment_generation_units_path(self) -> Path:
        return self.segment_dir / "generation_units.json"

    @property
    def segment_state_overview_path(self) -> Path:
        return self.segment_visuals_dir / "state_overview.png"

    @property
    def segment_frames_dir(self) -> Path:
        return self.segment_dir / "frames"

    @property
    def segment_keyframes_dir(self) -> Path:
        return self.segment_dir / "keyframes"

    @property
    def segment_calib_path(self) -> Path:
        return self.segment_dir / "calib.txt"

    @property
    def segment_timestamps_path(self) -> Path:
        return self.segment_dir / "timestamps.txt"

    @property
    def prompt_report_path(self) -> Path:
        return self.prompt_dir / "report.json"

    @property
    def prompt_manifest_path(self) -> Path:
        return self.prompt_dir / "prompt_manifest.json"

    @property
    def prompt_spans_path(self) -> Path:
        return self.prompt_dir / "prompt_spans.json"

    @property
    def infer_runs_dir(self) -> Path:
        return self.infer_dir / "runs"

    @property
    def infer_runs_plan_path(self) -> Path:
        return self.infer_dir / "runs_plan.json"

    @property
    def infer_report_path(self) -> Path:
        return self.infer_dir / "report.json"

    @property
    def merge_frames_dir(self) -> Path:
        return self.merge_dir / "frames"

    @property
    def merge_manifest_path(self) -> Path:
        return self.merge_dir / "merge_manifest.json"

    @property
    def merge_report_path(self) -> Path:
        return self.merge_dir / "report.json"

    @property
    def merge_calib_path(self) -> Path:
        return self.merge_dir / "calib.txt"

    @property
    def merge_timestamps_path(self) -> Path:
        return self.merge_dir / "timestamps.txt"

    @property
    def eval_slam_dir(self) -> Path:
        return self.eval_dir / "slam"

    @property
    def eval_slam_report_path(self) -> Path:
        return self.eval_slam_dir / "report.json"

    @property
    def eval_slam_primary_traj_path(self) -> Path:
        return self.eval_slam_dir / "traj_est.txt"

    @property
    def eval_report_path(self) -> Path:
        return self.eval_dir / "report.json"

    @property
    def eval_compression_path(self) -> Path:
        return self.eval_dir / "compression.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.eval_dir / "summary.txt"

    @property
    def eval_metrics_dir(self) -> Path:
        return self.eval_dir / "metrics"

    @property
    def eval_plots_dir(self) -> Path:
        return self.eval_dir / "plots"

    @property
    def eval_details_path(self) -> Path:
        return self.eval_dir / "details.csv"

    @property
    def eval_traj_metrics_path(self) -> Path:
        return self.eval_metrics_dir / "traj_eval.json"

    @property
    def eval_traj_plot_path(self) -> Path:
        return self.eval_plots_dir / "traj_xy.png"

    @property
    def eval_metrics_overview_path(self) -> Path:
        return self.eval_plots_dir / "metrics_overview.png"

    # Stage-name aliases for the current mainline.
    @property
    def input_dir(self) -> Path:
        return self.segment_dir

    @property
    def encode_dir(self) -> Path:
        return self.segment_dir

    @property
    def decode_dir(self) -> Path:
        return self.infer_dir
