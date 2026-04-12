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
    def input_dir(self) -> Path:
        return self.exp_dir / "input"

    @property
    def encode_dir(self) -> Path:
        return self.exp_dir / "encode"

    @property
    def decode_dir(self) -> Path:
        return self.exp_dir / "decode"

    @property
    def eval_dir(self) -> Path:
        return self.exp_dir / "eval"

    @property
    def export_dir(self) -> Path:
        return self.exp_dir / "export"

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs"

    @property
    def default_export_root(self) -> Path:
        return self.export_dir.resolve()

    @property
    def exp_meta_path(self) -> Path:
        return self.exp_dir / "exp_meta.json"

    @property
    def input_report_path(self) -> Path:
        return self.input_dir / "input_report.json"

    @property
    def input_frames_dir(self) -> Path:
        return self.input_dir / "frames"

    @property
    def encode_plan_path(self) -> Path:
        return self.encode_dir / "encode_plan.json"

    @property
    def prompt_spans_path(self) -> Path:
        return self.encode_dir / "prompt_spans.json"

    @property
    def encode_report_path(self) -> Path:
        return self.encode_dir / "encode_report.json"

    @property
    def encode_segmentation_overview_path(self) -> Path:
        return self.encode_dir / "encode_segmentation_overview.png"

    @property
    def decode_runs_dir(self) -> Path:
        return self.decode_dir / "runs"

    @property
    def decode_plan_path(self) -> Path:
        return self.decode_dir / "decode_plan.json"

    @property
    def decode_report_path(self) -> Path:
        return self.decode_dir / "decode_report.json"

    @property
    def decode_frames_dir(self) -> Path:
        return self.decode_dir / "frames"

    @property
    def decode_merge_report_path(self) -> Path:
        return self.decode_dir / "decode_merge_report.json"

    @property
    def decode_calib_path(self) -> Path:
        return self.decode_dir / "calib.txt"

    @property
    def decode_timestamps_path(self) -> Path:
        return self.decode_dir / "timestamps.txt"

    @property
    def decode_preview_path(self) -> Path:
        return self.decode_dir / "preview.mp4"

    @property
    def eval_slam_report_path(self) -> Path:
        return self.eval_dir / "eval_slam_report.json"

    @property
    def eval_traj_report_path(self) -> Path:
        return self.eval_dir / "eval_traj_report.json"

    @property
    def eval_compression_report_path(self) -> Path:
        return self.eval_dir / "eval_compression_report.json"

    @property
    def eval_report_path(self) -> Path:
        return self.eval_dir / "eval_report.json"

    @property
    def eval_summary_path(self) -> Path:
        return self.eval_dir / "eval_summary.txt"

    @property
    def eval_details_path(self) -> Path:
        return self.eval_dir / "eval_details.csv"

    @property
    def eval_traj_plot_path(self) -> Path:
        return self.eval_dir / "eval_traj_xy.png"

    @property
    def eval_metrics_overview_path(self) -> Path:
        return self.eval_dir / "eval_metrics_overview.png"

    @property
    def eval_slam_primary_traj_path(self) -> Path:
        return self.eval_dir / "traj_est.txt"

    @property
    def export_report_path(self) -> Path:
        return self.export_dir / "export_report.json"

    @property
    def export_dataset_report_path(self) -> Path:
        return self.export_dir / "export_dataset_report.json"

    @property
    def export_clips_dir(self) -> Path:
        return self.export_dir / "clips"

    @property
    def export_metadata_dir(self) -> Path:
        return self.export_dir / "metadata"

    @property
    def export_clip_manifests_dir(self) -> Path:
        return self.export_dir / "clip_manifests"
