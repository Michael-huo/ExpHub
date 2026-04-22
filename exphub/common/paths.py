from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from exphub.meta import ExperimentSpec


@dataclass(frozen=True)
class ExperimentPaths:
    exphub_root: Path
    mode: str
    dataset: str
    sequence: str
    exp_name: str
    scope: str
    exp_root_override: Optional[Path] = None

    @classmethod
    def from_spec(cls, spec: ExperimentSpec):
        return cls(
            exphub_root=Path(spec.exphub_root).resolve(),
            mode=str(spec.mode),
            dataset=str(spec.dataset),
            sequence=str(spec.sequence),
            exp_name=str(spec.exp_name),
            scope=str(spec.scope),
            exp_root_override=Path(spec.exp_root_override).resolve() if spec.exp_root_override is not None else None,
        )

    @property
    def exp_root(self) -> Path:
        if self.exp_root_override is not None:
            return self.exp_root_override
        if str(self.mode or "").strip().lower() == "train":
            return self.exphub_root / "artifacts" / "train" / self.dataset
        return self.exphub_root / "artifacts" / "infer" / self.dataset / self.sequence

    @property
    def exp_dir(self) -> Path:
        return (self.exp_root / self.exp_name).resolve()

    @property
    def prepare_dir(self) -> Path:
        return self.exp_dir / "prepare"

    @property
    def prepare_frames_dir(self) -> Path:
        return self.prepare_dir / "frames"

    @property
    def prepare_result_path(self) -> Path:
        return self.prepare_dir / "prepare_result.json"

    @property
    def prepare_sequences_dir(self) -> Path:
        return self.prepare_dir / "sequences"

    @property
    def prepare_dataset_index_path(self) -> Path:
        return self.prepare_dir / "dataset_prepare_index.json"

    def prepare_sequence_dir(self, sequence: str) -> Path:
        return self.prepare_sequences_dir / str(sequence)

    def prepare_sequence_frames_dir(self, sequence: str) -> Path:
        return self.prepare_sequence_dir(sequence) / "frames"

    def prepare_sequence_result_path(self, sequence: str) -> Path:
        return self.prepare_sequence_dir(sequence) / "prepare_result.json"

    @property
    def encode_dir(self) -> Path:
        return self.exp_dir / "encode"

    @property
    def encode_motion_segments_path(self) -> Path:
        return self.encode_dir / "motion_segments.json"

    @property
    def encode_semantic_anchors_path(self) -> Path:
        return self.encode_dir / "semantic_anchors.json"

    @property
    def encode_generation_units_path(self) -> Path:
        return self.encode_dir / "generation_units.json"

    @property
    def encode_prompts_path(self) -> Path:
        return self.encode_dir / "prompts.json"

    @property
    def encode_result_path(self) -> Path:
        return self.encode_dir / "encode_result.json"

    @property
    def encode_overview_path(self) -> Path:
        return self.encode_dir / "encode_overview.png"

    @property
    def encode_sequences_dir(self) -> Path:
        return self.encode_dir / "sequences"

    @property
    def encode_dataset_index_path(self) -> Path:
        return self.encode_dir / "dataset_encode_index.json"

    def encode_sequence_dir(self, sequence: str) -> Path:
        return self.encode_sequences_dir / str(sequence)

    def encode_sequence_motion_segments_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "motion_segments.json"

    def encode_sequence_semantic_anchors_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "semantic_anchors.json"

    def encode_sequence_generation_units_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "generation_units.json"

    def encode_sequence_prompts_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "prompts.json"

    def encode_sequence_result_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "encode_result.json"

    def encode_sequence_overview_path(self, sequence: str) -> Path:
        return self.encode_sequence_dir(sequence) / "encode_overview.png"

    @property
    def trainset_dir(self) -> Path:
        return self.exp_dir / "trainset"

    @property
    def trainset_clips_dir(self) -> Path:
        return self.trainset_dir / "clips"

    @property
    def trainset_metadata_path(self) -> Path:
        return self.trainset_dir / "metadata.json"

    @property
    def trainset_stats_path(self) -> Path:
        return self.trainset_dir / "stats.json"

    @property
    def decode_dir(self) -> Path:
        return self.exp_dir / "decode"

    @property
    def eval_dir(self) -> Path:
        return self.exp_dir / "eval"

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs"

    @property
    def run_meta_path(self) -> Path:
        return self.exp_dir / "run_meta.json"

    @property
    def decode_runs_dir(self) -> Path:
        return self.decode_dir / "runs"

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
