from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional


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

    @staticmethod
    def _canon_num_str(x):
        """Canonicalize numeric string: avoid sci notation, drop trailing zeros."""
        s = str(x or "").strip()
        if not s:
            return "0"
        d = Decimal(s)
        if d == d.to_integral():
            return str(int(d))
        s2 = format(d.normalize(), "f")
        if "." in s2:
            s2 = s2.rstrip("0").rstrip(".")
        return s2

    @staticmethod
    def _dot_to_p(s):
        return str(s).replace(".", "p")

    @staticmethod
    def resolve_kf_gap(fps, kf_gap):
        fps_i = int(round(float(fps)))
        g = int(kf_gap)
        if g <= 0:
            g = fps_i - (fps_i % 4)
        if g <= 0:
            g = max(1, fps_i)
        return g

    @staticmethod
    def build_exp_name(tag, w, h, start_sec, dur, fps, kf_gap):
        start_tag = ExperimentContext._dot_to_p(ExperimentContext._canon_num_str(start_sec))
        dur_tag = ExperimentContext._dot_to_p(ExperimentContext._canon_num_str(dur))
        fps_f = float(fps)
        fps_tag = str(int(round(fps_f))) if fps_f.is_integer() else ExperimentContext._canon_num_str(fps)
        return "{tag}_{w}x{h}_t{start}s_dur{dur}s_fps{fps}_gap{kf_gap}".format(
            tag=tag,
            w=int(w),
            h=int(h),
            start=start_tag,
            dur=dur_tag,
            fps=fps_tag,
            kf_gap=int(kf_gap),
        )

    @staticmethod
    def compute_segment_count(frames_avail, base_idx, kf_gap, requested_segments=0):
        frames_avail_i = int(frames_avail)
        base_idx_i = int(base_idx)
        kf_gap_i = int(kf_gap)
        if kf_gap_i <= 0:
            raise ValueError("kf_gap must be > 0")
        max_segments = (frames_avail_i - 1 - base_idx_i) // kf_gap_i
        req = int(requested_segments)
        if req > 0:
            return min(max_segments, req)
        return max_segments

    @property
    def kf_gap(self):
        return self.resolve_kf_gap(self.fps, self.kf_gap_input)

    @property
    def exp_name(self):
        return self.build_exp_name(
            tag=self.tag,
            w=self.w,
            h=self.h,
            start_sec=self.start_sec,
            dur=self.dur,
            fps=self.fps,
            kf_gap=self.kf_gap,
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
    def prompt_final_path(self):
        return self.prompt_dir / "final_prompt.json"

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

    def frames_available(self):
        if not self.segment_frames_dir.is_dir():
            return 0
        return len([p for p in self.segment_frames_dir.glob("*.png") if p.is_file()])

    def segment_count(self, base_idx, requested_segments=0):
        return self.compute_segment_count(
            frames_avail=self.frames_available(),
            base_idx=base_idx,
            kf_gap=self.kf_gap,
            requested_segments=requested_segments,
        )
