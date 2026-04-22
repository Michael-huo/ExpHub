from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


StageName = str
KeepLevel = str

STAGE_ORDER: Tuple[StageName, ...] = ("prepare", "encode", "decode", "eval")


def sanitize_token(text: str, max_len: int = 64) -> str:
    value = (text or "").strip()
    value = re.sub(r"[^0-9a-zA-Z_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if len(value) > max_len:
        value = value[:max_len]
    return value


def canon_num_str(value) -> str:
    text = str(value or "").strip()
    if not text:
        return "0"
    decimal_value = Decimal(text)
    if decimal_value == decimal_value.to_integral():
        return str(int(decimal_value))
    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def dot_to_p(value) -> str:
    return str(value).replace(".", "p")


def format_intlike(value) -> str:
    try:
        numeric = float(value)
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
    except Exception:
        pass
    return str(value)


def resolve_kf_gap(fps, kf_gap) -> int:
    fps_i = int(round(float(fps)))
    gap = int(kf_gap)
    if gap <= 0:
        gap = fps_i - (fps_i % 4)
    if gap <= 0:
        gap = max(1, fps_i)
    return gap


@dataclass(frozen=True)
class ExperimentSpec:
    exphub_root: Path
    mode: str
    dataset: str
    sequence: str
    tag: str
    start: str
    dur: str
    fps: float
    kf_gap_input: int
    exp_root_override: Optional[Path] = None

    @staticmethod
    def build_exp_name(tag, start, dur, fps) -> str:
        start_tag = dot_to_p(canon_num_str(start))
        dur_tag = dot_to_p(canon_num_str(dur))
        fps_f = float(fps)
        if fps_f.is_integer():
            fps_tag = str(int(round(fps_f)))
        else:
            fps_tag = dot_to_p(canon_num_str(fps))
        return "{tag}_fps{fps}_dur{dur}_start{start}".format(
            tag=tag,
            fps=fps_tag,
            dur=dur_tag,
            start=start_tag,
        )

    @staticmethod
    def build_train_exp_name(tag, fps, sequence="") -> str:
        fps_f = float(fps)
        if fps_f.is_integer():
            fps_tag = str(int(round(fps_f)))
        else:
            fps_tag = dot_to_p(canon_num_str(fps))
        base = "{tag}_fps{fps}".format(tag=sanitize_token(str(tag)), fps=fps_tag)
        seq = sanitize_token(str(sequence or ""))
        if seq:
            return "{}_{}".format(base, seq)
        return base

    @staticmethod
    def compute_segment_count(frames_avail, base_idx, kf_gap, requested_segments=0) -> int:
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
    def kf_gap(self) -> int:
        return resolve_kf_gap(self.fps, self.kf_gap_input)

    @property
    def exp_name(self) -> str:
        if str(self.mode or "").strip().lower() == "train":
            return self.build_train_exp_name(
                tag=self.tag,
                fps=self.fps,
                sequence=self.sequence,
            )
        return self.build_exp_name(
            tag=self.tag,
            start=self.start,
            dur=self.dur,
            fps=self.fps,
        )

    @property
    def scope(self) -> str:
        if str(self.mode or "").strip().lower() == "train" and str(self.sequence or "").strip():
            return "sequence"
        if str(self.mode or "").strip().lower() == "train":
            return "dataset"
        return "sequence"

    @property
    def fps_text(self) -> str:
        return format_intlike(self.fps)


@dataclass
class ExpPaths:
    exphub: Path
    exp_root: Path
    exp_dir: Path
    encode_dir: Path


def write_exp_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_exp_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ExpPaths",
    "ExperimentSpec",
    "KeepLevel",
    "STAGE_ORDER",
    "StageName",
    "canon_num_str",
    "dot_to_p",
    "format_intlike",
    "read_exp_meta",
    "resolve_kf_gap",
    "sanitize_token",
    "write_exp_meta",
]
