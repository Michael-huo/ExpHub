from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


StageName = str
KeepLevel = str

STAGE_ORDER: Tuple[StageName, ...] = ("encode", "decode", "eval")


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
    def build_exp_name(tag, w, h, start_sec, dur, fps, kf_gap) -> str:
        start_tag = dot_to_p(canon_num_str(start_sec))
        dur_tag = dot_to_p(canon_num_str(dur))
        fps_f = float(fps)
        if fps_f.is_integer():
            fps_tag = str(int(round(fps_f)))
        else:
            fps_tag = canon_num_str(fps)
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
