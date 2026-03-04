from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional


def sanitize_token(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^0-9a-zA-Z_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s


def canon_num_str(x: str) -> str:
    """Canonicalize numeric string: avoid sci notation, drop trailing zeros."""
    s = (x or "").strip()
    if not s:
        return "0"
    d = Decimal(s)
    if d == d.to_integral():
        return str(int(d))
    s2 = format(d.normalize(), "f")
    if "." in s2:
        s2 = s2.rstrip("0").rstrip(".")
    return s2


def dot_to_p(s: str) -> str:
    return s.replace(".", "p")


def auto_kf_gap(fps: float, kf_gap: int) -> int:
    fps_i = int(round(float(fps)))
    g = int(kf_gap)
    if g <= 0:
        g = fps_i - (fps_i % 4)
    if g <= 0:
        g = max(1, fps_i)
    return g


def build_exp_name(tag: str, w: int, h: int, start_sec: str, dur: str, fps: float, kf_gap: int) -> str:
    start_tag = dot_to_p(canon_num_str(start_sec))
    dur_tag = dot_to_p(canon_num_str(dur))
    fps_tag = str(int(round(float(fps)))) if float(fps).is_integer() else canon_num_str(str(fps))
    return f"{tag}_{w}x{h}_t{start_tag}s_dur{dur_tag}s_fps{fps_tag}_gap{kf_gap}"


@dataclass
class ExpPaths:
    exphub: Path
    exp_root: Path
    exp_dir: Path
    segment_dir: Path


def write_exp_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_exp_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
