from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict


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
