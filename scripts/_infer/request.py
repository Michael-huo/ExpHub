from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class InferRequest(object):
    frames_dir: Path
    exp_dir: Path
    prompt_manifest_path: Path
    fps: int
    kf_gap: int
    base_idx: int
    num_segments: int
    seed_base: int
    gpus: int
    schedule_source: str
    execution_backend: str
    prompt_policy: str = "structured"
    execution_segments: List[Dict[str, object]] = field(default_factory=list)
    infer_extra: List[str] = field(default_factory=list)
