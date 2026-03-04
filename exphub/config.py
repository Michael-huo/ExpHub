from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetResolved:
    dataset: str
    sequence: str
    bag: Path
    topic: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: List[float]


def load_datasets_cfg(cfg_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ConfigError(f"Failed to parse datasets config: {cfg_path} ({e})")


def resolve_dataset(cfg: Dict[str, Any], exphub_root: Path, dataset: str, sequence: str) -> DatasetResolved:
    ds_all = cfg.get("datasets")
    if not isinstance(ds_all, dict):
        raise ConfigError("datasets.json missing top-level 'datasets' dict")

    if dataset not in ds_all:
        raise ConfigError(f"Dataset not found in config: {dataset}")
    ds = ds_all[dataset] or {}

    fmt = str(ds.get("format", "")).strip().lower()
    if fmt and fmt != "rosbag":
        raise ConfigError(f"Unsupported dataset format: {fmt} (only rosbag)")

    root = str(ds.get("root", "")).strip()
    if not root:
        raise ConfigError(f"Missing datasets.{dataset}.root")

    topic = str(ds.get("topic", "")).strip()
    if not topic:
        raise ConfigError(f"Missing datasets.{dataset}.topic")

    intr = ds.get("intrinsics") or {}
    try:
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    except Exception:
        raise ConfigError(f"Missing/invalid intrinsics fx/fy/cx/cy in datasets.{dataset}.intrinsics")

    if fx <= 0 or fy <= 0:
        raise ConfigError(f"Invalid intrinsics (fx/fy must be >0): fx={fx} fy={fy}")

    dist_val = intr.get("dist") or []
    if dist_val is None:
        dist_val = []
    if not isinstance(dist_val, list):
        raise ConfigError("intrinsics.dist must be a list")
    dist = [float(x) for x in dist_val]

    seqs = ds.get("sequences") or {}
    if sequence not in seqs:
        raise ConfigError(f"Sequence not found in config: {dataset}/{sequence}")
    seq_obj = seqs[sequence] or {}
    bag_name = str(seq_obj.get("bag", "")).strip()
    if not bag_name:
        raise ConfigError(f"Missing bag in datasets.{dataset}.sequences.{sequence}.bag")

    root_abs = (exphub_root / root).resolve() if not Path(root).is_absolute() else Path(root).resolve()
    bag_path = (root_abs / bag_name).resolve()

    return DatasetResolved(
        dataset=dataset,
        sequence=sequence,
        bag=bag_path,
        topic=topic,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist=dist,
    )
