from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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


_PLATFORM_CONFIG = None  # type: Optional[Dict[str, Any]]


def load_datasets_cfg(cfg_path):
    try:
        return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError("Failed to parse datasets config: {} ({})".format(cfg_path, exc))


def resolve_dataset(cfg, exphub_root, dataset, sequence):
    ds_all = cfg.get("datasets")
    if not isinstance(ds_all, dict):
        raise ConfigError("datasets.json missing top-level 'datasets' dict")

    if dataset not in ds_all:
        raise ConfigError("Dataset not found in config: {}".format(dataset))
    ds = ds_all[dataset] or {}

    fmt = str(ds.get("format", "")).strip().lower()
    if fmt and fmt != "rosbag":
        raise ConfigError("Unsupported dataset format: {} (only rosbag)".format(fmt))

    root = str(ds.get("root", "")).strip()
    if not root:
        raise ConfigError("Missing datasets.{}.root".format(dataset))

    topic = str(ds.get("topic", "")).strip()
    if not topic:
        raise ConfigError("Missing datasets.{}.topic".format(dataset))

    intr = ds.get("intrinsics") or {}
    try:
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    except Exception:
        raise ConfigError("Missing/invalid intrinsics fx/fy/cx/cy in datasets.{}.intrinsics".format(dataset))

    if fx <= 0 or fy <= 0:
        raise ConfigError("Invalid intrinsics (fx/fy must be >0): fx={} fy={}".format(fx, fy))

    dist_val = intr.get("dist") or []
    if dist_val is None:
        dist_val = []
    if not isinstance(dist_val, list):
        raise ConfigError("intrinsics.dist must be a list")
    dist = [float(item) for item in dist_val]

    seqs = ds.get("sequences") or {}
    if sequence not in seqs:
        raise ConfigError("Sequence not found in config: {}/{}".format(dataset, sequence))
    seq_obj = seqs[sequence] or {}
    bag_name = str(seq_obj.get("bag", "")).strip()
    if not bag_name:
        raise ConfigError("Missing bag in datasets.{}.sequences.{}.bag".format(dataset, sequence))

    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (Path(exphub_root).resolve() / root_path).resolve()
    else:
        root_path = root_path.resolve()
    bag_path = (root_path / bag_name).resolve()

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


def get_platform_config(exphub_root=None):
    global _PLATFORM_CONFIG
    if _PLATFORM_CONFIG is not None:
        return _PLATFORM_CONFIG

    root = Path(exphub_root).resolve() if exphub_root else Path(__file__).resolve().parents[2]
    config_path = root / "config" / "platform.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Platform configuration not found: {}\n"
            "Please copy config/platform.yaml.example to config/platform.yaml and configure your paths.".format(
                config_path
            )
        )

    with open(str(config_path), "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if isinstance(loaded, dict):
        _PLATFORM_CONFIG = loaded
    else:
        _PLATFORM_CONFIG = {}
    return _PLATFORM_CONFIG


def get_phase_python_config(phase_name, exphub_root=None):
    cfg = get_platform_config(exphub_root=exphub_root)
    phases_cfg = cfg.get("environments", {}).get("phases", {})
    if not isinstance(phases_cfg, dict):
        return None

    phase_cfg = phases_cfg.get(str(phase_name), {})
    if not isinstance(phase_cfg, dict):
        return None

    python_bin = str(phase_cfg.get("python", "") or "").strip()
    if not python_bin:
        return None
    return python_bin
