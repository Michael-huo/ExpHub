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


_PLATFORM_CONFIG: Optional[Dict[str, Any]] = None


def load_datasets_cfg(cfg_path) -> Dict[str, Any]:
    try:
        return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError("Failed to parse datasets config: {} ({})".format(cfg_path, exc))


def resolve_dataset(cfg, exphub_root, dataset, sequence) -> DatasetResolved:
    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        raise ConfigError("datasets.json missing top-level 'datasets' dict")
    if dataset not in datasets:
        raise ConfigError("Dataset not found in config: {}".format(dataset))

    dataset_cfg = datasets[dataset] or {}
    fmt = str(dataset_cfg.get("format", "")).strip().lower()
    if fmt and fmt != "rosbag":
        raise ConfigError("Unsupported dataset format: {} (only rosbag)".format(fmt))

    root = str(dataset_cfg.get("root", "")).strip()
    topic = str(dataset_cfg.get("topic", "")).strip()
    if not root:
        raise ConfigError("Missing datasets.{}.root".format(dataset))
    if not topic:
        raise ConfigError("Missing datasets.{}.topic".format(dataset))

    intrinsics = dataset_cfg.get("intrinsics") or {}
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except Exception as exc:
        raise ConfigError(
            "Missing/invalid intrinsics fx/fy/cx/cy in datasets.{}.intrinsics".format(dataset)
        ) from exc
    if fx <= 0 or fy <= 0:
        raise ConfigError("Invalid intrinsics (fx/fy must be >0): fx={} fy={}".format(fx, fy))

    dist_value = intrinsics.get("dist") or []
    if dist_value is None:
        dist_value = []
    if not isinstance(dist_value, list):
        raise ConfigError("intrinsics.dist must be a list")
    dist = [float(item) for item in dist_value]

    sequences = dataset_cfg.get("sequences") or {}
    if sequence not in sequences:
        raise ConfigError("Sequence not found in config: {}/{}".format(dataset, sequence))
    sequence_cfg = sequences[sequence] or {}
    bag_name = str(sequence_cfg.get("bag", "")).strip()
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


def get_platform_config(exphub_root=None) -> Dict[str, Any]:
    global _PLATFORM_CONFIG
    if _PLATFORM_CONFIG is not None:
        return _PLATFORM_CONFIG

    root = Path(exphub_root).resolve() if exphub_root else Path(__file__).resolve().parents[1]
    config_path = root / "config" / "platform.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Platform configuration not found: {}\n"
            "Please copy config/platform.yaml.example to config/platform.yaml and configure your paths.".format(
                config_path
            )
        )

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    _PLATFORM_CONFIG = loaded if isinstance(loaded, dict) else {}
    return _PLATFORM_CONFIG


def get_phase_python_config(phase_name, exphub_root=None) -> Optional[str]:
    cfg = get_platform_config(exphub_root=exphub_root)
    phases_cfg = cfg.get("environments", {}).get("phases", {})
    if not isinstance(phases_cfg, dict):
        return None
    phase_cfg = phases_cfg.get(str(phase_name), {})
    if not isinstance(phase_cfg, dict):
        return None
    python_bin = str(phase_cfg.get("python", "") or "").strip()
    return python_bin or None


__all__ = [
    "ConfigError",
    "DatasetResolved",
    "get_phase_python_config",
    "get_platform_config",
    "load_datasets_cfg",
    "resolve_dataset",
]
