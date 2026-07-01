from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from exphub.common.io import read_json_dict, write_json_atomic, write_text_atomic, write_yaml_atomic
from exphub.config import get_platform_config


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def _run_git(root: Path, args) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *list(args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def _git_dirty(root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return bool(str(proc.stdout or "").strip())


def git_state(root: Path) -> Dict[str, object]:
    repo = Path(root).resolve()
    return {
        "commit": _run_git(repo, ["rev-parse", "HEAD"]),
        "branch": _run_git(repo, ["branch", "--show-current"]),
        "worktree_dirty": bool(_git_dirty(repo)),
    }


def command_text(argv) -> str:
    return shlex.join(["python3", "-m", "exphub", *[str(item) for item in tuple(argv or ())]]) + "\n"


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _platform_subset(runtime) -> Dict[str, object]:
    stages = set(runtime.execution_plan.stages)
    if not stages.intersection({"decode", "eval", "lora"}):
        return {}
    cfg = get_platform_config(exphub_root=runtime.exphub_root)
    out: Dict[str, object] = {}
    environments = dict(cfg.get("environments") or {})
    phases = dict(environments.get("phases") or {})
    selected_phases = {}
    for phase in ("semantic_openclip", "lora", "slam", "decode"):
        if phase in phases and (
            (phase == "semantic_openclip" and "encode" in stages)
            or (phase == "lora" and "lora" in stages)
            or (phase == "slam" and "eval" in stages)
            or (phase == "decode" and "decode" in stages)
        ):
            selected_phases[phase] = dict(phases.get(phase) or {})
    if selected_phases:
        out["environments"] = {"phases": selected_phases}
    if "decode" in stages:
        comfyui = dict(dict(cfg.get("services") or {}).get("comfyui") or {})
        active_profile = str(runtime.config.decode_profile or comfyui.get("active_profile") or "")
        profiles = dict(comfyui.get("profiles") or {})
        selected = dict(profiles.get(active_profile) or {}) if active_profile else {}
        out["services"] = {
                "comfyui": {
                    "active_profile": comfyui.get("active_profile"),
                    "selected_profile": active_profile,
                    "profile": selected,
                }
            }
    if "eval" in stages:
        out["repos"] = {"droid_slam": dict(cfg.get("repos") or {}).get("droid_slam")}
        out["models"] = {"droid": dict(dict(cfg.get("models") or {}).get("droid") or {})}
    if "lora" in stages:
        repos = dict(cfg.get("repos") or {})
        out.setdefault("repos", {})["videox_fun"] = repos.get("videox_fun")
    return _as_plain(out)


def _dataset_subset(runtime) -> Dict[str, object]:
    if "prepare" not in set(runtime.execution_plan.stages):
        return {}
    cfg = _read_json(runtime.cfg_path)
    dataset = str(runtime.config.dataset)
    sequence = str(runtime.config.sequence or "")
    datasets = cfg.get("datasets")
    if not isinstance(datasets, dict):
        return {"config_path": str(runtime.cfg_path)}
    selected = datasets.get(dataset)
    if not isinstance(selected, dict):
        return {"config_path": str(runtime.cfg_path), "dataset": dataset}
    out: Dict[str, object] = {"config_path": str(runtime.cfg_path), "dataset": dataset}
    for key in ("root", "bag_root", "rosbag_root", "topic", "image_topic", "camera_topic", "gt_root"):
        if key in selected:
            out[key] = selected.get(key)
    sequences = selected.get("sequences")
    if sequence and isinstance(sequences, dict) and isinstance(sequences.get(sequence), dict):
        out["sequence"] = {sequence: dict(sequences.get(sequence) or {})}
    return _as_plain(out)


def effective_config(runtime) -> Dict[str, object]:
    stages = tuple(runtime.execution_plan.stages)
    payload: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "mode": runtime.execution_plan.mode,
            "requested_step": runtime.execution_plan.requested_step,
            "resolved_step": runtime.execution_plan.resolved_step,
            "stages": list(stages),
            "experiments": list(runtime.execution_plan.experiments),
        },
        "inputs": {
            "dataset": runtime.config.dataset,
            "sequence": runtime.config.sequence,
            "tag": runtime.config.tag,
            "fps": runtime.config.fps,
            "start": runtime.config.start,
            "dur": runtime.config.dur,
            "seed": runtime.config.seed,
            "decode_profile": runtime.config.decode_profile,
            "log_level": runtime.config.log_level,
        },
    }
    if "encode" in stages:
        from exphub.encode.encode import ENCODE_SEGMENT_POLICY

        payload["encode"] = {"segment_policy": ENCODE_SEGMENT_POLICY}
    dataset_cfg = _dataset_subset(runtime)
    if dataset_cfg:
        payload["dataset_config"] = dataset_cfg
    platform_cfg = _platform_subset(runtime)
    if platform_cfg:
        payload["platform"] = platform_cfg
    return _as_plain(payload)


def _base_run_meta(runtime, status: str, *, start_time: str, end_time=None, error=None) -> Dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": runtime.execution_plan.mode,
        "requested_step": runtime.execution_plan.requested_step,
        "resolved_step": runtime.execution_plan.resolved_step,
        "stages": list(runtime.execution_plan.stages),
        "experiments": list(runtime.execution_plan.experiments),
        "dataset": runtime.config.dataset,
        "sequence": runtime.config.sequence,
        "tag": runtime.config.tag,
        "fps": runtime.config.fps,
        "start": runtime.config.start,
        "dur": runtime.config.dur,
        "seed": runtime.config.seed,
        "decode_profile": runtime.config.decode_profile,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "updated_at": _now(),
    }
    if error is not None:
        payload["error"] = str(error)
    return payload


def write_run_start(runtime, argv) -> str:
    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    start_time = _now()
    write_json_atomic(runtime.paths.run_meta_path, _base_run_meta(runtime, "running", start_time=start_time), indent=2)
    write_yaml_atomic(runtime.paths.effective_config_path, effective_config(runtime))
    write_text_atomic(runtime.paths.command_path, command_text(argv))
    write_json_atomic(runtime.paths.git_state_path, git_state(runtime.exphub_root), indent=2)
    return start_time


def update_run_status(runtime, *, status: str, start_time: str, error=None) -> None:
    existing = read_json_dict(runtime.paths.run_meta_path)
    start = str(existing.get("start_time") or start_time)
    concise_error = None
    if error is not None:
        concise_error = str(error).replace("\n", " ").strip()
        if len(concise_error) > 500:
            concise_error = concise_error[:497] + "..."
    write_json_atomic(
        runtime.paths.run_meta_path,
        _base_run_meta(runtime, status, start_time=start, end_time=_now(), error=concise_error),
        indent=2,
    )


__all__ = [
    "command_text",
    "effective_config",
    "git_state",
    "update_run_status",
    "write_run_start",
]
