from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from exphub.common.io import remove_path, write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_warn
from exphub.common.subprocess import RunError, run_cmd
from exphub.config import ConfigError, get_phase_python_config, get_platform_config


_LAUNCH_KEYS = {
    "num_processes",
    "mixed_precision",
    "use_fsdp",
    "fsdp_auto_wrap_policy",
    "fsdp_transformer_layer_cls_to_wrap",
    "fsdp_sharding_strategy",
    "fsdp_state_dict_type",
    "fsdp_backward_prefetch",
    "fsdp_cpu_ram_efficient_loading",
}
_EXPAND_ENV = {
    "cuda_visible_devices",
}
_CONTROL_KEYS = _LAUNCH_KEYS | _EXPAND_ENV | {"trainer", "script"}
_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError("Failed to parse json config: {} ({})".format(path, exc))
    if not isinstance(data, dict):
        raise ConfigError("Expected json object: {}".format(path))
    return data


def _lora_profiles_path(runtime) -> Path:
    return runtime.exphub_root / "config" / "lora_profiles.json"


def _profile_name(runtime, profiles_cfg: Dict[str, object]) -> str:
    requested = str(getattr(runtime.args, "lora_profile", "") or "").strip()
    if requested:
        return requested
    default_profile = str(profiles_cfg.get("default_profile", "") or "").strip()
    if not default_profile:
        raise ConfigError("config/lora_profiles.json missing default_profile")
    return default_profile


def _load_profile(runtime) -> Tuple[str, Dict[str, object]]:
    profiles_path = _lora_profiles_path(runtime)
    profiles_cfg = _read_json(profiles_path)
    profiles = profiles_cfg.get("profiles")
    if not isinstance(profiles, dict):
        raise ConfigError("config/lora_profiles.json missing profiles object")
    name = _profile_name(runtime, profiles_cfg)
    raw_profile = profiles.get(name)
    if not isinstance(raw_profile, dict):
        raise ConfigError("LoRA profile not found: {}".format(name))
    profile = dict(raw_profile)
    gpus = str(getattr(runtime.args, "lora_gpus", "") or "").strip()
    if gpus:
        profile["cuda_visible_devices"] = gpus
    epochs = getattr(runtime.args, "lora_epochs", None)
    if epochs is not None:
        profile["num_train_epochs"] = int(epochs)
    resume = str(getattr(runtime.args, "lora_resume", "none") or "none").strip().lower()
    if resume != "none":
        raise RuntimeError("resume={} is not implemented yet".format(resume))
    return name, profile


def _ensure_trainset(runtime) -> Tuple[Path, Path, Path]:
    trainset_dir = Path(runtime.paths.trainset_dir).resolve()
    videos_dir = Path(runtime.paths.trainset_videos_dir).resolve()
    metadata_path = Path(runtime.paths.trainset_metadata_path).resolve()
    stats_path = Path(runtime.paths.trainset_stats_path).resolve()

    if not trainset_dir.is_dir():
        raise RuntimeError("trainset not found, run train encode first: {}".format(trainset_dir))
    if not videos_dir.is_dir():
        raise RuntimeError("trainset/videos not found, run train encode first: {}".format(videos_dir))
    if not list(videos_dir.glob("*.mp4")):
        raise RuntimeError("trainset/videos contains no mp4: {}".format(videos_dir))
    if not metadata_path.is_file():
        raise RuntimeError("train metadata not found: {}".format(metadata_path))
    if not stats_path.is_file():
        raise RuntimeError("trainset stats not found: {}".format(stats_path))
    return trainset_dir, metadata_path, stats_path


def _trainer_paths(runtime, profile: Dict[str, object]) -> Tuple[Path, Path, Path]:
    platform = get_platform_config(runtime.exphub_root)
    repos = platform.get("repos", {})
    if not isinstance(repos, dict):
        repos = {}
    trainer = str(profile.get("trainer", "videox_fun") or "videox_fun").strip()
    if trainer != "videox_fun":
        raise ConfigError("unsupported lora trainer: {}".format(trainer))
    repo = Path(str(repos.get("videox_fun", "") or "").strip()).expanduser()
    if not repo.is_absolute():
        repo = (runtime.exphub_root / repo).resolve()
    else:
        repo = repo.resolve()
    if not repo.is_dir():
        raise RuntimeError("VideoX-Fun repo not found: {}".format(repo))

    script_rel = str(profile.get("script", "") or "").strip()
    if not script_rel:
        raise ConfigError("lora profile missing script")
    script_path = (repo / script_rel).resolve()
    if not script_path.is_file():
        raise RuntimeError("VideoX-Fun train_lora.py not found: {}".format(script_path))

    python_bin = get_phase_python_config("lora", exphub_root=runtime.exphub_root)
    if not python_bin:
        raise RuntimeError("Missing 'environments.phases.lora.python' in config/platform.yaml.")
    python_path = Path(str(python_bin)).expanduser()
    if not python_path.is_absolute():
        python_path = (runtime.exphub_root / python_path).resolve()
    else:
        python_path = python_path.resolve()
    if not python_path.is_file() or not os.access(str(python_path), os.X_OK):
        raise RuntimeError("lora python not found: {}".format(python_path))
    return repo, script_path, python_path


def _resolve_launcher(python_path: Path) -> List[str]:
    env_root = python_path.parent.parent
    accelerate_bin = python_path.parent / "accelerate"
    if accelerate_bin.is_file() and os.access(str(accelerate_bin), os.X_OK):
        return [str(accelerate_bin), "launch"]

    proc = subprocess.run(
        [str(python_path), "-m", "accelerate.commands.launch", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode == 0:
        return [str(python_path), "-m", "accelerate.commands.launch"]
    raise RuntimeError("accelerate launcher not found in lora environment: {}".format(env_root))


def _bool_text(value: object) -> str:
    return "True" if bool(value) else "False"


def _append_launch_args(argv: List[str], profile: Dict[str, object]) -> None:
    argv.append("--num_machines=1")
    argv.append("--dynamo_backend=no")
    if "num_processes" in profile:
        argv.append("--num_processes={}".format(profile["num_processes"]))
    if "mixed_precision" in profile:
        argv.append("--mixed_precision={}".format(profile["mixed_precision"]))
    if bool(profile.get("use_fsdp", False)):
        argv.append("--use_fsdp")
    for key in [
        "fsdp_auto_wrap_policy",
        "fsdp_transformer_layer_cls_to_wrap",
        "fsdp_sharding_strategy",
        "fsdp_state_dict_type",
        "fsdp_backward_prefetch",
    ]:
        value = profile.get(key)
        if value is not None and str(value).strip() != "":
            argv.extend(["--{}".format(key), str(value)])
    if "fsdp_cpu_ram_efficient_loading" in profile:
        argv.extend(["--fsdp_cpu_ram_efficient_loading", _bool_text(profile.get("fsdp_cpu_ram_efficient_loading"))])


def _append_trainer_args(
    argv: List[str],
    profile: Dict[str, object],
    trainset_dir: Path,
    metadata_path: Path,
    output_dir: Path,
) -> None:
    script = str(profile.get("script", "") or "").strip()
    argv.append(script)
    argv.extend(["--train_data_dir={}".format(trainset_dir)])
    argv.extend(["--train_data_meta={}".format(metadata_path)])
    argv.extend(["--output_dir={}".format(output_dir)])
    for key in sorted(profile.keys()):
        if key in _CONTROL_KEYS:
            continue
        if key in ("train_data_dir", "train_data_meta", "output_dir"):
            continue
        value = profile.get(key)
        if value is None:
            continue
        opt = "--{}".format(key)
        if isinstance(value, bool):
            if value:
                argv.append(opt)
            continue
        argv.append("{}={}".format(opt, value))


def _command_script(argv: List[str], repo: Path, env: Dict[str, str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd {}".format(shlex.quote(str(repo))),
        "export CUDA_VISIBLE_DEVICES={}".format(shlex.quote(str(env["CUDA_VISIBLE_DEVICES"]))),
        "export PYTORCH_CUDA_ALLOC_CONF={}".format(shlex.quote(str(env["PYTORCH_CUDA_ALLOC_CONF"]))),
        "export TOKENIZERS_PARALLELISM={}".format(shlex.quote(str(env["TOKENIZERS_PARALLELISM"]))),
        "",
    ]
    if not argv:
        return "\n".join(lines) + "\n"
    command_lines = []
    for idx, item in enumerate(argv):
        suffix = " \\" if idx < len(argv) - 1 else ""
        command_lines.append("  {}{}".format(shlex.quote(str(item)), suffix))
    if command_lines:
        command_lines[0] = command_lines[0].lstrip()
        lines.extend(command_lines)
    return "\n".join(lines) + "\n"


def _read_tail(path: Path, count: int = 30) -> List[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return lines[-int(count) :] if count > 0 else lines


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.search(path.name)
    if not match:
        match = _CHECKPOINT_RE.search(str(path.parent.name))
    if not match:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _make_top_level_compatible_links(lora_dir: Path, compatible: List[Path]) -> None:
    for path in compatible:
        if path.parent == lora_dir:
            continue
        step = _checkpoint_step(path)
        if step < 0:
            continue
        link_path = lora_dir / "checkpoint-{}-compatible_with_comfyui.safetensors".format(step)
        if link_path.exists() or link_path.is_symlink():
            try:
                current = link_path.resolve()
                if current == path.resolve():
                    continue
            except Exception:
                pass
            link_path.unlink()
        target = os.path.relpath(str(path.resolve()), str(lora_dir.resolve()))
        link_path.symlink_to(target)


def _scan_outputs(lora_dir: Path) -> Tuple[List[str], List[str], Optional[str]]:
    checkpoints = [
        p.absolute()
        for pattern in ("checkpoint-*.safetensors", "lora_diffusion_pytorch_model.safetensors")
        for p in lora_dir.rglob(pattern)
        if (p.is_file() or p.is_symlink()) and "compatible_with_comfyui" not in p.name
    ]
    compatible = [
        p.absolute()
        for pattern in (
            "checkpoint-*-compatible_with_comfyui.safetensors",
            "lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors",
        )
        for p in lora_dir.rglob(pattern)
        if p.is_file() or p.is_symlink()
    ]
    _make_top_level_compatible_links(lora_dir, compatible)
    checkpoints = [
        p.absolute()
        for pattern in ("checkpoint-*.safetensors", "lora_diffusion_pytorch_model.safetensors")
        for p in lora_dir.rglob(pattern)
        if (p.is_file() or p.is_symlink()) and "compatible_with_comfyui" not in p.name
    ]
    compatible = [
        p.absolute()
        for pattern in (
            "checkpoint-*-compatible_with_comfyui.safetensors",
            "lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors",
        )
        for p in lora_dir.rglob(pattern)
        if p.is_file() or p.is_symlink()
    ]
    checkpoints = sorted(set(checkpoints), key=lambda p: (_checkpoint_step(p), str(p)))
    compatible = sorted(set(compatible), key=lambda p: (_checkpoint_step(p), str(p)))
    top_level_compatible = [p for p in compatible if p.parent == lora_dir]
    latest_pool = top_level_compatible or compatible or checkpoints
    latest = str(latest_pool[-1]) if latest_pool else None
    return [str(p) for p in checkpoints], [str(p) for p in compatible], latest


def _write_result(
    runtime,
    returncode: int,
    elapsed_sec: float,
    profile_name: str,
    repo: Path,
    trainset_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    failure_message: str = "",
) -> Dict[str, object]:
    checkpoints, compatible, latest = _scan_outputs(output_dir)
    payload = {
        "returncode": int(returncode),
        "elapsed_sec": float(elapsed_sec),
        "profile": str(profile_name),
        "videox_fun_repo": str(repo),
        "train_data_dir": str(trainset_dir),
        "train_data_meta": str(metadata_path),
        "output_dir": str(output_dir),
        "command_path": str(Path(runtime.paths.lora_command_path).resolve()),
        "log_path": str(Path(runtime.paths.lora_log_path).resolve()),
        "checkpoints": checkpoints,
        "comfyui_compatible_loras": compatible,
        "best_or_latest_lora": latest,
    }
    if failure_message:
        payload["failure_message"] = str(failure_message)
        payload["tail_log"] = _read_tail(Path(runtime.paths.lora_log_path), 30)
    write_json_atomic(runtime.paths.lora_result_path, payload, indent=2)
    return payload


def _stop_comfyui_pool(runtime) -> None:
    script = runtime.exphub_root / "exphub" / "comfyui_pool.sh"
    if not script.is_file():
        return
    try:
        proc = subprocess.run(
            ["bash", str(script), "stop"],
            cwd=str(runtime.exphub_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            log_warn("comfyui pool stop failed rc={}: {}".format(proc.returncode, (proc.stdout or "").strip()))
    except Exception as exc:
        log_warn("comfyui pool stop failed: {}".format(exc))


def run(runtime):
    mode = str(runtime.args.mode or "").strip().lower()
    if mode != "train":
        raise RuntimeError("lora stage is only supported in train mode")

    trainset_dir, metadata_path, _stats_path = _ensure_trainset(runtime)
    profile_name, profile = _load_profile(runtime)
    repo, _script_path, python_path = _trainer_paths(runtime, profile)
    launcher = _resolve_launcher(python_path)

    lora_dir = Path(runtime.paths.lora_dir).resolve()
    runtime.assert_under_exp(lora_dir)
    remove_path(lora_dir)
    lora_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "CUDA_VISIBLE_DEVICES": str(profile.get("cuda_visible_devices", "") or "").strip(),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if not env["CUDA_VISIBLE_DEVICES"]:
        raise ConfigError("lora profile missing cuda_visible_devices")

    argv = list(launcher)
    _append_launch_args(argv, profile)
    _append_trainer_args(argv, profile, trainset_dir, metadata_path, lora_dir)

    resolved_config = {
        "profile": profile_name,
        "profile_config": profile,
        "overrides": {
            "lora_gpus": str(getattr(runtime.args, "lora_gpus", "") or ""),
            "lora_epochs": getattr(runtime.args, "lora_epochs", None),
            "lora_resume": str(getattr(runtime.args, "lora_resume", "none") or "none"),
        },
        "videox_fun_repo": str(repo),
        "python": str(python_path),
        "launcher": list(launcher),
        "train_data_dir": str(trainset_dir),
        "train_data_meta": str(metadata_path),
        "output_dir": str(lora_dir),
    }
    write_json_atomic(runtime.paths.lora_config_path, resolved_config, indent=2)
    write_text_atomic(runtime.paths.lora_command_path, _command_script(argv, repo, env))
    os.chmod(str(runtime.paths.lora_command_path), 0o755)

    runtime.write_meta_snapshot()
    _stop_comfyui_pool(runtime)
    log_info("lora training start profile={} repo={}".format(profile_name, repo))

    started = time.time()
    rc = -1
    try:
        rc = run_cmd(
            ["bash", str(Path(runtime.paths.lora_command_path).resolve())],
            cwd=repo,
            env=os.environ.copy(),
            check=False,
            log_path=runtime.paths.lora_log_path,
            log_level=getattr(runtime.args, "log_level", "info"),
            fail_tail_lines=30,
            display_phase_name="lora",
            stream_mode="tee",
        )
    except Exception as exc:
        elapsed = time.time() - started
        message = "lora training failed before process completion: {}".format(exc)
        _write_result(runtime, rc, elapsed, profile_name, repo, trainset_dir, metadata_path, lora_dir, message)
        raise

    elapsed = time.time() - started
    if rc != 0:
        message = "VideoX-Fun train_lora.py failed with exit code {}".format(rc)
        payload = _write_result(runtime, rc, elapsed, profile_name, repo, trainset_dir, metadata_path, lora_dir, message)
        raise RunError(
            message,
            returncode=rc,
            cmd=["bash", str(Path(runtime.paths.lora_command_path).resolve())],
            log_path=runtime.paths.lora_log_path,
            tail_lines=list(payload.get("tail_log") or []),
        )

    _write_result(runtime, rc, elapsed, profile_name, repo, trainset_dir, metadata_path, lora_dir)
    log_info("lora training done: {}".format(lora_dir))
    return lora_dir
