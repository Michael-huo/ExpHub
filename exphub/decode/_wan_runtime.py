#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from exphub.config import get_platform_config
from exphub.common.logging import log_info
from .runtime_manage import load_image_gen_runtime, resolve_image_gen_runtime_segments

from ._runtime_backend_base import DirectInferBackend, _run_filtered


WAN_GPU_MEMORY_MODES = (
    "model_full_load",
    "model_full_load_and_qfloat8",
    "model_cpu_offload",
    "model_cpu_offload_and_qfloat8",
    "sequential_cpu_offload",
)


@dataclass
class WanFunRuntimeConfig(object):
    backend_name: str
    default_phase: str
    model_config_keys: Tuple[str, ...] = field(default_factory=tuple)
    default_model_dir: str = ""
    default_config_path: str = ""
    gpu_memory_mode: str = "model_full_load"
    prefer_quantization: bool = False
    compile_dit: bool = False
    enable_teacache: bool = True
    teacache_threshold: float = 0.10
    cfg_skip_ratio: float = 0.0
    fsdp_text_encoder: bool = True
    backend_entry_type: str = "wan_fun_runtime"
    teacache_skip_start_steps: int = 5
    teacache_offload: bool = False
    enable_riflex: bool = False
    riflex_k: int = 6
    flow_shift: float = 5.0
    sampler_name: str = "Flow"
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    prompt: str = (
        "First-person camera moving forward along an outdoor park walkway. "
        "Photorealistic, natural lighting, stable exposure and white balance. "
        "Consistent perspective and geometry, level horizon, sharp textures on pavement, grass, "
        "and trees. No flicker, no warping, no artifacts."
    )
    negative_prompt: str = (
        "static camera, fixed viewpoint, blurry, distorted, flickering, warping, wobble, "
        "rolling shutter artifacts, ghosting, double edges, inconsistent geometry, wrong perspective, "
        "texture swimming, repeating patterns, text, watermark, low quality, JPEG compression artifacts, "
        "excessive noise, color shift, unnatural motion"
    )
    config_name: str = ""


@dataclass
class SegmentRunResult(object):
    infer_sec: float
    result_sample: object
    start_path: str
    end_path: str
    seed: int
    desired_frames: int
    run_frames: int
    prompt: str
    negative_prompt: str
    prompt_source: str
    num_inference_steps: int
    guidance_scale: float
    prompt_hash8: str


DEFAULT_WAN_FUN_RUNTIME_CONFIG = WanFunRuntimeConfig(
    backend_name="wan_fun_runtime",
    default_phase="infer",
)


def _normalize_backend_config(backend_config):
    # type: (object) -> WanFunRuntimeConfig
    if isinstance(backend_config, WanFunRuntimeConfig):
        config = WanFunRuntimeConfig(**backend_config.__dict__)
    elif isinstance(backend_config, dict):
        base = dict(DEFAULT_WAN_FUN_RUNTIME_CONFIG.__dict__)
        base.update(backend_config)
        config = WanFunRuntimeConfig(**base)
    else:
        config = WanFunRuntimeConfig(**DEFAULT_WAN_FUN_RUNTIME_CONFIG.__dict__)

    model_keys = tuple([str(x) for x in list(config.model_config_keys or ()) if str(x).strip()])
    config.model_config_keys = model_keys
    config.backend_name = str(config.backend_name or DEFAULT_WAN_FUN_RUNTIME_CONFIG.backend_name)
    config.default_phase = str(config.default_phase or DEFAULT_WAN_FUN_RUNTIME_CONFIG.default_phase)
    config.config_name = str(config.config_name or config.backend_name)
    if str(config.gpu_memory_mode or "") not in WAN_GPU_MEMORY_MODES:
        config.gpu_memory_mode = DEFAULT_WAN_FUN_RUNTIME_CONFIG.gpu_memory_mode
    return config


def _resolve_platform_model_defaults(cfg, backend_config):
    # type: (Dict[str, object], WanFunRuntimeConfig) -> Tuple[str, str]
    models_cfg = cfg.get("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
    for key in backend_config.model_config_keys:
        item = models_cfg.get(str(key), {})
        if not isinstance(item, dict) or not item:
            continue
        model_dir = str(item.get("path", "") or "").strip()
        config_path = str(item.get("config", "") or "").strip()
        if model_dir or config_path:
            return model_dir, config_path
    return "", ""


def _resolve_runtime_config(backend_config, cfg, model_dir="", config_path=""):
    # type: (WanFunRuntimeConfig, Dict[str, object], str, str) -> WanFunRuntimeConfig
    config = _normalize_backend_config(backend_config)
    platform_model_dir, platform_config_path = _resolve_platform_model_defaults(cfg, config)
    resolved_model_dir = str(model_dir or platform_model_dir or config.default_model_dir or "").strip()
    resolved_config_path = str(config_path or platform_config_path or config.default_config_path or "").strip()
    config.default_model_dir = resolved_model_dir
    config.default_config_path = resolved_config_path
    return config


def _format_backend_model_keys(backend_config):
    # type: (WanFunRuntimeConfig) -> str
    if not backend_config.model_config_keys:
        return "models.<unset>"
    return " / ".join(["models.{}".format(key) for key in backend_config.model_config_keys])


def _parse_tri_bool(text, default):
    # type: (str, bool) -> bool
    value = str(text or "auto").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _mode_uses_quantization(gpu_memory_mode):
    # type: (str) -> bool
    return "qfloat8" in str(gpu_memory_mode or "")


def _coerce_positive_int(value, default=1):
    # type: (object, int) -> int
    try:
        parsed = int(value)
    except Exception:
        text = str(value or "").strip()
        match = re.search(r"-?\d+", text)
        if match is None:
            return int(default)
        try:
            parsed = int(match.group(0))
        except Exception:
            return int(default)
    if parsed <= 0:
        return int(default)
    return int(parsed)


def _normalize_vae_compression_ratios(config):
    # type: (object) -> Tuple[int, int]
    vae_kwargs = config.get("vae_kwargs", {})
    temporal_ratio = _coerce_positive_int(vae_kwargs.get("temporal_compression_ratio", 1), default=1)
    spatial_ratio = _coerce_positive_int(vae_kwargs.get("spatial_compression_ratio", 1), default=1)
    vae_kwargs["temporal_compression_ratio"] = int(temporal_ratio)
    vae_kwargs["spatial_compression_ratio"] = int(spatial_ratio)
    return int(temporal_ratio), int(spatial_ratio)


def _format_segment_range(seg_spec):
    # type: (object) -> str
    if not isinstance(seg_spec, dict):
        return ""
    try:
        start_idx = int(seg_spec.get("start_idx"))
        end_idx = int(seg_spec.get("end_idx"))
    except Exception:
        return ""
    return " idx {}->{}".format(start_idx, end_idx)


def _visible_devices_text():
    text = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    return text or "<all>"


class WanFunInferBackend(DirectInferBackend):
    backend_config = DEFAULT_WAN_FUN_RUNTIME_CONFIG

    def __init__(
        self,
        videox_root,  # type: str
        model_ref="",  # type: str
        backend_python_phase="infer",  # type: str
    ):
        # type: (...) -> None
        super(WanFunInferBackend, self).__init__(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
        self._runtime_config = None  # type: Optional[WanFunRuntimeConfig]

    def _resolve_runtime_config_for_meta(self):
        # type: () -> WanFunRuntimeConfig
        if self._runtime_config is not None:
            return self._runtime_config
        if self._loaded:
            model_dir = self._model_dir or ""
            config_path = self._config_path or ""
        else:
            model_dir, _model_id, config_path = self._resolve_model_ref()
            model_dir = model_dir or ""
            config_path = config_path or ""
        return _resolve_runtime_config(self.backend_config, self._cfg, model_dir=model_dir, config_path=config_path)

    def load(self):
        # type: () -> None
        super(WanFunInferBackend, self).load()
        self._runtime_config = _resolve_runtime_config(
            self.backend_config,
            self._cfg,
            model_dir=str(self._model_dir or ""),
            config_path=str(self._config_path or ""),
        )

    def meta(self):
        # type: () -> dict
        meta = dict(super(WanFunInferBackend, self).meta())
        runtime_config = self._resolve_runtime_config_for_meta()
        meta.update(
            {
                "gpu_memory_mode": str(runtime_config.gpu_memory_mode),
                "quantized_transformer": bool(_mode_uses_quantization(runtime_config.gpu_memory_mode)),
                "backend_config_name": str(runtime_config.config_name),
            }
        )
        return meta

    def _worker_script_path(self):
        # type: () -> Path
        return Path(inspect.getfile(self.__class__)).resolve()

    def run(self, request):
        # type: (object) -> dict
        if not self._loaded:
            self.load()

        gpus = int(request.gpus)
        if gpus > 1:
            self.backend_entry_type = "torchrun_backend_worker"
            log_info(
                "infer backend launcher: backend={} mode=torchrun nproc_per_node={} gpus={}".format(
                    self.name,
                    gpus,
                    gpus,
                )
            )
            log_info(
                "infer launcher env: visible_devices={} requested_gpus={} world_size={} entry={}".format(
                    _visible_devices_text(),
                    gpus,
                    gpus,
                    self._worker_script_path(),
                )
            )
            env = os.environ.copy()
            old_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(self.videox_root) + (os.pathsep + old_pp if old_pp else "")
            cmd = [
                sys.executable if getattr(sys, "executable", "") else "python3",
                "-m",
                "torch.distributed.run",
                "--nproc_per_node={}".format(gpus),
                str(self._worker_script_path()),
            ] + self._build_cmd(request)
            rc = _run_filtered(cmd, cwd=self.videox_root, env=env)
            if rc != 0:
                raise SystemExit(rc)
            return self.meta()

        self.backend_entry_type = "direct_backend"

        log_info(
            "infer backend launcher: backend={} mode=direct gpus={}".format(
                self.name,
                gpus,
            )
        )
        log_info(
            "infer launcher env: visible_devices={} requested_gpus={} world_size={} entry={}".format(
                _visible_devices_text(),
                gpus,
                1,
                self._worker_script_path(),
            )
        )
        return super(WanFunInferBackend, self).run(request)

    def _run_direct(self, argv):
        # type: (list) -> None
        run_wan_fun_backend_cli(argv, backend_config=self._resolve_runtime_config_for_meta())


def run_wan_fun_backend_cli(argv=None, backend_config=None):
    # type: (object, object) -> None
    backend_config = _normalize_backend_config(backend_config)

    import warnings
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Accessing config attribute.*directly via.*")
    warnings.filterwarnings("ignore", message=".*torch.load with weights_only=False.*")
    warnings.filterwarnings("ignore", message=".*Padding mask is disabled.*")
    warnings.filterwarnings(
        "ignore",
        message=".*You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import argparse
    import atexit
    import ctypes
    import ctypes.util
    import gc
    import json
    import time
    from datetime import datetime

    import numpy as np
    import torch
    import torch.distributed as dist
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image

    from videox_fun.dist import set_multi_gpus_devices, shard_model
    from videox_fun.models import (
        AutoTokenizer,
        AutoencoderKLWan,
        AutoencoderKLWan3_8,
        Wan2_2Transformer3DModel,
        WanT5EncoderModel,
    )
    from videox_fun.models.cache_utils import get_teacache_coefficients
    from videox_fun.pipeline import Wan2_2FunInpaintPipeline
    from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from videox_fun.utils.fp8_optimization import (
        convert_model_weight_to_float8,
        convert_weight_dtype_wrapper,
        replace_parameters_by_name,
    )
    from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
    from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid

    import videox_fun

    videox_root = os.path.dirname(os.path.dirname(os.path.abspath(videox_fun.__file__)))

    t_all_start = time.time()
    allow_prefix = ("[PROG]", "[INFO]", "[WARN]", "[ERR]", "[BAR]", "[PROMPT]")

    def _is_primary_rank():
        # type: () -> bool
        rank_text = str(os.environ.get("RANK", "") or "").strip()
        if rank_text:
            try:
                return int(rank_text) == 0
            except Exception:
                return rank_text == "0"

        local_rank_text = str(os.environ.get("LOCAL_RANK", "") or "").strip()
        if local_rank_text:
            try:
                return int(local_rank_text) == 0
            except Exception:
                return local_rank_text == "0"

        if dist.is_available() and dist.is_initialized():
            try:
                return dist.get_rank() == 0
            except Exception:
                return True
        return True

    def rprint(msg):
        # type: (str) -> None
        if not _is_primary_rank():
            return
        if not str(msg).startswith(allow_prefix):
            return
        print(msg, flush=True)

    def malloc_trim():
        # type: () -> None
        try:
            libc_path = ctypes.util.find_library("c")
            if not libc_path:
                return
            libc = ctypes.CDLL(libc_path)
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (1 for single-GPU)")
    parser.add_argument("--batch", action="store_true", help="Run sequential segments in one process")
    parser.add_argument("--frames_dir", type=str, default="", help="Dataset frames directory")
    parser.add_argument("--base_idx", type=int, default=0, help="Start frame index in dataset")
    parser.add_argument("--num_segments", type=int, default=16, help="Number of segments to run")
    parser.add_argument("--dataset_fps", type=int, default=25, help="Input dataset FPS")
    parser.add_argument("--segment_seconds", type=float, default=1.0, help="Seconds per segment")
    parser.add_argument("--fps", type=int, default=25, help="Target output FPS")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt override")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt override")
    parser.add_argument("--prompt_file", type=str, default="", help="Prompt file json")
    parser.add_argument("--execution_plan", type=str, default="", help="Execution plan json")
    parser.add_argument("--guidance_scale", type=float, default=-1.0, help="CFG guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=-1, help="Denoising steps")
    parser.add_argument("--config_path", type=str, default="", help="Override YAML config path")
    parser.add_argument("--model_name", type=str, default="", help="Override model directory")
    parser.add_argument("--start_image", type=str, default="", help="Single mode start image")
    parser.add_argument("--end_image", type=str, default="", help="Single mode end image")
    parser.add_argument("--seed_base", type=int, default=43, help="Base seed")
    parser.add_argument("--kf_gap", type=int, default=0, help="Keyframe gap in frames")
    parser.add_argument("--runs_parent", type=str, default="", help="Parent directory for runs")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment folder name")
    parser.add_argument("--tag", type=str, default="", help="Optional short tag")
    parser.add_argument("--gpu_memory_mode", type=str, default="", help="Override runtime gpu memory mode")
    parser.add_argument(
        "--prefer_quantization",
        type=str,
        default="auto",
        help="Override backend config prefer_quantization with true/false/auto",
    )
    parser.add_argument(
        "--compile_dit",
        type=str,
        default="auto",
        help="Override compile_dit with true/false/auto",
    )
    parser.add_argument(
        "--enable_teacache",
        type=str,
        default="auto",
        help="Override enable_teacache with true/false/auto",
    )
    parser.add_argument(
        "--teacache_threshold",
        type=float,
        default=-1.0,
        help="Override TeaCache threshold",
    )
    parser.add_argument(
        "--cfg_skip_ratio",
        type=float,
        default=-1.0,
        help="Override cfg skip ratio",
    )
    parser.add_argument(
        "--fsdp_text_encoder",
        type=str,
        default="auto",
        help="Override fsdp_text_encoder with true/false/auto",
    )

    args = parser.parse_args(argv)
    if args.batch and not args.frames_dir.strip():
        raise SystemExit("[ERR] --frames_dir is required in --batch mode")

    cfg = get_platform_config()
    backend_config = _resolve_runtime_config(
        backend_config,
        cfg,
        model_dir=str(args.model_name or "").strip(),
        config_path=str(args.config_path or "").strip(),
    )

    if args.gpus > 1:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
        except Exception as exc:
            rprint("[WARN] failed to set local cuda device: {}".format(exc))

    rprint(
        "[INFO] infer worker env: visible_devices={} requested_gpus={} world_size={} rank={} local_rank={}".format(
            _visible_devices_text(),
            int(args.gpus),
            str(os.environ.get("WORLD_SIZE", str(int(args.gpus) if int(args.gpus) > 1 else 1))),
            str(os.environ.get("RANK", "0")),
            str(os.environ.get("LOCAL_RANK", "0")),
        )
    )

    gpu_memory_mode = str(args.gpu_memory_mode or backend_config.gpu_memory_mode).strip() or backend_config.gpu_memory_mode
    if gpu_memory_mode not in WAN_GPU_MEMORY_MODES:
        raise SystemExit("[ERR] unsupported --gpu_memory_mode: {}".format(gpu_memory_mode))
    prefer_quantization = _parse_tri_bool(args.prefer_quantization, backend_config.prefer_quantization)
    compile_dit = _parse_tri_bool(args.compile_dit, backend_config.compile_dit)
    enable_teacache = _parse_tri_bool(args.enable_teacache, backend_config.enable_teacache)
    teacache_threshold = (
        float(args.teacache_threshold)
        if float(args.teacache_threshold) >= 0
        else float(backend_config.teacache_threshold)
    )
    cfg_skip_ratio = (
        float(args.cfg_skip_ratio)
        if float(args.cfg_skip_ratio) >= 0
        else float(backend_config.cfg_skip_ratio)
    )
    fsdp_text_encoder_enabled = _parse_tri_bool(args.fsdp_text_encoder, backend_config.fsdp_text_encoder)
    fsdp_text_encoder = bool(fsdp_text_encoder_enabled and int(args.gpus) == 1)
    quantized_transformer = _mode_uses_quantization(gpu_memory_mode)
    if quantized_transformer and not prefer_quantization:
        rprint(
            "[WARN] backend={} config prefers non-quantized execution, but gpu_memory_mode={} enables qfloat8".format(
                backend_config.backend_name,
                gpu_memory_mode,
            )
        )

    ulysses_degree = int(args.gpus)
    ring_degree = 1
    fsdp_dit = False
    enable_riflex = bool(backend_config.enable_riflex)
    riflex_k = int(backend_config.riflex_k)
    teacache_offload = bool(backend_config.teacache_offload)
    num_skip_start_steps = int(backend_config.teacache_skip_start_steps)
    sampler_name = str(backend_config.sampler_name)
    shift = float(backend_config.flow_shift)

    default_model = str(backend_config.default_model_dir or "").strip()
    default_config = str(backend_config.default_config_path or "").strip()
    config_path = str(args.config_path or "").strip() or default_config
    model_name = str(args.model_name or "").strip() or default_model
    if not config_path:
        raise SystemExit(
            "[ERR] infer backend '{}' has no config configured. Set {}.config in config/platform.yaml or pass --config_path.".format(
                backend_config.backend_name,
                _format_backend_model_keys(backend_config),
            )
        )
    if not model_name:
        raise SystemExit(
            "[ERR] infer backend '{}' has no model configured. Set {}.path in config/platform.yaml or pass --model_name.".format(
                backend_config.backend_name,
                _format_backend_model_keys(backend_config),
            )
        )
    rprint(
        "[INFO] backend={} model_name={} config_path={}".format(
            backend_config.backend_name,
            model_name,
            config_path,
        )
    )
    rprint(
        "[INFO] runtime_config={} gpu_memory_mode={} quantized_transformer={} prefer_quantization={}".format(
            backend_config.config_name,
            gpu_memory_mode,
            quantized_transformer,
            prefer_quantization,
        )
    )

    transformer_path = None
    transformer_high_path = None
    vae_path = None
    lora_path = None
    lora_high_path = None
    lora_weight = 0.55
    lora_high_weight = 0.55

    sample_size = None
    fps = int(args.fps)
    video_length = None
    validation_image_start = str(args.start_image or "").strip() or None
    validation_image_end = str(args.end_image or "").strip() or None

    prompt = str(backend_config.prompt)
    negative_prompt = str(backend_config.negative_prompt)
    guidance_scale = float(backend_config.guidance_scale)
    seed = int(args.seed_base)
    num_inference_steps = int(backend_config.num_inference_steps)

    if str(args.prompt or "").strip():
        prompt = str(args.prompt).strip()
    if str(args.negative_prompt or "").strip():
        negative_prompt = str(args.negative_prompt).strip()
    if float(args.guidance_scale) >= 0:
        guidance_scale = float(args.guidance_scale)
    if int(args.num_inference_steps) > 0:
        num_inference_steps = int(args.num_inference_steps)

    task_name = "i2v"
    save_frames = True
    frame_ext = "png"
    default_samples_root = os.path.join(videox_root, "samples")

    def _canonical_json(obj):
        # type: (object) -> str
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _sha1_hex(text):
        # type: (str) -> str
        import hashlib

        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _write_text_atomic(path, text):
        # type: (str, str) -> None
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fobj:
            fobj.write(text)
        os.replace(tmp, path)

    def _write_json_atomic(path, obj):
        # type: (str, object) -> None
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fobj:
            json.dump(obj, fobj, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _samefile_safe_path(src, dst):
        # type: (Path, Path) -> bool
        src_s = os.path.abspath(str(src))
        dst_s = os.path.abspath(str(dst))
        try:
            return os.path.samefile(src_s, dst_s)
        except Exception:
            return src_s == dst_s

    def _load_execution_plan(path):
        # type: (str) -> dict
        with open(path, "r", encoding="utf-8") as fobj:
            payload = json.load(fobj)
        if not isinstance(payload, dict):
            raise ValueError("execution plan must be a JSON object")
        raw_segments = list(payload.get("segments", []) or [])
        segments = []
        for idx, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                continue
            start_idx = int(item.get("start_idx"))
            end_idx = int(item.get("end_idx"))
            segments.append(
                {
                    "seg": int(item.get("seg", idx)),
                    "segment_id": int(item.get("segment_id", idx)),
                    "schedule_source": str(item.get("schedule_source", payload.get("schedule_source", "")) or ""),
                    "execution_backend": str(item.get("execution_backend", payload.get("execution_backend", "")) or ""),
                    "raw_start_idx": int(item.get("raw_start_idx", start_idx)),
                    "raw_end_idx": int(item.get("raw_end_idx", end_idx)),
                    "deploy_start_idx": int(item.get("deploy_start_idx", start_idx)),
                    "deploy_end_idx": int(item.get("deploy_end_idx", end_idx)),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "raw_gap": int(item.get("raw_gap", end_idx - start_idx)),
                    "deploy_gap": int(item.get("deploy_gap", end_idx - start_idx)),
                    "num_frames": int(item.get("num_frames", end_idx - start_idx + 1)),
                }
            )
        return {
            "schedule_source": str(payload.get("schedule_source", "") or ""),
            "execution_backend": str(payload.get("execution_backend", "") or ""),
            "segments": segments,
        }

    def _fallback_batch_execution_segments(base_idx, stride, num_segments):
        # type: (int, int, int) -> list
        if stride <= 0:
            raise ValueError("fallback batch stride must be > 0")
        out = []
        for seg in range(int(num_segments)):
            start_idx = int(base_idx + seg * stride)
            end_idx = int(start_idx + stride)
            out.append(
                {
                    "seg": int(seg),
                    "segment_id": int(seg),
                    "schedule_source": "fallback_kf_gap",
                    "execution_backend": "fallback_uniform",
                    "raw_start_idx": int(start_idx),
                    "raw_end_idx": int(end_idx),
                    "deploy_start_idx": int(start_idx),
                    "deploy_end_idx": int(end_idx),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "raw_gap": int(stride),
                    "deploy_gap": int(stride),
                    "num_frames": int(stride + 1),
                }
            )
        return out

    def _escape_one_line(text):
        # type: (str) -> str
        return str(text or "").replace("\r", "").replace("\n", "\\n")

    def _probe_frame_path(frames_dir, idx):
        # type: (str, int) -> str
        base = os.path.join(frames_dir, "{:06d}".format(idx))
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        base2 = os.path.join(frames_dir, str(idx))
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = base2 + ext
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError("frame not found: idx={} under {}".format(idx, frames_dir))

    def _infer_sample_size_from_path(path):
        # type: (str) -> list
        image = Image.open(path)
        width, height = image.size
        image.close()
        return [height, width]

    if args.batch:
        sample_size = _infer_sample_size_from_path(_probe_frame_path(args.frames_dir, int(args.base_idx)))
    elif validation_image_start:
        sample_size = _infer_sample_size_from_path(validation_image_start)

    if args.batch:
        if int(args.kf_gap) > 0:
            video_length = int(args.kf_gap) + 1
        else:
            video_length = max(1, int(round(float(fps) * float(args.segment_seconds))))
    else:
        video_length = max(1, int(round(float(fps) * float(args.segment_seconds))))

    if sample_size is None:
        raise SystemExit("[ERR] cannot infer image size. Provide --frames_dir (batch) or --start_image (single).")

    def _auto_exp_name_short():
        # type: () -> str
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not sample_size:
            width, height = 0, 0
        else:
            width, height = int(sample_size[1]), int(sample_size[0])
        out_fps = int(fps)
        if args.batch:
            total_dur = float(args.num_segments) * float(args.segment_seconds)
        else:
            total_dur = float(video_length) / max(out_fps, 1)
        if abs(total_dur - round(total_dur)) < 1e-6:
            dur_str = str(int(round(total_dur)))
        else:
            dur_str = "{:g}".format(total_dur)
        name = "{}_dur{}s_{}fps_{}x{}".format(ts, dur_str, out_fps, width, height)
        if args.tag:
            name = "{}_{}".format(name, args.tag)
        return name

    default_runs_parent = os.path.join(default_samples_root, "wan-videos-fun-i2v", "runs")
    runs_parent = str(args.runs_parent or "").strip() or default_runs_parent
    exp_name = str(args.exp_name or "").strip() or _auto_exp_name_short()
    runs_root = os.path.join(runs_parent, exp_name)

    runs_parent_path = Path(runs_parent).resolve()
    if runs_parent_path.name == "infer":
        prompt_dir = runs_parent_path.parent / "prompt"
    else:
        prompt_dir = runs_parent_path / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_file_dst = prompt_dir / "image_gen_runtime.json"
    prompt_file_path = str(prompt_file_dst.resolve())
    if str(args.prompt_file or "").strip():
        src = Path(args.prompt_file).resolve()
        if not src.is_file():
            raise FileNotFoundError("prompt_file not found: {}".format(src))
        prompt_file_path = str(src)
        if _samefile_safe_path(src, prompt_file_dst.resolve()):
            rprint("[INFO] prompt_file already at standard path, using: {}".format(src))
        else:
            rprint("[INFO] prompt_file using custom path: {}".format(src))
    elif not prompt_file_dst.is_file():
        _write_json_atomic(
            str(prompt_file_dst),
            {
                "version": 1,
                "schema": "image_gen_runtime.v1",
                "base_prompt": prompt.strip(),
                "negative_prompt": negative_prompt.strip(),
                "source": "image_gen_runtime.default",
                "segments": [],
            },
        )
        prompt_file_path = str(prompt_file_dst.resolve())
    image_gen_runtime = load_image_gen_runtime(
        prompt_file_path,
        default_prompt=prompt,
        default_negative_prompt=negative_prompt,
    )
    prompt_digest8 = _sha1_hex(_canonical_json(image_gen_runtime["_raw"]))[:8]

    batch_execution_segments = []
    execution_schedule_source = ""
    execution_backend = ""
    if args.batch:
        if str(args.execution_plan or "").strip():
            execution_plan = _load_execution_plan(str(args.execution_plan))
            batch_execution_segments = list(execution_plan.get("segments", []) or [])
            execution_schedule_source = str(execution_plan.get("schedule_source", "") or "")
            execution_backend = str(execution_plan.get("execution_backend", "") or "")
        if not batch_execution_segments:
            if int(args.kf_gap) <= 0:
                raise SystemExit("[ERR] --kf_gap is required in --batch mode when execution plan is missing")
            if int(args.num_segments) <= 0:
                raise SystemExit("[ERR] --num_segments must be > 0 in batch mode when execution plan is missing")
            batch_execution_segments = _fallback_batch_execution_segments(int(args.base_idx), int(args.kf_gap), int(args.num_segments))
            execution_schedule_source = "fallback_kf_gap"
            execution_backend = "fallback_uniform"
        elif execution_schedule_source == "":
            execution_schedule_source = "execution_plan"
            if execution_backend == "":
                execution_backend = "custom"

    resolved_prompt_count = int(len(batch_execution_segments)) if args.batch else 1
    resolved_segments = resolve_image_gen_runtime_segments(
        image_gen_runtime,
        resolved_prompt_count,
        default_prompt=prompt,
        default_negative_prompt=negative_prompt,
        default_num_inference_steps=num_inference_steps,
        default_guidance_scale=guidance_scale,
    )
    for item in resolved_segments:
        item["prompt_hash8"] = _sha1_hex(item["resolved_prompt"] + "\n||NEG||\n" + item["negative_prompt"])[:8]

    if args.batch and batch_execution_segments:
        video_length = max([int(seg.get("num_frames", 1)) for seg in batch_execution_segments])

    t_init_start = time.time()
    t_quant_low = 0.0
    t_quant_high = 0.0
    rprint("[INFO] Initializing model pipeline and loading weights from disk...")
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    def _is_main_process():
        # type: () -> bool
        return _is_primary_rank()

    def _distributed_barrier():
        # type: () -> None
        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except Exception:
            pass

    cleanup_state = {"done": False}

    def _cleanup_process_group():
        # type: () -> None
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        if not (dist.is_available() and dist.is_initialized()):
            return
        _distributed_barrier()
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    atexit.register(_cleanup_process_group)

    config = OmegaConf.load(config_path)
    normalized_vae_temporal_ratio, normalized_vae_spatial_ratio = _normalize_vae_compression_ratios(config)

    weight_dtype_env = os.environ.get("WAN_WEIGHT_DTYPE", "").strip().lower()
    if weight_dtype_env in ("bf16", "bfloat16"):
        weight_dtype = torch.bfloat16
    elif weight_dtype_env in ("fp16", "float16", "half"):
        weight_dtype = torch.float16
    elif weight_dtype_env in ("fp32", "float32"):
        weight_dtype = torch.float32
    else:
        try:
            if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                weight_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                weight_dtype = torch.float16
            else:
                weight_dtype = torch.float32
        except Exception:
            weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if rank == 0:
        rprint("[INFO] weight_dtype={}".format(weight_dtype))

    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            model_name,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                model_name,
                config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer"),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    if transformer_path is not None:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        transformer.load_state_dict(state_dict, strict=False)

    if transformer_2 is not None and transformer_high_path is not None:
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        transformer_2.load_state_dict(state_dict, strict=False)

    chosen_autoencoder = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_autoencoder.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)
    vae.temporal_compression_ratio = int(normalized_vae_temporal_ratio)
    vae.spatial_compression_ratio = int(normalized_vae_spatial_ratio)
    if getattr(vae, "config", None) is not None:
        try:
            vae.config.temporal_compression_ratio = int(normalized_vae_temporal_ratio)
        except Exception:
            pass
        try:
            vae.config.spatial_compression_ratio = int(normalized_vae_spatial_ratio)
        except Exception:
            pass

    if vae_path is not None:
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        vae.load_state_dict(state_dict, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(
        **filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if ulysses_degree > 1 or ring_degree > 1:
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            def shard_fn(module):
                return shard_model(module, device_id=device, param_dtype=weight_dtype)

            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        if fsdp_text_encoder:
            def shard_fn(module):
                return shard_model(module, device_id=device, param_dtype=weight_dtype)

            pipeline.text_encoder = shard_fn(pipeline.text_encoder)

    if compile_dit:
        for idx in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[idx] = torch.compile(pipeline.transformer.blocks[idx])
        if transformer_2 is not None:
            for idx in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[idx] = torch.compile(pipeline.transformer_2.blocks[idx])

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        rprint("[INFO] Starting float8 quantization (CPU/Mem bound process, please wait...)")
        t0 = time.time()
        rprint("[INFO] Quantizing transformer 1/2...")
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        t_quant_low = time.time() - t0
        rprint("[INFO] Transformer 1/2 quantized in {:.2f}s".format(t_quant_low))
        if transformer_2 is not None:
            t0 = time.time()
            rprint("[INFO] Quantizing transformer 2/2...")
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            t_quant_high = time.time() - t0
            rprint("[INFO] Transformer 2/2 quantized in {:.2f}s".format(t_quant_high))
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_full_load_and_qfloat8":
        rprint("[INFO] Starting float8 quantization (CPU/Mem bound process, please wait...)")
        t0 = time.time()
        rprint("[INFO] Quantizing transformer 1/2...")
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        t_quant_low = time.time() - t0
        rprint("[INFO] Transformer 1/2 quantized in {:.2f}s".format(t_quant_low))
        if transformer_2 is not None:
            t0 = time.time()
            rprint("[INFO] Quantizing transformer 2/2...")
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            t_quant_high = time.time() - t0
            rprint("[INFO] Transformer 2/2 quantized in {:.2f}s".format(t_quant_high))
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        rprint(
            "[INFO] Enable TeaCache with threshold {} and skip the first {} steps.".format(
                teacache_threshold,
                num_skip_start_steps,
            )
        )
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            offload=teacache_offload,
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    if cfg_skip_ratio is not None:
        rprint("[INFO] Enable cfg_skip_ratio {}.".format(cfg_skip_ratio))
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        if transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    def _apply_segment_runtime_policy(current_num_inference_steps):
        # type: (int) -> None
        seg_steps = max(1, int(current_num_inference_steps))
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                seg_steps,
                teacache_threshold,
                num_skip_start_steps=num_skip_start_steps,
                offload=teacache_offload,
            )
            if transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        if cfg_skip_ratio is not None:
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, seg_steps)
            if transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None:
            pipeline = merge_lora(
                pipeline,
                lora_high_path,
                lora_high_weight,
                device=device,
                dtype=weight_dtype,
                sub_transformer_name="transformer_2",
            )

    pipeline.set_progress_bar_config(
        bar_format="[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        disable=(not _is_primary_rank()),
    )
    t_init_end = time.time()
    t_init_total = t_init_end - t_init_start
    t_quant_total = t_quant_low + t_quant_high
    t_load = t_init_total - t_quant_total
    rprint(
        "[INFO] Initialization completed in {:.2f}s (Loading: {:.2f}s, Quantization: {:.2f}s)".format(
            t_init_total,
            t_load,
            t_quant_total,
        )
    )

    video_length_desired = video_length
    vae_temporal_ratio = _coerce_positive_int(getattr(vae.config, "temporal_compression_ratio", 1), default=1)
    video_length = int((video_length - 1) // vae_temporal_ratio * vae_temporal_ratio) + 1 if video_length != 1 else 1
    video_length_run = video_length
    latent_frames = (video_length - 1) // vae_temporal_ratio + 1
    default_video_length_desired = int(video_length_desired)
    default_video_length_run = int(video_length_run)

    def _align_video_length_for_segment(desired_length):
        # type: (int) -> Tuple[int, int, int]
        desired = max(1, int(desired_length))
        run = int((desired - 1) // vae_temporal_ratio * vae_temporal_ratio) + 1 if desired != 1 else 1
        latent = (run - 1) // vae_temporal_ratio + 1
        return int(desired), int(run), int(latent)

    if enable_riflex:
        with torch.no_grad():
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
            if transformer_2 is not None:
                pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    def _frame_path(frames_dir, idx):
        # type: (str, int) -> str
        for ext in ("png", "jpg", "jpeg"):
            candidate = os.path.join(frames_dir, "{:06d}.{}".format(idx, ext))
            if os.path.exists(candidate):
                return candidate
        return os.path.join(frames_dir, "{:06d}.png".format(idx))

    save_ctx = {
        "task_name": task_name,
        "runs_root": runs_root,
        "exp_name": exp_name,
        "dataset_fps": int(args.dataset_fps) if args.batch else 30,
        "target_fps": int(fps),
        "sample_size": sample_size,
        "base_idx": int(batch_execution_segments[0]["start_idx"]) if args.batch and batch_execution_segments else (int(args.base_idx) if args.batch else None),
        "num_segments": int(len(batch_execution_segments)) if args.batch else None,
        "schedule_source": str(execution_schedule_source) if args.batch else None,
        "execution_backend": str(execution_backend) if args.batch else None,
        "image_gen_runtime_path": os.path.abspath(prompt_file_path),
        "image_gen_runtime_digest8": prompt_digest8,
        "image_gen_runtime_version": int(image_gen_runtime.get("version", 1)),
        "image_gen_runtime_source": str(image_gen_runtime.get("source", "") or ""),
        "save_frames": bool(save_frames),
        "frame_ext": frame_ext,
        "is_batch": bool(args.batch),
        "source_frames_dir": str(args.frames_dir) if args.batch else None,
        "default_num_inference_steps": int(num_inference_steps),
        "default_guidance_scale": float(guidance_scale),
        "vae_temporal_compression_ratio": int(vae_temporal_ratio),
    }

    def run_one_segment(
        start_path,
        end_path,
        seg_seed,
        segment_prompt,
        segment_negative_prompt,
        segment_prompt_source,
        segment_num_inference_steps,
        segment_guidance_scale,
        desired_num_frames=None,
    ):
        # type: (str, str, int, str, str, str, int, float, Optional[int]) -> SegmentRunResult
        current_seed = int(seg_seed)
        segment_generator = torch.Generator(device=device).manual_seed(current_seed)
        if desired_num_frames is None:
            current_video_length_desired = int(default_video_length_desired)
            current_video_length_run = int(default_video_length_run)
        else:
            current_video_length_desired, current_video_length_run, _latent = _align_video_length_for_segment(int(desired_num_frames))
        segment_prompt = str(segment_prompt)
        segment_negative_prompt = str(segment_negative_prompt)
        segment_prompt_source = str(segment_prompt_source or "")
        segment_num_inference_steps = max(1, int(segment_num_inference_steps))
        segment_guidance_scale = float(segment_guidance_scale)
        _apply_segment_runtime_policy(segment_num_inference_steps)
        with torch.no_grad():
            start_img = Image.open(start_path).convert("RGB")
            end_img = Image.open(end_path).convert("RGB")
            input_video, input_video_mask, clip_image = get_image_to_video_latent(
                [start_img],
                [end_img],
                video_length=current_video_length_run,
                sample_size=sample_size,
            )
            start_time = time.time()
            out = pipeline(
                segment_prompt,
                num_frames=current_video_length_run,
                negative_prompt=segment_negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=segment_generator,
                guidance_scale=segment_guidance_scale,
                num_inference_steps=segment_num_inference_steps,
                boundary=boundary,
                video=input_video,
                mask_video=input_video_mask,
                shift=shift,
            )
            if _is_main_process():
                result_sample_tensor = out.videos
            else:
                result_sample_tensor = None
            try:
                del out
                del input_video, input_video_mask, clip_image
                del start_img, end_img
            except Exception:
                pass
            infer_sec = float(time.time() - start_time)
            return SegmentRunResult(
                infer_sec=infer_sec,
                result_sample=result_sample_tensor,
                start_path=str(start_path),
                end_path=str(end_path),
                seed=int(current_seed),
                desired_frames=int(current_video_length_desired),
                run_frames=int(current_video_length_run),
                prompt=segment_prompt,
                negative_prompt=segment_negative_prompt,
                prompt_source=segment_prompt_source,
                num_inference_steps=int(segment_num_inference_steps),
                guidance_scale=float(segment_guidance_scale),
                prompt_hash8=_sha1_hex(segment_prompt + "\n||NEG||\n" + segment_negative_prompt)[:8],
            )

    def _parse_frame_idx(path):
        # type: (str) -> Optional[int]
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            return int(base)
        except Exception:
            return None

    def _make_run_name(task, width, height, target_fps, frame_count, start_idx, end_idx, seed_value):
        # type: (str, int, int, int, int, object, object, int) -> str
        s_idx = "{:06d}".format(start_idx) if isinstance(start_idx, int) else "NA"
        e_idx = "{:06d}".format(end_idx) if isinstance(end_idx, int) else "NA"
        return "{}_{}x{}_{}fps_L{}_s{}_e{}_seed{}".format(
            task,
            width,
            height,
            target_fps,
            frame_count,
            s_idx,
            e_idx,
            seed_value,
        )

    def _tensor_frame_to_uint8_hwc(frame_chw):
        # type: (torch.Tensor) -> np.ndarray
        frame = frame_chw.detach().float().cpu().clamp(0, 1)
        frame = frame.permute(1, 2, 0).numpy()
        frame = (frame * 255.0 + 0.5).astype(np.uint8)
        return frame

    def _save_frames(sample, frames_dir, ext="png"):
        # type: (torch.Tensor, str, str) -> None
        os.makedirs(frames_dir, exist_ok=True)
        total = sample.shape[2]
        for idx in range(total):
            arr = _tensor_frame_to_uint8_hwc(sample[0, :, idx])
            Image.fromarray(arr).save(os.path.join(frames_dir, "frame_{:06d}.{}".format(idx, ext)))

    def save_results(result, save_context):
        # type: (SegmentRunResult, dict) -> None
        if result.result_sample is None:
            return
        os.makedirs(save_context["runs_root"], exist_ok=True)
        width = int(save_context["sample_size"][1])
        height = int(save_context["sample_size"][0])
        s_idx = _parse_frame_idx(result.start_path) if result.start_path else None
        e_idx = _parse_frame_idx(result.end_path) if result.end_path else None
        run_name = _make_run_name(
            save_context["task_name"],
            width,
            height,
            save_context["target_fps"],
            int(result.desired_frames),
            s_idx,
            e_idx,
            result.seed,
        )
        run_dir = os.path.join(save_context["runs_root"], run_name)
        if os.path.isdir(run_dir):
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(run_dir, exist_ok=True)
        video_path = os.path.join(run_dir, "preview.mp4")
        save_videos_grid(result.result_sample, video_path, fps=save_context["target_fps"])
        frames_dir = os.path.join(run_dir, "frames")
        if save_context["save_frames"]:
            _save_frames(result.result_sample, frames_dir, ext=save_context["frame_ext"])
        saved_frame_count = int(result.result_sample.shape[2]) if result.result_sample is not None else int(result.run_frames)
        params = {
            "task": save_context["task_name"],
            "created_at": datetime.now().isoformat(),
            "experiment_name": save_context["exp_name"],
            "experiment_root": save_context["runs_root"],
            "dataset_fps": int(save_context["dataset_fps"]),
            "target_fps": int(save_context["target_fps"]),
            "width": int(width),
            "height": int(height),
            "video_length_desired": int(result.desired_frames),
            "video_length_run": int(result.run_frames),
            "saved_frame_count": int(saved_frame_count),
            "vae_temporal_compression_ratio": int(save_context["vae_temporal_compression_ratio"]),
            "start_idx": s_idx,
            "end_idx": e_idx,
            "start_path": result.start_path,
            "end_path": result.end_path,
            "batch": bool(save_context["is_batch"]),
            "source_frames_dir": save_context["source_frames_dir"],
            "base_idx": save_context["base_idx"],
            "num_segments": save_context["num_segments"],
            "segment_seconds": (
                float(max(0, int(result.desired_frames) - 1)) / float(max(int(save_context["dataset_fps"]), 1))
                if save_context["is_batch"]
                else None
            ),
            "schedule_source": save_context["schedule_source"],
            "execution_backend": save_context["execution_backend"],
            "num_inference_steps": int(result.num_inference_steps),
            "guidance_scale": float(result.guidance_scale),
            "default_num_inference_steps": int(save_context["default_num_inference_steps"]),
            "default_guidance_scale": float(save_context["default_guidance_scale"]),
            "seed": int(result.seed),
            "prompt": result.prompt,
            "negative_prompt": result.negative_prompt,
            "prompt_source": result.prompt_source,
            "prompt_hash8": result.prompt_hash8,
            "image_gen_runtime": save_context["image_gen_runtime_path"],
            "image_gen_runtime_digest8": save_context["image_gen_runtime_digest8"],
            "image_gen_runtime_version": int(save_context["image_gen_runtime_version"]),
            "output_dir": run_dir,
            "output_video": "preview.mp4",
            "frames_dir": "frames" if save_context["save_frames"] else None,
            "frame_ext": save_context["frame_ext"] if save_context["save_frames"] else None,
        }
        with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as fobj:
            json.dump(params, fobj, ensure_ascii=False, indent=2)

    infer_sum = 0.0
    segments_ran = 0
    total_generated_frames = 0

    if args.batch:
        total = int(len(batch_execution_segments))
        if total <= 0:
            raise ValueError("batch execution segments resolved to 0")
        base_idx_exec = int(batch_execution_segments[0]["start_idx"])
        mean_gap_exec = sum([int(seg.get("deploy_gap", 0)) for seg in batch_execution_segments]) / float(total)
        rprint(
            "[PROG] batch: segments={} base_idx={} schedule_source={} backend={} mean_gap={:.2f} fps={}".format(
                total,
                base_idx_exec,
                execution_schedule_source,
                execution_backend,
                mean_gap_exec,
                int(fps),
            )
        )
        if os.environ.get("RANK", "0") == "0":
            try:
                width = int(sample_size[1])
                height = int(sample_size[0])
            except Exception:
                width, height = 0, 0
            plan_path = os.path.join(runs_parent, "runs_plan.json")
            segs = []
            for seg, seg_spec in enumerate(batch_execution_segments):
                start_idx = int(seg_spec["start_idx"])
                end_idx = int(seg_spec["end_idx"])
                desired_frames = int(seg_spec.get("num_frames", end_idx - start_idx + 1))
                seg_seed = int(args.seed_base) + int(seg)
                run_name = _make_run_name(task_name, width, height, int(fps), desired_frames, start_idx, end_idx, seg_seed)
                resolved_info = resolved_segments[int(seg)] if int(seg) < len(resolved_segments) else {}
                segs.append(
                    {
                        "seg": int(seg),
                        "segment_id": int(seg_spec.get("segment_id", seg)),
                        "schedule_source": str(seg_spec.get("schedule_source", execution_schedule_source)),
                        "execution_backend": str(seg_spec.get("execution_backend", execution_backend)),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "raw_start_idx": int(seg_spec.get("raw_start_idx", start_idx)),
                        "raw_end_idx": int(seg_spec.get("raw_end_idx", end_idx)),
                        "desired_start_idx": int(seg_spec.get("desired_start_idx", start_idx)),
                        "desired_end_idx": int(seg_spec.get("desired_end_idx", end_idx)),
                        "desired_num_frames": int(seg_spec.get("desired_num_frames", desired_frames)),
                        "aligned_start_idx": int(seg_spec.get("aligned_start_idx", start_idx)),
                        "aligned_end_idx": int(seg_spec.get("aligned_end_idx", end_idx)),
                        "aligned_num_frames": int(seg_spec.get("aligned_num_frames", desired_frames)),
                        "deploy_start_idx": int(seg_spec.get("deploy_start_idx", start_idx)),
                        "deploy_end_idx": int(seg_spec.get("deploy_end_idx", end_idx)),
                        "raw_gap": int(seg_spec.get("raw_gap", end_idx - start_idx)),
                        "deploy_gap": int(seg_spec.get("deploy_gap", end_idx - start_idx)),
                        "num_frames": int(desired_frames),
                        "left_shift": int(seg_spec.get("left_shift", 0)),
                        "right_shift": int(seg_spec.get("right_shift", 0)),
                        "align_reason": str(seg_spec.get("align_reason", "") or ""),
                        "is_valid_for_decode": bool(seg_spec.get("is_valid_for_decode", False)),
                        "is_valid_for_export": bool(seg_spec.get("is_valid_for_export", False)),
                        "run_id": str(seg_spec.get("run_id", "") or ""),
                        "source_unit_id": str(seg_spec.get("source_unit_id", "") or ""),
                        "source_span_id": str(seg_spec.get("source_span_id", "") or ""),
                        "source_prompt_ref": dict(seg_spec.get("source_prompt_ref", {}) or {}),
                        "target_num_frames": int(seg_spec.get("target_num_frames", desired_frames) or desired_frames),
                        "seed": int(seg_seed),
                        "run_name": run_name,
                        "prompt": str(resolved_info.get("resolved_prompt", "") or ""),
                        "negative_prompt": str(resolved_info.get("negative_prompt", "") or ""),
                        "prompt_hash8": resolved_info.get("prompt_hash8"),
                        "prompt_source": str(resolved_info.get("prompt_source", "")),
                        "num_inference_steps": int(resolved_info.get("num_inference_steps", num_inference_steps)),
                        "guidance_scale": float(resolved_info.get("guidance_scale", guidance_scale)),
                    }
                )
            plan = {
                "version": 1,
                "created_at": datetime.now().isoformat(),
                "runs_parent": os.path.abspath(runs_parent),
                "runs_root": os.path.abspath(runs_root),
                "exp_name": exp_name,
                "task": task_name,
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "dataset_fps": int(args.dataset_fps),
                "schedule_source": str(execution_schedule_source),
                "execution_backend": str(execution_backend),
                "segment_seconds": float(args.segment_seconds),
                "segment_seconds_mean": float(mean_gap_exec) / float(max(int(fps), 1)),
                "kf_gap": int(args.kf_gap),
                "step": None,
                "stride": None,
                "base_idx": int(base_idx_exec),
                "num_segments": int(total),
                "seed_base": int(args.seed_base),
                "image_gen_runtime": os.path.abspath(prompt_file_path),
                "image_gen_runtime_digest8": prompt_digest8,
                "image_gen_runtime_version": int(image_gen_runtime.get("version", 1)),
                "image_gen_runtime_source": str(image_gen_runtime.get("source", "") or ""),
                "default_num_inference_steps": int(num_inference_steps),
                "default_guidance_scale": float(guidance_scale),
                "segments": segs,
            }
            tmp_path = plan_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fobj:
                json.dump(plan, fobj, ensure_ascii=False, indent=2)
            os.replace(tmp_path, plan_path)
        batch_start = time.time()
        for seg, seg_spec in enumerate(batch_execution_segments):
            start_idx = int(seg_spec["start_idx"])
            end_idx = int(seg_spec["end_idx"])
            desired_num_frames = int(seg_spec.get("num_frames", end_idx - start_idx + 1))
            start_path = _frame_path(args.frames_dir, start_idx)
            end_path = _frame_path(args.frames_dir, end_idx)
            seg_seed = int(args.seed_base) + int(seg)
            segment_prompt = prompt
            segment_negative_prompt = negative_prompt
            segment_prompt_source = str(image_gen_runtime.get("source", "") or "image_gen_runtime.default")
            segment_num_inference_steps = int(num_inference_steps)
            segment_guidance_scale = float(guidance_scale)
            seg_i = int(seg) + 1
            hash_str = ""
            if int(seg) < len(resolved_segments):
                prompt_info = resolved_segments[int(seg)]
                hash_str = " hash={}".format(prompt_info["prompt_hash8"])
            desired_check, aligned_check, latent_check = _align_video_length_for_segment(desired_num_frames)
            if aligned_check != desired_num_frames:
                rprint(
                    "[WARN] seg {}/{}: desired_num_frames={} aligned_to={} (r={} latent={})".format(
                        seg_i,
                        total,
                        desired_num_frames,
                        aligned_check,
                        int(vae_temporal_ratio),
                        latent_check,
                    )
                )
            rprint(
                "[PROG] seg {}/{}: idx {}->{} deploy_gap={} frames={} seed={}{}".format(
                    seg_i,
                    total,
                    start_idx,
                    end_idx,
                    int(seg_spec.get("deploy_gap", end_idx - start_idx)),
                    desired_num_frames,
                    seg_seed,
                    hash_str,
                )
            )
            if int(seg) < len(resolved_segments):
                prompt_info = resolved_segments[int(seg)]
                segment_prompt = prompt_info["resolved_prompt"]
                segment_negative_prompt = prompt_info["negative_prompt"]
                segment_prompt_source = str(prompt_info.get("prompt_source", "") or segment_prompt_source)
                segment_num_inference_steps = int(prompt_info.get("num_inference_steps", num_inference_steps))
                segment_guidance_scale = float(prompt_info.get("guidance_scale", guidance_scale))
                if segment_negative_prompt:
                    rprint("[PROMPT] seg {}/{} neg={}".format(seg_i, total, _escape_one_line(segment_negative_prompt)))
            rprint(
                "[PROMPT] seg {}/{} source={} steps={} guidance={} prompt={}".format(
                    seg_i,
                    total,
                    segment_prompt_source,
                    segment_num_inference_steps,
                    "{:.3f}".format(segment_guidance_scale),
                    _escape_one_line(segment_prompt),
                )
            )
            segment_result = run_one_segment(
                start_path,
                end_path,
                seg_seed,
                segment_prompt,
                segment_negative_prompt,
                segment_prompt_source,
                segment_num_inference_steps,
                segment_guidance_scale,
                desired_num_frames=desired_num_frames,
            )
            infer_sum += segment_result.infer_sec
            segments_ran += 1
            total_generated_frames += int(segment_result.run_frames)
            save_start = time.time()
            if _is_main_process():
                save_results(segment_result, save_ctx)
            save_sec = time.time() - save_start
            elapsed = time.time() - batch_start
            left = total - seg_i
            eta = (elapsed / seg_i * left) if seg_i > 0 and left > 0 else 0.0
            range_text = _format_segment_range(seg_spec)
            rprint(
                "[INFO] seg {}/{}:{} infer={:.2f}s save={:.2f}s elapsed={:.1f}s eta={:.1f}s".format(
                    seg_i,
                    total,
                    range_text,
                    segment_result.infer_sec,
                    save_sec,
                    elapsed,
                    eta,
                )
            )
            _distributed_barrier()
            try:
                segment_result.result_sample = None
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim()
            _distributed_barrier()
    else:
        segment_prompt = prompt
        segment_negative_prompt = negative_prompt
        segment_prompt_source = str(image_gen_runtime.get("source", "") or "image_gen_runtime.default")
        segment_num_inference_steps = int(num_inference_steps)
        segment_guidance_scale = float(guidance_scale)
        prompt_info = resolved_segments[0] if len(resolved_segments) > 0 else None
        if prompt_info is not None:
            segment_prompt = prompt_info["resolved_prompt"]
            segment_negative_prompt = prompt_info["negative_prompt"]
            segment_prompt_source = str(prompt_info.get("prompt_source", "") or segment_prompt_source)
            segment_num_inference_steps = int(prompt_info.get("num_inference_steps", num_inference_steps))
            segment_guidance_scale = float(prompt_info.get("guidance_scale", guidance_scale))
            if segment_negative_prompt:
                rprint("[PROMPT] single neg={}".format(_escape_one_line(segment_negative_prompt)))
        rprint(
            "[PROMPT] single source={} steps={} guidance={} prompt={}".format(
                segment_prompt_source,
                segment_num_inference_steps,
                "{:.3f}".format(segment_guidance_scale),
                _escape_one_line(segment_prompt),
            )
        )
        segment_result = run_one_segment(
            validation_image_start,
            validation_image_end,
            seed,
            segment_prompt,
            segment_negative_prompt,
            segment_prompt_source,
            segment_num_inference_steps,
            segment_guidance_scale,
        )
        infer_sum = segment_result.infer_sec
        segments_ran = 1
        total_generated_frames = int(segment_result.run_frames)
        if _is_main_process():
            save_results(segment_result, save_ctx)

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None:
            pipeline = unmerge_lora(
                pipeline,
                lora_high_path,
                lora_high_weight,
                device=device,
                dtype=weight_dtype,
                sub_transformer_name="transformer_2",
            )

    total_time = time.time() - t_all_start
    init_time = t_init_end - t_init_start
    avg_infer = (infer_sum / segments_ran) if segments_ran > 0 else 0.0
    total_frames = int(total_generated_frames)
    avg_infer_per_frame = (infer_sum / total_frames) if total_frames > 0 else 0.0
    rprint(
        "[INFO] done: segments={} frames={} init={:.2f}s infer_sum={:.2f}s avg_infer={:.2f}s avg_frame={:.3f}s total={:.2f}s".format(
            segments_ran,
            total_frames,
            init_time,
            infer_sum,
            avg_infer,
            avg_infer_per_frame,
            total_time,
        )
    )
    rprint("[PROG] runs_root: {}".format(runs_root))
    _cleanup_process_group()
