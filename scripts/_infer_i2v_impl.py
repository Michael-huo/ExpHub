import warnings
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()   # 只显示 error，屏蔽 warning/info
# 如果你连 error 也不想看到（不建议），可以：
# diffusers_logging.disable_default_handler()

# 1. 忽略 torch.autocast 弃用警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")
# 2. 忽略 VAE config 属性访问警告
warnings.filterwarnings("ignore", message=".*Accessing config attribute.*directly via.*")
# 3. 忽略 torch.load weights_only 警告
warnings.filterwarnings("ignore", message=".*torch.load with weights_only=False.*")
# 4. 忽略 padding mask 警告
warnings.filterwarnings("ignore", message=".*Padding mask is disabled.*")
# 5. 忽略 torch.load weights_only 警告
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning
)

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
import argparse
import time
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import torch.distributed as dist
import json
from datetime import datetime
from pathlib import Path
import ctypes
import ctypes.util
from _common import get_platform_config


# NOTE: ExpHub version relies on PYTHONPATH pointing to VideoX-Fun repo root for importing `videox_fun`.
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunInpaintPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# Resolve VideoX-Fun repo root (do NOT depend on this script's location).
import videox_fun
VIDEOX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(videox_fun.__file__)))


T_ALL_START = time.time()

# def rank0_print(msg: str):
#     if os.environ.get("RANK", "0") != "0":
#         return
#     print(msg, flush=True)

ALLOW_PREFIX = ("[PROG]", "[INFO]", "[WARN]", "[ERR]", "[BAR]", "[PROMPT]")

def rprint(msg: str):
    """Rank0-only print, and keep output minimal."""
    if os.environ.get("RANK", "0") != "0":
        return
    if not msg.startswith(ALLOW_PREFIX):
        return
    print(msg, flush=True)


def malloc_trim():
    """Best-effort: ask glibc to return freed heap pages back to the OS."""
    try:
        libc_path = ctypes.util.find_library("c")
        if not libc_path:
            return
        libc = ctypes.CDLL(libc_path)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        # Best-effort only.
        pass

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_cpu_offload_and_qfloat8"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
# ulysses_degree      = 1
# ring_degree         = 1

# ===== 新增：命令行参数解析控制GPU数量 =====
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (1 for single-GPU)")

# ===== Batch mode (run multiple 1s segments in one process) =====
parser.add_argument("--batch", action="store_true", help="Run sequential segments in one process")
parser.add_argument(
    "--frames_dir",
    type=str,
    default="",
    help="Dataset frames directory (e.g., <dataset>/frames). Required in --batch mode.",
)
parser.add_argument("--base_idx", type=int, default=0, help="Start frame index in dataset")
parser.add_argument("--num_segments", type=int, default=16, help="Number of 1-second segments to run")
parser.add_argument("--dataset_fps", type=int, default=25, help="Input dataset FPS (used to compute segment boundaries)")
parser.add_argument("--segment_seconds", type=float, default=1.0, help="Seconds per segment (default 1.0)")
parser.add_argument("--fps", type=int, default=25, help="Target output FPS for generation")
parser.add_argument("--prompt", type=str, default="", help="Text prompt (empty => use script default)")
parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (empty => use script default)")
parser.add_argument("--prompt_manifest", type=str, default="", help="Optional prompt manifest json; archived under prompt/manifest.json")
parser.add_argument("--guidance_scale", type=float, default=-1.0, help="CFG guidance scale (-1 => use default)")
parser.add_argument("--num_inference_steps", type=int, default=-1, help="Denoising steps (-1 => use default)")
parser.add_argument("--config_path", type=str, default="", help="Override YAML config path (optional)")
parser.add_argument("--model_name", type=str, default="", help="Override model directory (optional)")
parser.add_argument("--start_image", type=str, default="", help="(single) start image path")
parser.add_argument("--end_image", type=str, default="", help="(single) end image path")
parser.add_argument("--seed_base", type=int, default=43, help="Base seed; segment i uses seed_base+i")
parser.add_argument(
    "--kf_gap",
    type=int,
    default=0,
    help=(
        "(batch) Keyframe gap in frames. Segment i uses anchors [base+i*kf_gap, base+i*kf_gap+kf_gap]. "
        "Each segment outputs L=kf_gap+1 frames (includes both anchors). "
        "For r=4 models, choose kf_gap%4==0 to avoid temporal alignment truncation."
    ),
)

# ===== New: runs folder auto naming (short) =====
parser.add_argument(
    "--runs_parent",
    type=str,
    default="",
    help="Parent directory for runs. Default: <VideoX-Fun>/samples/wan-videos-fun-i2v/runs",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="",
    help="Experiment folder name. If empty, auto-generate a short name.",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Optional short tag appended to exp_name (e.g., seq01/promptA/ablation1).",
)

args = parser.parse_args()

if args.batch and not args.frames_dir.strip():
    raise SystemExit('[ERR] --frames_dir is required in --batch mode')


# ---- torchrun multi-gpu: bind current process to its local GPU early ----
# Some upstream helpers don't read LOCAL_RANK, leading to "local_rank=-1" logs and
# NCCL barrier warnings. Binding here makes torch.cuda.current_device() correct.
if args.gpus > 1:
    try:
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(_local_rank)
    except Exception as _e:
        rprint(f"[WARN] failed to set local cuda device: {_e}")

ulysses_degree = args.gpus
ring_degree = 1
fsdp_text_encoder = (args.gpus == 1)  # 单卡开 FSDP，多卡关
# =========================================
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
# fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# | Wan2.2-Fun-A14B-*   | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# # Config and model path
# config_path         = "config/wan2.2/wan_civitai_i2v.yaml"
# # model path
# model_name          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"

# Config and model path from platform config (allow CLI override).
cfg = get_platform_config()
default_config = cfg.get("models", {}).get("wan2_2", {}).get("config", "")
default_model = cfg.get("models", {}).get("wan2_2", {}).get("path", "")

config_path = args.config_path.strip() or default_config
model_name = args.model_name.strip() or default_model


# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
lora_path               = None
lora_high_path          = None 

# Other params (keep defaults, but make them configurable via CLI).
sample_size = None            # [H, W] inferred from input images
fps = int(args.fps)           # target FPS for generation
video_length = None           # will be set after probing input and segment_seconds

# (single) start/end images. Batch mode will override per segment.
validation_image_start = args.start_image.strip() or None
validation_image_end   = args.end_image.strip() or None


# =========================
# Prompt manifest (base + optional delta per segment)
# =========================

def _canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _sha1_hex(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _write_text_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def _write_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_prompt_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("manifest must be a JSON object")
    ver = m.get("version", 1)
    if int(ver) != 1:
        raise ValueError(f"unsupported manifest version: {ver}")
    base = str(m.get("base_prompt", "")).strip()
    if not base:
        raise ValueError("manifest.base_prompt is required and cannot be empty")
    base_neg = str(m.get("base_neg_prompt", "")).strip()
    seg_items = m.get("segments", [])
    seg_map = {}
    if seg_items is not None:
        if not isinstance(seg_items, list):
            raise ValueError("manifest.segments must be a list")
        for it in seg_items:
            if not isinstance(it, dict):
                continue
            if "seg" not in it:
                continue
            seg = int(it["seg"])
            dp = str(it.get("delta_prompt", "") or "").strip()
            dn = str(it.get("delta_neg_prompt", "") or "").strip()
            seg_map[seg] = {"delta_prompt": dp, "delta_neg_prompt": dn}
    return {"version": 1, "base_prompt": base, "base_neg_prompt": base_neg, "segments": seg_map, "_raw": m}

def _resolve_prompts(manifest: dict, nseg: int) -> list:
    base = manifest["base_prompt"]
    base_neg = manifest.get("base_neg_prompt", "") or ""
    seg_map = manifest.get("segments", {}) or {}
    out = []
    for seg in range(int(nseg)):
        dp = ""
        dn = ""
        if seg in seg_map:
            dp = seg_map[seg].get("delta_prompt", "") or ""
            dn = seg_map[seg].get("delta_neg_prompt", "") or ""
        final_p = base if not dp else (base + "\n" + dp)
        if base_neg and dn:
            final_n = base_neg + "\n" + dn
        elif base_neg:
            final_n = base_neg
        else:
            final_n = dn
        ph = _sha1_hex(final_p + "\n||NEG||\n" + final_n)[:8]
        out.append({
            "seg": seg,
            "base_prompt": base,
            "delta_prompt": dp,
            "final_prompt": final_p,
            "base_neg_prompt": base_neg,
            "delta_neg_prompt": dn,
            "final_neg_prompt": final_n,
            "prompt_hash8": ph,
            "has_delta": bool(dp or dn),
        })
    return out

def _escape_one_line(s: str) -> str:
    # Keep terminal output single-line to avoid filtered logs dropping lines.
    return (s or "").replace("\r", "").replace("\n", "\\n")

def _probe_frame_path(frames_dir: str, idx: int) -> str:
    """Find frame file by idx under frames_dir. Supports png/jpg/jpeg/webp."""
    base = os.path.join(frames_dir, f"{idx:06d}")
    for ext in ('.png', '.jpg', '.jpeg', '.webp'):
        p = base + ext
        if os.path.exists(p):
            return p
    # fallback: try without zero padding (rare)
    base2 = os.path.join(frames_dir, str(idx))
    for ext in ('.png', '.jpg', '.jpeg', '.webp'):
        p = base2 + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"frame not found: idx={idx} under {frames_dir}")

def _infer_sample_size_from_path(p: str):
    im = Image.open(p)
    w, h = im.size
    im.close()
    return [h, w]

# Infer resolution early so exp naming/logs reflect the actual dataset.
if args.batch:
    _p0 = _probe_frame_path(args.frames_dir, int(args.base_idx))
    sample_size = _infer_sample_size_from_path(_p0)
else:
    if validation_image_start:
        sample_size = _infer_sample_size_from_path(validation_image_start)

# Derive desired frame count.
# - single mode: video_length ~= fps * seconds
# - batch mode (ExpHub keyframe interpolation): video_length is driven by keyframe gap,
#   to keep the time-grid/indices well-defined and avoid "fps" semantic ambiguity.
if args.batch:
    if int(args.kf_gap) <= 0:
        raise SystemExit('[ERR] --kf_gap is required in --batch mode (keyframe interpolation)')
    video_length = int(args.kf_gap) + 1
else:
    video_length = max(1, int(round(float(fps) * float(args.segment_seconds))))

if sample_size is None:
    raise SystemExit('[ERR] cannot infer image size. Provide --frames_dir (batch) or --start_image (single).')


# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
prompt              = "First-person camera moving forward along an outdoor park walkway. Photorealistic, natural lighting, stable exposure and white balance. Consistent perspective and geometry, level horizon, sharp textures on pavement, grass, and trees. No flicker, no warping, no artifacts."
negative_prompt     = "static camera, fixed viewpoint, blurry, distorted, flickering, warping, wobble, rolling shutter artifacts, ghosting, double edges, inconsistent geometry, wrong perspective, texture swimming, repeating patterns, text, watermark, low quality, JPEG compression artifacts, excessive noise, color shift, unnatural motion"
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 50

# CLI overrides (keep default behavior when args are empty / -1).
if args.prompt.strip():
    prompt = args.prompt.strip()
if args.negative_prompt.strip():
    negative_prompt = args.negative_prompt.strip()
if float(args.guidance_scale) >= 0:
    guidance_scale = float(args.guidance_scale)
if int(args.num_inference_steps) > 0:
    num_inference_steps = int(args.num_inference_steps)

# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight         = 0.55
lora_high_weight    = 0.55

# =========================
# Output layout (run folder)
# =========================
TASK_NAME = "i2v"              # 目录名里的 task 标签
SAVE_FRAMES = True             # 同时导出逐帧图片
FRAME_EXT = "png"              # png 稳；也可改成 jpg 省空间

# Default output under VideoX-Fun/samples unless --runs_parent is provided.
DEFAULT_SAMPLES_ROOT = os.path.join(VIDEOX_ROOT, "samples")

def _auto_exp_name_short() -> str:
    """
    Generate a short experiment folder name (3-5 key params):
    <timestamp>_dur{total}s_{fps}fps_{W}x{H}[_tag]
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not sample_size:
        w, h = 0, 0
    else:
        w, h = int(sample_size[1]), int(sample_size[0])
    out_fps = int(fps)
    if args.batch:
        total_dur = float(args.num_segments) * float(args.segment_seconds)
    else:
        total_dur = float(video_length) / max(out_fps, 1)
    # Pretty duration
    if abs(total_dur - round(total_dur)) < 1e-6:
        dur_str = str(int(round(total_dur)))
    else:
        dur_str = f"{total_dur:g}"
    name = f"{ts}_dur{dur_str}s_{out_fps}fps_{w}x{h}"
    if args.tag:
        name = f"{name}_{args.tag}"
    return name

# Default parent keeps original layout
_default_runs_parent = os.path.join(DEFAULT_SAMPLES_ROOT, "wan-videos-fun-i2v", "runs")
RUNS_PARENT = args.runs_parent.strip() or _default_runs_parent
EXP_NAME = args.exp_name.strip() or _auto_exp_name_short()
RUNS_ROOT = os.path.join(RUNS_PARENT, EXP_NAME)


# ---- prompt dir (ExpHub: <exp_dir>/prompt; fallback: under RUNS_PARENT) ----
_runs_parent_p = Path(RUNS_PARENT).resolve()
if _runs_parent_p.name == "infer":
    _exp_dir_p = _runs_parent_p.parent
    PROMPTS_DIR = _exp_dir_p / "prompt"
else:
    PROMPTS_DIR = _runs_parent_p / "prompt"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

_manifest_dst = PROMPTS_DIR / "manifest.json"
if args.prompt_manifest.strip():
    src = Path(args.prompt_manifest).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"prompt_manifest not found: {src}")
    import shutil
    shutil.copy2(str(src), str(_manifest_dst))
elif not _manifest_dst.is_file():
    # Auto-generate minimal manifest from current (possibly CLI-overridden) prompts.
    auto_m = {
        "version": 1,
        "base_prompt": prompt.strip(),
        "base_neg_prompt": negative_prompt.strip(),
    }
    _write_json_atomic(str(_manifest_dst), auto_m)

manifest_path = str(_manifest_dst)
_manifest_loaded = _load_prompt_manifest(manifest_path)

# manifest digest (stable)
manifest_canon = _canonical_json(_manifest_loaded["_raw"])
manifest_digest8 = _sha1_hex(manifest_canon)[:8]
_write_text_atomic(str(PROMPTS_DIR / "digest.txt"), manifest_digest8 + "\n")

# resolve per segment (batch: num_segments; single: 1)
_resolve_nseg = int(args.num_segments) if args.batch else 1
resolved_prompts = _resolve_prompts(_manifest_loaded, _resolve_nseg)

# save resolved for reproducibility (rank0 only)
if os.environ.get("RANK", "0") == "0":
    _write_json_atomic(str(PROMPTS_DIR / "resolved.json"), {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "manifest_path": manifest_path,
        "manifest_digest8": manifest_digest8,
        "num_segments": _resolve_nseg,
        "items": resolved_prompts,
    })


T_INIT_START = time.time()
t_quant_low = 0.0
t_quant_high = 0.0
rprint("[INFO] Initializing model pipeline and loading weights from disk...")
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
else:
    rank = 0
config = OmegaConf.load(config_path)

# Pick a sane weight dtype (bf16 on supported GPUs; otherwise fp16).
# Can be overridden via env: WAN_WEIGHT_DTYPE=bf16|fp16|fp32
_wd = os.environ.get('WAN_WEIGHT_DTYPE', '').strip().lower()
if _wd in ('bf16', 'bfloat16'):
    weight_dtype = torch.bfloat16
elif _wd in ('fp16', 'float16', 'half'):
    weight_dtype = torch.float16
elif _wd in ('fp32', 'float32'):
    weight_dtype = torch.float32
else:
    try:
        if torch.cuda.is_available() and getattr(torch.cuda, 'is_bf16_supported', lambda: False)():
            weight_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
    except Exception:
        weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
if rank == 0:
    rprint(f"[INIT] weight_dtype={weight_dtype}")

boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    transformer_2 = None

if transformer_path is not None:
    rprint(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    rprint(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        rprint(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        rprint(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    rprint(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    rprint(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2FunInpaintPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        rprint("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        rprint("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    rprint("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    rprint("[INFO] Starting float8 quantization (CPU/Mem bound process, please wait...)")
    _t = time.time()
    rprint("[INFO] Quantizing transformer 1/2...")
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    t_quant_low = time.time() - _t
    rprint(f"[INFO] Transformer 1/2 quantized in {t_quant_low:.2f}s")

    if transformer_2 is not None:
        _t = time.time()
        rprint("[INFO] Quantizing transformer 2/2...")
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        t_quant_high = time.time() - _t
        rprint(f"[INFO] Transformer 2/2 quantized in {t_quant_high:.2f}s")

    pipeline.enable_model_cpu_offload(device=device)

elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
    
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    rprint("[INFO] Starting float8 quantization (CPU/Mem bound process, please wait...)")
    _t = time.time()
    rprint("[INFO] Quantizing transformer 1/2...")
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    t_quant_low = time.time() - _t
    rprint(f"[INFO] Transformer 1/2 quantized in {t_quant_low:.2f}s")

    if transformer_2 is not None:
        _t = time.time()
        rprint("[INFO] Quantizing transformer 2/2...")
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        t_quant_high = time.time() - _t
        rprint(f"[INFO] Transformer 2/2 quantized in {t_quant_high:.2f}s")

    pipeline.to(device=device)

else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    rprint(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    rprint(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

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

pipeline.set_progress_bar_config(bar_format="[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
T_INIT_END = time.time()
t_init_total = T_INIT_END - T_INIT_START
t_quant_total = t_quant_low + t_quant_high
t_load = t_init_total - t_quant_total
rprint(f"[INFO] Initialization completed in {t_init_total:.2f}s (Loading: {t_load:.2f}s, Quantization: {t_quant_total:.2f}s)")

# ===== Align video_length with VAE temporal compression =====
video_length_desired = video_length
r = int(getattr(vae.config, "temporal_compression_ratio", 1))
video_length = int((video_length - 1) // r * r) + 1 if video_length != 1 else 1
video_length_run = video_length
latent_frames = (video_length - 1) // r + 1

if enable_riflex:
    with torch.no_grad():
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)


def _frame_path(frames_dir: str, idx: int) -> str:
    """Resolve dataset frame path. Tries png/jpg/jpeg."""
    for ext in ("png", "jpg", "jpeg"):
        p = os.path.join(frames_dir, f"{idx:06d}.{ext}")
        if os.path.exists(p):
            return p
    # default to png (for error message / debugging)
    return os.path.join(frames_dir, f"{idx:06d}.png")


def run_one_segment(start_path: str, end_path: str, seg_seed: int) -> float:
    """Run one i2v segment in the current process (no re-init / no re-quant)."""
    global validation_image_start, validation_image_end, generator, sample, start_time, end_time, seed

    validation_image_start = start_path
    validation_image_end = end_path
    seed = int(seg_seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        start_img = Image.open(validation_image_start).convert("RGB")
        end_img = Image.open(validation_image_end).convert("RGB")
        input_video, input_video_mask, _clip_image = get_image_to_video_latent(
            [start_img],
            [end_img],
            video_length=video_length,
            sample_size=sample_size,
        )

        rprint(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFERENCE] Start seg "
            f"s={os.path.basename(validation_image_start)} e={os.path.basename(validation_image_end)} "
            f"seed={seed} ({video_length} frames, {sample_size[1]}×{sample_size[0]})..."
        )
        start_time = time.time()

        _out = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=shift,
        )

        # Keep the final video tensor only on rank0. Other ranks don't save artifacts,
        # and holding the output can blow up host RAM across segments.
        _rank = int(os.environ.get("RANK", "0"))
        if _rank == 0:
            sample = _out.videos
        else:
            sample = None

        # Aggressively release per-segment temporaries ASAP (helps avoid host-RAM growth)
        try:
            del _out
            del input_video, input_video_mask, _clip_image
            del start_img, end_img
        except Exception:
            pass

        end_time = time.time()
        dt = float(end_time - start_time)
        rprint(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFERENCE] Completed in {dt:.2f} s")
        return dt

def _parse_frame_idx(path: str):
    """Parse '000029.png' -> 29. Return None if not numeric."""
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        return int(base)
    except Exception:
        return None

def _make_run_name(task: str, width: int, height: int, fps: int, L: int, s_idx, e_idx, seed: int):
    # Deterministic run folder name (NO timestamp) so re-runs overwrite cleanly.
    s = f"{s_idx:06d}" if isinstance(s_idx, int) else "NA"
    e = f"{e_idx:06d}" if isinstance(e_idx, int) else "NA"
    return f"{task}_{width}x{height}_{fps}fps_L{L}_s{s}_e{e}_seed{seed}"

def _tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """
    frame_chw: [C,H,W] torch tensor (possibly on GPU / bf16).
    Return uint8 numpy [H,W,C] in [0,255].
    """
    frame = frame_chw.detach().float().cpu().clamp(0, 1)  # safe conversion
    frame = frame.permute(1, 2, 0).numpy()                 # HWC
    frame = (frame * 255.0 + 0.5).astype(np.uint8)         # rounding
    return frame

def _save_frames(sample: torch.Tensor, frames_dir: str, ext: str = "png"):
    os.makedirs(frames_dir, exist_ok=True)
    T = sample.shape[2]  # [B,C,T,H,W]
    for t in range(T):
        arr = _tensor_frame_to_uint8_hwc(sample[0, :, t])
        Image.fromarray(arr).save(os.path.join(frames_dir, f"frame_{t:06d}.{ext}"))

def save_results():
    # 只在 rank0 保存（你外面已有判断，这里再保险一次也无妨）
    if os.environ.get("RANK", "0") != "0":
        return

    os.makedirs(RUNS_ROOT, exist_ok=True)

    width, height = sample_size[1], sample_size[0]
    s_idx = _parse_frame_idx(validation_image_start) if validation_image_start else None
    e_idx = _parse_frame_idx(validation_image_end) if validation_image_end else None

    # video_length_run / r / video_length_desired 在上面 with torch.no_grad() 里生成
    run_name = _make_run_name(
        TASK_NAME, width, height, fps,
        int(video_length_desired if "video_length_desired" in globals() else video_length),
        s_idx, e_idx, seed
    )
    run_dir = os.path.join(RUNS_ROOT, run_name)
    # Overwrite run_dir to avoid stale frames when re-running with the same name.
    if os.path.isdir(run_dir):
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir, exist_ok=True)

    # 1) save mp4
    video_path = os.path.join(run_dir, "preview.mp4")
    save_videos_grid(sample, video_path, fps=fps)

    # 2) save frames
    frames_dir = os.path.join(run_dir, "frames")
    if SAVE_FRAMES:
        _save_frames(sample, frames_dir, ext=FRAME_EXT)

    # 3) write params.json
    params = {
        "task": TASK_NAME,
        "created_at": datetime.now().isoformat(),
        "experiment_name": EXP_NAME,
        "experiment_root": RUNS_ROOT,

        "dataset_fps": int(args.dataset_fps) if args.batch else 30,  # dataset FPS
        "target_fps": int(fps),

        "width": int(width),
        "height": int(height),

        "video_length_desired": int(video_length_desired) if "video_length_desired" in globals() else int(video_length),
        "video_length_run": int(video_length_run) if "video_length_run" in globals() else int(video_length),
        "vae_temporal_compression_ratio": int(r) if "r" in globals() else None,

        "start_idx": s_idx,
        "end_idx": e_idx,
        "start_path": validation_image_start,
        "end_path": validation_image_end,

        "batch": bool(args.batch),
        "source_frames_dir": args.frames_dir if args.batch else None,
        "base_idx": int(args.base_idx) if args.batch else None,
        "num_segments": int(args.num_segments) if args.batch else None,
        "segment_seconds": float(args.segment_seconds) if args.batch else None,

        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),

        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_hash8": _sha1_hex(prompt + "\n||NEG||\n" + negative_prompt)[:8],
        "prompt_manifest": os.path.abspath(manifest_path) if "manifest_path" in globals() else None,
        "prompt_manifest_digest8": manifest_digest8 if "manifest_digest8" in globals() else None,

        "output_dir": run_dir,
        "output_video": "preview.mp4",
        "frames_dir": "frames" if SAVE_FRAMES else None,
        "frame_ext": FRAME_EXT if SAVE_FRAMES else None,
    }

    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
        
    run_dir = os.path.dirname(video_path)
    rprint(f"[OK] run saved: {run_dir}")
    rprint(f"[OK] video: {video_path}")
    if SAVE_FRAMES:
        rprint(f"[OK] frames: {frames_dir}")

def _rank0_save():
    """Save artifacts for the current global `sample` (rank0 only)."""
    if ulysses_degree * ring_degree > 1:
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()


# =========================
# Main: single / batch run
# =========================
infer_sum = 0.0
segments_ran = 0


if args.batch:
    # Keyframe interpolation mode:
    # - stride = kf_gap (shared anchor between adjacent segments)
    # - step   = stride + 1 (output frames per segment, includes both anchors)
    stride = int(args.kf_gap)
    if stride <= 0:
        raise ValueError(f"invalid kf_gap={stride}, must be > 0")
    step = stride + 1

    rprint(f"[PROG] batch: segments={int(args.num_segments)} base_idx={int(args.base_idx)} step={int(step)} stride={int(stride)} fps={int(fps)}")
    # ---- write plan (rank0) so merger only consumes THIS run set ----
    if os.environ.get("RANK", "0") == "0":
        try:
            width, height = int(sample_size[1]), int(sample_size[0])
        except Exception:
            width, height = 0, 0
        L_desired = int(step)
        plan_path = os.path.join(RUNS_PARENT, "runs_plan.json")
        segs = []
        for seg in range(int(args.num_segments)):
            s = int(args.base_idx + seg * stride)
            e = int(s + stride)
            seg_seed = int(args.seed_base) + int(seg)
            run_name = _make_run_name(TASK_NAME, width, height, int(fps), L_desired, s, e, seg_seed)
            segs.append({
                "seg": int(seg),
                "start_idx": int(s),
                "end_idx": int(e),
                "seed": int(seg_seed),
                "run_name": run_name,
                "prompt_hash8": resolved_prompts[int(seg)]["prompt_hash8"] if int(seg) < len(resolved_prompts) else None,
                "has_delta": resolved_prompts[int(seg)]["has_delta"] if int(seg) < len(resolved_prompts) else False,
            })
        plan = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "runs_parent": os.path.abspath(RUNS_PARENT),
            "runs_root": os.path.abspath(RUNS_ROOT),
            "exp_name": EXP_NAME,
            "task": TASK_NAME,
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "dataset_fps": int(args.dataset_fps),
            "segment_seconds": float(args.segment_seconds),
            "kf_gap": int(stride),
            "step": int(step),
            "stride": int(stride),
            "base_idx": int(args.base_idx),
            "num_segments": int(args.num_segments),
            "seed_base": int(args.seed_base),
            "prompt_manifest": os.path.abspath(manifest_path),
            "prompt_manifest_digest8": manifest_digest8,
            "segments": segs,
        }
        tmp = plan_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        os.replace(tmp, plan_path)
    t_batch0 = time.time()

    for seg in range(int(args.num_segments)):
        s = int(args.base_idx + seg * stride)
        e = int(s + stride)
        start_path = _frame_path(args.frames_dir, s)
        end_path = _frame_path(args.frames_dir, e)
        seg_seed = int(args.seed_base) + int(seg)

        seg_i = int(seg) + 1
        total = int(args.num_segments)
        hash_str = ""
        if "resolved_prompts" in globals() and int(seg) < len(resolved_prompts):
            _rp = resolved_prompts[int(seg)]
            hash_str = f" hash={_rp['prompt_hash8']}"
        rprint(f"[PROG] seg {seg_i}/{total}: idx {s}->{e} seed={seg_seed}{hash_str}")
        # prompt per segment (base + optional delta)
        if "resolved_prompts" in globals() and int(seg) < len(resolved_prompts):
            _rp = resolved_prompts[int(seg)]
            prompt = _rp["final_prompt"]
            negative_prompt = _rp["final_neg_prompt"]
            # Print prompts without embedded newlines to keep logs clean and unambiguous.
            # We print base/delta separately (delta is optional). The model still receives
            # the real final prompt (base + "\n" + delta when delta exists).
            rprint(f"[PROMPT] seg {seg_i}/{total} base={_escape_one_line(_rp.get('base_prompt',''))}")
            if _rp.get("delta_prompt"):
                rprint(f"[PROMPT] seg {seg_i}/{total} delta={_escape_one_line(_rp.get('delta_prompt',''))}")
            if negative_prompt:
                rprint(f"[PROMPT] seg {seg_i}/{total} neg={_escape_one_line(negative_prompt)}")
        dt = run_one_segment(start_path, end_path, seg_seed)
        infer_sum += dt
        segments_ran += 1

        t_save0 = time.time()
        _rank0_save()
        t_save = time.time() - t_save0

        elapsed = time.time() - t_batch0
        left = total - seg_i
        eta = (elapsed / seg_i * left) if seg_i > 0 and left > 0 else 0.0
        rprint(f"[INFO] seg {seg_i}/{total}: infer={dt:.2f}s save={t_save:.2f}s elapsed={elapsed:.1f}s eta={eta:.1f}s")

        # ---- multi-gpu safety: keep ranks in lockstep across segments ----
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # ---- per-segment cleanup: avoid host RAM growth across segments ----
        import gc
        try:
            del sample
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        malloc_trim()

        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

else:
    # keep original single-run behavior
    _rp = None
    if "resolved_prompts" in globals() and len(resolved_prompts) > 0:
        _rp = resolved_prompts[0]
    if _rp is not None:
        prompt = _rp["final_prompt"]
        negative_prompt = _rp["final_neg_prompt"]
        rprint(f"[PROMPT] single base={_escape_one_line(_rp.get('base_prompt',''))}")
        if _rp.get("delta_prompt"):
            rprint(f"[PROMPT] single delta={_escape_one_line(_rp.get('delta_prompt',''))}")
        if negative_prompt:
            rprint(f"[PROMPT] single neg={_escape_one_line(negative_prompt)}")
    infer_sum = run_one_segment(validation_image_start, validation_image_end, seed)
    segments_ran = 1
    _rank0_save()


# Unmerge LoRA at the very end (keep behavior identical, but avoid re-merging per segment)
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


# ===== Performance report (rank0 only) =====
total_time = time.time() - T_ALL_START
init_time = T_INIT_END - T_INIT_START
avg_infer = (infer_sum / segments_ran) if segments_ran > 0 else 0.0
frames_per_seg = int(video_length_run if "video_length_run" in globals() else video_length)
total_frames = int(segments_ran) * int(frames_per_seg)
avg_infer_per_frame = (infer_sum / total_frames) if total_frames > 0 else 0.0
rprint(
    f"[INFO] done: segments={segments_ran} frames={total_frames} "
    f"init={init_time:.2f}s infer_sum={infer_sum:.2f}s avg_infer={avg_infer:.2f}s "
    f"avg_frame={avg_infer_per_frame:.3f}s total={total_time:.2f}s"
)
rprint(f"[PROG] runs_root: {RUNS_ROOT}")
