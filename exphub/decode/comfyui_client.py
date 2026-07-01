#!/usr/bin/env python3
"""
ComfyUI Wan2.2 5B InP decode client for ExpHub experiments.

Purpose:
- Use an exported ComfyUI API-format workflow JSON as a template.
- Inject start frame, end frame, prompt, negative prompt, resolution, length, seed, steps, cfg, fps.
- Submit the workflow to a running local/remote ComfyUI server.
- Wait for completion.
- Collect generated frame PNGs and preview video files from ComfyUI output.
- Write a decode_meta.json for downstream ExpHub merge/eval.

Expected workflow node IDs based on current template:
- 3  : KSampler
- 6  : CLIPTextEncode positive prompt
- 7  : CLIPTextEncode negative prompt
- 57 : CreateVideo
- 58 : SaveVideo
- 70 : LoadImage start frame
- 73 : WanFunInpaintToVideo
- 74 : LoadImage end frame
- 75 : SaveImage frames

Example:
python comfyui_client.py \
  --workflow-json /data/hx/ExpHub/config/workflows/comfyui/wan2_2_5b_inp_api.json \
  --start-frame /data/hx/input/start.png \
  --end-frame /data/hx/input/end.png \
  --positive-prompt "Maintain the first-person perspective and smoothly turn left." \
  --negative-prompt "flickering, warping, geometry drift, blurry, low quality" \
  --segment-id seg_0001 \
  --width 640 --height 480 --length 73 --fps 24 \
  --seed 123456 --steps 20 --cfg 6.0 \
  --comfy-url http://127.0.0.1:8188 \
  --comfy-output-root /home/hx/ComfyUI/output \
  --exp-output-dir /data/hx/ExpHub/tmp/comfyui_decode/seg_0001
"""

from __future__ import annotations

import copy
import json
import mimetypes
import random
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
import yaml

from exphub.common.io import ensure_dir, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog
from exphub.config import get_platform_config


COMFYUI_BACKEND = "comfyui_wan2_2_5b_inp"
REPORT_FILENAME = "decode_report.json"
_SEGMENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SHARED_DECODE_SEED_POLICY = "shared_decode_seed"


@dataclass
class DecodeRequest:
    segment_id: str
    start_frame: str
    end_frame: str
    positive_prompt: str
    negative_prompt: str
    width: int
    height: int
    length: int
    fps: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str = "uni_pc"
    scheduler: str = "simple"
    denoise: float = 1.0


@dataclass
class DecodeResult:
    backend: str
    segment_id: str
    prompt_id: str
    actual_seed: int
    status: str
    video_paths: list[str]
    frames_dir: str
    frame_paths: list[str]
    meta_path: str
    output_frames: int
    history_path: str | None = None
    cleanup_paths: list[str] | None = None
    error: str | None = None
    instance_name: str = ""
    instance_base_url: str = ""
    instance_output_root: str = ""
    generate_sec: float | None = None


@dataclass(frozen=True)
class WorkflowNodes:
    ksampler: str
    positive_prompt: str
    negative_prompt: str
    vae_decode: str
    create_video: str
    save_video: str
    start_image: str
    video_spec: str
    end_image: str
    save_image: str
    lora_loader: str | None = None


@dataclass(frozen=True)
class WorkflowProfile:
    name: str
    workflow_json: Path
    nodes: WorkflowNodes
    lora_enabled: bool = False
    lora_name: str = ""
    lora_strength_model: float | None = None
    lora_strength_clip: float | None = None


DEFAULT_WORKFLOW_NODES = WorkflowNodes(
    ksampler="3",
    positive_prompt="6",
    negative_prompt="7",
    vae_decode="8",
    create_video="57",
    save_video="58",
    start_image="70",
    video_spec="73",
    end_image="74",
    save_image="75",
)

WORKFLOW_CONTRACT_NODE_ROLES = (
    "ksampler",
    "positive_prompt",
    "negative_prompt",
    "vae_decode",
    "create_video",
    "save_video",
    "start_image",
    "video_spec",
    "end_image",
    "save_image",
)


@dataclass(frozen=True)
class ComfyUIInstance:
    name: str
    base_url: str
    output_root: Path

    def as_report(self) -> dict[str, str]:
        return {
            "name": str(self.name),
            "base_url": str(self.base_url),
            "output_root": str(self.output_root),
        }


class ComfyUIDecodeError(RuntimeError):
    pass


def _relative_path(base_dir: str | Path, target_path: str | Path) -> str:
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _image_size(path: str | Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to read ComfyUI decode input frame size") from exc
    with Image.open(str(path)) as image_obj:
        width, height = image_obj.size
    return int(width), int(height)


def _task_steps(task: dict[str, Any]) -> int:
    value = task.get("num_inference_steps")
    if value is None or str(value).strip() == "":
        return 20
    return int(value)


def _task_cfg(task: dict[str, Any]) -> float:
    value = task.get("guidance_scale")
    if value is None or str(value).strip() == "":
        return 6.0
    return float(value)


def _resolve_shared_decode_seed(seed_base: int) -> int:
    seed_value = int(seed_base)
    if seed_value == -1:
        return int(random.SystemRandom().randint(1, 2**31 - 1))
    if seed_value <= 0:
        raise ValueError("seed must be a positive integer or -1, got: {}".format(seed_value))
    return seed_value


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("invalid boolean value: {!r}".format(value))


def _as_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _workflow_profile_metadata(profile: WorkflowProfile) -> dict[str, Any]:
    return {
        "decode_profile": str(profile.name),
        "workflow_json": str(profile.workflow_json),
        "lora_enabled": bool(profile.lora_enabled),
        "lora_name": str(profile.lora_name or ""),
        "lora_strength_model": profile.lora_strength_model,
        "lora_strength_clip": profile.lora_strength_clip,
    }


def _parse_workflow_nodes(raw_nodes: Any, profile_name: str) -> WorkflowNodes:
    if not isinstance(raw_nodes, dict):
        raise RuntimeError("ComfyUI decode profile {} missing nodes mapping".format(profile_name))

    missing = [
        role
        for role in WORKFLOW_CONTRACT_NODE_ROLES
        if raw_nodes.get(role) is None or str(raw_nodes.get(role)).strip() == ""
    ]
    if missing:
        raise RuntimeError(
            "ComfyUI decode profile {} missing required node field(s): {}".format(
                profile_name,
                ", ".join(missing),
            )
        )

    lora_loader = raw_nodes.get("lora_loader")
    return WorkflowNodes(
        ksampler=str(raw_nodes["ksampler"]).strip(),
        positive_prompt=str(raw_nodes["positive_prompt"]).strip(),
        negative_prompt=str(raw_nodes["negative_prompt"]).strip(),
        vae_decode=str(raw_nodes["vae_decode"]).strip(),
        create_video=str(raw_nodes["create_video"]).strip(),
        save_video=str(raw_nodes["save_video"]).strip(),
        start_image=str(raw_nodes["start_image"]).strip(),
        video_spec=str(raw_nodes["video_spec"]).strip(),
        end_image=str(raw_nodes["end_image"]).strip(),
        save_image=str(raw_nodes["save_image"]).strip(),
        lora_loader=str(lora_loader).strip() if lora_loader is not None and str(lora_loader).strip() else None,
    )


def _resolve_workflow_profile(comfyui: dict[str, Any], decode_profile: str | None = None) -> WorkflowProfile:
    profiles = comfyui.get("profiles")
    profile_override = str(decode_profile or "").strip()

    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError("services.comfyui.profiles must be a non-empty mapping")

    active_profile = profile_override or str(comfyui.get("active_profile") or "base").strip() or "base"
    if active_profile not in profiles:
        raise RuntimeError("Unknown ComfyUI decode profile: {}".format(active_profile))

    raw_profile = profiles.get(active_profile)
    if not isinstance(raw_profile, dict):
        raise RuntimeError("services.comfyui.profiles.{} must be a mapping".format(active_profile))

    workflow_value = raw_profile.get("workflow_json")
    if workflow_value is None or str(workflow_value).strip() == "":
        raise RuntimeError("ComfyUI decode profile {} missing workflow_json".format(active_profile))
    workflow_json = Path(str(workflow_value)).expanduser().resolve()
    if not workflow_json.is_file():
        raise RuntimeError(
            "ComfyUI decode profile {} workflow_json not found: {}".format(
                active_profile,
                workflow_json,
            )
        )

    nodes = _parse_workflow_nodes(raw_profile.get("nodes"), active_profile)
    lora_cfg = raw_profile.get("lora") or {}
    if not isinstance(lora_cfg, dict):
        raise RuntimeError("services.comfyui.profiles.{}.lora must be a mapping".format(active_profile))
    lora_enabled = _as_bool(lora_cfg.get("enabled"), False)
    if lora_enabled and not nodes.lora_loader:
        raise RuntimeError("ComfyUI decode profile {} has lora_enabled=true but no lora_loader node".format(active_profile))

    return WorkflowProfile(
        name=str(active_profile),
        workflow_json=workflow_json,
        nodes=nodes,
        lora_enabled=bool(lora_enabled),
        lora_name=str(lora_cfg.get("name") or "").strip(),
        lora_strength_model=_as_optional_float(lora_cfg.get("strength_model")),
        lora_strength_clip=_as_optional_float(lora_cfg.get("strength_clip")),
    )


def _platform_path_from_root(exphub_root: str | Path | None = None) -> Path:
    if exphub_root:
        return Path(exphub_root).resolve() / "config" / "platform.yaml"
    return Path(__file__).resolve().parents[2] / "config" / "platform.yaml"


def resolve_comfyui_platform_config(
    platform_cfg: dict[str, Any] | None = None,
    *,
    exphub_root: str | Path | None = None,
    decode_profile: str | None = None,
) -> dict[str, Any]:
    cfg = platform_cfg if platform_cfg is not None else get_platform_config(exphub_root=exphub_root)
    services = cfg.get("services", {}) if isinstance(cfg, dict) else {}
    if not isinstance(services, dict):
        services = {}
    comfyui = services.get("comfyui", {})
    if not isinstance(comfyui, dict):
        comfyui = {}

    def require(key: str) -> Any:
        value = comfyui.get(key)
        if value is None or str(value).strip() == "":
            raise RuntimeError("Missing services.comfyui.{} in config/platform.yaml".format(key))
        return value

    raw_instances = comfyui.get("instances")
    instances: list[ComfyUIInstance] = []
    if raw_instances is not None:
        if not isinstance(raw_instances, list) or not raw_instances:
            raise RuntimeError("services.comfyui.instances must be a non-empty list when configured")
        for idx, raw_item in enumerate(raw_instances):
            if not isinstance(raw_item, dict):
                raise RuntimeError("services.comfyui.instances[{}] must be a mapping".format(idx))

            def require_instance(key: str) -> Any:
                value = raw_item.get(key)
                if value is None or str(value).strip() == "":
                    raise RuntimeError("Missing services.comfyui.instances[{}].{}".format(idx, key))
                return value

            instances.append(
                ComfyUIInstance(
                    name=str(require_instance("name")).strip(),
                    base_url=str(require_instance("base_url")).strip().rstrip("/"),
                    output_root=Path(str(require_instance("output_root"))).expanduser().resolve(),
                )
            )
    else:
        instances.append(
            ComfyUIInstance(
                name="single",
                base_url=str(require("base_url")).strip().rstrip("/"),
                output_root=Path(str(require("output_root"))).expanduser().resolve(),
            )
        )

    workflow_profile = _resolve_workflow_profile(comfyui, decode_profile=decode_profile)
    workflow_meta = _workflow_profile_metadata(workflow_profile)

    return {
        "base_url": str(instances[0].base_url),
        "workflow_json": workflow_profile.workflow_json,
        "workflow_nodes": workflow_profile.nodes,
        "workflow_profile": workflow_profile,
        **workflow_meta,
        "output_root": Path(instances[0].output_root),
        "parallel": _as_bool(comfyui.get("parallel"), False),
        "schedule": str(comfyui.get("schedule", "longest_first") or "longest_first").strip(),
        "instances": instances,
        "timeout_sec": int(comfyui.get("timeout_sec", 1800) or 1800),
        "poll_interval_sec": float(comfyui.get("poll_interval_sec", 2.0) or 2.0),
        "platform_config": _platform_path_from_root(exphub_root).resolve(),
    }


def _validate_run_output(task: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    run_root = ensure_dir(run_dir, "decode unit run dir")
    params_path = run_root / "params.json"
    if not params_path.is_file():
        raise RuntimeError("ComfyUI decode did not write params.json for {}".format(task["unit_id"]))
    frames_dir = ensure_dir(run_root / "frames", "decode unit frames dir")
    frames = list_frames_sorted(frames_dir)
    if not frames:
        raise RuntimeError("ComfyUI decode produced zero frames for {}".format(task["unit_id"]))
    params = read_json_dict(params_path)
    expected = int(params.get("video_length_run", task["length"]) or 0)
    if expected != len(frames):
        raise RuntimeError(
            "decode unit frame count mismatch for {}: params.video_length_run={} files={}".format(
                task["unit_id"],
                expected,
                len(frames),
            )
        )
    if int(params.get("start_idx", task["start_idx"]) or 0) != int(task["start_idx"]):
        raise RuntimeError("decode unit {} params start_idx mismatch".format(task["unit_id"]))
    if int(params.get("end_idx", task["end_idx"]) or 0) != int(task["end_idx"]):
        raise RuntimeError("decode unit {} params end_idx mismatch".format(task["unit_id"]))
    return {
        "params_path": params_path,
        "frames_dir": frames_dir,
        "num_frames": int(len(frames)),
        "params": params,
    }


def _execution_segments(tasks_payload: dict[str, Any], backend_name: str = COMFYUI_BACKEND) -> list[dict[str, Any]]:
    tasks = list(tasks_payload.get("tasks") or [])
    segments: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        start_idx = int(task["start_idx"])
        end_idx = int(task["end_idx"])
        length = int(task["length"])
        segments.append(
            {
                "seg": int(idx),
                "segment_id": int(idx),
                "unit_id": str(task["unit_id"]),
                "source_unit_id": str(task["unit_id"]),
                "source_span_id": str(dict(task.get("source_prompt_ref") or {}).get("span_id", "")),
                "source_prompt_ref": dict(task.get("source_prompt_ref") or {}),
                "run_id": "run_{:03d}".format(idx),
                "run_name": str(task["run_name"]),
                "schedule_source": "decode.native_tasks",
                "execution_backend": str(backend_name),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "raw_start_idx": int(start_idx),
                "raw_end_idx": int(end_idx),
                "desired_start_idx": int(start_idx),
                "desired_end_idx": int(end_idx),
                "desired_num_frames": int(length),
                "aligned_start_idx": int(start_idx),
                "aligned_end_idx": int(end_idx),
                "aligned_num_frames": int(length),
                "deploy_start_idx": int(start_idx),
                "deploy_end_idx": int(end_idx),
                "raw_gap": int(end_idx - start_idx),
                "deploy_gap": int(end_idx - start_idx),
                "num_frames": int(length),
                "target_num_frames": int(length),
                "align_reason": str(task.get("align_reason", "generation_unit_shared_anchor") or "generation_unit_shared_anchor"),
                "is_valid_for_decode": True,
                "state_label": str(task.get("scene_label", "") or ""),
                "motion_label": str(task.get("motion_label", "") or ""),
                "prompt_source": str(task.get("prompt_source", "prompts.prompt_positive") or "prompts.prompt_positive"),
                "resolved_prompt": str(task["prompt"]),
                "negative_prompt": str(task.get("negative_prompt", "") or ""),
                "prompt": str(task["prompt"]),
                "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
                "num_inference_steps": task.get("num_inference_steps"),
                "guidance_scale": task.get("guidance_scale"),
            }
        )
    return segments


def _write_comfyui_params(
    task: dict[str, Any],
    run_dir: str | Path,
    result: DecodeResult,
    width: int,
    height: int,
    fps: int,
    steps: int,
    cfg: float,
    seed_base: int,
    seed_policy: str,
    resolved_seed: int,
    backend_name: str = COMFYUI_BACKEND,
    workflow_profile: WorkflowProfile | None = None,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    profile_meta = _workflow_profile_metadata(workflow_profile) if workflow_profile else {}
    params = {
        "task": str(task["unit_id"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": "runs",
        "experiment_root": str(run_root.parent),
        "dataset_fps": int(fps),
        "target_fps": int(fps),
        "width": int(width),
        "height": int(height),
        "video_length_desired": int(task["length"]),
        "video_length_run": int(task["length"]),
        "saved_frame_count": int(result.output_frames),
        "start_idx": int(task["start_idx"]),
        "end_idx": int(task["end_idx"]),
        "start_path": str(task["start_frame_path"]),
        "end_path": str(task["end_frame_path"]),
        "batch": True,
        "source_frames_dir": str(Path(task["start_frame_path"]).parent),
        "base_idx": int(task["start_idx"]),
        "num_segments": 1,
        "segment_seconds": float(max(0, int(task["length"]) - 1)) / float(max(int(fps), 1)),
        "schedule_source": "decode.native_tasks",
        "execution_backend": str(backend_name),
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg),
        "seed": int(resolved_seed),
        "actual_seed": int(resolved_seed),
        "seed_base": int(seed_base),
        "resolved_seed": int(resolved_seed),
        "seed_policy": str(seed_policy),
        "prompt": str(task["prompt"]),
        "negative_prompt": str(task.get("negative_prompt", "") or ""),
        "prompt_source": str(task.get("prompt_source", "prompts.prompt_positive") or "prompts.prompt_positive"),
        "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
        "output_dir": str(run_root),
        "output_video": Path(result.video_paths[0]).name if result.video_paths else "",
        "frames_dir": "frames",
        "frame_ext": "png",
        "comfyui_prompt_id": str(result.prompt_id),
        "instance_name": str(result.instance_name),
        "instance_base_url": str(result.instance_base_url),
        "instance_output_root": str(result.instance_output_root),
        "generate_sec": result.generate_sec,
        **profile_meta,
    }
    write_json_atomic(run_root / "params.json", params, indent=2)
    return params


class ComfyUIWanInpClient:
    def __init__(
        self,
        comfy_url: str,
        workflow_json: Path | None,
        comfy_output_root: Path,
        exp_output_dir: Path,
        timeout_sec: int = 1800,
        poll_interval_sec: float = 2.0,
        platform_config: Path | None = None,
        instance_name: str = "single",
        workflow_profile: WorkflowProfile | None = None,
        workflow_nodes: WorkflowNodes | None = None,
    ) -> None:
        self.comfy_url = comfy_url.rstrip("/")
        if workflow_profile is None:
            raise ValueError("ComfyUIWanInpClient requires workflow_profile")
        self.workflow_profile = workflow_profile
        self.nodes = workflow_profile.nodes
        self.workflow_json = workflow_profile.workflow_json
        self.comfy_output_root = comfy_output_root
        self.exp_output_dir = exp_output_dir
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.platform_config = platform_config
        self.instance_name = str(instance_name or "single")

        if not self.workflow_json.exists():
            raise FileNotFoundError(
                "workflow json not found for profile {}: {}".format(
                    self.workflow_profile.name,
                    self.workflow_json,
                )
            )

        self.workflow_template = self._load_workflow(self.workflow_json, self.workflow_profile)

    @staticmethod
    def resolve_seed(seed: int) -> int:
        seed_value = int(seed)
        if seed_value == -1:
            return int(random.SystemRandom().randint(1, 2**63 - 1))
        if seed_value <= 0:
            raise ValueError(f"seed must be a positive integer or -1, got: {seed_value}")
        return seed_value

    @staticmethod
    def _load_workflow(path: Path, profile: WorkflowProfile) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise ValueError("workflow JSON must be API format: top-level object keyed by node id")
        roles = list(WORKFLOW_CONTRACT_NODE_ROLES)
        if profile.lora_enabled:
            if not profile.nodes.lora_loader:
                raise ValueError(
                    "ComfyUI decode profile {} has lora_enabled=true but no lora_loader node".format(
                        profile.name,
                    )
                )
            roles.append("lora_loader")
        for role in roles:
            node_id = getattr(profile.nodes, role)
            if node_id not in workflow:
                contract_note = " (profile contract node)" if role == "vae_decode" else ""
                raise ValueError(
                    "workflow JSON missing node for profile {}: role={} node_id={}{}".format(
                        profile.name,
                        role,
                        node_id,
                        contract_note,
                    )
                )
        return workflow

    def upload_image(self, image_path: Path, subfolder: str = "exphub") -> str:
        """Upload image to ComfyUI input and return the name to place into LoadImage inputs."""
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        with image_path.open("rb") as f:
            files = {"image": (image_path.name, f, mime_type)}
            data = {
                "type": "input",
                "subfolder": subfolder,
                "overwrite": "true",
            }
            resp = requests.post(
                f"{self.comfy_url}/upload/image",
                files=files,
                data=data,
                timeout=120,
            )
        self._raise_for_json_error(resp, "upload image")
        payload = resp.json()

        # ComfyUI normally returns: {"name": "...", "subfolder": "...", "type": "input"}
        name = payload.get("name")
        returned_subfolder = payload.get("subfolder") or subfolder

        if not name:
            raise ComfyUIDecodeError(f"upload image response missing name: {payload}")

        # LoadImage accepts "subfolder/name" for input images.
        if returned_subfolder:
            return f"{returned_subfolder}/{name}"
        return name

    def build_workflow(self, req: DecodeRequest, start_image_name: str, end_image_name: str) -> dict[str, Any]:
        wf = copy.deepcopy(self.workflow_template)
        nodes = self.nodes

        wf[nodes.positive_prompt]["inputs"]["text"] = req.positive_prompt
        wf[nodes.negative_prompt]["inputs"]["text"] = req.negative_prompt

        wf[nodes.start_image]["inputs"]["image"] = start_image_name
        wf[nodes.end_image]["inputs"]["image"] = end_image_name

        wf[nodes.video_spec]["inputs"]["width"] = int(req.width)
        wf[nodes.video_spec]["inputs"]["height"] = int(req.height)
        wf[nodes.video_spec]["inputs"]["length"] = int(req.length)
        wf[nodes.video_spec]["inputs"]["batch_size"] = 1

        wf[nodes.ksampler]["inputs"]["seed"] = int(req.seed)
        wf[nodes.ksampler]["inputs"]["steps"] = int(req.steps)
        wf[nodes.ksampler]["inputs"]["cfg"] = float(req.cfg)
        wf[nodes.ksampler]["inputs"]["sampler_name"] = req.sampler_name
        wf[nodes.ksampler]["inputs"]["scheduler"] = req.scheduler
        wf[nodes.ksampler]["inputs"]["denoise"] = float(req.denoise)

        wf[nodes.create_video]["inputs"]["fps"] = int(req.fps)

        # These prefixes are under ComfyUI/output.
        wf[nodes.save_video]["inputs"]["filename_prefix"] = f"video/{req.segment_id}"
        wf[nodes.save_image]["inputs"]["filename_prefix"] = f"frames/{req.segment_id}/frame"

        if self.workflow_profile.lora_enabled:
            if not nodes.lora_loader:
                raise ComfyUIDecodeError(
                    "ComfyUI decode profile {} has lora_enabled=true but no lora_loader node".format(
                        self.workflow_profile.name,
                    )
                )
            lora_inputs = wf[nodes.lora_loader].setdefault("inputs", {})
            if self.workflow_profile.lora_name:
                lora_inputs["lora_name"] = self.workflow_profile.lora_name
            if self.workflow_profile.lora_strength_model is not None:
                lora_inputs["strength_model"] = float(self.workflow_profile.lora_strength_model)
            if self.workflow_profile.lora_strength_clip is not None:
                lora_inputs["strength_clip"] = float(self.workflow_profile.lora_strength_clip)

        return wf

    def submit_prompt(self, workflow: dict[str, Any]) -> str:
        resp = requests.post(
            f"{self.comfy_url}/prompt",
            json={"prompt": workflow},
            timeout=120,
        )
        self._raise_for_json_error(resp, "submit prompt")
        payload = resp.json()
        if "prompt_id" not in payload:
            raise ComfyUIDecodeError(f"prompt response missing prompt_id: {payload}")
        return str(payload["prompt_id"])

    def wait_history(self, prompt_id: str) -> dict[str, Any]:
        start = time.time()
        last_payload: dict[str, Any] | None = None

        while time.time() - start < self.timeout_sec:
            resp = requests.get(f"{self.comfy_url}/history/{prompt_id}", timeout=60)
            self._raise_for_json_error(resp, "get history")
            payload = resp.json()
            last_payload = payload
            if prompt_id in payload:
                return payload[prompt_id]
            time.sleep(self.poll_interval_sec)

        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out; last response={last_payload}")

    def run(self, req: DecodeRequest) -> DecodeResult:
        self.exp_output_dir.mkdir(parents=True, exist_ok=True)

        status = "ok"
        error = None
        prompt_id = ""
        generate_started = time.time()
        actual_seed = self.resolve_seed(req.seed)
        if int(req.seed) != actual_seed:
            req = DecodeRequest(
                segment_id=req.segment_id,
                start_frame=req.start_frame,
                end_frame=req.end_frame,
                positive_prompt=req.positive_prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                length=req.length,
                fps=req.fps,
                seed=actual_seed,
                steps=req.steps,
                cfg=req.cfg,
                sampler_name=req.sampler_name,
                scheduler=req.scheduler,
                denoise=req.denoise,
            )
        cleanup_paths = self.cleanup_segment_outputs(req.segment_id)

        try:
            start_name = self.upload_image(Path(req.start_frame), subfolder=f"exphub/{req.segment_id}")
            end_name = self.upload_image(Path(req.end_frame), subfolder=f"exphub/{req.segment_id}")

            workflow = self.build_workflow(req, start_name, end_name)

            workflow_debug_path = self.exp_output_dir / "submitted_workflow.json"
            write_json_atomic(workflow_debug_path, workflow, indent=2)

            prompt_id = self.submit_prompt(workflow)
            history_item = self.wait_history(prompt_id)

            history_path = self.exp_output_dir / "comfyui_history.json"
            write_json_atomic(history_path, history_item, indent=2)

            frame_paths = self._collect_frame_paths(req.segment_id)
            video_paths = self._collect_video_paths(req.segment_id)

            if not frame_paths:
                raise ComfyUIDecodeError(
                    f"No frame outputs found under {self.comfy_output_root / 'frames' / req.segment_id}"
                )

            local_frames_dir = self.exp_output_dir / "frames"
            local_frame_paths = self._copy_frames(frame_paths, local_frames_dir)

            local_video_paths = self._copy_videos(video_paths, self.exp_output_dir)

            result = DecodeResult(
                backend=COMFYUI_BACKEND,
                segment_id=req.segment_id,
                prompt_id=prompt_id,
                actual_seed=actual_seed,
                status=status,
                video_paths=[str(p) for p in local_video_paths],
                frames_dir=str(local_frames_dir),
                frame_paths=[str(p) for p in local_frame_paths],
                meta_path=str(self.exp_output_dir / "decode_meta.json"),
                output_frames=len(local_frame_paths),
                history_path=str(history_path),
                cleanup_paths=cleanup_paths,
                error=error,
                instance_name=str(self.instance_name),
                instance_base_url=str(self.comfy_url),
                instance_output_root=str(self.comfy_output_root),
                generate_sec=float(time.time() - generate_started),
            )

        except Exception as exc:
            status = "failed"
            error = str(exc)
            result = DecodeResult(
                backend=COMFYUI_BACKEND,
                segment_id=req.segment_id,
                prompt_id=prompt_id,
                actual_seed=actual_seed,
                status=status,
                video_paths=[],
                frames_dir=str(self.exp_output_dir / "frames"),
                frame_paths=[],
                meta_path=str(self.exp_output_dir / "decode_meta.json"),
                output_frames=0,
                history_path=None,
                cleanup_paths=cleanup_paths,
                error=error,
                instance_name=str(self.instance_name),
                instance_base_url=str(self.comfy_url),
                instance_output_root=str(self.comfy_output_root),
                generate_sec=float(time.time() - generate_started),
            )

        profile_meta = _workflow_profile_metadata(self.workflow_profile)
        meta = {
            "backend": result.backend,
            "segment_id": req.segment_id,
            "prompt_id": prompt_id,
            "actual_seed": actual_seed,
            "instance_name": str(result.instance_name),
            "instance_base_url": str(result.instance_base_url),
            "instance_output_root": str(result.instance_output_root),
            "generate_sec": result.generate_sec,
            "workflow_json": str(self.workflow_json),
            "platform_config": str(self.platform_config) if self.platform_config else None,
            "comfy_url": self.comfy_url,
            "comfy_output_root": str(self.comfy_output_root),
            "exp_output_dir": str(self.exp_output_dir),
            "cleanup_paths": list(cleanup_paths),
            "request": asdict(req),
            "result": asdict(result),
            "workflow_template": str(self.workflow_json),
            "output_frames": int(result.output_frames),
            "video_paths": list(result.video_paths),
            "frames_dir": str(result.frames_dir),
            "frame_paths": list(result.frame_paths),
            "status": str(result.status),
            "error": result.error,
            "expected_comfy_frames_dir": str(self.comfy_output_root / "frames" / req.segment_id),
            "expected_comfy_video_prefix": str(self.comfy_output_root / "video" / req.segment_id),
            **profile_meta,
        }
        Path(result.meta_path).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(Path(result.meta_path), meta, indent=2)

        if result.status != "ok":
            raise ComfyUIDecodeError(
                "ComfyUI decode failed: unit_id={} instance_name={} base_url={} prompt_id={} error={}".format(
                    req.segment_id,
                    self.instance_name,
                    self.comfy_url,
                    prompt_id or "",
                    error or "unknown",
                )
            )

        return result

    @staticmethod
    def _validate_segment_id(segment_id: str) -> str:
        value = str(segment_id or "").strip()
        if not value or not _SEGMENT_ID_RE.match(value) or value in (".", "..") or ".." in value:
            raise ValueError(f"unsafe segment_id for cleanup: {segment_id!r}")
        return value

    def cleanup_segment_outputs(self, segment_id: str) -> list[str]:
        segment = self._validate_segment_id(segment_id)
        cleanup_paths: list[str] = []

        def remove_dir(path: Path) -> None:
            cleanup_paths.append(str(path))
            if path.exists():
                if not path.is_dir() or path.is_symlink():
                    raise ComfyUIDecodeError(f"refusing to recursively delete non-directory path: {path}")
                shutil.rmtree(path)

        def remove_file(path: Path) -> None:
            cleanup_paths.append(str(path))
            if path.exists():
                if path.is_dir():
                    raise ComfyUIDecodeError(f"refusing to delete directory as file: {path}")
                path.unlink()

        remove_dir(self.comfy_output_root / "frames" / segment)
        remove_dir(self.exp_output_dir / "frames")
        remove_file(self.exp_output_dir / "submitted_workflow.json")
        remove_file(self.exp_output_dir / "comfyui_history.json")
        remove_file(self.exp_output_dir / "decode_meta.json")
        remove_file(self.exp_output_dir / "params.json")

        video_dir = self.comfy_output_root / "video"
        if video_dir.exists():
            if not video_dir.is_dir() or video_dir.is_symlink():
                raise ComfyUIDecodeError(f"refusing to scan non-directory video output path: {video_dir}")
            for suffix in ("mp4", "webm", "mov", "mkv"):
                for path in sorted(video_dir.glob(f"{segment}*.{suffix}")):
                    cleanup_paths.append(str(path))
                    if path.is_dir():
                        raise ComfyUIDecodeError(f"refusing to delete directory matched as video output: {path}")
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

        if self.exp_output_dir.exists():
            if not self.exp_output_dir.is_dir() or self.exp_output_dir.is_symlink():
                raise ComfyUIDecodeError(f"refusing to scan non-directory unit output path: {self.exp_output_dir}")
            for suffix in ("mp4", "webm", "mov", "mkv"):
                for path in sorted(self.exp_output_dir.glob(f"{segment}*.{suffix}")):
                    cleanup_paths.append(str(path))
                    if path.is_dir():
                        raise ComfyUIDecodeError(f"refusing to delete directory matched as local video output: {path}")
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

        return cleanup_paths

    def _collect_frame_paths(self, segment_id: str) -> list[Path]:
        frames_dir = self.comfy_output_root / "frames" / segment_id
        if not frames_dir.exists():
            return []
        return sorted(frames_dir.glob("*.png"))

    def _collect_video_paths(self, segment_id: str) -> list[Path]:
        video_dir = self.comfy_output_root / "video"
        if not video_dir.exists():
            return []

        patterns = [
            f"{segment_id}*.mp4",
            f"{segment_id}*.webm",
            f"{segment_id}*.mov",
            f"{segment_id}*.mkv",
        ]
        matches: list[Path] = []
        for pat in patterns:
            matches.extend(video_dir.glob(pat))
        return sorted(set(matches))

    @staticmethod
    def _copy_frames(source_frames: list[Path], dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for idx, src in enumerate(source_frames):
            dst = dest_dir / f"{idx:06d}.png"
            shutil.copy2(src, dst)
            copied.append(dst)
        return copied

    @staticmethod
    def _copy_videos(source_videos: list[Path], dest_dir: Path) -> list[Path]:
        copied: list[Path] = []
        for src in source_videos:
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
        return copied

    @staticmethod
    def _raise_for_json_error(resp: requests.Response, action: str) -> None:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = resp.text[:2000]
            raise ComfyUIDecodeError(f"ComfyUI {action} HTTP error: {exc}; body={body}") from exc

        try:
            payload = resp.json()
        except Exception:
            return

        if isinstance(payload, dict) and "error" in payload:
            raise ComfyUIDecodeError(
                f"ComfyUI {action} error: {json.dumps(payload, ensure_ascii=False)[:4000]}"
            )


def run_comfyui_decode(
    req: DecodeRequest,
    platform_cfg: dict[str, Any] | None = None,
    *,
    comfy_url: str | None = None,
    workflow_json: str | Path | None = None,
    workflow_profile: WorkflowProfile | None = None,
    workflow_nodes: WorkflowNodes | None = None,
    comfy_output_root: str | Path | None = None,
    exp_output_dir: str | Path,
    timeout_sec: int | None = None,
    poll_interval_sec: float | None = None,
    platform_config: str | Path | None = None,
    instance_name: str = "single",
    decode_profile: str | None = None,
) -> DecodeResult:
    if platform_cfg is not None:
        resolved = resolve_comfyui_platform_config(platform_cfg, decode_profile=decode_profile)
        comfy_url = comfy_url or str(resolved["base_url"])
        workflow_json = workflow_json or resolved["workflow_json"]
        workflow_profile = workflow_profile or resolved["workflow_profile"]
        workflow_nodes = workflow_nodes or resolved["workflow_nodes"]
        comfy_output_root = comfy_output_root or resolved["output_root"]
        timeout_sec = int(timeout_sec if timeout_sec is not None else resolved["timeout_sec"])
        poll_interval_sec = float(
            poll_interval_sec if poll_interval_sec is not None else resolved["poll_interval_sec"]
        )
        platform_config = platform_config or resolved["platform_config"]
    if comfy_url is None or workflow_profile is None or comfy_output_root is None:
        raise ValueError("run_comfyui_decode requires platform_cfg or explicit ComfyUI runtime config")
    client = ComfyUIWanInpClient(
        comfy_url=comfy_url,
        workflow_json=Path(workflow_json),
        comfy_output_root=Path(comfy_output_root),
        exp_output_dir=Path(exp_output_dir),
        timeout_sec=int(timeout_sec if timeout_sec is not None else 1800),
        poll_interval_sec=float(poll_interval_sec if poll_interval_sec is not None else 2.0),
        platform_config=Path(platform_config).resolve() if platform_config else None,
        instance_name=str(instance_name or "single"),
        workflow_profile=workflow_profile,
        workflow_nodes=workflow_nodes,
    )
    return client.run(req)


def _check_comfyui_instance_health(instance: ComfyUIInstance) -> None:
    url = "{}/queue".format(str(instance.base_url).rstrip("/"))
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        raise ComfyUIDecodeError(
            "ComfyUI health check failed: instance_name={} base_url={} endpoint={} error={}".format(
                instance.name,
                instance.base_url,
                url,
                exc,
            )
        ) from exc


def _check_comfyui_instances_health(instances: list[ComfyUIInstance]) -> None:
    for instance in instances:
        _check_comfyui_instance_health(instance)


def _shutdown_executor(executor: ThreadPoolExecutor, *, wait_for_running: bool) -> None:
    try:
        executor.shutdown(wait=wait_for_running, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=wait_for_running)


def _task_schedule_length(task: dict[str, Any]) -> int:
    value = task.get("video_length_run")
    if value is None or str(value).strip() == "":
        value = task.get("length")
    return int(value)


def _run_comfyui_unit(
    *,
    task: dict[str, Any],
    task_index: int,
    total_tasks: int,
    runtime: Any,
    cfg: dict[str, Any],
    instance: ComfyUIInstance,
    seed_base: int,
    seed_policy: str,
    resolved_decode_seed: int,
) -> dict[str, Any]:
    run_dir = Path(task["output_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    width, height = _image_size(task["start_frame_path"])
    steps = _task_steps(task)
    cfg_scale = _task_cfg(task)
    requested_seed = int(resolved_decode_seed)

    log_prog(
        "decode generate comfyui: unit={}/{} id={} frames={} seed_request={} instance={} base_url={}".format(
            task_index + 1,
            total_tasks,
            task["unit_id"],
            int(task["length"]),
            int(requested_seed),
            instance.name,
            instance.base_url,
        )
    )

    req = DecodeRequest(
        segment_id=str(task["unit_id"]),
        start_frame=str(task["start_frame_path"]),
        end_frame=str(task["end_frame_path"]),
        positive_prompt=str(task["prompt"]),
        negative_prompt=str(task.get("negative_prompt", "") or ""),
        width=int(width),
        height=int(height),
        length=int(task["length"]),
        fps=int(float(runtime.fps_arg)),
        seed=int(resolved_decode_seed),
        steps=int(steps),
        cfg=float(cfg_scale),
    )

    try:
        result = run_comfyui_decode(
            req,
            comfy_url=instance.base_url,
            workflow_json=cfg["workflow_json"],
            workflow_profile=cfg["workflow_profile"],
            workflow_nodes=cfg["workflow_nodes"],
            comfy_output_root=instance.output_root,
            exp_output_dir=run_dir,
            timeout_sec=int(cfg["timeout_sec"]),
            poll_interval_sec=float(cfg["poll_interval_sec"]),
            platform_config=cfg["platform_config"],
            instance_name=instance.name,
        )
        if int(result.output_frames) != int(task["length"]):
            raise RuntimeError(
                "ComfyUI output frame count mismatch: expected={} actual={}".format(
                    int(task["length"]),
                    int(result.output_frames),
                )
            )

        _write_comfyui_params(
            task=task,
            run_dir=run_dir,
            result=result,
            width=width,
            height=height,
            fps=int(float(runtime.fps_arg)),
            steps=steps,
            cfg=cfg_scale,
            seed_base=seed_base,
            seed_policy=seed_policy,
            resolved_seed=resolved_decode_seed,
            backend_name=COMFYUI_BACKEND,
            workflow_profile=cfg["workflow_profile"],
        )
        actual = _validate_run_output(task, run_dir)
        params = dict(actual.get("params") or {})
        report_seed = params.get("seed", task.get("seed"))
        if report_seed is None or str(report_seed).strip() == "":
            report_seed = resolved_decode_seed
        report_seed = int(report_seed)
        return {
            "unit_id": str(task["unit_id"]),
            "status": "success",
            "error": "",
            "start_idx": int(task["start_idx"]),
            "end_idx": int(task["end_idx"]),
            "length": int(task["length"]),
            "seed": int(report_seed),
            "actual_seed": int(report_seed),
            "requested_seed": int(requested_seed),
            "seed_base": int(seed_base),
            "resolved_seed": int(resolved_decode_seed),
            "seed_policy": str(seed_policy),
            "instance_name": str(result.instance_name),
            "instance_base_url": str(result.instance_base_url),
            "instance_output_root": str(result.instance_output_root),
            "generate_sec": result.generate_sec,
            "frame_count": int(actual["num_frames"]),
            "output_dir": _relative_path(runtime.paths.exp_dir, run_dir),
            "frames_dir": _relative_path(runtime.paths.exp_dir, actual["frames_dir"]),
            "params_path": _relative_path(runtime.paths.exp_dir, actual["params_path"]),
            "decode_meta_path": _relative_path(runtime.paths.exp_dir, run_dir / "decode_meta.json"),
            "num_frames": int(actual["num_frames"]),
            "prompt_hash8": str(task.get("prompt_hash8", "") or ""),
        }
    except Exception as exc:
        raise ComfyUIDecodeError(
            "ComfyUI unit failed: unit_id={} instance_name={} base_url={} error={}".format(
                task.get("unit_id", ""),
                instance.name,
                instance.base_url,
                exc,
            )
        ) from exc


def run_comfyui_decode_tasks(
    tasks_payload: dict[str, Any],
    runtime: Any,
    platform_cfg: dict[str, Any] | None = None,
    *,
    decode_profile: str = "",
    seed_base: int = 12345,
) -> dict[str, Any]:
    cfg = resolve_comfyui_platform_config(
        platform_cfg,
        exphub_root=runtime.exphub_root,
        decode_profile=decode_profile,
    )
    tasks = list(tasks_payload.get("tasks") or [])
    execution_segments = _execution_segments(tasks_payload, COMFYUI_BACKEND)
    if not execution_segments:
        raise RuntimeError("generation task builder produced zero executable tasks")
    instances = list(cfg.get("instances") or [])
    if not instances:
        raise RuntimeError("ComfyUI platform config resolved zero instances")
    parallel_requested = bool(cfg.get("parallel", False))
    parallel_active = bool(parallel_requested and len(instances) > 1)
    configured_schedule = str(cfg.get("schedule") or "longest_first").strip().lower()
    schedule = "longest_first" if parallel_active else "serial"
    if parallel_active and configured_schedule != "longest_first":
        raise RuntimeError(
            "unsupported services.comfyui.schedule for parallel decode: {}".format(cfg.get("schedule"))
        )

    seed_base = int(seed_base)
    resolved_decode_seed = _resolve_shared_decode_seed(seed_base)
    seed_policy = SHARED_DECODE_SEED_POLICY

    _check_comfyui_instances_health(instances)

    log_info(
        "decode seed policy: seed_base={} resolved_seed={} unit_policy=shared".format(
            int(seed_base),
            int(resolved_decode_seed),
        )
    )

    log_prog(
        "decode generate: backend={} tasks={} fps={} seed_policy={} resolved_seed={} parallel={} instances={} schedule={}".format(
            COMFYUI_BACKEND,
            len(execution_segments),
            int(float(runtime.fps_arg)),
            seed_policy,
            int(resolved_decode_seed),
            str(parallel_active).lower(),
            len(instances),
            schedule,
        )
    )
    unit_reports_by_index: list[dict[str, Any] | None] = [None] * len(tasks)
    started = time.time()

    if parallel_active:
        idle_instances: Queue[ComfyUIInstance] = Queue()
        for instance in instances:
            idle_instances.put(instance)

        def run_scheduled(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            idx, task = item
            instance = idle_instances.get()
            try:
                report_item = _run_comfyui_unit(
                    task=task,
                    task_index=idx,
                    total_tasks=len(tasks),
                    runtime=runtime,
                    cfg=cfg,
                    instance=instance,
                    seed_base=seed_base,
                    seed_policy=seed_policy,
                    resolved_decode_seed=resolved_decode_seed,
                )
                return idx, report_item
            finally:
                idle_instances.put(instance)

        scheduled = sorted(
            list(enumerate(tasks)),
            key=lambda item: (-_task_schedule_length(item[1]), int(item[0])),
        )
        executor = ThreadPoolExecutor(max_workers=len(instances))
        futures = [executor.submit(run_scheduled, item) for item in scheduled]
        failed = False
        try:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for future in done:
                    exc = future.exception()
                    if exc is not None:
                        for pending_future in pending:
                            pending_future.cancel()
                        failed = True
                        _shutdown_executor(executor, wait_for_running=False)
                        raise exc
                    idx, report_item = future.result()
                    unit_reports_by_index[idx] = report_item
        finally:
            if not failed:
                _shutdown_executor(executor, wait_for_running=True)
    else:
        instance = instances[0]
        for idx, task in enumerate(tasks):
            unit_reports_by_index[idx] = _run_comfyui_unit(
                task=task,
                task_index=idx,
                total_tasks=len(tasks),
                runtime=runtime,
                cfg=cfg,
                instance=instance,
                seed_base=seed_base,
                seed_policy=seed_policy,
                resolved_decode_seed=resolved_decode_seed,
            )

    missing = [idx for idx, item in enumerate(unit_reports_by_index) if item is None]
    if missing:
        raise RuntimeError("ComfyUI decode missing unit reports for task indexes: {}".format(missing))
    unit_reports = [dict(item) for item in unit_reports_by_index if item is not None]
    elapsed = float(time.time() - started)
    sum_unit_generate_sec = sum(
        float(item["generate_sec"])
        for item in unit_reports
        if item.get("generate_sec") is not None
    )
    parallel_speedup = (
        float(sum_unit_generate_sec) / float(elapsed)
        if elapsed > 0 and sum_unit_generate_sec > 0
        else None
    )
    instances_report = [instance.as_report() for instance in instances]
    workflow_meta = _workflow_profile_metadata(cfg["workflow_profile"])
    report = {
        "schema": "decode_report",
        "stage": "decode",
        "substage": "comfyui_decode",
        "status": "success",
        "run_id": str(runtime.spec.exp_name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_profile": "base_motion_prompt",
        "anchor_backend": "image_embedding_visual_anchor",
        "backend_name": COMFYUI_BACKEND,
        "backend_meta": {
            "backend": COMFYUI_BACKEND,
            "client": "exphub.decode.comfyui_client",
            "call_mode": "direct_python",
        },
        "backend_result": {
            "backend": COMFYUI_BACKEND,
            "mode": "direct_python_comfyui_client",
            "workflow_json": str(cfg["workflow_json"]),
            **workflow_meta,
            "comfy_url": str(cfg["base_url"]),
            "comfy_output_root": str(cfg["output_root"]),
            "platform_config": str(cfg["platform_config"]),
            "execution_segments": int(len(execution_segments)),
            "seed_policy": str(seed_policy),
            "requested_seed": int(seed_base),
            "seed_base": int(seed_base),
            "resolved_seed": int(resolved_decode_seed),
            "total_runtime_sec": float(elapsed),
            "wall_generate_sec": float(elapsed),
            "sum_unit_generate_sec": float(sum_unit_generate_sec),
            "parallel_speedup": parallel_speedup,
            "parallel": bool(parallel_active),
            "schedule": str(schedule),
            "instance_count": int(len(instances)),
            "instances": instances_report,
        },
        "parallel": bool(parallel_active),
        **workflow_meta,
        "schedule": str(schedule),
        "instance_count": int(len(instances)),
        "instances": instances_report,
        "wall_generate_sec": float(elapsed),
        "sum_unit_generate_sec": float(sum_unit_generate_sec),
        "parallel_speedup": parallel_speedup,
        "seed_policy": str(seed_policy),
        "requested_seed": int(seed_base),
        "seed_base": int(seed_base),
        "resolved_seed": int(resolved_decode_seed),
        "num_tasks": int(len(unit_reports)),
        "source_inputs": dict(tasks_payload.get("source_inputs") or {}),
        "task_summary": dict(tasks_payload.get("summary") or {}),
        "units": unit_reports,
        "runs": unit_reports,
        "outputs": {
            "runs_dir": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_runs_dir),
            "report": "decode/{}".format(REPORT_FILENAME),
        },
        "total_runtime_sec": float(elapsed),
    }
    write_json_atomic(runtime.paths.decode_report_path, report, indent=2)
    log_info("decode generate report: {}".format(runtime.paths.decode_report_path))
    return report
