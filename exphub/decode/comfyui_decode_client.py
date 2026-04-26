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
python comfyui_decode_client.py \
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

import argparse
import copy
import json
import mimetypes
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


# Node IDs in the exported API workflow template.
NODE_KSAMPLER = "3"
NODE_POSITIVE_PROMPT = "6"
NODE_NEGATIVE_PROMPT = "7"
NODE_CREATE_VIDEO = "57"
NODE_SAVE_VIDEO = "58"
NODE_START_IMAGE = "70"
NODE_VIDEO_SPEC = "73"
NODE_END_IMAGE = "74"
NODE_SAVE_IMAGE = "75"


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


class ComfyUIDecodeError(RuntimeError):
    pass


class ComfyUIWanInpClient:
    def __init__(
        self,
        comfy_url: str,
        workflow_json: Path,
        comfy_output_root: Path,
        exp_output_dir: Path,
        timeout_sec: int = 1800,
        poll_interval_sec: float = 2.0,
        platform_config: Path | None = None,
    ) -> None:
        self.comfy_url = comfy_url.rstrip("/")
        self.workflow_json = workflow_json
        self.comfy_output_root = comfy_output_root
        self.exp_output_dir = exp_output_dir
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.platform_config = platform_config

        if not self.workflow_json.exists():
            raise FileNotFoundError(f"workflow json not found: {self.workflow_json}")

        self.workflow_template = self._load_workflow(self.workflow_json)

    @staticmethod
    def resolve_seed(seed: int) -> int:
        seed_value = int(seed)
        if seed_value == -1:
            return int(random.SystemRandom().randint(1, 2**63 - 1))
        if seed_value <= 0:
            raise ValueError(f"seed must be a positive integer or -1, got: {seed_value}")
        return seed_value

    @staticmethod
    def _load_workflow(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise ValueError("workflow JSON must be API format: top-level object keyed by node id")
        for node_id in (
            NODE_KSAMPLER,
            NODE_POSITIVE_PROMPT,
            NODE_NEGATIVE_PROMPT,
            NODE_CREATE_VIDEO,
            NODE_SAVE_VIDEO,
            NODE_START_IMAGE,
            NODE_VIDEO_SPEC,
            NODE_END_IMAGE,
            NODE_SAVE_IMAGE,
        ):
            if node_id not in workflow:
                raise ValueError(f"workflow missing required node id {node_id}")
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

        wf[NODE_POSITIVE_PROMPT]["inputs"]["text"] = req.positive_prompt
        wf[NODE_NEGATIVE_PROMPT]["inputs"]["text"] = req.negative_prompt

        wf[NODE_START_IMAGE]["inputs"]["image"] = start_image_name
        wf[NODE_END_IMAGE]["inputs"]["image"] = end_image_name

        wf[NODE_VIDEO_SPEC]["inputs"]["width"] = int(req.width)
        wf[NODE_VIDEO_SPEC]["inputs"]["height"] = int(req.height)
        wf[NODE_VIDEO_SPEC]["inputs"]["length"] = int(req.length)
        wf[NODE_VIDEO_SPEC]["inputs"]["batch_size"] = 1

        wf[NODE_KSAMPLER]["inputs"]["seed"] = int(req.seed)
        wf[NODE_KSAMPLER]["inputs"]["steps"] = int(req.steps)
        wf[NODE_KSAMPLER]["inputs"]["cfg"] = float(req.cfg)
        wf[NODE_KSAMPLER]["inputs"]["sampler_name"] = req.sampler_name
        wf[NODE_KSAMPLER]["inputs"]["scheduler"] = req.scheduler
        wf[NODE_KSAMPLER]["inputs"]["denoise"] = float(req.denoise)

        wf[NODE_CREATE_VIDEO]["inputs"]["fps"] = int(req.fps)

        # These prefixes are under ComfyUI/output.
        wf[NODE_SAVE_VIDEO]["inputs"]["filename_prefix"] = f"video/{req.segment_id}"
        wf[NODE_SAVE_IMAGE]["inputs"]["filename_prefix"] = f"frames/{req.segment_id}/frame"

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
            workflow_debug_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

            prompt_id = self.submit_prompt(workflow)
            history_item = self.wait_history(prompt_id)

            history_path = self.exp_output_dir / "comfyui_history.json"
            history_path.write_text(json.dumps(history_item, ensure_ascii=False, indent=2), encoding="utf-8")

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
                backend="comfyui_wan2_2_5b_inp",
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
            )

        except Exception as exc:
            status = "failed"
            error = str(exc)
            result = DecodeResult(
                backend="comfyui_wan2_2_5b_inp",
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
            )

        meta = {
            "backend": result.backend,
            "segment_id": req.segment_id,
            "prompt_id": prompt_id,
            "actual_seed": actual_seed,
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
        }
        Path(result.meta_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result.meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if result.status != "ok":
            raise ComfyUIDecodeError(error or "ComfyUI decode failed")

        return result

    @staticmethod
    def _validate_segment_id(segment_id: str) -> str:
        value = str(segment_id or "").strip()
        if not value:
            raise ValueError("segment_id must not be empty")
        if value in (".", "..") or ".." in value or "/" in value or "\\" in value:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Wan2.2 5B InP ComfyUI workflow for one ExpHub segment.")

    p.add_argument("--platform-config", default=None, help="Path to ExpHub platform.yaml with services.comfyui config.")
    p.add_argument("--workflow-json", default=None, help="Path to exported ComfyUI API-format workflow JSON.")
    p.add_argument("--start-frame", required=True, help="Path to start frame image.")
    p.add_argument("--end-frame", required=True, help="Path to end frame image.")
    p.add_argument("--positive-prompt", required=True, help="Positive prompt.")
    p.add_argument("--negative-prompt", default="", help="Negative prompt.")
    p.add_argument("--segment-id", required=True, help="Unique segment id, e.g. seg_0001.")

    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--sampler-name", default="uni_pc")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--denoise", type=float, default=1.0)

    p.add_argument("--comfy-url", default=None)
    p.add_argument("--comfy-output-root", default=None, help="ComfyUI output directory, e.g. /home/hx/ComfyUI/output.")
    p.add_argument("--exp-output-dir", required=True, help="Where standardized decode outputs will be copied.")
    p.add_argument("--timeout-sec", type=int, default=None)
    p.add_argument("--poll-interval-sec", type=float, default=None)

    return p.parse_args()


def _load_comfyui_platform_config(platform_config: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    if not platform_config:
        return {}, None
    path = Path(platform_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"platform config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"platform config must be a YAML object: {path}")
    services = loaded.get("services") or {}
    if not isinstance(services, dict):
        raise ValueError(f"platform config services must be a YAML object: {path}")
    comfyui = services.get("comfyui") or {}
    if not isinstance(comfyui, dict):
        raise ValueError(f"platform config services.comfyui must be a YAML object: {path}")
    return comfyui, path


def _first_config_value(cli_value: Any, cfg: dict[str, Any], key: str, label: str, default: Any = None) -> Any:
    value = cli_value
    if value is None:
        value = cfg.get(key)
    if value is None:
        value = default
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {label}; pass CLI argument or set services.comfyui.{key} in platform.yaml")
    return value


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    comfyui_cfg, platform_path = _load_comfyui_platform_config(args.platform_config)
    return {
        "workflow_json": Path(
            _first_config_value(args.workflow_json, comfyui_cfg, "workflow_json", "--workflow-json")
        ).expanduser().resolve(),
        "comfy_url": str(
            _first_config_value(args.comfy_url, comfyui_cfg, "base_url", "--comfy-url")
        ).strip(),
        "comfy_output_root": Path(
            _first_config_value(args.comfy_output_root, comfyui_cfg, "output_root", "--comfy-output-root")
        ).expanduser().resolve(),
        "timeout_sec": int(_first_config_value(args.timeout_sec, comfyui_cfg, "timeout_sec", "--timeout-sec", 1800)),
        "poll_interval_sec": float(
            _first_config_value(args.poll_interval_sec, comfyui_cfg, "poll_interval_sec", "--poll-interval-sec", 2.0)
        ),
        "platform_config": platform_path,
    }


def run_comfyui_decode(
    req: DecodeRequest,
    *,
    comfy_url: str,
    workflow_json: str | Path,
    comfy_output_root: str | Path,
    exp_output_dir: str | Path,
    timeout_sec: int = 1800,
    poll_interval_sec: float = 2.0,
    platform_config: str | Path | None = None,
) -> DecodeResult:
    client = ComfyUIWanInpClient(
        comfy_url=comfy_url,
        workflow_json=Path(workflow_json),
        comfy_output_root=Path(comfy_output_root),
        exp_output_dir=Path(exp_output_dir),
        timeout_sec=int(timeout_sec),
        poll_interval_sec=float(poll_interval_sec),
        platform_config=Path(platform_config).resolve() if platform_config else None,
    )
    return client.run(req)


def main() -> int:
    args = parse_args()
    runtime_cfg = resolve_runtime_config(args)

    req = DecodeRequest(
        segment_id=args.segment_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        length=args.length,
        fps=args.fps,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        denoise=args.denoise,
    )

    client = ComfyUIWanInpClient(
        comfy_url=runtime_cfg["comfy_url"],
        workflow_json=runtime_cfg["workflow_json"],
        comfy_output_root=runtime_cfg["comfy_output_root"],
        exp_output_dir=Path(args.exp_output_dir),
        timeout_sec=runtime_cfg["timeout_sec"],
        poll_interval_sec=runtime_cfg["poll_interval_sec"],
        platform_config=runtime_cfg["platform_config"],
    )

    result = client.run(req)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
