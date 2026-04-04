from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, write_json_atomic
from exphub.common.logging import log_info, log_prog
from exphub.contracts import prompt as prompt_contract
from exphub.pipeline.prompt.base_prompt import build_base_prompt_payload
from exphub.pipeline.prompt.scene_encoding import build_state_scene_encoding
from exphub.pipeline.prompt.reporting import (
    build_prompt_report,
    cleanup_legacy_prompt_outputs,
    write_prompt_report,
)
from exphub.pipeline.prompt.runtime_plan import build_runtime_prompt_plan
from exphub.pipeline.prompt.state_manifest import build_state_prompt_manifest, load_segment_prompt_inputs


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_NATURAL_RE = re.compile(r"(\d+)")


def run(runtime):
    contract = prompt_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")
    ensure_file(runtime.paths.segment_manifest_path, "segment manifest")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(runtime.paths.prompt_dir)

    prompt_phase = runtime.prompt_phase_name()
    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "prompt" / "service.py").resolve()

    cmd = [
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--segment_manifest",
        str(runtime.paths.segment_manifest_path),
        "--fps",
        runtime.fps_arg,
        "--backend_python_phase",
        str(prompt_phase),
    ]
    prompt_model_ref = str(runtime.prompt_model_ref() or "").strip()
    if prompt_model_ref:
        cmd.extend(["--prompt_model_ref", prompt_model_ref])
    runtime.step_runner.run_env_python(cmd, phase_name=prompt_phase, log_name="prompt.log", cwd=runtime.exphub_root)

    ensure_file(contract.artifacts[prompt_contract.REPORT], "prompt report")
    ensure_file(contract.artifacts[prompt_contract.BASE_PROMPT], "prompt base prompt")
    ensure_file(contract.artifacts[prompt_contract.STATE_PROMPT_MANIFEST], "state prompt manifest")
    ensure_file(contract.artifacts[prompt_contract.RUNTIME_PROMPT_PLAN], "runtime prompt plan")
    return contract.artifacts[prompt_contract.REPORT]


def _natural_sort_key(path_text):
    # type: (str) -> List[object]
    text = str(path_text)
    name = Path(text).name
    parts = _NATURAL_RE.split(name)
    out = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def _list_images(image_dir):
    # type: (Path) -> List[str]
    image_root = Path(image_dir).resolve()
    items = []
    for fp in image_root.iterdir():
        if fp.is_file() and fp.suffix.lower() in _IMG_EXTS:
            items.append(str(fp.resolve()))
    items.sort(key=_natural_sort_key)
    return items


def _run_formal_mainline(args):
    # type: (argparse.Namespace) -> Path
    total_t0 = time.time()
    exp_dir = Path(args.exp_dir).resolve()
    prompt_dir = (exp_dir / "prompt").resolve()
    prompt_dir.mkdir(parents=True, exist_ok=True)

    segment_manifest_path = ensure_file(args.segment_manifest, "segment manifest")
    segment_inputs = load_segment_prompt_inputs(segment_manifest_path)
    frames_dir = ensure_dir(exp_dir / "segment" / "frames", "segment frames dir")
    frame_files = _list_images(frames_dir)
    if not frame_files:
        raise RuntimeError("no image files found in {}".format(frames_dir))

    base_prompt_payload = build_base_prompt_payload()
    state_prompt_manifest = build_state_prompt_manifest(
        segment_inputs=segment_inputs,
        prompt_dir=prompt_dir,
        base_prompt_path=prompt_dir / "base_prompt.json",
    )
    state_scene_encoding = build_state_scene_encoding(
        segment_inputs=segment_inputs,
        frames_dir=frames_dir,
        prompt_model_ref=str(args.prompt_model_ref or ""),
    )
    backend_meta = dict(state_scene_encoding.get("backend_meta") or {})
    if backend_meta:
        log_info("processor loaded in {:.2f}s".format(float(backend_meta.get("processor_load_sec", 0.0) or 0.0)))
        log_info("model weights loaded in {:.2f}s".format(float(backend_meta.get("model_load_sec", 0.0) or 0.0)))
    runtime_prompt_plan = build_runtime_prompt_plan(
        segment_inputs=segment_inputs,
        state_prompt_manifest=state_prompt_manifest,
        state_scene_encoding=state_scene_encoding,
        base_prompt_payload=base_prompt_payload,
        prompt_dir=prompt_dir,
    )

    write_json_atomic(prompt_dir / "base_prompt.json", base_prompt_payload, indent=2)
    write_json_atomic(prompt_dir / "state_prompt_manifest.json", state_prompt_manifest, indent=2)
    write_json_atomic(prompt_dir / "runtime_prompt_plan.json", runtime_prompt_plan, indent=2)

    prompt_report = build_prompt_report(
        prompt_dir=prompt_dir,
        base_prompt_payload=base_prompt_payload,
        state_prompt_manifest=state_prompt_manifest,
        runtime_prompt_plan=runtime_prompt_plan,
        frame_files_count=len(frame_files),
        total_sec=float(time.time() - total_t0),
        assembly_notes={
            "clip_profile_mode": "removed_from_mainline",
            "scene_prompt_mode": str(state_scene_encoding.get("scene_prompt_mode", "") or "state_v2t_primary_frame"),
            "scene_encoding_backend": str(state_scene_encoding.get("backend", "") or ""),
            "state_scene_segment_count": int(len(list(state_scene_encoding.get("state_segments") or []))),
            "state_control_mode": "minimal_state_control",
            "backend_python_phase": str(args.backend_python_phase or ""),
        },
    )
    prompt_report["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    prompt_report["frames_dir"] = str(frames_dir)
    report_path = write_prompt_report(prompt_dir, prompt_report)
    cleanup_legacy_prompt_outputs(prompt_dir)

    log_prog("prompt runtime plan assembled from invariant base + per-state scene encoding + minimal state control")
    log_info(
        "state control sources: segment_manifest={} state_segments={} deploy_schedule={}".format(
            str(((segment_inputs.get("source_files") or {}).get("segment_manifest", "")) or "<missing>"),
            str(((segment_inputs.get("source_files") or {}).get("state_segments", "")) or "<missing>"),
            str(((segment_inputs.get("source_files") or {}).get("deploy_schedule", "")) or "<missing>"),
        )
    )
    log_info("state control manifest generated: count={}".format(int(state_prompt_manifest.get("state_segment_count", 0) or 0)))
    log_info(
        "state scene encoding generated: count={} backend={}".format(
            int(len(list(state_scene_encoding.get("state_segments") or []))),
            str(state_scene_encoding.get("backend", "") or "<missing>"),
        )
    )
    log_info("runtime prompt plan generated: count={}".format(int(runtime_prompt_plan.get("deploy_segment_count", 0) or 0)))
    log_info("scene prompt mode: {}".format(str(runtime_prompt_plan.get("scene_prompt_mode", "") or "state_v2t_primary_frame")))
    log_info("wrote: {}".format(prompt_dir / "base_prompt.json"))
    log_info("wrote: {}".format(prompt_dir / "state_prompt_manifest.json"))
    log_info("wrote: {}".format(prompt_dir / "runtime_prompt_plan.json"))
    log_info("wrote: {}".format(report_path))
    return report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Formal ExpHub prompt mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--segment_manifest", required=True, help="formal segment_manifest.json path")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--prompt_model_ref", default="")
    parser.add_argument("--backend_python_phase", default="prompt_smol")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
