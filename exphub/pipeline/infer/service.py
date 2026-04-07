from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, write_json_atomic
from exphub.common.logging import log_info, log_prog
from exphub.contracts import infer as infer_contract
from exphub.pipeline.infer.backends import create_backend
from exphub.pipeline.infer.reporting import (
    build_infer_report,
    cleanup_obsolete_infer_outputs,
    write_infer_report,
)
from exphub.pipeline.infer.request import InferRequest
from exphub.pipeline.infer.runtime_plan import (
    build_execution_plan,
    build_prompt_resolution,
    load_runtime_prompt_plan,
    merge_prompt_resolution_into_runs_plan,
    write_execution_plan,
)


def run(runtime):
    contract = infer_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")
    ensure_file(runtime.paths.prompt_runtime_plan_path, "runtime prompt plan")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(runtime.paths.infer_dir)

    infer_phase = runtime.infer_phase_name()
    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "infer" / "service.py").resolve()

    cmd = [
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--frames_dir",
        str(runtime.paths.segment_frames_dir),
        "--runtime_prompt_plan",
        str(runtime.paths.prompt_runtime_plan_path),
        "--videox_root",
        str(runtime.args.videox_root),
        "--gpus",
        str(runtime.args.gpus),
        "--fps",
        runtime.fps_arg,
        "--kf_gap",
        str(runtime.spec.kf_gap),
        "--seed_base",
        str(runtime.args.seed_base),
        "--infer_backend",
        str(runtime.args.infer_backend),
        "--infer_model_dir",
        str(runtime.args.infer_model_dir),
        "--backend_python_phase",
        str(infer_phase),
    ]
    if runtime.args.infer_extra:
        cmd.extend(["--infer_extra", str(runtime.args.infer_extra)])

    runtime.step_runner.run_env_python(cmd, phase_name=infer_phase, log_name="infer.log", cwd=runtime.exphub_root)

    ensure_dir(contract.artifacts[infer_contract.RUNS_DIR], "infer runs dir")
    ensure_file(contract.artifacts[infer_contract.RUNS_PLAN], "infer runs plan")
    ensure_file(contract.artifacts[infer_contract.REPORT], "infer report")
    return contract.artifacts[infer_contract.REPORT]


def _mean(values):
    # type: (list) -> float
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _list_frame_count(frames_dir):
    # type: (Path) -> int
    count = 0
    for item in Path(frames_dir).resolve().iterdir():
        if item.is_file():
            count += 1
    return int(count)


def _normalize_extra(extra_args):
    # type: (str) -> list
    text = str(extra_args or "").strip()
    if not text:
        return []
    extra = shlex.split(text)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _validate_execution_segments(frames_avail, execution_segments):
    # type: (int, list) -> None
    if frames_avail <= 0:
        raise RuntimeError("segment has no frames")
    for idx, item in enumerate(list(execution_segments or [])):
        start_idx = int(item.get("start_idx", 0) or 0)
        end_idx = int(item.get("end_idx", 0) or 0)
        if start_idx < 0 or end_idx < start_idx:
            raise RuntimeError("invalid execution segment range at index {}".format(idx))
        if end_idx >= int(frames_avail):
            raise RuntimeError(
                "execution segment {} exceeds frames_dir range: end_idx={} frames_avail={}".format(
                    idx,
                    end_idx,
                    int(frames_avail),
                )
            )


def _run_formal_mainline(args):
    # type: (argparse.Namespace) -> Path
    frames_dir = ensure_dir(args.frames_dir, "segment frames dir")
    exp_dir = Path(args.exp_dir).resolve()
    infer_dir = (exp_dir / "infer").resolve()
    infer_dir.mkdir(parents=True, exist_ok=True)

    frames_avail = _list_frame_count(frames_dir)
    runtime_prompt_plan = load_runtime_prompt_plan(str(args.runtime_prompt_plan))
    execution_plan = build_execution_plan(runtime_prompt_plan)
    execution_segments = list(execution_plan.get("segments") or [])
    if not execution_segments:
        raise RuntimeError("runtime prompt plan resolved to zero execution segments")
    _validate_execution_segments(frames_avail, execution_segments)

    prompt_resolution = build_prompt_resolution(runtime_prompt_plan, execution_segments, exp_dir=exp_dir)
    schedule_source = str(execution_plan.get("schedule_source", "") or "")
    execution_backend = str(execution_plan.get("execution_backend", "") or "")
    execution_plan_path = write_execution_plan(infer_dir, execution_plan)

    gpus = int(args.gpus)
    fps = int(float(args.fps))
    kf_gap = int(args.kf_gap)
    segments = int(len(execution_segments))
    used_start_idx = int(execution_segments[0]["start_idx"])
    used_end_idx = int(execution_segments[-1]["end_idx"])
    used_frames = int(used_end_idx - used_start_idx + 1)
    mean_deploy_gap = float(_mean([int(seg.get("deploy_gap", 0) or 0) for seg in execution_segments]))

    request = InferRequest(
        frames_dir=frames_dir,
        exp_dir=exp_dir,
        prompt_file_path=Path(str(args.runtime_prompt_plan)).resolve(),
        execution_plan_path=execution_plan_path,
        fps=int(fps),
        kf_gap=int(kf_gap),
        base_idx=int(used_start_idx),
        num_segments=int(segments),
        seed_base=int(args.seed_base),
        gpus=int(gpus),
        schedule_source=str(schedule_source),
        execution_backend=str(execution_backend),
        execution_segments=list(execution_segments),
        infer_extra=_normalize_extra(args.infer_extra),
    )

    backend = create_backend(
        backend_name=str(args.infer_backend),
        videox_root=str(args.videox_root),
        model_ref=str(args.infer_model_dir or ""),
        backend_python_phase=str(args.backend_python_phase or "infer"),
    )
    backend.load()
    backend_meta = dict(backend.meta() or {})

    log_prog(
        "infer config: backend={} segments={} fps={} gpus={}".format(
            backend_meta.get("infer_backend", args.infer_backend),
            segments,
            fps,
            gpus,
        )
    )
    log_info(
        "infer detail: schedule_source={} execution_backend={} used_frames={}".format(
            schedule_source or "runtime_prompt_plan",
            execution_backend or "runtime_prompt_plan",
            used_frames,
        )
    )

    t0 = time.time()
    try:
        backend_result = dict(backend.run(request) or {})
    finally:
        try:
            if execution_plan_path.is_file():
                execution_plan_path.unlink()
        except Exception:
            pass
    dt = float(time.time() - t0)

    runs_plan_path = ensure_file(infer_dir / "runs_plan.json", "runs_plan")
    plan_obj = json.loads(runs_plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan_obj, dict):
        raise RuntimeError("invalid runs_plan.json: {}".format(runs_plan_path))
    plan_obj = merge_prompt_resolution_into_runs_plan(plan_obj, prompt_resolution.get("segment_resolutions", []))
    plan_obj["state_prompt_enabled"] = bool(prompt_resolution.get("state_prompt_enabled", False))
    plan_obj["state_prompt_segment_count"] = int(prompt_resolution.get("state_prompt_segment_count", 0) or 0)
    plan_obj["matched_execution_segment_count"] = int(prompt_resolution.get("matched_execution_segment_count", 0) or 0)
    plan_obj["runtime_prompt_plan_version"] = int(prompt_resolution.get("runtime_prompt_plan_version", 1) or 1)
    plan_obj["runtime_prompt_plan_source"] = str(prompt_resolution.get("runtime_prompt_plan_source", "") or "")
    plan_obj["prompt_source_counts"] = dict(prompt_resolution.get("prompt_source_counts", {}) or {})
    plan_obj["state_label_counts"] = dict(prompt_resolution.get("state_label_counts", {}) or {})
    write_json_atomic(runs_plan_path, plan_obj, indent=2)

    runtime_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": int(gpus),
        "fps": int(fps),
        "kf_gap": int(kf_gap),
        "frames_avail": int(frames_avail),
        "segments": int(len(list(plan_obj.get("segments", []) or []))),
        "used_frames": int(used_frames),
        "used_start_idx": int(used_start_idx),
        "used_end_idx": int(used_end_idx),
        "schedule_source": str(schedule_source),
        "execution_backend": str(execution_backend),
        "mean_deploy_gap": float(mean_deploy_gap),
        "state_prompt_enabled": bool(prompt_resolution.get("state_prompt_enabled", False)),
        "state_prompt_segment_count": int(prompt_resolution.get("state_prompt_segment_count", 0) or 0),
        "matched_execution_segment_count": int(prompt_resolution.get("matched_execution_segment_count", 0) or 0),
        "runtime_prompt_plan_version": int(plan_obj.get("runtime_prompt_plan_version", 1) or 1),
        "runtime_prompt_plan_source": str(plan_obj.get("runtime_prompt_plan_source", "") or ""),
        "prompt_source_counts": dict(prompt_resolution.get("prompt_source_counts", {}) or {}),
        "state_label_counts": dict(prompt_resolution.get("state_label_counts", {}) or {}),
    }
    infer_report = build_infer_report(
        infer_dir=infer_dir,
        runs_plan_obj=plan_obj,
        prompt_resolution=prompt_resolution,
        backend_meta=backend_meta,
        backend_result=backend_result,
        runtime_summary=runtime_summary,
    )
    report_path = write_infer_report(infer_dir, infer_report)
    cleanup_obsolete_infer_outputs(infer_dir)

    log_info("infer finished: {:.2f}s".format(dt))
    log_info("report written: {}".format(report_path))
    return report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Formal ExpHub infer mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--frames_dir", required=True, help="segment frames dir")
    parser.add_argument("--runtime_prompt_plan", required=True, help="formal runtime_prompt_plan.json path")
    parser.add_argument("--videox_root", required=True, help="VideoX-Fun repo root")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--kf_gap", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=43)
    parser.add_argument("--infer_backend", default="wan_fun_5b_inp", choices=["wan_fun_a14b_inp", "wan_fun_5b_inp"])
    parser.add_argument("--infer_model_dir", default="")
    parser.add_argument("--backend_python_phase", default="infer")
    parser.add_argument("--infer_extra", default="")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
