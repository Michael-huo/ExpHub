from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .cleanup import normalize_keep_level
from .config import ConfigError, get_platform_config
from .common.logging import set_cli_log_level
from .meta import sanitize_token
from .runner import build_runtime, run_runtime
from .runner import RunError


FORMAL_ENCODE_POLICY = "encode_pass1"


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_STEP = "\033[1;36m"
_STEP_SEPARATOR = "=" * 70
_CLI_LOG_LEVEL = "info"


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _runtime_info(msg: str) -> None:
    if _CLI_LOG_LEVEL != "quiet":
        _info(msg)


def _run(msg: str) -> None:
    print(f"[RUN] {msg}")


def _step(msg: str) -> None:
    line = "{}[STEP] {}{}".format(_ANSI_STEP, msg, _ANSI_RESET)
    sep = "{}{}{}".format(_ANSI_BOLD, _STEP_SEPARATOR, _ANSI_RESET)
    lower_msg = msg.strip().lower()
    is_start = (" start " in lower_msg) or lower_msg.endswith(" start")
    is_done = (" done " in lower_msg) or lower_msg.endswith(" done")
    is_fail = (" fail " in lower_msg) or lower_msg.endswith(" fail")

    if is_start:
        print(sep)
        print(line)
        print(sep)
        return

    print(line)
    if is_done or is_fail:
        print(sep)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _die(msg: str) -> None:
    raise SystemExit(f"[ERR] {msg}")


def _ensure(p: Path, kind: str = "file") -> None:
    if kind == "file":
        if not p.is_file():
            _die(f"file not found: {p}")
    else:
        if not p.is_dir():
            _die(f"dir not found: {p}")


def _print_experiment_summary(
    mode: str,
    step: str,
    dataset: str,
    sequence: str,
    tag: str,
    fps_text: str,
    dur_text: str,
    start_text: str,
    gpus: int,
    keep_level: str,
    exp_dir: Path,
) -> None:
    sep = "=" * 70
    rows = [
        ("Mode", mode),
        ("Step", step),
        ("Dataset", dataset),
        ("Sequence", sequence),
        ("Tag", tag),
        ("FPS", fps_text),
        ("Duration", dur_text),
        ("Start", start_text),
        ("GPUs", str(gpus)),
        ("Keep Level", keep_level),
        ("Artifact Root", str(exp_dir)),
    ]
    key_w = max(len(k) for k, _ in rows)
    _info(sep)
    _info("EXPERIMENT SUMMARY")
    _info(sep)
    for key, val in rows:
        _info("{:<{w}} : {}".format(key, val, w=key_w))
    _info(sep)


def _read_json_dict(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def _get_nested(obj: Dict[str, object], path: List[str]) -> object:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float_or_none(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_int_or_none(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _fmt_seconds(value: float, digits: int = 2, suffix: str = "s") -> str:
    return "{:.{digits}f}{}".format(float(value), suffix, digits=digits)


def _fmt_phase_seconds(value: float, total_time: float) -> str:
    pct = (float(value) / float(total_time) * 100.0) if total_time > 0 else 0.0
    return "{:8.2f}s ({:5.1f}%)".format(float(value), pct)


def _fmt_metric(value: float, unit: str = "", digits: int = 4) -> str:
    text = "{:.{digits}f}".format(float(value), digits=digits).rstrip("0").rstrip(".")
    if not text:
        text = "0"
    if unit:
        return "{} {}".format(text, unit)
    return text


def _fmt_ratio(value: float) -> str:
    return "{:.4f}x".format(float(value))


def _fmt_reduction(value: float) -> str:
    return "{:.2f}%".format(float(value) * 100.0)


def _fmt_count(value: int) -> str:
    return str(int(value))


def _fmt_bytes(value: int) -> str:
    size = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return "{} {}".format(int(size), units[unit_idx])
    return "{:.2f} {}".format(size, units[unit_idx])


def _pick_float(obj: Dict[str, object], path: List[str]) -> Optional[float]:
    return _as_float_or_none(_get_nested(obj, path))


def _pick_int(obj: Dict[str, object], path: List[str]) -> Optional[int]:
    return _as_int_or_none(_get_nested(obj, path))


def _decode_generation_summary(exp_dir: Path) -> Dict[str, object]:
    decode_dir = exp_dir / "decode"
    decode_report = _read_json_dict(decode_dir / "decode_report.json")
    merge_report = _read_json_dict(decode_dir / "decode_merge_report.json")
    units = list(decode_report.get("units") or [])

    backend = str(
        decode_report.get("backend_name")
        or _get_nested(decode_report, ["backend_meta", "backend"])
        or _get_nested(decode_report, ["backend_result", "backend"])
        or ""
    ).strip()
    unit_count = _as_int_or_none(decode_report.get("num_tasks"))
    if unit_count is None and units:
        unit_count = len(units)

    merged_frames = (
        _pick_int(merge_report, ["summary", "merged_frame_count"])
        or _pick_int(merge_report, ["outputs", "frame_count"])
    )
    generated_frames = _pick_int(merge_report, ["summary", "execution_frame_count"])
    if generated_frames is None and units:
        total = 0
        for item in units:
            if isinstance(item, dict):
                count = _as_int_or_none(item.get("num_frames"))
                if count is not None:
                    total += int(count)
        generated_frames = total if total > 0 else None
    if generated_frames is None:
        generated_frames = merged_frames

    generate_sec = (
        _as_float_or_none(decode_report.get("wall_generate_sec"))
        or _as_float_or_none(decode_report.get("total_runtime_sec"))
        or _pick_float(decode_report, ["backend_result", "wall_generate_sec"])
        or _pick_float(decode_report, ["backend_result", "total_runtime_sec"])
    )
    avg_fps = None
    if generated_frames is not None and generate_sec is not None and float(generate_sec) > 0:
        avg_fps = float(generated_frames) / float(generate_sec)
    parallel = decode_report.get("parallel")
    if not isinstance(parallel, bool):
        parallel = _get_nested(decode_report, ["backend_result", "parallel"])
    if not isinstance(parallel, bool):
        parallel = None
    schedule = str(decode_report.get("schedule") or _get_nested(decode_report, ["backend_result", "schedule"]) or "").strip()
    instance_count = (
        _as_int_or_none(decode_report.get("instance_count"))
        or _pick_int(decode_report, ["backend_result", "instance_count"])
    )

    return {
        "backend": backend,
        "units": unit_count,
        "frames": merged_frames if merged_frames is not None else generated_frames,
        "generate_sec": generate_sec,
        "parallel": parallel,
        "instances": instance_count,
        "schedule": schedule or None,
        "sum_unit_sec": (
            _as_float_or_none(decode_report.get("sum_unit_generate_sec"))
            or _pick_float(decode_report, ["backend_result", "sum_unit_generate_sec"])
        ),
        "speedup": (
            _as_float_or_none(decode_report.get("parallel_speedup"))
            or _pick_float(decode_report, ["backend_result", "parallel_speedup"])
        ),
        "avg_fps": avg_fps,
    }


def _load_experiment_report(exp_dir: Path, step_times: Dict[str, float]) -> Dict[str, object]:
    phase_names = ["encode", "decode", "eval"]
    total_time = sum(float(x) for x in step_times.values())

    eval_dir = exp_dir / "eval"
    traj_metrics = _read_json_dict(eval_dir / "eval_traj_report.json")
    decode_generation = _decode_generation_summary(exp_dir)

    compression_snapshot = _read_json_dict(eval_dir / "eval_compression_report.json")

    ori_bytes = _pick_int(compression_snapshot, ["orig_size_bytes"])
    comp_size = _pick_int(compression_snapshot, ["comp_size_bytes"])
    ratio_bytes = _pick_float(compression_snapshot, ["ratio"])
    unit_boundary_count = _pick_int(compression_snapshot, ["unit_boundary_count"])

    reduction_ratio = None  # type: Optional[float]
    if ratio_bytes is not None:
        reduction_ratio = 1.0 - float(ratio_bytes)
    else:
        reduction_pct = _pick_float(compression_snapshot, ["reduction_pct"])
        if reduction_pct is not None:
            reduction_ratio = float(reduction_pct) / 100.0

    phase_times = {}
    for phase_name in phase_names:
        phase_times[phase_name] = _as_float_or_none(step_times.get(phase_name))

    report = {
        "exp_dir": str(exp_dir),
        "total_time": float(total_time),
        "phase_times": phase_times,
        "decode_generation": decode_generation,
        "quality": {
            "ape_rmse": _pick_float(traj_metrics, ["ape_trans", "rmse"]),
            "rpe_trans": _pick_float(traj_metrics, ["rpe_trans", "rmse"]),
            "rpe_rot": _pick_float(traj_metrics, ["rpe_rot", "rmse"]),
            "ori_len": _pick_float(traj_metrics, ["ori_path_length_m"]),
            "gen_len": _pick_float(traj_metrics, ["gen_path_length_m"]),
            "poses": _pick_int(traj_metrics, ["matched_pose_count"]),
        },
        "compression": {
            "ratio": ratio_bytes,
            "reduction": reduction_ratio,
            "orig_size": ori_bytes,
            "comp_size": comp_size,
            "unit_boundaries": unit_boundary_count,
        },
    }
    return report


def _print_rows(rows: List[tuple]) -> None:
    if not rows:
        return
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        _info("{:<{w}} : {}".format(str(key), value, w=width))


def _add_row(rows: List[tuple], key: str, value: object, formatter) -> None:
    if value is None:
        return
    rows.append((key, formatter(value)))


def _print_experiment_report(exp_dir: Path, step_times: Dict[str, float]) -> None:
    report = _load_experiment_report(exp_dir, step_times)
    total_time = float(report.get("total_time") or 0.0)
    phase_times = dict(report.get("phase_times") or {})
    decode_generation = dict(report.get("decode_generation") or {})
    quality = dict(report.get("quality") or {})
    compression = dict(report.get("compression") or {})

    sep = "=" * 70
    div = "-" * 70

    time_rows = []
    for phase_name in ["encode", "decode", "eval"]:
        value = _as_float_or_none(phase_times.get(phase_name))
        if value is not None:
            time_rows.append((phase_name, _fmt_phase_seconds(value, total_time)))

    detail_rows = []
    backend = str(decode_generation.get("backend") or "").strip()
    if backend:
        detail_rows.append(("decode.backend", backend))
    if isinstance(decode_generation.get("parallel"), bool):
        detail_rows.append(("decode.parallel", "true" if bool(decode_generation.get("parallel")) else "false"))
    instances = _as_int_or_none(decode_generation.get("instances"))
    if instances is not None and int(instances) > 0:
        detail_rows.append(("decode.instances", _fmt_count(instances)))
    schedule = str(decode_generation.get("schedule") or "").strip()
    if schedule:
        detail_rows.append(("decode.schedule", schedule))
    _add_row(detail_rows, "decode.units", _as_int_or_none(decode_generation.get("units")), _fmt_count)
    _add_row(detail_rows, "decode.frames", _as_int_or_none(decode_generation.get("frames")), _fmt_count)
    _add_row(detail_rows, "decode.generate_sec", _as_float_or_none(decode_generation.get("generate_sec")), _fmt_seconds)
    _add_row(detail_rows, "decode.sum_unit_sec", _as_float_or_none(decode_generation.get("sum_unit_sec")), _fmt_seconds)
    _add_row(detail_rows, "decode.speedup", _as_float_or_none(decode_generation.get("speedup")), lambda v: "{:.3f}x".format(float(v)))
    _add_row(detail_rows, "decode.avg_fps", _as_float_or_none(decode_generation.get("avg_fps")), lambda v: "{:.3f} fps".format(float(v)))
    if total_time > 0:
        detail_rows.append(("total", _fmt_seconds(total_time)))

    quality_rows = []
    _add_row(quality_rows, "ape_rmse", _as_float_or_none(quality.get("ape_rmse")), lambda v: _fmt_metric(v, unit="m"))
    _add_row(quality_rows, "rpe_trans", _as_float_or_none(quality.get("rpe_trans")), lambda v: _fmt_metric(v, unit="m"))
    _add_row(quality_rows, "rpe_rot", _as_float_or_none(quality.get("rpe_rot")), lambda v: _fmt_metric(v, unit="deg"))
    _add_row(quality_rows, "ori_len", _as_float_or_none(quality.get("ori_len")), lambda v: _fmt_metric(v, unit="m"))
    _add_row(quality_rows, "gen_len", _as_float_or_none(quality.get("gen_len")), lambda v: _fmt_metric(v, unit="m"))
    _add_row(quality_rows, "poses", _as_int_or_none(quality.get("poses")), _fmt_count)

    compression_rows = []
    _add_row(compression_rows, "ratio", _as_float_or_none(compression.get("ratio")), _fmt_ratio)
    _add_row(compression_rows, "reduction", _as_float_or_none(compression.get("reduction")), _fmt_reduction)
    _add_row(compression_rows, "orig_size", _as_int_or_none(compression.get("orig_size")), _fmt_bytes)
    _add_row(compression_rows, "comp_size", _as_int_or_none(compression.get("comp_size")), _fmt_bytes)
    _add_row(compression_rows, "unit_boundaries", _as_int_or_none(compression.get("unit_boundaries")), _fmt_count)

    _info(sep)
    _info("EXPERIMENT REPORT")
    _info(sep)
    _info("[Time]")
    _print_rows(time_rows)
    if detail_rows:
        _info(div)
        _print_rows(detail_rows)
    if quality_rows:
        _info(div)
        _info("[Quality]")
        _print_rows(quality_rows)
    if compression_rows:
        _info(div)
        _info("[Compression]")
        _print_rows(compression_rows)
    _info(div)
    _print_rows([("exp_dir", str(report.get("exp_dir") or exp_dir))])
    _info(sep)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exphub.cli", add_help=True)

    try:
        _cfg = get_platform_config()
        _def_droid_repo = _cfg.get("repos", {}).get("droid_slam", "")
        _def_droid_w = _cfg.get("models", {}).get("droid", {}).get("path", "")
    except Exception:
        _def_droid_repo = ""
        _def_droid_w = ""

    ap.add_argument(
        "--mode",
        required=True,
        choices=["infer", "train"],
        help="execution mode",
    )
    ap.add_argument(
        "--step",
        required=True,
        choices=["prepare", "encode", "decode", "eval", "all"],
        help="pipeline step",
    )
    ap.add_argument("--exphub", default=os.environ.get("EXPHUB", ""), help="ExpHub root (default: $EXPHUB or cwd)")

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", default="")
    ap.add_argument("--tag", required=True)

    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--dur", type=str, default=None)
    ap.add_argument("--start", type=str, default=None)

    ap.add_argument(
        "--segment_policy",
        default=FORMAL_ENCODE_POLICY,
        help="encode policy for the current workflow; default is encode_pass1",
    )
    ap.add_argument("--train_clip_num_frames", type=int, default=73)
    ap.add_argument("--train_clip_stride", type=int, default=36)
    ap.add_argument("--seed", type=int, default=-1, dest="seed_base")
    ap.add_argument("--gpus", type=int, default=2)

    ap.add_argument("--infer_extra", default="", help="extra args passed through the formal infer service (quoted string)")

    ap.add_argument("--exp_root", default="", help="override artifact root (default: <exphub>/artifacts/infer/<dataset>/<sequence>)")

    ap.add_argument(
        "--keep_level",
        default="max",
        choices=["max", "min"],
        help="artifact retention level: max (keep all) or min (batch-optimized cleanup)",
    )
    ap.add_argument("--log_level", default="info", choices=["info", "quiet"], help="child process terminal verbosity")

    ap.add_argument(
        "--no_auto_conda",
        action="store_false",
        dest="auto_conda",
        default=True,
        help="disable automatic conda activation and use current shell env for child commands",
    )
    ap.add_argument("--droid_repo", default=_def_droid_repo)
    ap.add_argument("--droid_weights", default=_def_droid_w)

    ap.add_argument(
        "--prompt-python",
        default=os.environ.get("EXPHUB_PROMPT_PYTHON", "/home/hx/anaconda3/envs/blip2/bin/python"),
        help="python executable for the BLIP-2 prompt caption backend",
    )
    ap.add_argument(
        "--prompt-backend",
        default="blip2",
        choices=["blip2"],
        help="prompt semantic caption backend; pass1 supports only blip2",
    )
    ap.add_argument(
        "--prompt-blip2-model",
        default=os.environ.get("EXPHUB_BLIP2_MODEL", "Salesforce/blip2-opt-2.7b"),
        help="BLIP-2 Hugging Face repo id or local model path",
    )

    ap.add_argument("--ros_setup", default=os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))

    # Eval-stage SLAM sequence selection.
    # Default is "both" so eval can compare ori/gen unless explicitly overridden.
    ap.add_argument("--droid_seq", default="both", choices=["auto", "ori", "gen", "both"])
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no_viz", action="store_true")

    args = ap.parse_args(argv)
    if int(args.seed_base) <= 0 and int(args.seed_base) != -1:
        _die("--seed must be a positive integer or -1")
    args.datasets_cfg = ""
    args.kf_gap = 0
    args.start_idx = -1
    args.base_idx = 0
    global _CLI_LOG_LEVEL
    _CLI_LOG_LEVEL = str(args.log_level or "info").strip().lower()
    set_cli_log_level(_CLI_LOG_LEVEL)
    args.keep_level = normalize_keep_level(args.keep_level)
    args.segment_policy = str(args.segment_policy or FORMAL_ENCODE_POLICY).strip() or FORMAL_ENCODE_POLICY
    if int(args.train_clip_num_frames) <= 0:
        _die("--train_clip_num_frames must be > 0")
    if int(args.train_clip_stride) <= 0:
        _die("--train_clip_stride must be > 0")
    prompt_python = str(args.prompt_python or "").strip()
    if not prompt_python:
        _die("--prompt-python is required for BLIP-2 prompt captions")
    if os.path.isabs(prompt_python) or os.sep in prompt_python:
        if not (os.path.isfile(os.path.expanduser(prompt_python)) and os.access(os.path.expanduser(prompt_python), os.X_OK)):
            _die(
                "--prompt-python not found or not executable: {}. Create the blip2 conda environment or pass --prompt-python.".format(
                    prompt_python
                )
            )
    if str(args.prompt_backend or "").strip().lower() != "blip2":
        _die("--prompt-backend must be blip2")

    args.mode = str(args.mode or "").strip().lower()
    args.step = str(args.step or "").strip().lower()
    if args.mode == "infer":
        if not args.sequence:
            _die("--sequence is required for --mode infer")
        if args.start is None or str(args.start).strip() == "":
            _die("--start is required for --mode infer")
        if args.dur is None or str(args.dur).strip() == "":
            _die("--dur is required for --mode infer")
    elif args.mode == "train":
        if args.step in ("decode", "eval"):
            _die("train mode does not support --step {}".format(args.step))
        if args.start is not None and str(args.start).strip() != "":
            _die("train mode does not accept --start")
        if args.dur is not None and str(args.dur).strip() != "":
            _die("train mode does not accept --dur")
        args.start = ""
        args.dur = ""
    else:
        _die("unsupported mode: {}".format(args.mode))

    args.dataset = sanitize_token(args.dataset)
    args.sequence = sanitize_token(args.sequence) if str(args.sequence or "").strip() else ""
    args.tag = sanitize_token(args.tag)
    if not args.dataset or not args.tag:
        _die("dataset/tag becomes empty after sanitize")
    if args.mode == "infer" and not args.sequence:
        _die("dataset/sequence/tag becomes empty after sanitize")

    runtime = build_runtime(args)

    if _CLI_LOG_LEVEL != "quiet":
        _print_experiment_summary(
            mode=args.mode,
            step=args.step,
            dataset=runtime.spec.dataset,
            sequence=runtime.spec.sequence or "<all>",
            tag=runtime.spec.tag,
            fps_text=runtime.fps_arg,
            dur_text=str(args.dur),
            start_text=str(args.start),
            gpus=int(args.gpus),
            keep_level=str(args.keep_level),
            exp_dir=runtime.paths.exp_dir,
        )

    try:
        result = run_runtime(runtime)
        step_norm = str(args.step or "").strip().lower()
        if args.mode == "infer" and step_norm != "prepare":
            _print_experiment_report(result.exp_dir, result.step_times)
    except (ConfigError, RunError, RuntimeError) as e:
        _die(str(e))


if __name__ == "__main__":
    main()
