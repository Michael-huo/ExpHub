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


FORMAL_ENCODE_POLICY = "encode_mainline"


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


def _fmt_plain_seconds(value: object) -> str:
    parsed = _as_float_or_none(value)
    if parsed is None:
        return "n/a"
    return "{:.2f}".format(float(parsed))


def _fmt_plain_metric(value: object, digits: int = 3) -> str:
    parsed = _as_float_or_none(value)
    if parsed is None:
        return "n/a"
    return "{:.{digits}f}".format(float(parsed), digits=int(digits))


def _fmt_plain_ratio(value: object) -> str:
    parsed = _as_float_or_none(value)
    if parsed is None:
        return "n/a"
    return "{:.4f}".format(float(parsed))


def _fmt_plain_int(value: object) -> str:
    parsed = _as_int_or_none(value)
    if parsed is None:
        return "n/a"
    return str(int(parsed))


def _bytes_to_mib(value: object) -> Optional[float]:
    parsed = _as_float_or_none(value)
    if parsed is None:
        return None
    return float(parsed) / (1024.0 * 1024.0)


def _alignment_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "sim3":
        return "Sim3 (-a -s)"
    return text or "n/a"


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
    decode_profile = str(
        decode_report.get("decode_profile")
        or _get_nested(decode_report, ["backend_result", "decode_profile"])
        or ""
    ).strip()
    workflow_json = str(
        decode_report.get("workflow_json")
        or _get_nested(decode_report, ["backend_result", "workflow_json"])
        or ""
    ).strip()
    lora_enabled = decode_report.get("lora_enabled")
    if not isinstance(lora_enabled, bool):
        lora_enabled = _get_nested(decode_report, ["backend_result", "lora_enabled"])
    if not isinstance(lora_enabled, bool):
        lora_enabled = None
    lora_name = str(
        decode_report.get("lora_name")
        or _get_nested(decode_report, ["backend_result", "lora_name"])
        or ""
    ).strip()
    lora_strength_model = _as_float_or_none(decode_report.get("lora_strength_model"))
    if lora_strength_model is None:
        lora_strength_model = _pick_float(decode_report, ["backend_result", "lora_strength_model"])
    lora_strength_clip = _as_float_or_none(decode_report.get("lora_strength_clip"))
    if lora_strength_clip is None:
        lora_strength_clip = _pick_float(decode_report, ["backend_result", "lora_strength_clip"])

    return {
        "backend": backend,
        "decode_profile": decode_profile or None,
        "workflow_json": workflow_json or None,
        "lora_enabled": lora_enabled,
        "lora_name": lora_name or None,
        "lora_strength_model": lora_strength_model,
        "lora_strength_clip": lora_strength_clip,
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
    encode_result = _read_json_dict(exp_dir / "encode" / "encode_result.json")
    evo_summary = _read_json_dict(eval_dir / "evo_summary.json")
    decode_generation = _decode_generation_summary(exp_dir)

    compression_snapshot = _read_json_dict(eval_dir / "eval_compression_report.json")

    ori_bytes = _pick_int(compression_snapshot, ["orig_size_bytes"])
    comp_size = _pick_int(compression_snapshot, ["comp_size_bytes"])
    ratio_bytes = _pick_float(compression_snapshot, ["ratio"])
    raw_frame_count = _pick_int(compression_snapshot, ["raw_frame_count"])
    transmitted_frame_count = _pick_int(compression_snapshot, ["transmitted_frame_count"])
    if transmitted_frame_count is None:
        transmitted_frame_count = _pick_int(compression_snapshot, ["unit_boundary_count"])
    reduction_pct = _pick_float(compression_snapshot, ["reduction_pct"])
    if reduction_pct is None and ratio_bytes is not None:
        reduction_pct = (1.0 - float(ratio_bytes)) * 100.0

    phase_times = {}
    for phase_name in phase_names:
        phase_times[phase_name] = _as_float_or_none(step_times.get(phase_name))

    report = {
        "exp_dir": str(exp_dir),
        "total_time": float(total_time),
        "phase_times": phase_times,
        "encode_profile": encode_result.get("profile") if isinstance(encode_result.get("profile"), dict) else {},
        "decode_generation": decode_generation,
        "quality": {
            "metric_source": _get_nested(evo_summary, ["metric_source"]),
            "alignment": _get_nested(evo_summary, ["alignment"]),
            "gt_path_length_m": _pick_float(evo_summary, ["gt_path_length_m"]),
            "ori_ape_rmse": _pick_float(evo_summary, ["ori_ape_rmse"]),
            "gen_ape_rmse": _pick_float(evo_summary, ["gen_ape_rmse"]),
            "rmse_delta_gen_minus_ori": _pick_float(evo_summary, ["rmse_delta_gen_minus_ori"]),
            "ori_rpe_trans_rmse": _pick_float(evo_summary, ["ori_rpe_trans_rmse"]),
            "gen_rpe_trans_rmse": _pick_float(evo_summary, ["gen_rpe_trans_rmse"]),
            "rpe_delta_trans": _pick_float(evo_summary, ["rpe_delta_trans"]),
            "ori_rpe_rot_rmse_deg": _pick_float(evo_summary, ["ori_rpe_rot_rmse_deg"]),
            "gen_rpe_rot_rmse_deg": _pick_float(evo_summary, ["gen_rpe_rot_rmse_deg"]),
            "rpe_delta_rot_deg": _pick_float(evo_summary, ["rpe_delta_rot_deg"]),
            "eval_reliability": _get_nested(evo_summary, ["eval_reliability"]),
        },
        "compression": {
            "ratio": ratio_bytes,
            "reduction_pct": reduction_pct,
            "orig_size": ori_bytes,
            "comp_size": comp_size,
            "raw_frames": raw_frame_count,
            "transmitted_frames": transmitted_frame_count,
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
    encode_profile = dict(report.get("encode_profile") or {})
    encode_motion_profile = dict(encode_profile.get("motion") or {})
    decode_generation = dict(report.get("decode_generation") or {})
    quality = dict(report.get("quality") or {})
    compression = dict(report.get("compression") or {})

    sep = "=" * 70
    div = "-" * 70

    time_rows = [
        ("total_sec", _fmt_plain_seconds(total_time)),
        ("encode_sec", _fmt_plain_seconds(phase_times.get("encode"))),
        ("decode_sec", _fmt_plain_seconds(phase_times.get("decode"))),
        ("eval_sec", _fmt_plain_seconds(phase_times.get("eval"))),
        ("encode.motion_segment_sec", _fmt_plain_seconds(encode_profile.get("motion_segment_sec"))),
        ("encode.semantic_anchor_sec", _fmt_plain_seconds(encode_profile.get("semantic_anchor_sec"))),
        ("encode.result_writer_sec", _fmt_plain_seconds(encode_profile.get("result_writer_sec"))),
        ("motion.phase_correlation_sec", _fmt_plain_seconds(encode_motion_profile.get("phase_correlation_sec"))),
        ("motion.orb_tracking_sec", _fmt_plain_seconds(encode_motion_profile.get("orb_tracking_sec"))),
        ("motion.optical_flow_sec", _fmt_plain_seconds(encode_motion_profile.get("optical_flow_sec"))),
        ("decode.generate_sec", _fmt_plain_seconds(decode_generation.get("generate_sec"))),
        ("decode.avg_fps", _fmt_plain_metric(decode_generation.get("avg_fps"), digits=2)),
    ]

    quality_rows = [
        ("alignment", _alignment_label(quality.get("alignment"))),
        ("gt_path_length_m", _fmt_plain_metric(quality.get("gt_path_length_m"), digits=2)),
        ("ape.ori_rmse_m", _fmt_plain_metric(quality.get("ori_ape_rmse"), digits=3)),
        ("ape.gen_rmse_m", _fmt_plain_metric(quality.get("gen_ape_rmse"), digits=3)),
        ("ape.delta_gen_minus_ori_m", _fmt_plain_metric(quality.get("rmse_delta_gen_minus_ori"), digits=3)),
        ("rpe.ori_trans_rmse_m", _fmt_plain_metric(quality.get("ori_rpe_trans_rmse"), digits=3)),
        ("rpe.gen_trans_rmse_m", _fmt_plain_metric(quality.get("gen_rpe_trans_rmse"), digits=3)),
        ("rpe.delta_trans_m", _fmt_plain_metric(quality.get("rpe_delta_trans"), digits=3)),
        ("rpe.ori_rot_rmse_deg", _fmt_plain_metric(quality.get("ori_rpe_rot_rmse_deg"), digits=2)),
        ("rpe.gen_rot_rmse_deg", _fmt_plain_metric(quality.get("gen_rpe_rot_rmse_deg"), digits=2)),
        ("rpe.delta_rot_deg", _fmt_plain_metric(quality.get("rpe_delta_rot_deg"), digits=2)),
        ("eval_reliability", str(quality.get("eval_reliability") or "n/a")),
    ]

    compression_rows = [
        ("raw_size_mib", _fmt_plain_metric(_bytes_to_mib(compression.get("orig_size")), digits=2)),
        ("hvm_size_mib", _fmt_plain_metric(_bytes_to_mib(compression.get("comp_size")), digits=2)),
        ("transmission_ratio", _fmt_plain_ratio(compression.get("ratio"))),
        ("reduction_pct", _fmt_plain_metric(compression.get("reduction_pct"), digits=2)),
        ("raw_frames", _fmt_plain_int(compression.get("raw_frames"))),
        ("transmitted_frames", _fmt_plain_int(compression.get("transmitted_frames"))),
    ]

    _info(sep)
    _info("EXPERIMENT REPORT")
    _info(sep)
    _info("[Time]")
    _print_rows(time_rows)
    _info(div)
    _info("[Quality: evo]")
    _print_rows(quality_rows)
    _info(div)
    _info("[Compression]")
    _print_rows(compression_rows)
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
        choices=["prepare", "encode", "decode", "eval", "lora", "all"],
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
        help="encode policy for the current workflow; default is encode_mainline",
    )
    ap.add_argument("--train_clip_num_frames", type=int, default=73)
    ap.add_argument("--train_clip_stride", type=int, default=36)
    ap.add_argument("--seed", type=int, default=-1, dest="seed_base")
    ap.add_argument("--decode_profile", default="", help="ComfyUI decode workflow profile override")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--lora-profile", default="")
    ap.add_argument("--lora-gpus", default="")
    ap.add_argument("--lora-epochs", type=int, default=None)
    ap.add_argument("--lora-resume", default="none", choices=["none", "latest"])

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
    if args.lora_epochs is not None and int(args.lora_epochs) <= 0:
        _die("--lora-epochs must be > 0")
    args.mode = str(args.mode or "").strip().lower()
    args.step = str(args.step or "").strip().lower()
    if args.mode == "infer":
        if args.step == "lora":
            _die("infer mode does not support --step lora")
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
