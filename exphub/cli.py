from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .cleanup import normalize_keep_level
from .common.config import ConfigError, get_platform_config
from .common.logging import set_cli_log_level
from .common.types import sanitize_token
from .contracts.segment import FORMAL_SEGMENT_POLICY, require_formal_segment_policy
from .pipeline.orchestrator import build_runtime, run_runtime
from .runner import RunError


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


def _debug_info(msg: str) -> None:
    if _CLI_LOG_LEVEL == "debug":
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
    dataset: str,
    sequence: str,
    tag: str,
    w: int,
    h: int,
    fps_text: str,
    dur_text: str,
    gpus: int,
    keep_level: str,
    exp_dir: Path,
) -> None:
    sep = "=" * 70
    rows = [
        ("Mode", mode),
        ("Dataset", dataset),
        ("Sequence", sequence),
        ("Tag", tag),
        ("Resolution", "{}x{}".format(w, h)),
        ("FPS", fps_text),
        ("Duration", dur_text),
        ("GPUs", str(gpus)),
        ("Keep Level", keep_level),
        ("Exp Dir", str(exp_dir)),
    ]
    key_w = max(len(k) for k, _ in rows)
    _info(sep)
    _info("EXPERIMENT SUMMARY")
    _info(sep)
    for key, val in rows:
        _info("{:<{w}} : {}".format(key, val, w=key_w))
    _info(sep)


def _strip_info_prefix(line: str) -> str:
    s = line.strip()
    if s.startswith("[INFO] "):
        return s[len("[INFO] "):].strip()
    return s


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


def _fmt_seconds(value: Optional[float], digits: int = 2, suffix: str = "s") -> str:
    if value is None:
        return "unavailable"
    return "{:.{digits}f}{}".format(float(value), suffix, digits=digits)


def _fmt_phase_seconds(value: Optional[float], total_time: float) -> str:
    if value is None:
        return "unavailable"
    pct = (float(value) / float(total_time) * 100.0) if total_time > 0 else 0.0
    return "{:8.2f}s ({:5.1f}%)".format(float(value), pct)


def _fmt_metric(value: Optional[float], unit: str = "", digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    text = "{:.{digits}f}".format(float(value), digits=digits).rstrip("0").rstrip(".")
    if not text:
        text = "0"
    if unit:
        return "{} {}".format(text, unit)
    return text


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    return "{:.4f}x".format(float(value))


def _fmt_reduction(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    return "{:.2f}%".format(float(value) * 100.0)


def _fmt_count(value: Optional[int]) -> str:
    if value is None:
        return "unavailable"
    return str(int(value))


def _fmt_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unavailable"
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


def _pick_first_int(obj: Dict[str, object], paths: List[List[str]]) -> Optional[int]:
    for path in paths:
        value = _pick_int(obj, path)
        if value is not None:
            return value
    return None


def _pick_first_float(obj: Dict[str, object], paths: List[List[str]]) -> Optional[float]:
    for path in paths:
        value = _pick_float(obj, path)
        if value is not None:
            return value
    return None


def _parse_infer_log_details(log_path: Path) -> Dict[str, object]:
    out = {}  # type: Dict[str, object]
    if not log_path.is_file():
        return out

    init_re = re.compile(
        r"Initialization completed in ([0-9.]+)s \(Loading: ([0-9.]+)s, Quantization: ([0-9.]+)s\)"
    )
    done_re = re.compile(
        r"done: segments=(\d+) frames=(\d+) init=([0-9.]+)s infer_sum=([0-9.]+)s "
        r"avg_infer=([0-9.]+)s avg_frame=([0-9.]+)s total=([0-9.]+)s"
    )

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for raw in lines:
        line = _strip_info_prefix(raw)
        init_match = init_re.search(line)
        if init_match:
            out["infer.init"] = float(init_match.group(1))
            out["infer.load"] = float(init_match.group(2))
            out["infer.quant"] = float(init_match.group(3))
        done_match = done_re.search(line)
        if done_match:
            out["infer.segments"] = int(done_match.group(1))
            out["infer.frames"] = int(done_match.group(2))
            out["infer.init"] = float(done_match.group(3))
            out["infer.run"] = float(done_match.group(4))
            out["infer.avg"] = float(done_match.group(5))
            out["infer.avg_fr"] = float(done_match.group(6))
            out["infer.total"] = float(done_match.group(7))
    return out


def _load_experiment_report(exp_dir: Path, step_times: Dict[str, float]) -> Dict[str, object]:
    phase_names = ["encode", "decode", "eval"]
    total_time = sum(float(x) for x in step_times.values())

    eval_report = _read_json_dict(exp_dir / "eval" / "report.json")
    traj_metrics = dict(eval_report.get("traj_eval") or {}) if isinstance(eval_report.get("traj_eval"), dict) else {}
    if not traj_metrics:
        traj_metrics = _read_json_dict(exp_dir / "eval" / "metrics" / "traj_eval.json")
    infer_details = _parse_infer_log_details(exp_dir / "logs" / "infer.log")

    compression_obj = {}
    if isinstance(eval_report.get("compression"), dict):
        compression_obj = dict(eval_report.get("compression") or {})
    compression_snapshot = _read_json_dict(exp_dir / "eval" / "compression.json")

    ori_bytes = _pick_first_int(
        compression_obj,
        [["ori_bytes"]],
    )
    if ori_bytes is None:
        ori_bytes = _pick_first_int(compression_snapshot, [["orig_size"]])

    keyframes_bytes = _pick_first_int(
        compression_obj,
        [["keyframes_bytes"]],
    )

    prompt_bytes = _pick_first_int(
        compression_obj,
        [["prompt_bytes"]],
    )

    comp_size = None  # type: Optional[int]
    if keyframes_bytes is not None and prompt_bytes is not None:
        comp_size = int(keyframes_bytes + prompt_bytes)
    else:
        comp_size = _pick_first_int(compression_snapshot, [["comp_size"]])

    ratio_bytes = _pick_first_float(
        compression_obj,
        [["ratio_bytes"]],
    )
    if ratio_bytes is None:
        ratio_bytes = _pick_first_float(compression_snapshot, [["ratio"]])

    keyframes_frames = _pick_first_int(
        compression_obj,
        [["keyframes_frames"]],
    )
    if keyframes_frames is None:
        keyframes_frames = _pick_first_int(compression_snapshot, [["keyframes"]])

    reduction_ratio = None  # type: Optional[float]
    if ratio_bytes is not None:
        reduction_ratio = 1.0 - float(ratio_bytes)
    else:
        reduction_ratio = _pick_first_float(compression_snapshot, [["reduction"]])

    phase_times = {}
    for phase_name in phase_names:
        phase_times[phase_name] = _as_float_or_none(step_times.get(phase_name))

    report = {
        "exp_dir": str(exp_dir),
        "total_time": float(total_time),
        "phase_times": phase_times,
        "infer_details": infer_details,
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
            "keyframes": keyframes_frames,
        },
    }
    return report


def _print_rows(rows: List[tuple]) -> None:
    if not rows:
        return
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        _info("{:<{w}} : {}".format(str(key), value, w=width))


def _print_experiment_report(exp_dir: Path, step_times: Dict[str, float]) -> None:
    report = _load_experiment_report(exp_dir, step_times)
    total_time = float(report.get("total_time") or 0.0)
    phase_times = dict(report.get("phase_times") or {})
    infer_details = dict(report.get("infer_details") or {})
    quality = dict(report.get("quality") or {})
    compression = dict(report.get("compression") or {})

    sep = "=" * 70
    div = "-" * 70

    time_rows = []
    for phase_name in ["encode", "decode", "eval"]:
        time_rows.append((phase_name, _fmt_phase_seconds(_as_float_or_none(phase_times.get(phase_name)), total_time)))

    detail_rows = [
        ("image_gen.load", _fmt_seconds(_as_float_or_none(infer_details.get("infer.load")))),
        ("image_gen.quant", _fmt_seconds(_as_float_or_none(infer_details.get("infer.quant")))),
        ("image_gen.run", _fmt_seconds(_as_float_or_none(infer_details.get("infer.run")))),
        ("image_gen.avg_fr", _fmt_seconds(_as_float_or_none(infer_details.get("infer.avg_fr")), digits=3, suffix="s/frame")),
        ("total", _fmt_seconds(total_time)),
    ]

    quality_rows = [
        ("ape_rmse", _fmt_metric(_as_float_or_none(quality.get("ape_rmse")), unit="m")),
        ("rpe_trans", _fmt_metric(_as_float_or_none(quality.get("rpe_trans")), unit="m")),
        ("rpe_rot", _fmt_metric(_as_float_or_none(quality.get("rpe_rot")), unit="deg")),
        ("ori_len", _fmt_metric(_as_float_or_none(quality.get("ori_len")), unit="m")),
        ("gen_len", _fmt_metric(_as_float_or_none(quality.get("gen_len")), unit="m")),
        ("poses", _fmt_count(_as_int_or_none(quality.get("poses")))),
    ]

    compression_rows = [
        ("ratio", _fmt_ratio(_as_float_or_none(compression.get("ratio")))),
        ("reduction", _fmt_reduction(_as_float_or_none(compression.get("reduction")))),
        ("orig_size", _fmt_bytes(_as_int_or_none(compression.get("orig_size")))),
        ("comp_size", _fmt_bytes(_as_int_or_none(compression.get("comp_size")))),
        ("keyframes", _fmt_count(_as_int_or_none(compression.get("keyframes")))),
    ]

    _info(sep)
    _info("EXPERIMENT REPORT")
    _info(sep)
    _info("[Time]")
    _print_rows(time_rows)
    _info(div)
    _print_rows(detail_rows)
    _info(div)
    _info("[Quality]")
    _print_rows(quality_rows)
    _info(div)
    _info("[Compression]")
    _print_rows(compression_rows)
    _info(div)
    _print_rows([("exp_dir", str(report.get("exp_dir") or exp_dir))])
    _info(sep)


def _load_export_report(export_root: Path, step_times: Dict[str, float]) -> Dict[str, object]:
    dataset_report = _read_json_dict(export_root / "dataset_report.json")
    summary = dict(dataset_report.get("summary") or {})
    outputs = dict(dataset_report.get("outputs") or {})
    return {
        "export_root": str(export_root),
        "export_source": str(dataset_report.get("export_source", "") or ""),
        "total_time": float(sum(float(x) for x in step_times.values())),
        "phase_times": {"export": _as_float_or_none(step_times.get("export"))},
        "summary": summary,
        "outputs": outputs,
    }


def _print_export_report(export_root: Path, step_times: Dict[str, float]) -> None:
    payload = _load_export_report(export_root, step_times)
    total_time = float(payload.get("total_time") or 0.0)
    phase_times = dict(payload.get("phase_times") or {})
    summary = dict(payload.get("summary") or {})
    outputs = dict(payload.get("outputs") or {})
    export_source = str(payload.get("export_source", "") or "unknown")

    sep = "=" * 70
    div = "-" * 70
    _info(sep)
    _info("EXPORT REPORT")
    _info(sep)
    _print_rows(
        [
            ("source", export_source),
            ("export", _fmt_phase_seconds(_as_float_or_none(phase_times.get("export")), total_time)),
            ("total", _fmt_seconds(total_time)),
        ]
    )
    _info(div)
    _print_rows(
        [
            ("bags", _fmt_count(_as_int_or_none(summary.get("bag_count")))),
            ("clips", _fmt_count(_as_int_or_none(summary.get("exported_clip_count")))),
            ("skipped", _fmt_count(_as_int_or_none(summary.get("skipped_clip_count")))),
            ("spans", _fmt_count(_as_int_or_none(summary.get("prompt_span_count")))),
            ("units", _fmt_count(_as_int_or_none(summary.get("total_units_consumed")))),
            ("short_spans", _fmt_count(_as_int_or_none(summary.get("skipped_short_span_count")))),
            ("train", _fmt_count(_as_int_or_none(_get_nested(summary, ["split_counts", "train"])))),
            ("val", _fmt_count(_as_int_or_none(_get_nested(summary, ["split_counts", "val"])))),
            ("test", _fmt_count(_as_int_or_none(_get_nested(summary, ["split_counts", "test"])))),
        ]
    )
    _info(div)
    _print_rows(
        [
            ("reuse_ratio", str(summary.get("prompt_reuse_ratio", "unavailable"))),
            ("clips/span", str(summary.get("mean_clips_per_span", "unavailable"))),
            ("shared_units", _fmt_count(_as_int_or_none(summary.get("shared_prompt_unit_count")))),
        ]
    )
    _info(div)
    _print_rows(
        [
            ("clips_dir", str(outputs.get("clips_dir") or "unavailable")),
            ("metadata_dir", str(outputs.get("metadata_dir") or "unavailable")),
            ("export_root", str(payload.get("export_root") or export_root)),
        ]
    )
    _info(sep)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exphub", add_help=True)

    try:
        _cfg = get_platform_config()
        _def_videox = _cfg.get("repos", {}).get("videox_fun", "")
        _def_droid_repo = _cfg.get("repos", {}).get("droid_slam", "")
        _def_droid_w = _cfg.get("models", {}).get("droid", {}).get("path", "")
    except Exception:
        _def_videox = ""
        _def_droid_repo = ""
        _def_droid_w = ""

    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "workflow", "encode", "decode", "eval", "export", "doctor"],
        help="pipeline stage",
    )
    ap.add_argument("--exphub", default=os.environ.get("EXPHUB", ""), help="ExpHub root (default: $EXPHUB or cwd)")

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--w", type=int, default=832)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--dur", type=str, default="3")
    ap.add_argument("--start_sec", type=str, default="0")
    ap.add_argument("--start_idx", type=int, default=-1)

    ap.add_argument("--kf_gap", type=int, default=0, help="0 means auto")
    ap.add_argument("--keyframes_mode", default="symlink", choices=["symlink", "hardlink", "copy"], help="how to materialize segment/keyframes")
    ap.add_argument(
        "--segment_policy",
        default=FORMAL_SEGMENT_POLICY,
        help="segment policy for the current workflow; only 'state' is accepted",
    )
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43, dest="seed_base")
    ap.add_argument("--gpus", type=int, default=2)

    ap.add_argument("--infer_extra", default="", help="extra args passed through the formal infer service (quoted string)")
    ap.add_argument(
        "--infer_backend",
        default="wan_fun_5b_inp",
        choices=["wan_fun_5b_inp"],
        help="infer backend for the current workflow",
    )
    ap.add_argument("--decode_source", default="aligned", choices=["aligned", "generation_units"])
    ap.add_argument(
        "--infer_model_dir",
        default="",
        help="override infer backend model dir or model id",
    )

    ap.add_argument("--datasets_cfg", default="", help="datasets.json path (default: <exphub>/config/datasets.json)")
    ap.add_argument("--exp_root", default="", help="override experiments root (default: <exphub>/experiments/<dataset>/<sequence>)")
    ap.add_argument("--export_root", default="", help="override export root (default: <exphub>/exports/<scope>/<focus>/<tag>)")
    ap.add_argument("--export_scope", default="single", choices=["single", "dataset", "focus"])
    ap.add_argument("--export_focus", default="ncd_scand", choices=["ncd_scand", "ncd", "scand"])
    ap.add_argument("--export_source", default="aligned", choices=["aligned", "generation_units"])
    ap.add_argument("--export_target_fps", type=int, default=24)
    ap.add_argument("--export_target_num_frames", type=int, default=73)
    ap.add_argument("--export_target_width", type=int, default=832)
    ap.add_argument("--export_target_height", type=int, default=480)
    ap.add_argument("--export_harvest_sec", type=float, default=0.0)
    ap.add_argument("--export_stride_sec", type=float, default=0.0)
    ap.add_argument("--export_max_bags", type=int, default=0)
    ap.add_argument("--export_max_bags_per_dataset", type=int, default=0)
    ap.add_argument("--export_max_clips_per_bag", type=int, default=0)
    ap.add_argument("--export_split_seed", type=int, default=13)

    ap.add_argument(
        "--keep_level",
        default="max",
        choices=["max", "min"],
        help="artifact retention level: max (keep all) or min (batch-optimized cleanup)",
    )
    ap.add_argument("--log_level", default="info", choices=["info", "debug", "quiet"], help="child process terminal verbosity")

    ap.add_argument(
        "--no_auto_conda",
        action="store_false",
        dest="auto_conda",
        default=True,
        help="disable automatic conda activation and use current shell env for child commands",
    )
    ap.add_argument("--videox_root", default=_def_videox)
    ap.add_argument("--droid_repo", default=_def_droid_repo)
    ap.add_argument("--droid_weights", default=_def_droid_w)

    ap.add_argument(
        "--prompt_model_dir",
        default="",
        help="override prompt scene-encoding model dir or model id",
    )

    ap.add_argument("--ros_setup", default=os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))

    # Eval-stage SLAM sequence selection.
    # Default is "both" so eval can compare ori/gen unless explicitly overridden.
    ap.add_argument("--droid_seq", default="both", choices=["auto", "ori", "gen", "both"])
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no_viz", action="store_true")

    args = ap.parse_args(argv)
    global _CLI_LOG_LEVEL
    _CLI_LOG_LEVEL = str(args.log_level or "info").strip().lower()
    set_cli_log_level(_CLI_LOG_LEVEL)
    args.keep_level = normalize_keep_level(args.keep_level)
    try:
        args.segment_policy = require_formal_segment_policy(args.segment_policy)
    except ValueError as exc:
        _die(str(exc))

    args.dataset = sanitize_token(args.dataset)
    args.sequence = sanitize_token(args.sequence)
    args.tag = sanitize_token(args.tag)
    if not args.dataset or not args.sequence or not args.tag:
        _die("dataset/sequence/tag becomes empty after sanitize")

    runtime = build_runtime(args)

    if _CLI_LOG_LEVEL != "quiet":
        _print_experiment_summary(
            mode=args.mode,
            dataset=runtime.spec.dataset,
            sequence=runtime.spec.sequence,
            tag=runtime.spec.tag,
            w=int(args.w),
            h=int(args.h),
            fps_text=runtime.fps_arg,
            dur_text=str(args.dur),
            gpus=int(args.gpus),
            keep_level=str(args.keep_level),
            exp_dir=runtime.paths.exp_dir,
        )

    try:
        result = run_runtime(runtime)
        mode_norm = str(args.mode or "").strip().lower()
        if mode_norm == "export":
            _print_export_report(result.result_root, result.step_times)
        elif mode_norm != "doctor":
            _print_experiment_report(result.exp_dir, result.step_times)
    except (ConfigError, RunError, RuntimeError) as e:
        _die(str(e))


if __name__ == "__main__":
    main()
