from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .cleanup import normalize_keep_level
from .common.config import ConfigError, get_platform_config
from .common.logging import set_cli_log_level
from .common.types import sanitize_token
from .contracts.segment import FORMAL_SEGMENT_POLICY, require_formal_segment_policy
from .context import ExperimentContext
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


def _sum_files(p: Path, glob_pat: str, follow_symlinks: bool = True) -> (int, int):
    n = 0
    b = 0
    if not p.exists():
        return 0, 0
    for fp in sorted(p.glob(glob_pat)):
        if fp.is_file():
            n += 1
            try:
                st = fp.resolve().stat() if follow_symlinks else fp.lstat()
                b += int(st.st_size)
            except Exception:
                pass
    return n, b


def write_compression_stats(ctx: ExperimentContext) -> None:
    frames_dir = ctx.segment_frames_dir
    keyframes_dir = ctx.segment_keyframes_dir

    prompt_files = [
        ctx.prompt_report_path,
        ctx.prompt_base_path,
        ctx.prompt_dir / "state_prompt_manifest.json",
        ctx.prompt_runtime_plan_path,
    ]

    ori_n, ori_b = _sum_files(frames_dir, "*.png", follow_symlinks=True)
    kf_n, kf_b = _sum_files(keyframes_dir, "*.png", follow_symlinks=True)

    prompt_n = 0
    prompt_b = 0
    for f in prompt_files:
        if not f.exists() or not f.is_file():
            continue
        prompt_n += 1
        try:
            prompt_b += int(f.stat().st_size)
        except Exception:
            pass

    compressed_b = int(kf_b + prompt_b)
    ratio_bytes = (compressed_b / ori_b) if ori_b > 0 else None
    ratio_frames = (kf_n / ori_n) if ori_n > 0 else None

    out = {
        "ori": {
            "frames_dir": str(frames_dir),
            "frame_count": int(ori_n),
            "bytes_sum": int(ori_b),
        },
        "compressed": {
            "keyframes_dir": str(keyframes_dir),
            "keyframe_count": int(kf_n),
            "keyframe_bytes_sum": int(kf_b),
            "prompt_files": [str(p) for p in prompt_files],
            "prompt_file_count": int(prompt_n),
            "prompt_bytes_sum": int(prompt_b),
            "total_bytes_sum": int(compressed_b),
        },
        "ratios": {
            "bytes": ratio_bytes,
            "frames": ratio_frames,
        },
    }

    stats_dir = ctx.stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)
    ctx.stats_compression_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _rm_tree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _rm_any(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except FileNotFoundError:
        return



def _fmt_intlike(x: float) -> str:
    """Format numeric that is often used as int. If x is integer-like, return int string.

    This avoids passing values like '24.0' to downstream scripts whose argparse expects int.
    Compatible with Python 3.7.
    """
    try:
        xf = float(x)
        # treat near-integers as int
        if abs(xf - round(xf)) < 1e-9:
            return str(int(round(xf)))
    except Exception:
        pass
    return str(x)


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
    phase_names = ["segment", "prompt", "infer", "merge", "slam", "eval", "stats"]
    total_time = sum(float(x) for x in step_times.values())

    eval_report = _read_json_dict(exp_dir / "eval" / "report.json")
    traj_metrics = dict(eval_report.get("traj_eval") or {}) if isinstance(eval_report.get("traj_eval"), dict) else {}
    image_metrics = dict(eval_report.get("image_eval") or {}) if isinstance(eval_report.get("image_eval"), dict) else {}
    slam_metrics = dict(eval_report.get("slam_friendly_eval") or {}) if isinstance(eval_report.get("slam_friendly_eval"), dict) else {}
    if not traj_metrics:
        traj_metrics = _read_json_dict(exp_dir / "eval" / "traj_metrics.json")
    if not image_metrics:
        image_metrics = _read_json_dict(exp_dir / "eval" / "image_metrics.json")
    if not slam_metrics:
        slam_metrics = _read_json_dict(exp_dir / "eval" / "slam_metrics.json")
    stats_report = _read_json_dict(exp_dir / "stats" / "final_report.json")
    if not stats_report:
        stats_report = _read_json_dict(exp_dir / "stats" / "report.json")
    stats_legacy = _read_json_dict(exp_dir / "stats" / "compression.json")
    infer_details = _parse_infer_log_details(exp_dir / "logs" / "infer.log")

    compression_obj = {}
    if isinstance(stats_report.get("compression"), dict):
        compression_obj = dict(stats_report.get("compression") or {})

    legacy_ori = dict(stats_legacy.get("ori") or {}) if isinstance(stats_legacy.get("ori"), dict) else {}
    legacy_comp = dict(stats_legacy.get("compressed") or {}) if isinstance(stats_legacy.get("compressed"), dict) else {}
    legacy_ratios = dict(stats_legacy.get("ratios") or {}) if isinstance(stats_legacy.get("ratios"), dict) else {}

    ori_bytes = _pick_first_int(
        compression_obj,
        [["ori_bytes"]],
    )
    if ori_bytes is None:
        ori_bytes = _pick_first_int(legacy_ori, [["bytes_sum"]])

    keyframes_bytes = _pick_first_int(
        compression_obj,
        [["keyframes_bytes"]],
    )
    if keyframes_bytes is None:
        keyframes_bytes = _pick_first_int(legacy_comp, [["keyframe_bytes_sum"]])

    prompt_bytes = _pick_first_int(
        compression_obj,
        [["prompt_bytes"]],
    )
    if prompt_bytes is None:
        prompt_bytes = _pick_first_int(legacy_comp, [["prompt_bytes_sum"]])

    comp_size = None  # type: Optional[int]
    if keyframes_bytes is not None and prompt_bytes is not None:
        comp_size = int(keyframes_bytes + prompt_bytes)
    else:
        comp_size = _pick_first_int(legacy_comp, [["total_bytes_sum"]])

    ratio_bytes = _pick_first_float(
        compression_obj,
        [["ratio_bytes"]],
    )
    if ratio_bytes is None:
        ratio_bytes = _pick_first_float(legacy_ratios, [["bytes"]])

    keyframes_frames = _pick_first_int(
        compression_obj,
        [["keyframes_frames"]],
    )
    if keyframes_frames is None:
        keyframes_frames = _pick_first_int(legacy_comp, [["keyframe_count"], ["keyframes_frame_count"]])

    reduction_ratio = None  # type: Optional[float]
    if ratio_bytes is not None:
        reduction_ratio = 1.0 - float(ratio_bytes)

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
            "img_psnr": _pick_float(image_metrics, ["psnr", "mean"]),
            "img_ms_ssim": _pick_float(image_metrics, ["ms_ssim", "mean"]),
            "img_lpips": _pick_float(image_metrics, ["lpips", "mean"]),
            "img_frames": _pick_int(image_metrics, ["frame_count"]),
            "slam_inlier": _pick_float(slam_metrics, ["inlier_ratio", "mean"]),
            "slam_pose_sr": _pick_float(slam_metrics, ["pose_success_rate"]),
            "slam_ref": str(slam_metrics.get("reference_source", "unavailable") or "unavailable"),
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
    for phase_name in ["segment", "prompt", "infer", "merge", "slam", "eval", "stats"]:
        time_rows.append((phase_name, _fmt_phase_seconds(_as_float_or_none(phase_times.get(phase_name)), total_time)))

    detail_rows = [
        ("infer.load", _fmt_seconds(_as_float_or_none(infer_details.get("infer.load")))),
        ("infer.quant", _fmt_seconds(_as_float_or_none(infer_details.get("infer.quant")))),
        ("infer.run", _fmt_seconds(_as_float_or_none(infer_details.get("infer.run")))),
        ("infer.avg_fr", _fmt_seconds(_as_float_or_none(infer_details.get("infer.avg_fr")), digits=3, suffix="s/frame")),
        ("total", _fmt_seconds(total_time)),
    ]

    quality_rows = [
        ("ape_rmse", _fmt_metric(_as_float_or_none(quality.get("ape_rmse")), unit="m")),
        ("rpe_trans", _fmt_metric(_as_float_or_none(quality.get("rpe_trans")), unit="m")),
        ("rpe_rot", _fmt_metric(_as_float_or_none(quality.get("rpe_rot")), unit="deg")),
        ("ori_len", _fmt_metric(_as_float_or_none(quality.get("ori_len")), unit="m")),
        ("gen_len", _fmt_metric(_as_float_or_none(quality.get("gen_len")), unit="m")),
        ("img.psnr", _fmt_metric(_as_float_or_none(quality.get("img_psnr")), unit="dB")),
        ("img.ms_ssim", _fmt_metric(_as_float_or_none(quality.get("img_ms_ssim")))),
        ("img.lpips", _fmt_metric(_as_float_or_none(quality.get("img_lpips")))),
        ("slam.inlier", _fmt_metric(_as_float_or_none(quality.get("slam_inlier")))),
        ("slam.pose_sr", _fmt_metric(_as_float_or_none(quality.get("slam_pose_sr")))),
        ("slam.ref", str(quality.get("slam_ref") or "unavailable")),
        ("poses", _fmt_count(_as_int_or_none(quality.get("poses")))),
        ("img_frames", _fmt_count(_as_int_or_none(quality.get("img_frames")))),
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
        choices=["all", "workflow", "segment", "prompt", "stats", "infer", "merge", "slam", "eval", "doctor"],
        help="pipeline stage",
    )
    ap.add_argument("--exphub", default=os.environ.get("EXPHUB", ""), help="ExpHub root (default: $EXPHUB or cwd)")

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--dur", type=str, required=True)
    ap.add_argument("--start_sec", type=str, required=True)
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
        choices=["wan_fun_a14b_inp", "wan_fun_5b_inp"],
        help="infer backend for the current workflow",
    )
    ap.add_argument(
        "--infer_model_dir",
        default="",
        help="override infer backend model dir or model id",
    )

    ap.add_argument("--datasets_cfg", default="", help="datasets.json path (default: <exphub>/config/datasets.json)")
    ap.add_argument("--exp_root", default="", help="override experiments root (default: <exphub>/experiments/<dataset>/<sequence>)")

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
    ap.add_argument("--conda_env_vlm", default=os.environ.get("CONDA_ENV_VLM", "vlm_prompt"))
    ap.add_argument("--conda_env_videox", default=os.environ.get("CONDA_ENV_VIDEOX", "videox"))
    ap.add_argument("--conda_env_droid", default=os.environ.get("CONDA_ENV_DROID", "droid"))

    ap.add_argument("--videox_root", default=_def_videox)
    ap.add_argument("--droid_repo", default=_def_droid_repo)
    ap.add_argument("--droid_weights", default=_def_droid_w)

    ap.add_argument(
        "--prompt_backend",
        default="smolvlm2",
        choices=["smolvlm2"],
        help="prompt backend for the current workflow",
    )
    ap.add_argument(
        "--prompt_model_dir",
        default="",
        help="override SmolVLM2 model dir or model id",
    )
    ap.add_argument(
        "--prompt_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="SmolVLM2 torch dtype hint",
    )
    ap.add_argument(
        "--prompt_sample_mode",
        default="even",
        choices=["quartiles", "even", "first", "last"],
        help="frame sampling strategy inside each prompt clip",
    )
    ap.add_argument(
        "--prompt_num_images",
        type=int,
        default=5,
        help="number of representative images sampled for prompt generation",
    )

    ap.add_argument("--ros_setup", default=os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))
    ap.add_argument(
        "--skip_analyze",
        action="store_true",
        help="reserved no-op flag; ignored in the current workflow",
    )

    # SLAM sequence selection.
    # Default is "both" so that `--mode slam` runs both ori/gen unless explicitly overridden.
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
        if str(args.mode or "").strip().lower() != "doctor":
            _print_experiment_report(result.exp_dir, result.step_times)
    except (ConfigError, RunError, RuntimeError) as e:
        _die(str(e))


if __name__ == "__main__":
    main()
