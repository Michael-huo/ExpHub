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

from .cleanup import apply_keep_level, normalize_keep_level
from .config import ConfigError, load_datasets_cfg, resolve_dataset
from .context import ExperimentContext
from .meta import sanitize_token, write_exp_meta
from .runner import (
    RunnerConfig,
    StepRunner,
    detect_conda_base,
    get_phase_python_config,
    resolve_phase_python,
    run_cmd,
    RunError,
)
from scripts._segment.policies.naming import (
    OFFICIAL_POLICY_NAMES,
    is_supported_policy_name,
    normalize_policy_name,
)


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
        try:
            from scripts._common import get_platform_config
        except Exception:
            local_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
            if str(local_scripts_dir) not in sys.path:
                sys.path.insert(0, str(local_scripts_dir))
            from _common import get_platform_config

        _cfg = get_platform_config()
        _def_qwen = _cfg.get("models", {}).get("qwen2_vl", {}).get("path", "")
        _def_videox = _cfg.get("repos", {}).get("videox_fun", "")
        _def_droid_repo = _cfg.get("repos", {}).get("droid_slam", "")
        _def_droid_w = _cfg.get("models", {}).get("droid", {}).get("path", "")
    except Exception:
        _def_qwen = ""
        _def_videox = ""
        _def_droid_repo = ""
        _def_droid_w = ""

    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "segment", "prompt", "stats", "infer", "merge", "slam", "eval", "doctor"],
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
        default="uniform",
        help="official segment keyframe policy for the current mainline: uniform | state",
    )
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43, dest="seed_base")
    ap.add_argument("--gpus", type=int, default=2)

    ap.add_argument("--infer_extra", default="", help="extra args passed to infer_i2v.py (quoted string)")
    ap.add_argument(
        "--infer_backend",
        default="wan_fun_5b_inp",
        choices=["wan_fun_a14b_inp", "wan_fun_5b_inp"],
        help="infer backend used by scripts/infer_i2v.py",
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

    # Prompt generator is now managed under ExpHub/scripts.
    # Qwen2-VL model path default is sourced from config/platform.yaml.
    ap.add_argument(
        "--qwen_model_dir",
        default=_def_qwen,
        help="Qwen2-VL model dir used by prompt generator",
    )
    ap.add_argument(
        "--prompt_backend",
        default="smolvlm2",
        choices=["qwen", "smolvlm2"],
        help="prompt backend used by scripts/prompt_gen.py",
    )
    ap.add_argument(
        "--prompt_model_dir",
        default="",
        help="override prompt backend model dir or model id; qwen falls back to --qwen_model_dir when empty",
    )
    ap.add_argument(
        "--prompt_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="prompt backend torch dtype hint",
    )
    ap.add_argument(
        "--prompt_sample_mode",
        default="even",
        choices=["quartiles", "even", "first", "last", "all"],
        help="frame sampling strategy inside each prompt clip",
    )
    ap.add_argument(
        "--prompt_num_images",
        type=int,
        default=5,
        help="number of representative images passed into the prompt backend",
    )

    ap.add_argument("--ros_setup", default=os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))
    ap.add_argument(
        "--skip_analyze",
        action="store_true",
        help="legacy no-op flag; post-segment analyze sidecar is no longer run by default",
    )

    # SLAM sequence selection.
    # Default is "both" so that `--mode slam` runs both ori/gen unless explicitly overridden.
    ap.add_argument("--droid_seq", default="both", choices=["auto", "ori", "gen", "both"])
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no_viz", action="store_true")

    args = ap.parse_args(argv)
    global _CLI_LOG_LEVEL
    _CLI_LOG_LEVEL = str(args.log_level or "info").strip().lower()
    args.keep_level = normalize_keep_level(args.keep_level)
    raw_segment_policy = str(args.segment_policy or "uniform")
    if not is_supported_policy_name(raw_segment_policy):
        _die(
            "unsupported segment policy: {} (expected one of: {})".format(
                raw_segment_policy,
                ", ".join(OFFICIAL_POLICY_NAMES),
            )
        )
    args.segment_policy = normalize_policy_name(raw_segment_policy)

    fps_arg = _fmt_intlike(args.fps)

    exphub_root = Path(args.exphub).resolve() if args.exphub else Path.cwd().resolve()
    if not (exphub_root / "scripts").exists():
        # Try to infer if user runs from subdir.
        cur = Path.cwd().resolve()
        found = None
        for p in [cur] + list(cur.parents):
            if (p / "scripts").exists() and (p / "config").exists():
                found = p
                break
        if found:
            exphub_root = found
        else:
            _warn(f"Cannot verify ExpHub root at {exphub_root}; continuing")

    dataset = sanitize_token(args.dataset)
    sequence = sanitize_token(args.sequence)
    tag = sanitize_token(args.tag)
    if not dataset or not sequence or not tag:
        _die("dataset/sequence/tag becomes empty after sanitize")

    exp_root_override = Path(args.exp_root).resolve() if args.exp_root else None
    ctx = ExperimentContext(
        exphub_root=exphub_root,
        dataset=dataset,
        sequence=sequence,
        tag=tag,
        w=args.w,
        h=args.h,
        start_sec=args.start_sec,
        dur=args.dur,
        fps=args.fps,
        kf_gap_input=args.kf_gap,
        exp_root_override=exp_root_override,
    )

    kf_gap = ctx.kf_gap
    if kf_gap % 4 != 0:
        _warn(f"kf_gap={kf_gap} not divisible by 4 (r=4). model may truncate length.")

    exp_name = ctx.exp_name
    exp_dir = ctx.exp_dir
    segment_dir = ctx.segment_dir
    prompt_dir = ctx.prompt_dir
    infer_dir = ctx.infer_dir
    merge_dir = ctx.merge_dir
    slam_root = ctx.slam_dir
    eval_dir = ctx.eval_dir

    cfg_path = Path(args.datasets_cfg) if args.datasets_cfg else (exphub_root / "config" / "datasets.json")
    if not cfg_path.is_absolute():
        cfg_path = (exphub_root / cfg_path).resolve()

    runner_cfg = RunnerConfig(
        auto_conda=bool(args.auto_conda),
        conda_base=detect_conda_base() if args.auto_conda else None,
        ros_setup=Path(args.ros_setup) if args.ros_setup else None,
    )
    phase_python_cache = {}  # type: Dict[str, str]

    # script paths
    scripts_dir = exphub_root / "scripts"
    seg_py = scripts_dir / "segment_make.py"
    infer_py = scripts_dir / "infer_i2v.py"
    merge_py = scripts_dir / "merge_seq.py"
    droid_py = scripts_dir / "slam_droid.py"
    stats_py = scripts_dir / "stats_collect.py"
    eval_main_py = scripts_dir / "eval_main.py"
    eval_traj_py = scripts_dir / "eval_traj.py"
    prompt_gen_py = (scripts_dir / "prompt_gen.py").resolve()
    logs_dir = ctx.logs_dir
    child_pass_prefixes = ("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]")
    fail_tail_lines = 30
    step_runner = StepRunner(
        logs_dir=logs_dir,
        log_level=args.log_level,
        runner_cfg=runner_cfg,
        pass_prefixes=child_pass_prefixes,
        fail_tail_lines=fail_tail_lines,
    )
    step_times = {}  # type: Dict[str, float]

    def _read_log_tail(log_path: Path, n: int) -> List[str]:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if n <= 0:
                return lines
            return lines[-n:]
        except Exception:
            return []

    def _format_out_hint(out_hint: str) -> str:
        text = str(out_hint or "").strip()
        if not text:
            return ""
        try:
            target = Path(text).resolve()
        except Exception:
            return text
        try:
            rel = target.relative_to(exp_dir.resolve())
            short_text = rel.as_posix()
        except ValueError:
            short_text = text
        if target.is_dir() and short_text and not short_text.endswith("/"):
            short_text += "/"
        return short_text or "."

    def _run_step(step_name: str, fn, out_hint: str = "") -> None:
        t0 = time.time()
        _step(f"{step_name} start mode={args.mode}")
        try:
            fn()
        except RunError as e:
            sec = time.time() - t0
            rc = e.returncode if e.returncode is not None else -1
            log_path = str(e.log_path) if e.log_path else "-"
            _step(f"{step_name} FAIL sec={sec:.2f} rc={rc} log={log_path}")
            tail_lines = []
            if e.log_path and Path(e.log_path).is_file():
                tail_lines = _read_log_tail(Path(e.log_path), fail_tail_lines)
            if not tail_lines:
                tail_lines = list(e.tail_lines)
            if tail_lines:
                _warn(f"{step_name} last {len(tail_lines)} lines:")
                for line in tail_lines:
                    print(f"[TAIL] {line}")
            raise SystemExit(f"[ERR] step failed: {step_name}")
        sec = time.time() - t0
        step_times[step_name] = float(sec)
        out_hint_short = _format_out_hint(out_hint)
        if out_hint_short:
            _step(f"{step_name} done sec={sec:.2f} out={out_hint_short}")
        else:
            _step(f"{step_name} done sec={sec:.2f}")

    def _phase_python(phase_name: str) -> str:
        phase_key = str(phase_name)
        if phase_key not in phase_python_cache:
            try:
                phase_python_cache[phase_key] = resolve_phase_python(phase_key)
            except RuntimeError as e:
                _die(str(e))
        return phase_python_cache[phase_key]

    def _prompt_phase_name() -> str:
        backend = str(args.prompt_backend or "smolvlm2").strip().lower()
        if backend == "smolvlm2":
            return "prompt_smol"
        return "prompt"

    def _prompt_model_ref() -> str:
        if str(args.prompt_model_dir or "").strip():
            return str(args.prompt_model_dir).strip()
        if str(args.prompt_backend or "smolvlm2").strip().lower() == "qwen":
            return str(args.qwen_model_dir or "").strip()
        return ""

    def _infer_phase_name() -> str:
        backend = str(args.infer_backend or "wan_fun_5b_inp").strip().lower()
        if backend == "wan_fun_5b_inp":
            return "infer_fun_5b"
        return "infer"

    def step_doctor() -> int:
        _info("STEP doctor: begin")
        has_critical_missing = False
        phase_names = ["segment", "prompt", _infer_phase_name(), "slam"]
        if str(args.prompt_backend or "smolvlm2").strip().lower() == "smolvlm2":
            phase_names.append("prompt_smol")
        for phase_name in phase_names:
            python_bin = get_phase_python_config(phase_name)
            exists = False
            if python_bin:
                phase_path = Path(str(python_bin)).expanduser()
                exists = phase_path.is_file() and os.access(str(phase_path), os.X_OK)
            _info(
                "DOCTOR phase={} python={} exists={}".format(
                    phase_name,
                    python_bin or "<missing>",
                    exists,
                )
            )
            if not python_bin or not exists:
                has_critical_missing = True

        if has_critical_missing:
            _warn("DOCTOR result=FAIL")
            return 2
        _info("DOCTOR result=PASS")
        return 0

    if _CLI_LOG_LEVEL != "quiet":
        _print_experiment_summary(
            mode=args.mode,
            dataset=dataset,
            sequence=sequence,
            tag=tag,
            w=int(args.w),
            h=int(args.h),
            fps_text=fps_arg,
            dur_text=str(args.dur),
            gpus=int(args.gpus),
            keep_level=str(args.keep_level),
            exp_dir=exp_dir,
        )

    if args.mode == "doctor":
        rc = step_doctor()
        if rc != 0:
            raise SystemExit(rc)
        _info("DONE. MODE=doctor")
        return

    cfg = load_datasets_cfg(cfg_path)
    ds = resolve_dataset(cfg, exphub_root, dataset, sequence)
    if not ds.bag.exists():
        _die(f"bag not found: {ds.bag}")

    # viz default policy: auto => on for slam/eval, off otherwise
    if args.viz and args.no_viz:
        _die("--viz and --no_viz are mutually exclusive")
    if args.viz:
        viz_enable = True
    elif args.no_viz:
        viz_enable = False
    else:
        viz_enable = args.mode in ("slam", "eval")

    for p in [seg_py, infer_py, merge_py, droid_py, stats_py, eval_main_py, eval_traj_py]:
        _ensure(p, "file")

    _ensure(prompt_gen_py, "file")

    def _assert_under_exp(p: Path) -> None:
        base = exp_dir.resolve()
        target = p.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            _die(f"unsafe path outside EXP_DIR: {target} (exp_dir={base})")

    def _rm_in_exp(p: Path) -> None:
        _assert_under_exp(p)
        _rm_any(p)

    def ensure_clean_exp_dir() -> None:
        if exp_dir.exists():
            _debug_info(f"overwrite enabled: rm -rf {exp_dir}")
            _rm_tree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

    def write_meta_snapshot() -> None:
        meta = {
            "dataset": dataset,
            "sequence": sequence,
            "tag": tag,
            "exp_name": exp_name,
            "exp_dir": str(exp_dir),
            "inputs": {
                "bag": str(ds.bag),
                "topic": ds.topic,
                "intrinsics": {
                    "fx": ds.fx,
                    "fy": ds.fy,
                    "cx": ds.cx,
                    "cy": ds.cy,
                    "dist": ds.dist,
                },
            },
            "params": {
                "w": args.w,
                "h": args.h,
                "fps": args.fps,
                "dur": args.dur,
                "start_sec": args.start_sec,
                "start_idx": args.start_idx,
                "kf_gap": kf_gap,
                "segment_policy": args.segment_policy,
                "base_idx": args.base_idx,
                "seed_base": args.seed_base,
                "gpus": args.gpus,
                "prompt_backend": args.prompt_backend,
                "prompt_model_dir": args.prompt_model_dir,
                "infer_backend": args.infer_backend,
                "infer_model_dir": args.infer_model_dir,
                "prompt_sample_mode": args.prompt_sample_mode,
                "prompt_num_images": args.prompt_num_images,
                "droid_seq": args.droid_seq,
                "viz_enable": viz_enable,
                "keep_level": args.keep_level,
            },
            "paths": {
                "segment_dir": str(segment_dir),
                "segment_python": _phase_python("segment"),
                "videox_root": args.videox_root,
                "droid_repo": args.droid_repo,
            },
        }
        write_exp_meta(ctx.exp_meta_path, meta)

    def step_segment() -> None:
        ensure_clean_exp_dir()
        write_meta_snapshot()
        segment_python = _phase_python("segment")
        _debug_info("STEP segment: interpreter={}".format(segment_python))

        dist_args: List[str] = []
        if ds.dist:
            dist_args = ["--dist", *[str(x) for x in ds.dist]]

        cmd = [
            segment_python,
            str(seg_py),
            "--bag",
            str(ds.bag),
            "--topic",
            ds.topic,
            "--out_root",
            str(exp_dir),
            "--name",
            "segment",
            "--duration",
            str(args.dur),
            "--fps",
            fps_arg,
            "--kf_gap",
            str(kf_gap),
            "--keyframes_mode",
            str(args.keyframes_mode),
            "--segment_policy",
            str(args.segment_policy),
            "--start_idx",
            str(args.start_idx),
            "--start_sec",
            str(args.start_sec),
            "--width",
            str(args.w),
            "--height",
            str(args.h),
            "--fx",
            str(ds.fx),
            "--fy",
            str(ds.fy),
            "--cx",
            str(ds.cx),
            "--cy",
            str(ds.cy),
            *dist_args,
        ]

        step_runner.run_ros(cmd, log_name="segment.log", cwd=exphub_root)
        _ensure(ctx.segment_calib_path, "file")
        _ensure(ctx.segment_timestamps_path, "file")
        # recommended keep
        _ensure(ctx.segment_preprocess_meta_path, "file")

    def step_prompt() -> None:
        if not segment_dir.is_dir():
            _die(f'missing required input dir: {segment_dir}. Run "--mode segment" first.')
        frames_dir = ctx.segment_frames_dir
        if not frames_dir.is_dir():
            _die(f'missing required input dir: {frames_dir}. Run "--mode segment" first to generate segment/frames.')

        exp_dir.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_phase = _prompt_phase_name()
        prompt_python = get_phase_python_config(prompt_phase)
        if not prompt_python:
            if prompt_phase == "prompt_smol":
                _die(
                    "missing prompt_smol phase config in config/platform.yaml. "
                    "Please set environments.phases.prompt_smol.python (and optional conda metadata) before using --prompt_backend smolvlm2."
                )
            _die("missing prompt phase config in config/platform.yaml")

        cmd = [
            "python",
            str(prompt_gen_py),
            "--frames_dir",
            str(frames_dir),
            "--exp_dir",
            str(exp_dir),
            "--fps",
            fps_arg,
            "--backend",
            str(args.prompt_backend),
            "--model_dir",
            str(_prompt_model_ref()),
            "--dtype",
            str(args.prompt_dtype),
            "--sample_mode",
            str(args.prompt_sample_mode),
            "--num_images",
            str(args.prompt_num_images),
            "--backend_python_phase",
            str(prompt_phase),
        ]

        step_runner.run_env_python(cmd, phase_name=prompt_phase, log_name="prompt.log", cwd=exphub_root)
        _ensure(ctx.prompt_report_path, "file")
        _ensure(ctx.prompt_base_path, "file")
        _ensure(ctx.prompt_dir / "state_prompt_manifest.json", "file")
        _ensure(ctx.prompt_runtime_plan_path, "file")

    def step_stats() -> None:
        cmd = [
            "python",
            str(stats_py),
            "--exp_dir",
            str(exp_dir),
        ]
        step_runner.run_env_python(cmd, phase_name="prompt", log_name="stats.log", cwd=exphub_root)
        _ensure(ctx.stats_report_path, "file")

    def step_infer() -> None:
        prompt_file = ctx.prompt_runtime_plan_path
        if not prompt_file.is_file():
            _die(
                'missing prompt/runtime_prompt_plan.json. Run "--mode prompt" first or provide a valid prompt file.'
            )

        exp_dir.mkdir(parents=True, exist_ok=True)
        _rm_in_exp(infer_dir)
        infer_phase = _infer_phase_name()
        if not get_phase_python_config(infer_phase):
            if infer_phase == "infer_fun_5b":
                _die(
                    "missing infer_fun_5b phase config in config/platform.yaml. "
                    "Please set environments.phases.infer_fun_5b.python before using --infer_backend wan_fun_5b_inp."
                )
            _die("missing infer phase config in config/platform.yaml")

        cmd_infer = [
            "python",
            str(infer_py),
            "--segment_dir",
            str(segment_dir),
            "--exp_dir",
            str(exp_dir),
            "--videox_root",
            str(args.videox_root),
            "--gpus",
            str(args.gpus),
            "--fps",
            fps_arg,
            "--kf_gap",
            str(kf_gap),
            "--base_idx",
            str(args.base_idx),
            "--seed_base",
            str(args.seed_base),
            "--prompt_file",
            str(prompt_file),
            "--infer_backend",
            str(args.infer_backend),
            "--infer_model_dir",
            str(args.infer_model_dir),
            "--backend_python_phase",
            str(infer_phase),
        ]
        if args.infer_extra:
            import shlex as _sh
            cmd_infer.extend(_sh.split(args.infer_extra))
        step_runner.run_env_python(cmd_infer, phase_name=infer_phase, log_name="infer.log", cwd=exphub_root)
        _ensure(ctx.infer_runs_dir, "dir")
        _ensure(ctx.infer_runs_plan_path, "file")
        _ensure(ctx.infer_report_path, "file")

    def step_merge() -> None:
        _ensure(segment_dir, "dir")
        _ensure(ctx.infer_runs_dir, "dir")
        _ensure(ctx.infer_runs_plan_path, "file")

        _rm_in_exp(merge_dir)
        infer_phase = _infer_phase_name()

        cmd_merge = [
            "python",
            str(merge_py),
            "--segment_dir",
            str(segment_dir),
            "--exp_dir",
            str(exp_dir),
            "--runs_root",
            str(ctx.infer_runs_dir),
            "--plan",
            str(ctx.infer_runs_plan_path),
            "--out_dir",
            str(merge_dir),
        ]
        step_runner.run_env_python(cmd_merge, phase_name=infer_phase, log_name="merge.log", cwd=exphub_root)

        _ensure(ctx.merge_frames_dir, "dir")
        _ensure(ctx.merge_calib_path, "file")
        _ensure(ctx.merge_timestamps_path, "file")

    def step_slam() -> None:
        # Decide which sequences to run.
        seq = args.droid_seq
        if seq == "auto":
            seq = "both"

        def _run(tag_name: str, seg_path: Path) -> None:
            dst_dir = ctx.slam_track_dir(tag_name)
            _rm_in_exp(dst_dir)
            _runtime_info("slam run={}".format(tag_name))

            cmd = [
                "python",
                str(droid_py),
                "--segment_dir",
                str(seg_path),
                "--droid_repo",
                str(args.droid_repo),
                "--weights",
                str(args.droid_weights),
                "--out_dir",
                str(exp_dir),
                "--slam_out_dir",
                str(dst_dir),
                "--fps",
                fps_arg,
                "--undistort_mode",
                "auto",
                "--resize_interp",
                "linear",
                "--intr_scale_mode",
                "demo",
            ]
            if not viz_enable:
                cmd.append("--disable_vis")

            step_runner.run_env_python(cmd, phase_name="slam", log_name=f"slam_{tag_name}.log", cwd=exphub_root)
            _ensure(ctx.slam_traj_path(tag_name), "file")
            _ensure(ctx.slam_run_meta_path(tag_name), "file")

            # Ensure run_meta paths point to final track directory.
            try:
                meta_obj = json.loads(ctx.slam_run_meta_path(tag_name).read_text(encoding="utf-8"))
            except Exception:
                meta_obj = {}
            tum_meta = Path(str(meta_obj.get("tum_path", ""))).resolve() if meta_obj.get("tum_path") else None
            npz_meta = Path(str(meta_obj.get("npz_path", ""))).resolve() if meta_obj.get("npz_path") else None
            tum_expect = ctx.slam_traj_path(tag_name).resolve()
            npz_expect = ctx.slam_npz_path(tag_name).resolve()
            if tum_meta != tum_expect or npz_meta != npz_expect:
                _die(f"slam run_meta path mismatch for track={tag_name}: tum={tum_meta} npz={npz_meta}")
            _debug_info(f"[OK] slam {tag_name} saved: {ctx.slam_traj_path(tag_name)}")

        if seq in ("ori", "both"):
            _ensure(ctx.segment_frames_dir, "dir")
        if seq in ("gen", "both"):
            _ensure(ctx.merge_frames_dir, "dir")

        if seq == "ori":
            _run("ori", segment_dir)
        elif seq == "gen":
            _run("gen", merge_dir)
        else:
            _run("ori", segment_dir)
            _run("gen", merge_dir)

    def step_eval() -> None:
        eval_plot_enable = not bool(args.no_viz)
        _debug_info("STEP eval: phase=slam plots={}".format(eval_plot_enable))

        _rm_tree(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        tum_ori = ctx.slam_traj_path("ori")
        tum_gen = ctx.slam_traj_path("gen")
        cmd = [
            "python",
            str(eval_main_py),
            "--exp_dir",
            str(exp_dir),
            "--reference",
            str(tum_ori),
            "--estimate",
            str(tum_gen),
            "--out_dir",
            str(eval_dir),
            "--reference_name",
            "ori",
            "--estimate_name",
            "gen",
            "--alignment_mode",
            "se3",
        ]
        if not eval_plot_enable:
            cmd.append("--skip_plots")

        step_runner.run_env_python(
            cmd,
            phase_name="slam",
            log_name="eval.log",
            cwd=exphub_root,
            check=False,
        )

        report_path = ctx.eval_artifact_path("report.json")
        details_path = ctx.eval_artifact_path("details.csv")
        metrics_plot_path = eval_dir / "plots" / "metrics_overview.png"
        if report_path.is_file():
            _debug_info("STEP eval: report={}".format(report_path))
        else:
            _warn("eval report missing: {}".format(report_path))
        if details_path.is_file():
            _debug_info("STEP eval: details={}".format(details_path))
        else:
            _warn("eval details missing: {}".format(details_path))
        if metrics_plot_path.is_file():
            _debug_info("STEP eval: metrics_overview={}".format(metrics_plot_path))
        else:
            _warn("eval metrics overview missing: {}".format(metrics_plot_path))

    def maybe_run_post_analyze() -> None:
        if args.mode not in ("segment", "all"):
            return
        if args.skip_analyze:
            _debug_info("post analyze disabled: --skip_analyze (legacy no-op)")
            return
        _debug_info("post analyze disabled by default: analysis/research sidecar is now manual-only")


    # Execute mode
    try:
        if args.mode == "segment":
            _run_step("segment", step_segment, str(segment_dir))
            maybe_run_post_analyze()
        elif args.mode == "prompt":
            _run_step("prompt", step_prompt, str(ctx.prompt_report_path))
        elif args.mode == "infer":
            _run_step("infer", step_infer, str(ctx.infer_report_path))
        elif args.mode == "merge":
            _run_step("merge", step_merge, str(merge_dir))
        elif args.mode == "slam":
            _run_step("slam", step_slam, str(slam_root))
        elif args.mode == "eval":
            _run_step("eval", step_eval, str(eval_dir))
        elif args.mode == "stats":
            _run_step("stats", step_stats, str(ctx.stats_report_path))
        else:  # all
            _run_step("segment", step_segment, str(segment_dir))
            maybe_run_post_analyze()
            _run_step("prompt", step_prompt, str(ctx.prompt_report_path))
            _run_step("infer", step_infer, str(ctx.infer_report_path))
            _run_step("merge", step_merge, str(merge_dir))
            _run_step("slam", step_slam, str(slam_root))
            _run_step("eval", step_eval, str(eval_dir))
            _run_step("stats", step_stats, str(ctx.stats_report_path))

        apply_keep_level(exp_dir, args.keep_level)
        _print_experiment_report(exp_dir, step_times)

    except (ConfigError, RunError) as e:
        _die(str(e))

    _runtime_info("DONE.")


if __name__ == "__main__":
    main()
