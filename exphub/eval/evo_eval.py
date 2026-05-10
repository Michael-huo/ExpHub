from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_file, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_warn


T_MAX_DIFF = 0.03
STAT_KEYS = ("rmse", "mean", "median", "std", "min", "max", "sse")


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _find_zip_member(zf, basename):
    for name in zf.namelist():
        if Path(name).name == basename:
            return name
    return None


def _json_from_zip(path_obj, basename):
    path = Path(path_obj).resolve()
    with zipfile.ZipFile(str(path), "r") as zf:
        member = _find_zip_member(zf, basename)
        if member is None:
            raise RuntimeError("{} missing from evo result zip: {}".format(basename, path))
        with zf.open(member, "r") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("{} in evo result zip is not a JSON object: {}".format(basename, path))
    return payload


def _npy_len_from_zip(path_obj, basename):
    import numpy as np

    path = Path(path_obj).resolve()
    with zipfile.ZipFile(str(path), "r") as zf:
        member = _find_zip_member(zf, basename)
        if member is None:
            return None
        with zf.open(member, "r") as handle:
            arr = np.load(BytesIO(handle.read()), allow_pickle=False)
    try:
        return int(len(arr))
    except Exception:
        return None


def _as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_stats(zip_path, comparison, warnings):
    payload = _json_from_zip(zip_path, "stats.json")
    stats = {key: _as_float(payload.get(key)) for key in STAT_KEYS}
    pose_pairs = _npy_len_from_zip(zip_path, "error_array.npy")
    if pose_pairs is None:
        pose_pairs = _npy_len_from_zip(zip_path, "timestamps.npy")
    if pose_pairs is None:
        warnings.append("pose pair count unavailable for {}; error_array.npy and timestamps.npy not found".format(comparison))
    return stats, pose_pairs


def _stderr_summary(text):
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return "n/a"
    return " | ".join(lines[-8:])


def _format_stream(text):
    value = str(text or "")
    if value:
        return value
    return "<empty>\n"


def _write_failure_log(out_dir, name, cmd, completed):
    path = Path(out_dir).resolve() / "evo_failure.log"
    lines = [
        "comparison: {}".format(name),
        "command: {}".format(" ".join(str(item) for item in cmd)),
        "return_code: {}".format(completed.returncode),
        "",
        "[stdout]",
        _format_stream(completed.stdout).rstrip("\n"),
        "",
        "[stderr]",
        _format_stream(completed.stderr).rstrip("\n"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _resolve_evo_ape():
    checked = []
    env_value = str(os.environ.get("EXPHUB_EVO_APE") or "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        checked.append(str(candidate))
        if not candidate.exists():
            raise RuntimeError(
                "evo_ape not found for the current eval Python environment.\n"
                "EXPHUB_EVO_APE is set but does not exist: {candidate}\n"
                "current Python executable: {executable}\n"
                "current Python prefix: {prefix}\n"
                "checked candidate paths: {checked}\n"
                "PATH: {path}\n"
                "recommended install command: {executable} -m pip install evo".format(
                    candidate=candidate,
                    executable=sys.executable,
                    prefix=sys.prefix,
                    checked=checked,
                    path=os.environ.get("PATH", ""),
                )
            )
        if not candidate.is_file():
            raise RuntimeError(
                "evo_ape not found for the current eval Python environment.\n"
                "EXPHUB_EVO_APE is not a file: {candidate}\n"
                "current Python executable: {executable}\n"
                "current Python prefix: {prefix}\n"
                "checked candidate paths: {checked}\n"
                "PATH: {path}\n"
                "recommended install command: {executable} -m pip install evo".format(
                    candidate=candidate,
                    executable=sys.executable,
                    prefix=sys.prefix,
                    checked=checked,
                    path=os.environ.get("PATH", ""),
                )
            )
        return [str(candidate)]

    sibling = Path(sys.executable).with_name("evo_ape")
    checked.append(str(sibling))
    if sibling.exists() and sibling.is_file():
        return [str(sibling)]

    path_candidate = shutil.which("evo_ape")
    if path_candidate:
        checked.append(str(path_candidate))
        return [str(path_candidate)]
    checked.append("PATH:evo_ape")

    raise RuntimeError(
        "evo_ape not found for the current eval Python environment.\n"
        "current Python executable: {executable}\n"
        "current Python prefix: {prefix}\n"
        "checked candidate paths: {checked}\n"
        "PATH: {path}\n"
        "recommended install command: {executable} -m pip install evo".format(
            executable=sys.executable,
            prefix=sys.prefix,
            checked=checked,
            path=os.environ.get("PATH", ""),
        )
    )


def _evo_executable(evo_cmd):
    if not evo_cmd:
        return None
    if len(evo_cmd) >= 3 and evo_cmd[1:3] == ["-m", "evo.main_ape"]:
        return "{} -m evo.main_ape".format(evo_cmd[0])
    return evo_cmd[0]


def _run_evo_ape(evo_cmd, name, gt_traj, est_traj, result_zip, out_dir, t_max_diff):
    cmd = list(evo_cmd) + [
        "tum",
        str(gt_traj),
        str(est_traj),
        "-a",
        "-s",
        "--t_max_diff",
        str(t_max_diff),
        "--save_results",
        str(result_zip),
    ]
    log_prog("evo_ape {} vs GT".format(name.upper()))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        _write_failure_log(out_dir, name, cmd, completed)
        raise RuntimeError(
            "evo_ape failed for {name}: return code {code}; stderr: {stderr}; output directory: {out_dir}".format(
                name=name,
                code=completed.returncode,
                stderr=_stderr_summary(completed.stderr),
                out_dir=Path(out_dir).resolve(),
            )
        )


def _safe_delta_metrics(ori_rmse, gen_rmse, warnings):
    if ori_rmse is None or gen_rmse is None:
        return None, None, None
    delta = float(gen_rmse) - float(ori_rmse)
    if float(ori_rmse) == 0.0:
        warnings.append("ORI APE RMSE is zero; RMSE percentage increase and ratio are undefined")
        return delta, None, None
    return delta, delta / float(ori_rmse) * 100.0, float(gen_rmse) / float(ori_rmse)


def _apply_plot_result(summary, plot_result):
    plot_result = plot_result if isinstance(plot_result, dict) else {}
    summary["plot_status"] = str(plot_result.get("plot_status") or "skipped")
    summary["trajectory_overlay_path"] = plot_result.get("trajectory_overlay_path")
    summary["selected_plot_plane"] = plot_result.get("selected_plot_plane")
    summary["gt_plot_mode"] = plot_result.get("gt_plot_mode")
    summary["plot_common_start"] = plot_result.get("plot_common_start")
    summary["plot_common_end"] = plot_result.get("plot_common_end")
    for warning in list(plot_result.get("warnings") or []):
        if warning not in summary["warnings"]:
            summary["warnings"].append(str(warning))


def run_evo_eval(config):
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    exp_dir = Path(_get_arg(config, "exp_dir", out_dir.parent)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_traj = ensure_file(_get_arg(config, "gt_traj"), "ground truth trajectory")
    ori_traj = ensure_file(_get_arg(config, "ori_traj"), "ORI trajectory")
    gen_traj = ensure_file(_get_arg(config, "gen_traj"), "GEN trajectory")
    skip_plots = bool(_get_arg(config, "skip_plots", False))
    t_max_diff = float(_get_arg(config, "t_max_diff", T_MAX_DIFF))

    evo_cmd = _resolve_evo_ape()

    ori_zip = out_dir / "ori" / "evo_ape.zip"
    gen_zip = out_dir / "gen" / "evo_ape.zip"
    ori_zip.parent.mkdir(parents=True, exist_ok=True)
    gen_zip.parent.mkdir(parents=True, exist_ok=True)
    warnings = []

    _run_evo_ape(evo_cmd, "ori", gt_traj, ori_traj, ori_zip, out_dir, t_max_diff)
    _run_evo_ape(evo_cmd, "gen", gt_traj, gen_traj, gen_zip, out_dir, t_max_diff)

    ori_stats, ori_pose_pairs = _extract_stats(ori_zip, "ori", warnings)
    gen_stats, gen_pose_pairs = _extract_stats(gen_zip, "gen", warnings)
    ori_rmse = ori_stats.get("rmse")
    gen_rmse = gen_stats.get("rmse")
    delta, increase_pct, ratio = _safe_delta_metrics(ori_rmse, gen_rmse, warnings)

    summary = {
        "version": 1,
        "source": "exphub.eval.evo_eval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_source": "evo_ape",
        "evo_command": list(evo_cmd),
        "evo_executable": _evo_executable(evo_cmd),
        "eval_python_executable": sys.executable,
        "eval_python_prefix": sys.prefix,
        "alignment": "sim3",
        "align": True,
        "correct_scale": True,
        "t_max_diff": t_max_diff,
        "pose_relation": "translation_part",
        "gt_path": str(gt_traj),
        "ori_path": str(ori_traj),
        "gen_path": str(gen_traj),
        "ori_result_zip": _relative_path(exp_dir, ori_zip),
        "gen_result_zip": _relative_path(exp_dir, gen_zip),
        "ori_ape_rmse": ori_rmse,
        "gen_ape_rmse": gen_rmse,
        "rmse_delta_gen_minus_ori": delta,
        "rmse_increase_pct": increase_pct,
        "rmse_ratio_gen_over_ori": ratio,
        "ori_pose_pairs": ori_pose_pairs,
        "gen_pose_pairs": gen_pose_pairs,
        "ori_stats": ori_stats,
        "gen_stats": gen_stats,
        "plot_status": "skipped",
        "trajectory_overlay_path": None,
        "selected_plot_plane": None,
        "gt_plot_mode": None,
        "plot_common_start": None,
        "plot_common_end": None,
        "status": "success",
        "warnings": warnings,
    }

    if skip_plots:
        summary["warnings"].append("trajectory overlay skipped by --skip_plots/--no_viz")
    else:
        try:
            from exphub.eval.trajectory_plot import generate_trajectory_overlay

            _apply_plot_result(
                summary,
                generate_trajectory_overlay(
                    out_dir=out_dir,
                    exp_dir=exp_dir,
                    gt_path=gt_traj,
                    ori_path=ori_traj,
                    gen_path=gen_traj,
                    t_max_diff=t_max_diff,
                    ori_pose_pairs=ori_pose_pairs,
                    gen_pose_pairs=gen_pose_pairs,
                ),
            )
        except Exception as exc:
            message = "trajectory overlay skipped: {}".format(exc)
            summary["warnings"].append(message)
            log_warn(message)

    summary_path = out_dir / "evo_summary.json"
    write_json_atomic(summary_path, summary, indent=2)
    log_info("evo summary: {}".format(summary_path))
    return {"summary_path": summary_path, "summary": summary, "out_dir": out_dir}


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--exp_dir", required=False)
    parser.add_argument("--gt_traj", required=True)
    parser.add_argument("--ori_traj", required=True)
    parser.add_argument("--gen_traj", required=True)
    parser.add_argument("--t_max_diff", type=float, default=T_MAX_DIFF)
    parser.add_argument("--skip_plots", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    run_evo_eval(vars(args))


if __name__ == "__main__":
    main()
