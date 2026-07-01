from __future__ import annotations

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
from exphub.common.logging import log_prog, log_warn


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


def _display_track_label(name):
    return str(name or "").upper()


def _display_comparison_name(name):
    return str(name or "")


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


def _write_failure_log(out_dir, name, cmd, completed, filename="evo_failure.log"):
    path = Path(out_dir).resolve() / str(filename)
    lines = [
        "comparison: {}".format(_display_comparison_name(name)),
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


def _resolve_evo_tool(tool_name, env_name, module_name):
    checked = []
    env_value = str(os.environ.get(env_name) or "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        checked.append(str(candidate))
        if not candidate.exists():
            raise RuntimeError(
                "{tool} not found for the current eval Python environment.\n"
                "{env_name} is set but does not exist: {candidate}\n"
                "current Python executable: {executable}\n"
                "current Python prefix: {prefix}\n"
                "checked candidate paths: {checked}\n"
                "PATH: {path}\n"
                "recommended install command: {executable} -m pip install evo".format(
                    tool=tool_name,
                    env_name=env_name,
                    candidate=candidate,
                    executable=sys.executable,
                    prefix=sys.prefix,
                    checked=checked,
                    path=os.environ.get("PATH", ""),
                )
            )
        if not candidate.is_file():
            raise RuntimeError(
                "{tool} not found for the current eval Python environment.\n"
                "{env_name} is not a file: {candidate}\n"
                "current Python executable: {executable}\n"
                "current Python prefix: {prefix}\n"
                "checked candidate paths: {checked}\n"
                "PATH: {path}\n"
                "recommended install command: {executable} -m pip install evo".format(
                    tool=tool_name,
                    env_name=env_name,
                    candidate=candidate,
                    executable=sys.executable,
                    prefix=sys.prefix,
                    checked=checked,
                    path=os.environ.get("PATH", ""),
                )
            )
        return [str(candidate)]

    sibling = Path(sys.executable).with_name(tool_name)
    checked.append(str(sibling))
    if sibling.exists() and sibling.is_file():
        return [str(sibling)]

    path_candidate = shutil.which(tool_name)
    if path_candidate:
        checked.append(str(path_candidate))
        return [str(path_candidate)]
    checked.append("PATH:{}".format(tool_name))

    module_check = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    checked.append("{} -m {}".format(sys.executable, module_name))
    if module_check.returncode == 0:
        return [sys.executable, "-m", module_name]

    raise RuntimeError(
        "{tool} not found for the current eval Python environment.\n"
        "current Python executable: {executable}\n"
        "current Python prefix: {prefix}\n"
        "checked candidate paths: {checked}\n"
        "PATH: {path}\n"
        "recommended install command: {executable} -m pip install evo".format(
            tool=tool_name,
            executable=sys.executable,
            prefix=sys.prefix,
            checked=checked,
            path=os.environ.get("PATH", ""),
        )
    )


def _resolve_evo_ape():
    return _resolve_evo_tool("evo_ape", "EXPHUB_EVO_APE", "evo.main_ape")


def _evo_executable(evo_cmd):
    if not evo_cmd:
        return None
    if len(evo_cmd) >= 3 and evo_cmd[1] == "-m":
        return "{} -m {}".format(evo_cmd[0], evo_cmd[2])
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
    log_prog("evo_ape {} vs GT".format(_display_track_label(name)))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        _write_failure_log(out_dir, name, cmd, completed)
        raise RuntimeError(
            "evo_ape failed for {name}: return code {code}; stderr: {stderr}; output directory: {out_dir}".format(
                name=_display_comparison_name(name),
                code=completed.returncode,
                stderr=_stderr_summary(completed.stderr),
                out_dir=Path(out_dir).resolve(),
            )
        )
    return cmd


def _safe_delta(ori_rmse, rec_rmse):
    if ori_rmse is None or rec_rmse is None:
        return None
    return float(rec_rmse) - float(ori_rmse)


def _load_tum_trajectory(path_obj):
    from evo.tools import file_interface

    path = str(Path(path_obj).resolve())
    if hasattr(file_interface, "read_tum_trajectory_file"):
        return file_interface.read_tum_trajectory_file(path)
    if hasattr(file_interface, "load_tum_trajectory_file"):
        return file_interface.load_tum_trajectory_file(path)
    raise RuntimeError("No compatible evo TUM trajectory loader found")


def _positions(traj):
    import numpy as np

    value = getattr(traj, "positions_xyz", None)
    arr = np.asarray(value, dtype=np.float64) if value is not None else None
    if arr is None or arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    return arr[:, :3]


def _timestamps(traj):
    import numpy as np

    value = getattr(traj, "timestamps", None)
    arr = np.asarray(value, dtype=np.float64) if value is not None else None
    if arr is None or arr.ndim != 1 or arr.shape[0] == 0:
        return None
    return arr


def _path_length_for_window(gt_traj, start, end):
    import numpy as np

    positions = _positions(gt_traj)
    timestamps = _timestamps(gt_traj)
    if positions is None or timestamps is None:
        return None
    mask = (timestamps >= float(start)) & (timestamps <= float(end))
    if not np.any(mask):
        return None
    selected = positions[mask]
    if selected.shape[0] < 2:
        return 0.0
    deltas = np.diff(selected, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _associated_gt_window(gt_traj, ori_traj, rec_traj, t_max_diff, warnings):
    from evo.core import sync

    gt_ori, _ori_assoc = sync.associate_trajectories(gt_traj, ori_traj, max_diff=float(t_max_diff))
    gt_rec, _rec_assoc = sync.associate_trajectories(gt_traj, rec_traj, max_diff=float(t_max_diff))
    ori_ts = _timestamps(gt_ori)
    rec_ts = _timestamps(gt_rec)
    if ori_ts is None or rec_ts is None:
        warnings.append("gt path length unavailable: associated GT timestamps are missing")
        return None, None
    common_start = max(float(ori_ts[0]), float(rec_ts[0]))
    common_end = min(float(ori_ts[-1]), float(rec_ts[-1]))
    if common_end > common_start:
        return common_start, common_end
    fallback_start = min(float(ori_ts[0]), float(rec_ts[0]))
    fallback_end = max(float(ori_ts[-1]), float(rec_ts[-1]))
    if fallback_end > fallback_start:
        warnings.append("gt path length used associated timestamp span because common overlap was unavailable")
        return fallback_start, fallback_end
    warnings.append("gt path length unavailable: invalid associated timestamp window")
    return None, None


def _compute_gt_path_length(gt_path, ori_path, rec_path, t_max_diff, warnings):
    try:
        gt = _load_tum_trajectory(gt_path)
        ori = _load_tum_trajectory(ori_path)
        rec = _load_tum_trajectory(rec_path)
        start, end = _associated_gt_window(gt, ori, rec, t_max_diff, warnings)
        if start is None or end is None:
            return None
        length = _path_length_for_window(gt, start, end)
        if length is None:
            warnings.append("gt path length unavailable: GT trajectory has insufficient samples in eval window")
        return length
    except Exception as exc:
        warnings.append("gt path length unavailable: {}".format(exc))
        return None


def _reliability(summary):
    ape_ok = summary.get("ori_ape_rmse") is not None and summary.get("rec_ape_rmse") is not None
    if not ape_ok:
        return "FAIL"
    for key in ("ori_pose_pairs", "rec_pose_pairs"):
        value = summary.get(key)
        if value is None or int(value) < 2:
            return "WARN"
    if str(summary.get("plot_status") or "skipped").lower() != "success":
        return "WARN"
    if list(summary.get("warnings") or []):
        return "WARN"
    return "OK"


def _apply_plot_result(summary, plot_result):
    plot_result = plot_result if isinstance(plot_result, dict) else {}
    summary["plot_status"] = str(plot_result.get("plot_status") or "skipped")
    summary["trajectory_overlay_path"] = plot_result.get("trajectory_overlay_path")
    summary["trajectory_interactive_path"] = plot_result.get("trajectory_interactive_path")
    summary["trajectory_interactive_status"] = str(plot_result.get("trajectory_interactive_status") or "skipped_error")
    summary["trajectory_interactive_marker_count"] = int(plot_result.get("trajectory_interactive_marker_count", 0) or 0)
    summary["trajectory_interactive_unmatched_marker_count"] = int(
        plot_result.get("trajectory_interactive_unmatched_marker_count", 0) or 0
    )
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
    rec_traj = ensure_file(_get_arg(config, "rec_traj"), "REC trajectory")
    skip_plots = bool(_get_arg(config, "skip_plots", False))
    t_max_diff = float(_get_arg(config, "t_max_diff", T_MAX_DIFF))
    prepare_result = _get_arg(config, "prepare_result", "")
    generation_units = _get_arg(config, "generation_units", "")

    ape_cmd = _resolve_evo_ape()
    warnings = []

    ori_zip = out_dir / "ori" / "evo_ape.zip"
    rec_zip = out_dir / "rec" / "evo_ape.zip"
    ori_zip.parent.mkdir(parents=True, exist_ok=True)
    rec_zip.parent.mkdir(parents=True, exist_ok=True)
    ori_ape_command = _run_evo_ape(ape_cmd, "ori", gt_traj, ori_traj, ori_zip, out_dir, t_max_diff)
    rec_ape_command = _run_evo_ape(ape_cmd, "rec", gt_traj, rec_traj, rec_zip, out_dir, t_max_diff)

    ori_stats, ori_pose_pairs = _extract_stats(ori_zip, "ori", warnings)
    rec_stats, rec_pose_pairs = _extract_stats(rec_zip, "rec", warnings)
    ori_rmse = ori_stats.get("rmse")
    rec_rmse = rec_stats.get("rmse")
    delta = _safe_delta(ori_rmse, rec_rmse)

    gt_path_length_m = _compute_gt_path_length(gt_traj, ori_traj, rec_traj, t_max_diff, warnings)

    summary = {
        "version": 2,
        "source": "exphub.eval.evo_eval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_source": "evo",
        "ape_command": {
            "ori": list(ori_ape_command),
            "rec": list(rec_ape_command),
        },
        "ape_parameters": {
            "format": "tum",
            "align": True,
            "correct_scale": True,
            "t_max_diff": t_max_diff,
            "pose_relation": "trans_part",
        },
        "evo_command": list(ape_cmd),
        "evo_executable": _evo_executable(ape_cmd),
        "evo_ape_executable": _evo_executable(ape_cmd),
        "eval_python_executable": sys.executable,
        "eval_python_prefix": sys.prefix,
        "alignment": "sim3",
        "align": True,
        "correct_scale": True,
        "t_max_diff": t_max_diff,
        "pose_relation": "trans_part",
        "gt_path": str(gt_traj),
        "ori_path": str(ori_traj),
        "rec_path": str(rec_traj),
        "ori_result_zip": _relative_path(exp_dir, ori_zip),
        "rec_result_zip": _relative_path(exp_dir, rec_zip),
        "ori_ape_rmse": ori_rmse,
        "rec_ape_rmse": rec_rmse,
        "rmse_delta_rec_minus_ori": delta,
        "ori_pose_pairs": ori_pose_pairs,
        "rec_pose_pairs": rec_pose_pairs,
        "ori_stats": ori_stats,
        "rec_stats": rec_stats,
        "ape": {
            "ori": {
                "rmse": ori_rmse,
                "stats": ori_stats,
                "pose_pairs": ori_pose_pairs,
                "result_zip": _relative_path(exp_dir, ori_zip),
                "command": list(ori_ape_command),
            },
            "rec": {
                "rmse": rec_rmse,
                "stats": rec_stats,
                "pose_pairs": rec_pose_pairs,
                "result_zip": _relative_path(exp_dir, rec_zip),
                "command": list(rec_ape_command),
            },
            "delta_rec_minus_ori": delta,
        },
        "gt_path_length_m": gt_path_length_m,
        "plot_status": "skipped",
        "trajectory_overlay_path": None,
        "trajectory_interactive_path": None,
        "trajectory_interactive_status": "skipped_error",
        "trajectory_interactive_marker_count": 0,
        "trajectory_interactive_unmatched_marker_count": 0,
        "selected_plot_plane": None,
        "gt_plot_mode": None,
        "plot_common_start": None,
        "plot_common_end": None,
        "status": "success",
        "warnings": warnings,
    }
    if skip_plots:
        summary["warnings"].append("trajectory overlay skipped by skip_plots")
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
                    rec_path=rec_traj,
                    t_max_diff=t_max_diff,
                    ori_pose_pairs=ori_pose_pairs,
                    rec_pose_pairs=rec_pose_pairs,
                    prepare_result_path=prepare_result,
                    generation_units_path=generation_units,
                ),
            )
        except Exception as exc:
            message = "trajectory overlay skipped: {}".format(exc)
            summary["warnings"].append(message)
            log_warn(message)

    summary["eval_reliability"] = _reliability(summary)
    return {"summary": summary, "out_dir": out_dir}

def run_evo_eval_single_track(config):
    out_dir = Path(_get_arg(config, "out_dir")).resolve()
    exp_dir = Path(_get_arg(config, "exp_dir", out_dir.parent)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_traj = ensure_file(_get_arg(config, "gt_traj"), "ground truth trajectory")
    est_traj = ensure_file(_get_arg(config, "est_traj"), "estimated trajectory")
    method_key = str(_get_arg(config, "method_key", "method") or "method")
    display_name = str(_get_arg(config, "display_name", method_key) or method_key)
    t_max_diff = float(_get_arg(config, "t_max_diff", T_MAX_DIFF))

    warnings = []
    ape_cmd = _resolve_evo_ape()

    ape_zip = out_dir / "evo_ape.zip"
    ape_command = _run_evo_ape(ape_cmd, method_key, gt_traj, est_traj, ape_zip, out_dir, t_max_diff)
    ape_stats, ape_pose_pairs = _extract_stats(ape_zip, method_key, warnings)

    gt_path_length_m = _compute_gt_path_length(gt_traj, est_traj, est_traj, t_max_diff, warnings)
    summary = {
        "version": 1,
        "source": "exphub.eval.evo_eval.single_track",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_key": method_key,
        "display_name": display_name,
        "metric_source": "evo",
        "alignment": "sim3",
        "align": True,
        "correct_scale": True,
        "t_max_diff": float(t_max_diff),
        "pose_relation": "trans_part",
        "gt_path": str(gt_traj),
        "est_path": str(est_traj),
        "ape_rmse": ape_stats.get("rmse"),
        "ape_stats": ape_stats,
        "ape_pose_pairs": ape_pose_pairs,
        "ape_result_zip": _relative_path(exp_dir, ape_zip),
        "ape_command": list(ape_command),
        "gt_path_length_m": gt_path_length_m,
        "status": "success",
        "warnings": warnings,
    }
    summary_path = out_dir / "ape_summary.json"
    write_json_atomic(summary_path, summary, indent=2)
    return {"summary_path": summary_path, "summary": summary, "out_dir": out_dir}
