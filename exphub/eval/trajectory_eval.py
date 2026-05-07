from __future__ import annotations

import argparse
import datetime
import os
import tempfile
from pathlib import Path

import numpy as np

from exphub.common.io import write_json_atomic
from exphub.common.logging import log_info, log_warn


_GT_ASSOC_MAX_DIFF_SEC = 0.03
_MIN_COMMON_MATCHED_SAMPLES = 3
_GT_ALIGNMENT_MODE = "sim3"
_STAT_KEYS = ["rmse", "mean", "median", "max", "min", "std"]
_PLOT_DPI = 220


def _get_arg(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _resolve_path(value):
    return Path(value).expanduser().resolve()


def _empty_stats(status="failed", matched_samples=0):
    payload = dict((key, None) for key in _STAT_KEYS)
    payload["matched_samples"] = int(matched_samples)
    payload["status"] = str(status)
    payload["alignment_mode"] = _GT_ALIGNMENT_MODE
    payload["scale"] = None
    return payload


def _append_warning(metrics_obj, message):
    text = str(message or "").strip()
    if not text:
        return
    warnings_list = metrics_obj.setdefault("warnings", [])
    if text not in warnings_list:
        warnings_list.append(text)
    log_warn(text)


def _write_report(out_dir, metrics_obj):
    path = Path(out_dir).resolve() / "eval_traj_report.json"
    write_json_atomic(path, metrics_obj, indent=2)
    return path


def _timestamp_range(timestamps):
    arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return {
            "samples": int(arr.size),
            "timestamp_min": None,
            "timestamp_max": None,
            "duration_sec": None,
        }
    t_min = float(np.min(finite))
    t_max = float(np.max(finite))
    return {
        "samples": int(arr.size),
        "timestamp_min": t_min,
        "timestamp_max": t_max,
        "duration_sec": float(t_max - t_min),
    }


def _format_range_diag(label, diag):
    item = dict(diag or {})
    return "{} samples={} range=[{}, {}]".format(
        label,
        int(item.get("samples") or 0),
        item.get("timestamp_min"),
        item.get("timestamp_max"),
    )


def _timestamp_diag_text(timestamp_diagnostics):
    diag = dict(timestamp_diagnostics or {})
    return "; ".join(
        [
            _format_range_diag("gt", diag.get("gt")),
            _format_range_diag("ori", diag.get("ori")),
            _format_range_diag("gen", diag.get("gen")),
        ]
    )


def _read_tum(path_obj):
    path = Path(path_obj).resolve()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise RuntimeError("malformed TUM trajectory row: {}:{} expected 8 columns".format(path, line_no))
            try:
                values = [float(item) for item in parts[:8]]
            except Exception as exc:
                raise RuntimeError("malformed TUM trajectory row: {}:{} non-numeric value".format(path, line_no)) from exc
            if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
                raise RuntimeError("malformed TUM trajectory row: {}:{} non-finite value".format(path, line_no))
            rows.append(values)
    if not rows:
        raise RuntimeError("empty TUM trajectory: {}".format(path))

    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[order]
    return {
        "path": str(path),
        "timestamps": arr[:, 0].copy(),
        "xyz": arr[:, 1:4].copy(),
        "quat_xyzw": arr[:, 4:8].copy(),
    }


def _nearest_index(timestamps, target_ts):
    arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return None, None
    pos = int(np.searchsorted(arr, float(target_ts)))
    candidates = []
    if pos < arr.shape[0]:
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best_idx = None
    best_diff = None
    for idx in candidates:
        diff = abs(float(arr[idx]) - float(target_ts))
        if best_diff is None or diff < best_diff:
            best_idx = int(idx)
            best_diff = float(diff)
    return best_idx, best_diff


def _associate_to_gt(gt_ts, est_ts, max_diff_sec):
    matches = []
    for gt_idx, timestamp in enumerate(np.asarray(gt_ts, dtype=np.float64).reshape(-1)):
        est_idx, diff = _nearest_index(est_ts, timestamp)
        if est_idx is None or diff is None:
            matches.append(None)
            continue
        if float(diff) <= float(max_diff_sec):
            matches.append(int(est_idx))
        else:
            matches.append(None)
    return matches


def _align_sim3_to_gt(est_xyz, gt_xyz):
    """Align estimate points into the GT frame: aligned = scale * (R @ est.T).T + t."""
    est = np.asarray(est_xyz, dtype=np.float64)
    gt = np.asarray(gt_xyz, dtype=np.float64)
    if est.ndim != 2 or gt.ndim != 2 or est.shape != gt.shape or est.shape[1] != 3:
        raise RuntimeError("Sim3 alignment requires matching Nx3 point arrays")
    if est.shape[0] < _MIN_COMMON_MATCHED_SAMPLES:
        raise RuntimeError("Sim3 alignment requires at least {} samples".format(_MIN_COMMON_MATCHED_SAMPLES))
    if not np.all(np.isfinite(est)) or not np.all(np.isfinite(gt)):
        raise RuntimeError("Sim3 alignment requires finite point arrays")

    est_center = np.mean(est, axis=0)
    gt_center = np.mean(gt, axis=0)
    est_zero = est - est_center
    gt_zero = gt - gt_center
    source_variance = float(np.mean(np.sum(est_zero * est_zero, axis=1)))
    if not np.isfinite(source_variance) or source_variance <= 1e-12:
        raise RuntimeError("Sim3 alignment source variance is too small")
    covariance = (gt_zero.T @ est_zero) / float(est.shape[0])
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except Exception as exc:
        raise RuntimeError("Sim3 alignment SVD failed: {}".format(exc)) from exc
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Sim3 alignment scale must be finite and positive")
    translation = gt_center - scale * (rotation @ est_center)
    aligned = scale * (rotation @ est.T).T + translation
    return aligned, scale, rotation, translation


def _stats_from_errors(errors, scale=None):
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return _empty_stats(status="failed", matched_samples=0)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "min": float(np.min(finite)),
        "std": float(np.std(finite)),
        "matched_samples": int(finite.size),
        "status": "success",
        "alignment_mode": _GT_ALIGNMENT_MODE,
        "scale": float(scale) if scale is not None else None,
    }


def _build_overview(ori_stats, gen_stats, ori_scale=None, gen_scale=None):
    ori_rmse = ori_stats.get("rmse")
    gen_rmse = gen_stats.get("rmse")
    delta = None
    ratio = None
    better = "tie"
    if ori_rmse is not None and gen_rmse is not None:
        delta = float(gen_rmse) - float(ori_rmse)
        if abs(delta) <= 1e-12:
            better = "tie"
        elif delta < 0.0:
            better = "gen"
        else:
            better = "ori"
        if abs(float(ori_rmse)) > 1e-12:
            ratio = float(gen_rmse) / float(ori_rmse)
    return {
        "ori_vs_gt_rmse": ori_rmse,
        "gen_vs_gt_rmse": gen_rmse,
        "rmse_delta_gen_minus_ori": delta,
        "rmse_ratio_gen_over_ori": ratio,
        "better": better,
        "ori_scale": float(ori_scale) if ori_scale is not None else None,
        "gen_scale": float(gen_scale) if gen_scale is not None else None,
    }


def _base_report(gt_path, ori_path, gen_path):
    return {
        "version": 2,
        "source": "eval.trajectory_eval",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "gt_path": str(Path(gt_path).resolve()),
        "ori_path": str(Path(ori_path).resolve()),
        "gen_path": str(Path(gen_path).resolve()),
        "association": {
            "mode": "gt_nearest",
            "t_max_diff_sec": float(_GT_ASSOC_MAX_DIFF_SEC),
            "gt_samples": 0,
            "ori_matched_samples": 0,
            "gen_matched_samples": 0,
            "common_matched_samples": 0,
        },
        "alignment": {
            "mode": _GT_ALIGNMENT_MODE,
            "target": "gt",
            "applied_to": ["ori", "gen"],
            "ori_scale": None,
            "gen_scale": None,
            "ori": {},
            "gen": {},
        },
        "timestamp_diagnostics": {
            "gt": _timestamp_range([]),
            "ori": _timestamp_range([]),
            "gen": _timestamp_range([]),
        },
        "comparisons": {
            "ori_vs_gt": _empty_stats(),
            "gen_vs_gt": _empty_stats(),
        },
        "overview": _build_overview(_empty_stats(), _empty_stats()),
        "plot_path": "eval/eval_traj_xy.png",
        "warnings": [],
    }


def _setup_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="exphub_mpl_")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_xy(out_dir, gt_xyz, ori_xyz, gen_xyz, ori_rmse, gen_rmse):
    plt = _setup_matplotlib()
    plot_path = Path(out_dir).resolve() / "eval_traj_xy.png"

    gt = np.asarray(gt_xyz, dtype=np.float64)
    ori = np.asarray(ori_xyz, dtype=np.float64)
    gen = np.asarray(gen_xyz, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.2, 6.6), dpi=_PLOT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#d9dee5", linewidth=0.8, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#c7cfd8")
        spine.set_linewidth(0.8)

    ax.plot(gt[:, 0], gt[:, 1], color="#1f4e79", linewidth=2.4, label="GT")
    ax.plot(
        ori[:, 0],
        ori[:, 1],
        color="#c56a2d",
        linewidth=2.0,
        label="ORI Sim3 vs GT RMSE = {:.4f} m".format(float(ori_rmse)),
    )
    ax.plot(
        gen[:, 0],
        gen[:, 1],
        color="#2f7d5c",
        linewidth=2.0,
        label="GEN Sim3 vs GT RMSE = {:.4f} m".format(float(gen_rmse)),
    )
    ax.scatter(gt[0, 0], gt[0, 1], color="#1f4e79", s=38, marker="o", edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(gt[-1, 0], gt[-1, 1], color="#1f4e79", s=48, marker="D", edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_title("Trajectory vs GT (Sim3 aligned)", fontsize=13, color="#1f2a35", pad=10)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.06, y=0.08)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#c7cfd8", framealpha=0.96, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return plot_path


def _comparison_records(comparisons):
    rows = []
    for name in ["ori_vs_gt", "gen_vs_gt"]:
        item = dict((comparisons or {}).get(name) or {})
        rows.append(
            {
                "comparison": name,
                "alignment_mode": item.get("alignment_mode"),
                "scale": item.get("scale"),
                "rmse": item.get("rmse"),
                "mean": item.get("mean"),
                "median": item.get("median"),
                "min": item.get("min"),
                "max": item.get("max"),
                "std": item.get("std"),
                "matched_samples": item.get("matched_samples"),
            }
        )
    return rows


def run_trajectory_eval(config):
    out_dir = _resolve_path(_get_arg(config, "out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = _resolve_path(_get_arg(config, "gt_traj"))
    ori_path = _resolve_path(_get_arg(config, "ori_traj"))
    gen_path = _resolve_path(_get_arg(config, "gen_traj"))
    metrics_obj = _base_report(gt_path, ori_path, gen_path)

    for path_obj, label in [(gt_path, "ground truth trajectory"), (ori_path, "ori trajectory"), (gen_path, "gen trajectory")]:
        if not path_obj.is_file():
            _append_warning(metrics_obj, "missing {}: {}".format(label, path_obj))
            _write_report(out_dir, metrics_obj)
            raise RuntimeError("missing {}: {}".format(label, path_obj))

    try:
        log_info("eval load TUM trajectories")
        gt = _read_tum(gt_path)
        ori = _read_tum(ori_path)
        gen = _read_tum(gen_path)
    except Exception as exc:
        _append_warning(metrics_obj, "failed to load TUM trajectories: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        raise

    metrics_obj["timestamp_diagnostics"] = {
        "gt": _timestamp_range(gt["timestamps"]),
        "ori": _timestamp_range(ori["timestamps"]),
        "gen": _timestamp_range(gen["timestamps"]),
    }

    log_info("eval associate ORI/GEN to GT timestamps")
    ori_matches = _associate_to_gt(gt["timestamps"], ori["timestamps"], _GT_ASSOC_MAX_DIFF_SEC)
    gen_matches = _associate_to_gt(gt["timestamps"], gen["timestamps"], _GT_ASSOC_MAX_DIFF_SEC)
    common = [
        (gt_idx, int(ori_idx), int(gen_idx))
        for gt_idx, (ori_idx, gen_idx) in enumerate(zip(ori_matches, gen_matches))
        if ori_idx is not None and gen_idx is not None
    ]

    metrics_obj["association"] = {
        "mode": "gt_nearest",
        "t_max_diff_sec": float(_GT_ASSOC_MAX_DIFF_SEC),
        "gt_samples": int(len(gt["timestamps"])),
        "ori_matched_samples": int(sum(1 for idx in ori_matches if idx is not None)),
        "gen_matched_samples": int(sum(1 for idx in gen_matches if idx is not None)),
        "common_matched_samples": int(len(common)),
    }

    if len(common) < _MIN_COMMON_MATCHED_SAMPLES:
        message = (
            "too few common GT timestamp matches: common_matched_samples={} min_required={} {}; "
            "check whether GT/ORI/GEN timestamp domains are consistent"
        ).format(len(common), _MIN_COMMON_MATCHED_SAMPLES, _timestamp_diag_text(metrics_obj["timestamp_diagnostics"]))
        _append_warning(metrics_obj, message)
        _write_report(out_dir, metrics_obj)
        raise RuntimeError(message)

    common_gt_indices = np.asarray([item[0] for item in common], dtype=np.int64)
    common_ori_indices = np.asarray([item[1] for item in common], dtype=np.int64)
    common_gen_indices = np.asarray([item[2] for item in common], dtype=np.int64)

    gt_common_xyz = gt["xyz"][common_gt_indices]
    ori_common_xyz = ori["xyz"][common_ori_indices]
    gen_common_xyz = gen["xyz"][common_gen_indices]

    log_info("eval Sim3 align ORI/GEN to GT")
    try:
        ori_aligned_xyz, ori_scale, ori_rotation, ori_translation = _align_sim3_to_gt(ori_common_xyz, gt_common_xyz)
        gen_aligned_xyz, gen_scale, gen_rotation, gen_translation = _align_sim3_to_gt(gen_common_xyz, gt_common_xyz)
    except Exception as exc:
        _append_warning(metrics_obj, "failed to align trajectories to GT: {}".format(exc))
        _write_report(out_dir, metrics_obj)
        raise

    ori_errors = np.linalg.norm(ori_aligned_xyz - gt_common_xyz, axis=1)
    gen_errors = np.linalg.norm(gen_aligned_xyz - gt_common_xyz, axis=1)
    ori_stats = _stats_from_errors(ori_errors, scale=ori_scale)
    gen_stats = _stats_from_errors(gen_errors, scale=gen_scale)
    metrics_obj["comparisons"] = {
        "ori_vs_gt": ori_stats,
        "gen_vs_gt": gen_stats,
    }
    metrics_obj["overview"] = _build_overview(ori_stats, gen_stats, ori_scale=ori_scale, gen_scale=gen_scale)
    metrics_obj["alignment"] = {
        "mode": _GT_ALIGNMENT_MODE,
        "target": "gt",
        "applied_to": ["ori", "gen"],
        "ori_scale": float(ori_scale),
        "gen_scale": float(gen_scale),
        "ori": {
            "rotation": ori_rotation.tolist(),
            "translation": ori_translation.tolist(),
        },
        "gen": {
            "rotation": gen_rotation.tolist(),
            "translation": gen_translation.tolist(),
        },
    }
    metrics_obj["eval_status"] = "success"

    plot_path = None
    if not bool(_get_arg(config, "skip_plots", False)):
        plot_path = _plot_xy(
            out_dir,
            gt_common_xyz,
            ori_aligned_xyz,
            gen_aligned_xyz,
            ori_stats["rmse"],
            gen_stats["rmse"],
        )
        metrics_obj["plot_path"] = str(plot_path)

    report_path = _write_report(out_dir, metrics_obj)
    log_info("eval trajectory report: {}".format(report_path))
    return {
        "metrics": metrics_obj,
        "overview": dict(metrics_obj["overview"]),
        "records": _comparison_records(metrics_obj["comparisons"]),
        "artifacts": {
            "traj_xy_plot": str(plot_path) if plot_path is not None else "",
        },
    }


def run_sim3_sanity_check():
    est = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [2.0, -0.4, 0.3],
            [2.7, 0.9, 0.8],
            [3.4, -1.1, 1.5],
            [4.2, 0.3, 2.1],
        ],
        dtype=np.float64,
    )
    known_scale = 2.35
    angle = 0.63
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotation = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([1.7, -3.2, 0.45], dtype=np.float64)
    gt = known_scale * (rotation @ est.T).T + translation

    aligned, recovered_scale, _recovered_rotation, _recovered_translation = _align_sim3_to_gt(est, gt)
    errors = np.linalg.norm(aligned - gt, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    if rmse >= 1e-9:
        raise RuntimeError("Sim3 sanity check failed: rmse={}".format(rmse))
    if not np.isfinite(recovered_scale) or recovered_scale <= 0.0:
        raise RuntimeError("Sim3 sanity check failed: recovered scale is not finite positive")
    if abs(float(recovered_scale) - float(known_scale)) >= 1e-9:
        raise RuntimeError(
            "Sim3 sanity check failed: recovered scale={} expected={}".format(
                recovered_scale,
                known_scale,
            )
        )
    return {
        "rmse": rmse,
        "known_scale": float(known_scale),
        "recovered_scale": float(recovered_scale),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-native-trajectory", action="store_true")
    parser.add_argument("--run-sim3-sanity", action="store_true")
    parser.add_argument("--exp_dir", required=False)
    parser.add_argument("--out_dir", required=False)
    parser.add_argument("--prepare_result", required=False)
    parser.add_argument("--generation_units", required=False)
    parser.add_argument("--gt_traj", required=False)
    parser.add_argument("--ori_traj", required=False)
    parser.add_argument("--gen_traj", required=False)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args(argv)
    if args.run_sim3_sanity:
        result = run_sim3_sanity_check()
        print(
            "sim3 sanity ok: rmse={:.12g} scale={:.12g}".format(
                float(result["rmse"]),
                float(result["recovered_scale"]),
            )
        )
        return
    if not args.run_native_trajectory:
        raise SystemExit("eval trajectory helper requires --run-native-trajectory")
    for name in ("out_dir", "gt_traj", "ori_traj", "gen_traj"):
        if not getattr(args, name, None):
            raise SystemExit("eval trajectory helper requires --{}".format(name))
    run_trajectory_eval(vars(args))


if __name__ == "__main__":
    main()
