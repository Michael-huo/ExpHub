#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from _common import list_frames_sorted, log_info
from _eval.io import (
    append_warning,
    empty_stats,
    fmt_value,
    metric_stats,
    load_final_keyframe_context,
    read_json,
    read_timestamps,
    resolve_image_eval_inputs,
    write_csv,
    write_json,
    write_text,
)


_PLOT_DPI = 220
_FIG_FACE = "white"
_GRID_COLOR = "#d9dee5"
_SPINE_COLOR = "#c7cfd8"
_TEXT_COLOR = "#1f2a35"
_PSNR_COLOR = "#1f4e79"
_MSSSIM_COLOR = "#4d7f4b"
_LPIPS_COLOR = "#b35c2e"
_KEYFRAME_LINE_COLOR = "#b5bec8"
_KEYFRAME_LINE_ALPHA = 0.55
_KEYFRAME_LINE_WIDTH = 0.75
_KEYFRAME_LINE_STYLE = (0, (3.2, 3.2))


def _base_metrics(exp_dir):
    paths = resolve_image_eval_inputs(exp_dir)
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_status": "failed",
        "warnings": [],
        "frame_count": 0,
        "metric_units": {
            "psnr": "dB",
            "ms_ssim": "score",
            "lpips": "score",
        },
        "frame_index_base": None,
        "frame_index_source": "",
        "reference_frames_dir": str(paths["segment_frames_dir"]),
        "estimate_frames_dir": str(paths["merge_frames_dir"]),
        "timestamps_path": str(paths["merge_timestamps_path"]),
        "runs_plan_path": str(paths["runs_plan_path"]),
        "psnr": empty_stats(),
        "ms_ssim": empty_stats(),
        "lpips": empty_stats(),
    }


def _pairing_context(exp_dir, metrics_obj):
    paths = resolve_image_eval_inputs(exp_dir)
    ref_dir = Path(paths["segment_frames_dir"]).resolve()
    est_dir = Path(paths["merge_frames_dir"]).resolve()

    if not ref_dir.is_dir():
        append_warning(metrics_obj, "missing reference frames directory: {}".format(ref_dir))
        return None
    if not est_dir.is_dir():
        append_warning(metrics_obj, "missing estimate frames directory: {}".format(est_dir))
        return None

    ref_frames = list_frames_sorted(ref_dir)
    est_frames = list_frames_sorted(est_dir)
    if not ref_frames:
        append_warning(metrics_obj, "no reference frames found: {}".format(ref_dir))
        return None
    if not est_frames:
        append_warning(metrics_obj, "no estimate frames found: {}".format(est_dir))
        return None

    base_idx = 0
    source = "sequential_from_merge"
    plan_obj = read_json(paths["runs_plan_path"])
    if plan_obj is not None:
        try:
            base_idx = int(plan_obj.get("base_idx", 0))
            source = "infer/runs_plan.json"
            segments = list(plan_obj.get("segments") or [])
            if segments:
                last_end = int(segments[-1].get("end_idx", base_idx - 1))
                expected_count = int(last_end - base_idx + 1)
                if expected_count != len(est_frames):
                    append_warning(
                        metrics_obj,
                        "merge frame count mismatch with runs_plan: merge={} plan={}".format(
                            len(est_frames), expected_count
                        ),
                    )
        except Exception as exc:
            append_warning(metrics_obj, "failed to parse runs_plan for image eval: {}".format(exc))
    else:
        merge_meta = read_json(paths["merge_meta_path"])
        if merge_meta is not None and merge_meta.get("merged_start_idx") is not None:
            try:
                base_idx = int(merge_meta.get("merged_start_idx"))
                source = "merge/merge_meta.json"
                append_warning(
                    metrics_obj,
                    "image eval frame mapping fallback: using merge_meta merged_start_idx={}".format(base_idx),
                )
            except Exception as exc:
                append_warning(metrics_obj, "failed to parse merge_meta for image eval: {}".format(exc))
        else:
            append_warning(
                metrics_obj,
                "image eval frame mapping fallback: using merge sequence offset without runs_plan/merge_meta base index",
            )

    timestamps = read_timestamps(paths["merge_timestamps_path"])
    if not timestamps:
        append_warning(metrics_obj, "missing merge timestamps for image eval: {}".format(paths["merge_timestamps_path"]))

    metrics_obj["frame_index_base"] = int(base_idx)
    metrics_obj["frame_index_source"] = str(source)

    pairs = []
    missing_pairs = 0
    for seq_idx, est_path in enumerate(est_frames):
        frame_idx = int(base_idx + seq_idx)
        if frame_idx < 0 or frame_idx >= len(ref_frames):
            missing_pairs += 1
            continue
        timestamp = None
        if seq_idx < len(timestamps):
            timestamp = timestamps[seq_idx]
        pairs.append(
            {
                "seq_idx": int(seq_idx),
                "frame_idx": int(frame_idx),
                "timestamp": timestamp,
                "reference_path": str(ref_frames[frame_idx]),
                "estimate_path": str(est_path),
            }
        )
    if missing_pairs > 0:
        append_warning(metrics_obj, "skip {} frames without valid ori pairing".format(int(missing_pairs)))
    if not pairs:
        append_warning(metrics_obj, "no valid frame pairs available for image eval")
        return None
    return pairs


def _load_rgb_array(path):
    img = Image.open(str(path)).convert("RGB")
    return np.asarray(img, dtype=np.uint8).copy()


def _psnr_value(ref_arr, est_arr):
    ref_f = ref_arr.astype(np.float64)
    est_f = est_arr.astype(np.float64)
    mse = float(np.mean((ref_f - est_f) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def _init_torch_runtime(metrics_obj):
    try:
        import torch
        import torch.nn.functional as func
    except Exception as exc:
        append_warning(metrics_obj, "torch unavailable; skip MS-SSIM and LPIPS: {}".format(exc))
        return None
    return {
        "torch": torch,
        "func": func,
        "device": "cpu",
    }


def _tensor_from_uint8(runtime, arr):
    torch = runtime["torch"]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor.to(runtime["device"])


def _gaussian_window(runtime, channel, window_size, sigma):
    torch = runtime["torch"]
    coords = torch.arange(window_size, dtype=torch.float32, device=runtime["device"])
    coords = coords - float(window_size // 2)
    g = torch.exp(-(coords ** 2) / float(2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = torch.matmul(g.unsqueeze(1), g.unsqueeze(0))
    return kernel.unsqueeze(0).unsqueeze(0).expand(int(channel), 1, window_size, window_size).contiguous()


def _ssim_per_channel(runtime, x, y, data_range, window_size=11, sigma=1.5):
    func = runtime["func"]
    channel = int(x.size(1))
    window = _gaussian_window(runtime, channel, window_size, sigma)
    mu_x = func.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu_y = func.conv2d(y, window, padding=window_size // 2, groups=channel)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = func.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu_x_sq
    sigma_y_sq = func.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu_y_sq
    sigma_xy = func.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu_xy

    c1 = float((0.01 * data_range) ** 2)
    c2 = float((0.03 * data_range) ** 2)
    sigma_x_sq = runtime["torch"].clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = runtime["torch"].clamp(sigma_y_sq, min=0.0)

    cs_map = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = ((2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)) * cs_map
    dims = [1, 2, 3]
    return ssim_map.mean(dim=dims), cs_map.mean(dim=dims)


def _ms_ssim_value(runtime, ref_arr, est_arr):
    torch = runtime["torch"]
    func = runtime["func"]
    x = _tensor_from_uint8(runtime, ref_arr)
    y = _tensor_from_uint8(runtime, est_arr)

    weights_full = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    levels = 1
    min_side = int(min(x.size(2), x.size(3)))
    while levels < len(weights_full) and min_side >= 2:
        min_side = int((min_side + 1) // 2)
        levels += 1
        if min_side < 11:
            break
    weights = torch.tensor(weights_full[:levels], dtype=torch.float32, device=runtime["device"])

    mcs = []
    for level in range(levels):
        ssim_val, cs_val = _ssim_per_channel(runtime, x, y, data_range=1.0)
        ssim_val = torch.clamp(ssim_val, min=0.0, max=1.0)
        cs_val = torch.clamp(cs_val, min=0.0, max=1.0)
        if level < levels - 1:
            mcs.append(cs_val)
            x = func.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            y = func.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=True)
    if levels == 1:
        return float(ssim_val.mean().item())
    value = torch.ones_like(ssim_val)
    for level in range(levels - 1):
        value = value * (mcs[level] ** weights[level])
    value = value * (ssim_val ** weights[levels - 1])
    return float(value.mean().item())


def _init_lpips_model(runtime, metrics_obj):
    try:
        import lpips
    except Exception as exc:
        append_warning(metrics_obj, "LPIPS unavailable: {}".format(exc))
        return None
    try:
        model = lpips.LPIPS(net="alex")
        model = model.to(runtime["device"])
        model.eval()
    except Exception as exc:
        append_warning(metrics_obj, "LPIPS init failed: {}".format(exc))
        return None
    return model


def _lpips_value(runtime, model, ref_arr, est_arr):
    torch = runtime["torch"]
    ref_tensor = _tensor_from_uint8(runtime, ref_arr) * 2.0 - 1.0
    est_tensor = _tensor_from_uint8(runtime, est_arr) * 2.0 - 1.0
    with torch.no_grad():
        value = model(ref_tensor, est_tensor)
    return float(value.detach().cpu().view(-1)[0].item())


def _csv_rows_from_records(records):
    rows = []
    for item in records:
        timestamp = item.get("timestamp")
        rows.append(
            {
                "frame_idx": int(item["frame_idx"]),
                "timestamp": "" if timestamp is None else timestamp,
                "psnr": "" if item.get("psnr") is None else "{:.8f}".format(float(item["psnr"])),
                "ms_ssim": "" if item.get("ms_ssim") is None else "{:.8f}".format(float(item["ms_ssim"])),
                "lpips": "" if item.get("lpips") is None else "{:.8f}".format(float(item["lpips"])),
            }
        )
    return rows


def _setup_matplotlib():
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="exphub_mpl_")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax):
    ax.set_facecolor(_FIG_FACE)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#2f3b4a", labelsize=9.5)


def _draw_keyframe_vlines(ax, positions):
    if not positions:
        return
    for x_value in positions:
        ax.axvline(
            float(x_value),
            color=_KEYFRAME_LINE_COLOR,
            linewidth=_KEYFRAME_LINE_WIDTH,
            linestyle=_KEYFRAME_LINE_STYLE,
            alpha=_KEYFRAME_LINE_ALPHA,
            zorder=2.1,
        )


def _keyframe_frame_positions(frame_indices, keyframe_context):
    if not frame_indices or not isinstance(keyframe_context, dict):
        return []
    frame_set = set(int(value) for value in frame_indices)
    out = []
    seen = set()
    for value in list(keyframe_context.get("frame_indices") or []):
        try:
            frame_idx = int(value)
        except Exception:
            continue
        if frame_idx not in frame_set or frame_idx in seen:
            continue
        seen.add(frame_idx)
        out.append(frame_idx)
    return out


def _plot_metric_axis(ax, frame_indices, values, color, title, ylabel, unavailable_text, keyframe_xs=None):
    _style_axes(ax)
    ax.set_title(title, fontsize=11.5, color=_TEXT_COLOR, pad=6)
    ax.set_xlabel("Frame Index", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    if values:
        ax.plot(frame_indices, values, color=color, linewidth=1.7, zorder=3)
        _draw_keyframe_vlines(ax, keyframe_xs)
    else:
        ax.text(
            0.5,
            0.5,
            unavailable_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6a7480",
        )


def _write_plot(out_path, records, metrics_obj, keyframe_context=None):
    plt = _setup_matplotlib()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.4), dpi=_PLOT_DPI, sharex=False)
    fig.patch.set_facecolor(_FIG_FACE)
    frame_indices = [int(item["frame_idx"]) for item in records]
    keyframe_xs = _keyframe_frame_positions(frame_indices, keyframe_context)
    psnr_values = [float(item["psnr"]) for item in records if item.get("psnr") is not None]
    ms_ssim_values = [float(item["ms_ssim"]) for item in records if item.get("ms_ssim") is not None]
    lpips_values = [float(item["lpips"]) for item in records if item.get("lpips") is not None]

    _plot_metric_axis(
        axes[0],
        frame_indices[: len(psnr_values)],
        psnr_values,
        _PSNR_COLOR,
        "PSNR",
        "dB",
        "PSNR unavailable",
        keyframe_xs=keyframe_xs,
    )
    _plot_metric_axis(
        axes[1],
        frame_indices[: len(ms_ssim_values)],
        ms_ssim_values,
        _MSSSIM_COLOR,
        "MS-SSIM",
        "Score",
        "MS-SSIM unavailable",
        keyframe_xs=keyframe_xs,
    )
    lpips_unavailable = "LPIPS unavailable"
    warnings_text = "\n".join(metrics_obj.get("warnings", []) or [])
    if "LPIPS unavailable" in warnings_text or "LPIPS init failed" in warnings_text:
        lpips_unavailable = "LPIPS unavailable"
    _plot_metric_axis(
        axes[2],
        frame_indices[: len(lpips_values)],
        lpips_values,
        _LPIPS_COLOR,
        "LPIPS",
        "Score",
        lpips_unavailable,
        keyframe_xs=keyframe_xs,
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _update_status(metrics_obj):
    frame_count = int(metrics_obj.get("frame_count") or 0)
    available = []
    for key in ["psnr", "ms_ssim", "lpips"]:
        if metrics_obj[key].get("mean") is not None:
            available.append(key)
    if frame_count <= 0 or not available:
        metrics_obj["eval_status"] = "failed"
        return
    if len(available) == 3 and not metrics_obj.get("warnings"):
        metrics_obj["eval_status"] = "success"
        return
    metrics_obj["eval_status"] = "partial"


def write_image_outputs(out_dir, metrics_obj, records, keyframe_context=None):
    image_metrics_path = out_dir / "image_metrics.json"
    image_csv_path = out_dir / "image_per_frame.csv"
    image_plot_path = out_dir / "plots" / "image_metrics_curve.png"

    write_json(image_metrics_path, metrics_obj, indent=2)
    write_csv(image_csv_path, ["frame_idx", "timestamp", "psnr", "ms_ssim", "lpips"], _csv_rows_from_records(records))
    try:
        _write_plot(image_plot_path, records, metrics_obj, keyframe_context=keyframe_context)
    except Exception as exc:
        append_warning(metrics_obj, "failed to generate image plot: {}".format(exc))
        write_json(image_metrics_path, metrics_obj, indent=2)


def run_image_eval(exp_dir, out_dir):
    exp_root = Path(exp_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_obj = _base_metrics(exp_root)
    log_info("image eval start: exp_dir={} out_dir={}".format(exp_root, out_path))

    pairs = _pairing_context(exp_root, metrics_obj)
    records = []
    if pairs is not None:
        runtime = _init_torch_runtime(metrics_obj)
        lpips_model = _init_lpips_model(runtime, metrics_obj) if runtime is not None else None
        for pair in pairs:
            try:
                ref_arr = _load_rgb_array(pair["reference_path"])
                est_arr = _load_rgb_array(pair["estimate_path"])
            except Exception as exc:
                append_warning(
                    metrics_obj,
                    "failed to load frame pair idx={} ref={} est={}: {}".format(
                        pair["frame_idx"],
                        pair["reference_path"],
                        pair["estimate_path"],
                        exc,
                    ),
                )
                continue
            if ref_arr.shape != est_arr.shape:
                append_warning(
                    metrics_obj,
                    "skip frame idx={} due to image shape mismatch ref={} est={}".format(
                        pair["frame_idx"], ref_arr.shape, est_arr.shape
                    ),
                )
                continue

            record = {
                "seq_idx": int(pair["seq_idx"]),
                "frame_idx": int(pair["frame_idx"]),
                "timestamp": pair.get("timestamp"),
                "psnr": _psnr_value(ref_arr, est_arr),
                "ms_ssim": None,
                "lpips": None,
            }
            if runtime is not None:
                try:
                    record["ms_ssim"] = _ms_ssim_value(runtime, ref_arr, est_arr)
                except Exception as exc:
                    append_warning(metrics_obj, "MS-SSIM compute failed at frame {}: {}".format(pair["frame_idx"], exc))
                    runtime = None
            if runtime is not None and lpips_model is not None:
                try:
                    record["lpips"] = _lpips_value(runtime, lpips_model, ref_arr, est_arr)
                except Exception as exc:
                    append_warning(metrics_obj, "LPIPS compute failed at frame {}: {}".format(pair["frame_idx"], exc))
                    lpips_model = None
            records.append(record)

    metrics_obj["frame_count"] = int(len(records))
    metrics_obj["psnr"] = metric_stats([item.get("psnr") for item in records if item.get("psnr") is not None])
    metrics_obj["ms_ssim"] = metric_stats([item.get("ms_ssim") for item in records if item.get("ms_ssim") is not None])
    metrics_obj["lpips"] = metric_stats([item.get("lpips") for item in records if item.get("lpips") is not None])
    _update_status(metrics_obj)
    keyframe_context = load_final_keyframe_context([exp_root], metrics_obj=metrics_obj, warning_prefix="image plot keyframes")

    return {
        "metrics": metrics_obj,
        "records": records,
        "keyframe_context": keyframe_context,
    }
