from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
from PIL import Image

from exphub.common.io import list_frames_sorted
from .reporting import append_warning, empty_stats, metric_stats, read_json, read_timestamps, resolve_formal_eval_inputs


def _base_metrics(exp_dir):
    inputs = resolve_formal_eval_inputs(exp_dir)
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
        "reference_frames_dir": str(inputs["segment_frames_dir"]),
        "estimate_frames_dir": str(inputs["merge_frames_dir"]),
        "merge_manifest_path": str(inputs["merge_manifest_path"]),
        "runs_plan_path": str(inputs["runs_plan_path"]),
        "psnr": empty_stats(),
        "ms_ssim": empty_stats(),
        "lpips": empty_stats(),
    }


def _load_rgb_array(path_obj):
    image = Image.open(str(path_obj)).convert("RGB")
    return np.asarray(image, dtype=np.uint8).copy()


def _psnr_value(ref_arr, est_arr):
    ref_float = ref_arr.astype(np.float64)
    est_float = est_arr.astype(np.float64)
    mse = float(np.mean((ref_float - est_float) ** 2))
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
    tensor = runtime["torch"].from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
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
    ref_tensor = _tensor_from_uint8(runtime, ref_arr) * 2.0 - 1.0
    est_tensor = _tensor_from_uint8(runtime, est_arr) * 2.0 - 1.0
    with runtime["torch"].no_grad():
        value = model(ref_tensor, est_tensor)
    return float(value.detach().cpu().view(-1)[0].item())


def _resolve_pairs(exp_dir, metrics_obj):
    inputs = resolve_formal_eval_inputs(exp_dir)
    ref_dir = Path(inputs["segment_frames_dir"]).resolve()
    est_dir = Path(inputs["merge_frames_dir"]).resolve()
    if not ref_dir.is_dir():
        append_warning(metrics_obj, "missing reference frames directory: {}".format(ref_dir))
        return None
    if not est_dir.is_dir():
        append_warning(metrics_obj, "missing estimate frames directory: {}".format(est_dir))
        return None

    ref_frames = list_frames_sorted(ref_dir)
    est_frames = list_frames_sorted(est_dir)
    if not ref_frames or not est_frames:
        append_warning(metrics_obj, "image eval requires both segment/frames and merge/frames")
        return None

    merge_manifest = read_json(inputs["merge_manifest_path"]) or {}
    runs_plan = read_json(inputs["runs_plan_path"]) or {}
    base_idx = None
    source = ""
    if isinstance(merge_manifest.get("summary"), dict) and merge_manifest["summary"].get("merged_start_idx") is not None:
        base_idx = int(merge_manifest["summary"].get("merged_start_idx"))
        source = "merge/merge_manifest.json"
    elif runs_plan.get("base_idx") is not None:
        base_idx = int(runs_plan.get("base_idx"))
        source = "infer/runs_plan.json"
    else:
        base_idx = 0
        source = "sequential_fallback"
        append_warning(metrics_obj, "image eval using sequential frame pairing fallback")

    timestamps = read_timestamps(inputs["merge_timestamps_path"])
    metrics_obj["frame_index_base"] = int(base_idx)
    metrics_obj["frame_index_source"] = str(source)

    pairs = []
    skipped = 0
    for seq_idx, est_path in enumerate(est_frames):
        frame_idx = int(base_idx + seq_idx)
        if frame_idx < 0 or frame_idx >= len(ref_frames):
            skipped += 1
            continue
        pairs.append(
            {
                "seq_idx": int(seq_idx),
                "frame_idx": int(frame_idx),
                "timestamp": timestamps[seq_idx] if seq_idx < len(timestamps) else None,
                "reference_path": str(ref_frames[frame_idx]),
                "estimate_path": str(est_path),
            }
        )
    if skipped > 0:
        append_warning(metrics_obj, "skip {} frames without valid ori pairing".format(int(skipped)))
    if not pairs:
        append_warning(metrics_obj, "no valid frame pairs available for image eval")
        return None
    return pairs


def run_image_eval(exp_dir, out_dir):
    metrics_obj = _base_metrics(exp_dir)
    pairs = _resolve_pairs(exp_dir, metrics_obj)
    if not pairs:
        return {"metrics": metrics_obj, "records": []}

    runtime = _init_torch_runtime(metrics_obj)
    lpips_model = _init_lpips_model(runtime, metrics_obj) if runtime is not None else None

    psnr_values = []
    ms_ssim_values = []
    lpips_values = []
    records = []
    warned_ms_ssim = False
    warned_lpips = False

    for item in pairs:
        ref_arr = _load_rgb_array(item["reference_path"])
        est_arr = _load_rgb_array(item["estimate_path"])
        if ref_arr.shape != est_arr.shape:
            append_warning(
                metrics_obj,
                "skip frame_idx={} due to shape mismatch {} vs {}".format(
                    item["frame_idx"],
                    tuple(ref_arr.shape),
                    tuple(est_arr.shape),
                ),
            )
            continue

        psnr_value = _psnr_value(ref_arr, est_arr)
        ms_ssim_value = None
        if runtime is not None:
            try:
                ms_ssim_value = _ms_ssim_value(runtime, ref_arr, est_arr)
            except Exception as exc:
                if not warned_ms_ssim:
                    append_warning(metrics_obj, "MS-SSIM unavailable for current image geometry: {}".format(exc))
                    warned_ms_ssim = True
        lpips_value = None
        if runtime is not None and lpips_model is not None:
            try:
                lpips_value = _lpips_value(runtime, lpips_model, ref_arr, est_arr)
            except Exception as exc:
                if not warned_lpips:
                    append_warning(metrics_obj, "LPIPS unavailable for current image geometry: {}".format(exc))
                    warned_lpips = True

        psnr_values.append(float(psnr_value))
        if ms_ssim_value is not None:
            ms_ssim_values.append(float(ms_ssim_value))
        if lpips_value is not None:
            lpips_values.append(float(lpips_value))

        records.append(
            {
                "seq_idx": int(item["seq_idx"]),
                "frame_idx": int(item["frame_idx"]),
                "timestamp": item.get("timestamp"),
                "psnr": float(psnr_value),
                "ms_ssim": ms_ssim_value,
                "lpips": lpips_value,
            }
        )

    metrics_obj["frame_count"] = int(len(records))
    metrics_obj["psnr"] = metric_stats(psnr_values)
    metrics_obj["ms_ssim"] = metric_stats(ms_ssim_values)
    metrics_obj["lpips"] = metric_stats(lpips_values)
    metrics_obj["eval_status"] = "success" if records else "failed"
    return {
        "metrics": metrics_obj,
        "records": records,
    }
