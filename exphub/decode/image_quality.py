from __future__ import annotations

import csv
import io
import inspect
import math
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import urlparse

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic, write_text_atomic


BATCH_SIZE = 16
METRICS = ["lpips", "ssim", "fid"]
RESIZE_POLICY = "resize_generated_to_original_prepared_frame_size"
FID_INPUT_POLICY = "torchmetrics_uint8_nchw_0_255_normalize_false"
SMALL_FID_SAMPLE_WARNING = "FID may be unstable because evaluated frame count is below 50."
OPTIONAL_PACKAGE_HINT = "lpips, scikit-image, torchmetrics, torch-fidelity"
MISSING_INPUT_HINT = (
    "decode image quality requires prepared frames and merged generated frames. "
    "Please run prepare/decode first."
)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _metric_stats(values):
    items = [float(item) for item in list(values or [])]
    if not items:
        raise RuntimeError("cannot summarize empty metric values")
    items_sorted = sorted(items)
    count = len(items_sorted)
    mean = sum(items_sorted) / float(count)
    variance = sum((item - mean) ** 2 for item in items_sorted) / float(count)
    mid = count // 2
    if count % 2:
        median = items_sorted[mid]
    else:
        median = (items_sorted[mid - 1] + items_sorted[mid]) / 2.0
    return {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "median": float(median),
        "min": float(items_sorted[0]),
        "max": float(items_sorted[-1]),
    }


def _import_optional_dependencies(python_executable=None):
    missing = []
    modules = {}
    for name, import_name in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("lpips", "lpips"),
        ("scikit-image", "skimage"),
        ("torchmetrics", "torchmetrics"),
        ("torch-fidelity", "torch_fidelity"),
    ):
        try:
            modules[name] = __import__(import_name)
        except Exception:
            missing.append(name)

    try:
        from skimage.metrics import structural_similarity

        modules["structural_similarity"] = structural_similarity
    except Exception:
        if "scikit-image" not in missing:
            missing.append("scikit-image")

    try:
        from skimage.transform import resize

        modules["resize"] = resize
    except Exception:
        if "scikit-image" not in missing:
            missing.append("scikit-image")

    try:
        from skimage.io import imread

        modules["imread"] = imread
    except Exception:
        if "scikit-image" not in missing:
            missing.append("scikit-image")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        modules["FrechetInceptionDistance"] = FrechetInceptionDistance
    except Exception:
        if "torchmetrics" not in missing:
            missing.append("torchmetrics")

    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        python_text = str(python_executable or "").strip() or "<unknown>"
        raise RuntimeError(
            "decode image quality was executed with:\n{}\n\n"
            "Missing optional packages:\n{}\n\n"
            "Install them in the decode Python environment, or configure "
            "environments.phases.decode.python correctly. This optional evaluation "
            "is only required when --decode_image_quality is enabled.".format(python_text, missing_text)
        )

    return modules


def _resolve_device(torch, requested):
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("decode image quality requested cuda, but torch.cuda.is_available() is false")
    if value not in ("cuda", "cpu"):
        raise RuntimeError("--decode_image_quality_device must be one of auto, cuda, cpu")
    return value


def _frame_stem_index(path):
    stem = Path(path).stem
    if not stem.isdigit():
        return None
    try:
        return int(stem)
    except Exception:
        return None


def _collect_original_frames(prepare_result, prepare_frames_dir):
    frame_dir = ensure_dir(prepare_frames_dir, "prepare frames dir")
    frame_count = int(_as_dict(prepare_result).get("num_frames", 0) or 0)
    if frame_count <= 0:
        raise RuntimeError("prepare_result.num_frames must be > 0 for decode image quality")

    frames = {}
    for path in list_frames_sorted(frame_dir):
        idx = _frame_stem_index(path)
        if idx is not None and 0 <= idx < frame_count:
            frames[int(idx)] = Path(path).resolve()
    return frames


def _explicit_index_map(merge_report):
    candidates = []
    for parent_key in ("frame_index_map", "index_map"):
        raw = _as_dict(merge_report.get(parent_key))
        candidates.append(raw)
    for parent_key in ("outputs", "artifacts", "summary"):
        raw = _as_dict(_as_dict(merge_report.get(parent_key)).get("frame_index_map"))
        candidates.append(raw)

    for raw in candidates:
        if not raw:
            continue
        pairs = raw.get("generated_to_prepared") or raw.get("decode_to_prepared") or raw.get("output_to_prepared")
        if isinstance(pairs, list) and pairs:
            out = {}
            for item in pairs:
                if isinstance(item, dict):
                    gen = item.get("generated_index", item.get("output_index", item.get("decode_index")))
                    prep = item.get("prepared_index", item.get("frame_index"))
                    if gen is not None and prep is not None:
                        out[int(gen)] = int(prep)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out[int(item[0])] = int(item[1])
            if out:
                return out

        prepared_indices = raw.get("prepared_indices") or raw.get("prepared_frame_indices")
        generated_indices = raw.get("generated_indices") or raw.get("output_indices") or raw.get("decode_indices")
        if isinstance(prepared_indices, list) and isinstance(generated_indices, list) and len(prepared_indices) == len(generated_indices):
            return {int(gen): int(prep) for gen, prep in zip(generated_indices, prepared_indices)}
    return {}


def _collect_generated_frames(merge_report, decode_frames_dir):
    frame_dir = ensure_dir(decode_frames_dir, "decode frames dir")
    generated_by_output = {}
    for path in list_frames_sorted(frame_dir):
        idx = _frame_stem_index(path)
        if idx is not None:
            generated_by_output[int(idx)] = Path(path).resolve()
    if not generated_by_output:
        return {}, "continuous_from_zero", 0

    explicit = _explicit_index_map(_as_dict(merge_report))
    if explicit:
        frames = {}
        for output_idx, path in generated_by_output.items():
            if int(output_idx) in explicit:
                frames[int(explicit[int(output_idx)])] = path
        return frames, "decode_merge_report_explicit_index_map", len(generated_by_output)

    summary = _as_dict(_as_dict(merge_report).get("summary"))
    if summary.get("merged_start_idx") is not None:
        start_idx = int(summary.get("merged_start_idx"))
        return (
            {int(start_idx) + int(output_idx): path for output_idx, path in generated_by_output.items()},
            "continuous_from_merged_start_idx",
            len(generated_by_output),
        )

    return {int(output_idx): path for output_idx, path in generated_by_output.items()}, "continuous_from_zero", len(generated_by_output)


def _sample_pairs(original_frames, generated_frames, stride, max_frames):
    common_indices = sorted(set(original_frames.keys()).intersection(set(generated_frames.keys())))
    sampled_indices = common_indices[:: int(stride)]
    if int(max_frames) > 0:
        sampled_indices = sampled_indices[: int(max_frames)]
    return common_indices, [
        {
            "frame_index": int(idx),
            "original_path": original_frames[int(idx)],
            "generated_path": generated_frames[int(idx)],
        }
        for idx in sampled_indices
    ]


def _read_rgb_uint8(path, deps):
    np = deps["numpy"]
    arr = deps["imread"](str(path))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim != 3:
        raise RuntimeError("expected image with 2 or 3 dimensions: {}".format(path))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] != 3:
        raise RuntimeError("expected RGB-compatible image channels for {}".format(path))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_generated_to_original(original, generated, deps):
    np = deps["numpy"]
    if generated.shape == original.shape:
        return np.ascontiguousarray(generated)
    resized = deps["resize"](
        generated,
        (int(original.shape[0]), int(original.shape[1]), int(original.shape[2])),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return np.ascontiguousarray(np.clip(resized, 0, 255).astype(np.uint8))


def _load_pair_images(pair, deps):
    original = _read_rgb_uint8(pair["original_path"], deps)
    generated = _read_rgb_uint8(pair["generated_path"], deps)
    generated = _resize_generated_to_original(original, generated, deps)
    if original.shape != generated.shape:
        raise RuntimeError(
            "image size mismatch after resize for frame {}: original={} generated={}".format(
                int(pair["frame_index"]),
                original.shape,
                generated.shape,
            )
        )
    return original, generated


def _lpips_tensor(images, deps, device):
    torch = deps["torch"]
    np = deps["numpy"]
    stacked = np.stack(images, axis=0).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).to(device)
    return tensor


def _fid_tensor(images, deps, device):
    torch = deps["torch"]
    np = deps["numpy"]
    stacked = np.stack(images, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().to(device)
    return tensor


@contextmanager
def _block_downloads(torch):
    original_download = getattr(torch.hub, "download_url_to_file", None)
    original_load_state = getattr(torch.hub, "load_state_dict_from_url", None)
    fidelity_module = sys.modules.get("torch_fidelity.feature_extractor_inceptionv3")
    original_fidelity_load_state = getattr(fidelity_module, "load_state_dict_from_url", None) if fidelity_module is not None else None

    def blocked_download(*args, **kwargs):
        raise RuntimeError(
            "FID/LPIPS weight asset is not available in the local cache and network downloads are disabled. "
            "Image quality evaluation is optional and only required when --decode_image_quality is enabled."
        )

    def cached_only_load_state_dict_from_url(url, model_dir=None, *args, **kwargs):
        filename = str(kwargs.get("file_name") or os.path.basename(urlparse(str(url)).path))
        cache_dir = Path(model_dir).expanduser() if model_dir else Path(torch.hub.get_dir()).expanduser() / "checkpoints"
        cached_path = cache_dir / filename
        if not cached_path.is_file():
            raise RuntimeError(
                "FID/LPIPS weight asset is not available in the local cache and network downloads are disabled: "
                "{}. Image quality evaluation is optional and only required when --decode_image_quality is enabled.".format(
                    cached_path
                )
            )
        return original_load_state(url, model_dir=model_dir, *args, **kwargs)

    if original_download is not None:
        torch.hub.download_url_to_file = blocked_download
    if original_load_state is not None:
        torch.hub.load_state_dict_from_url = cached_only_load_state_dict_from_url
    if fidelity_module is not None and original_fidelity_load_state is not None:
        fidelity_module.load_state_dict_from_url = cached_only_load_state_dict_from_url
    try:
        yield
    finally:
        if original_download is not None:
            torch.hub.download_url_to_file = original_download
        if original_load_state is not None:
            torch.hub.load_state_dict_from_url = original_load_state
        if fidelity_module is not None and original_fidelity_load_state is not None:
            fidelity_module.load_state_dict_from_url = original_fidelity_load_state


def _compute_metrics(pairs, deps, device):
    torch = deps["torch"]
    lpips_module = deps["lpips"]
    structural_similarity = deps["structural_similarity"]
    FrechetInceptionDistance = deps["FrechetInceptionDistance"]

    lpips_values = []
    ssim_values = []
    rows = []
    fid_input_policy = FID_INPUT_POLICY

    try:
        with _block_downloads(torch):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                lpips_model = lpips_module.LPIPS(net="alex").to(device)
                lpips_model.eval()
            fid_kwargs = {"feature": 2048}
            try:
                signature = inspect.signature(FrechetInceptionDistance)
                if "normalize" in signature.parameters:
                    fid_kwargs["normalize"] = False
            except Exception:
                fid_kwargs["normalize"] = False
            if "normalize" not in fid_kwargs:
                fid_input_policy = "torchmetrics_uint8_nchw_0_255_legacy_default"
            try:
                fid = FrechetInceptionDistance(**fid_kwargs).to(device)
            except TypeError:
                if "normalize" not in fid_kwargs:
                    raise
                fid_kwargs.pop("normalize", None)
                fid_input_policy = "torchmetrics_uint8_nchw_0_255_legacy_default"
                fid = FrechetInceptionDistance(**fid_kwargs).to(device)
    except Exception as exc:
        raise RuntimeError(
            "failed to initialize decode image quality metrics. Optional packages/assets are required: {}. "
            "Network downloads are disabled; install/cache LPIPS and FID assets before enabling "
            "--decode_image_quality. Original error: {}".format(OPTIONAL_PACKAGE_HINT, exc)
        ) from exc

    for start in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[start : start + BATCH_SIZE]
        originals = []
        generated = []
        for pair in batch_pairs:
            original_image, generated_image = _load_pair_images(pair, deps)
            originals.append(original_image)
            generated.append(generated_image)
            ssim_value = structural_similarity(original_image, generated_image, channel_axis=2, data_range=255)
            ssim_values.append(float(ssim_value))

        with torch.no_grad():
            original_lpips = _lpips_tensor(originals, deps, device)
            generated_lpips = _lpips_tensor(generated, deps, device)
            batch_lpips = lpips_model(original_lpips, generated_lpips).detach().flatten().cpu().tolist()
            lpips_values.extend(float(item) for item in batch_lpips)

            original_fid = _fid_tensor(originals, deps, device)
            generated_fid = _fid_tensor(generated, deps, device)
            fid.update(original_fid, real=True)
            fid.update(generated_fid, real=False)

        del originals, generated
        try:
            del original_lpips, generated_lpips, original_fid, generated_fid
        except Exception:
            pass
        if str(device) == "cuda":
            torch.cuda.empty_cache()

        for pair, lpips_value, ssim_value in zip(batch_pairs, batch_lpips, ssim_values[-len(batch_pairs) :]):
            rows.append(
                {
                    "frame_index": int(pair["frame_index"]),
                    "original_path": pair["original_path"],
                    "generated_path": pair["generated_path"],
                    "lpips": float(lpips_value),
                    "ssim": float(ssim_value),
                }
            )

    try:
        with _block_downloads(torch):
            with torch.no_grad():
                fid_value = float(fid.compute().detach().cpu().item())
    except Exception as exc:
        raise RuntimeError(
            "failed to compute FID. torchmetrics/torch-fidelity assets must be installed and cached locally; "
            "image quality evaluation is optional and only required when --decode_image_quality is enabled. "
            "Original error: {}".format(exc)
        ) from exc

    return {
        "lpips": _metric_stats(lpips_values),
        "ssim": _metric_stats(ssim_values),
        "fid": fid_value,
        "fid_input_policy": fid_input_policy,
        "rows": rows,
    }


def _write_details_csv(path, rows, exp_dir):
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "original_path", "generated_path", "lpips", "ssim"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_index": int(row["frame_index"]),
                    "original_path": _relative_path(exp_dir, row["original_path"]),
                    "generated_path": _relative_path(exp_dir, row["generated_path"]),
                    "lpips": "{:.10f}".format(float(row["lpips"])),
                    "ssim": "{:.10f}".format(float(row["ssim"])),
                }
            )
    tmp_path.replace(resolved)


def _format_metric(value):
    try:
        return "{:.6f}".format(float(value))
    except Exception:
        return "nan"


def _write_summary(path, report):
    warnings = list(report.get("warnings") or [])
    text = "\n".join(
        [
            "[Image Quality]",
            "matched_frames: {}".format(int(report.get("frame_count_matched", 0) or 0)),
            "evaluated_frames: {}".format(int(report.get("frame_count_evaluated", 0) or 0)),
            "device: {}".format(str(report.get("device", ""))),
            "stride: {}".format(int(report.get("stride", 1) or 1)),
            "max_frames: {}".format(int(report.get("max_frames", 0) or 0)),
            "LPIPS mean: {}".format(_format_metric(_as_dict(report.get("lpips")).get("mean"))),
            "SSIM mean: {}".format(_format_metric(_as_dict(report.get("ssim")).get("mean"))),
            "FID: {}".format(_format_metric(report.get("fid"))),
            "warnings: {}".format("; ".join(str(item) for item in warnings) if warnings else ""),
            "",
        ]
    )
    write_text_atomic(path, text)


def _resolve_decode_frames_dir(run_root, decode_merge_report_path, merge_report):
    base = Path(run_root).resolve()
    report_dir = Path(decode_merge_report_path).resolve().parent
    for parent_key in ("outputs", "artifacts"):
        raw = str(_as_dict(merge_report.get(parent_key)).get("frames_dir", "") or "").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_dir():
            return candidate.resolve()
    return (report_dir / "frames").resolve()


def run_image_quality_evaluation(
    run_root,
    prepare_result_path,
    decode_merge_report_path,
    output_report_path,
    output_summary_path,
    output_details_csv_path,
    stride=1,
    max_frames=0,
    device="auto",
    python_executable=None,
    execution_mode="subprocess_decode_python",
):
    run_root_path = Path(run_root).resolve()
    prepare_result_path = Path(prepare_result_path).resolve()
    decode_merge_report_path = Path(decode_merge_report_path).resolve()
    output_report_path = Path(output_report_path).resolve()
    output_summary_path = Path(output_summary_path).resolve()
    output_details_csv_path = Path(output_details_csv_path).resolve()

    for path, label in (
        (prepare_result_path, "prepare result"),
        (decode_merge_report_path, "decode merge report"),
    ):
        try:
            ensure_file(path, label)
        except RuntimeError as exc:
            raise RuntimeError("{} Missing {}: {}".format(MISSING_INPUT_HINT, label, path)) from exc

    prepare_frames_dir = (prepare_result_path.parent / "frames").resolve()
    try:
        ensure_dir(prepare_frames_dir, "prepare frames dir")
    except RuntimeError as exc:
        raise RuntimeError("{} {}".format(MISSING_INPUT_HINT, exc)) from exc

    deps = _import_optional_dependencies(python_executable=python_executable)
    device = _resolve_device(deps["torch"], device)
    stride = int(stride)
    max_frames = int(max_frames)

    prepare_result = read_json_dict(prepare_result_path)
    merge_report = read_json_dict(decode_merge_report_path)
    if not prepare_result or not merge_report:
        raise RuntimeError("{} Invalid prepare or decode merge report.".format(MISSING_INPUT_HINT))

    decode_frames_dir = _resolve_decode_frames_dir(run_root_path, decode_merge_report_path, merge_report)
    try:
        ensure_dir(decode_frames_dir, "decode frames dir")
    except RuntimeError as exc:
        raise RuntimeError("{} {}".format(MISSING_INPUT_HINT, exc)) from exc

    original_frames = _collect_original_frames(prepare_result, prepare_frames_dir)
    generated_frames, frame_index_mode, generated_file_count = _collect_generated_frames(merge_report, decode_frames_dir)
    common_indices, pairs = _sample_pairs(original_frames, generated_frames, stride, max_frames)

    if not common_indices:
        raise RuntimeError("decode image quality matched zero frames between prepared and generated outputs")
    if not pairs:
        raise RuntimeError("decode image quality evaluated zero frames after stride/max_frames sampling")

    warnings = []
    if len(pairs) < 50:
        warnings.append(SMALL_FID_SAMPLE_WARNING)

    metrics = _compute_metrics(pairs, deps, device)
    _write_details_csv(output_details_csv_path, metrics["rows"], run_root_path)

    report = {
        "enabled": True,
        "python_executable": str(python_executable or ""),
        "execution_mode": str(execution_mode or "subprocess_decode_python"),
        "device": str(device),
        "metrics": list(METRICS),
        "stride": int(stride),
        "max_frames": int(max_frames),
        "batch_size": int(BATCH_SIZE),
        "frame_count_original": int(len(original_frames)),
        "frame_count_generated": int(generated_file_count),
        "frame_count_matched": int(len(common_indices)),
        "frame_count_evaluated": int(len(pairs)),
        "frame_index_mode": str(frame_index_mode),
        "resize_policy": RESIZE_POLICY,
        "fid_input_policy": str(metrics.get("fid_input_policy") or FID_INPUT_POLICY),
        "lpips": metrics["lpips"],
        "ssim": metrics["ssim"],
        "fid": float(metrics["fid"]),
        "warnings": warnings,
        "outputs": {
            "report": _relative_path(run_root_path, output_report_path),
            "summary": _relative_path(run_root_path, output_summary_path),
            "details_csv": _relative_path(run_root_path, output_details_csv_path),
        },
    }
    _write_summary(output_summary_path, report)
    write_json_atomic(output_report_path, report, indent=2)
    return report
