from __future__ import annotations

import time

from exphub.common.io import ensure_file
from exphub.common.logging import log_prog
from exphub.config import get_platform_config
from exphub.decode import decode as decode_pipeline
from exphub.decode.image_quality_subprocess import run_decode_image_quality_subprocess
from exphub.encode import encode as encode_pipeline


IMAGE_QUALITY_STRIDE = 1
IMAGE_QUALITY_MAX_FRAMES = 0
IMAGE_QUALITY_DEVICE = "auto"


def _droid_config(runtime) -> tuple[str, str]:
    platform_cfg = get_platform_config(exphub_root=runtime.exphub_root)
    return (
        str(platform_cfg.get("repos", {}).get("droid_slam", "") or ""),
        str(platform_cfg.get("models", {}).get("droid", {}).get("path", "") or ""),
    )


def _run_compression_benchmark(runtime):
    encode_pipeline.run_compression_benchmark_encode_extra(runtime)
    decode_pipeline.run_compression_benchmark_decode_extra(runtime)
    ensure_file(runtime.paths.prepare_gt_traj_path, "prepared ground truth trajectory")
    droid_repo, droid_weights = _droid_config(runtime)
    cmd = [
        "-m",
        "exphub.eval.compression_benchmark",
        "--run-compression-benchmark-eval",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--out_dir",
        str(runtime.paths.eval_compression_dir),
        "--decode_benchmark_report",
        str(runtime.paths.decode_compression_report_path),
        "--main_eval_summary",
        str(runtime.paths.eval_canonical_summary_path),
        "--prepare_result",
        str(runtime.paths.prepare_result_path),
        "--prepare_frames_dir",
        str(runtime.paths.prepare_frames_dir),
        "--generation_units",
        str(runtime.paths.encode_generation_units_path),
        "--encode_result",
        str(runtime.paths.encode_result_path),
        "--decode_frames_dir",
        str(runtime.paths.decode_frames_dir),
        "--decode_calib",
        str(runtime.paths.decode_calib_path),
        "--decode_timestamps",
        str(runtime.paths.decode_timestamps_path),
        "--decode_report",
        str(runtime.paths.decode_report_path),
        "--gt_traj",
        str(runtime.paths.prepare_gt_traj_path),
        "--droid_repo",
        droid_repo,
        "--weights",
        droid_weights,
        "--fps",
        str(float(runtime.spec.fps)),
        "--clip_duration",
        str(runtime.spec.dur or ""),
        "--t_max_diff",
        "0.03",
        "--disable_vis",
    ]
    runtime.step_runner.run_env_python(
        cmd,
        phase_name="slam",
        log_name="compression_benchmark_eval.log",
        cwd=runtime.exphub_root,
    )
    return runtime.paths.eval_compression_dir


def run_requested_experiments(runtime, execution_plan):
    last_out = runtime.paths.exp_dir
    experiment_times = {}
    for experiment in tuple(execution_plan.experiments or ()):
        started = time.perf_counter()
        log_prog("experiment {} start".format(experiment))
        if experiment == "motion-benchmark":
            encode_pipeline.run_motion_benchmark_extra(runtime)
            last_out = runtime.paths.encode_motion_benchmark_dir
        elif experiment == "compression-benchmark":
            _run_compression_benchmark(runtime)
            ensure_file(runtime.paths.eval_compression_summary_path, "compression benchmark summary.json")
            ensure_file(runtime.paths.eval_compression_summary_csv_path, "compression benchmark summary.csv")
            last_out = runtime.paths.eval_compression_dir
        elif experiment == "image-quality":
            run_decode_image_quality_subprocess(
                runtime,
                stride=IMAGE_QUALITY_STRIDE,
                max_frames=IMAGE_QUALITY_MAX_FRAMES,
                device=IMAGE_QUALITY_DEVICE,
            )
            last_out = runtime.paths.decode_image_quality_dir
        else:
            raise RuntimeError("unsupported experiment: {}".format(experiment))
        experiment_times[str(experiment)] = float(time.perf_counter() - started)
        log_prog("experiment {} done".format(experiment))
    return last_out, experiment_times


__all__ = [
    "IMAGE_QUALITY_DEVICE",
    "IMAGE_QUALITY_MAX_FRAMES",
    "IMAGE_QUALITY_STRIDE",
    "run_requested_experiments",
]
