from __future__ import annotations

from exphub.common.io import ensure_file
from exphub.contracts import segment as segment_contract


def run(runtime):
    """Formal segment stage entry.

    Step 0 keeps the implementation as a thin bridge to the legacy script while
    making this service the unique formal entry used by the orchestrator.
    """
    contract = segment_contract.build_contract(runtime.paths)
    runtime.ensure_clean_exp_dir()
    runtime.write_meta_snapshot()
    dataset = runtime.dataset()
    segment_python = runtime.phase_python("segment")

    dist_args = []
    if dataset.dist:
        dist_args = ["--dist"] + [str(item) for item in dataset.dist]

    cmd = [
        segment_python,
        str(runtime.script_path("segment_make.py")),
        "--bag",
        str(dataset.bag),
        "--topic",
        dataset.topic,
        "--out_root",
        str(runtime.paths.exp_dir),
        "--name",
        "segment",
        "--duration",
        str(runtime.spec.dur),
        "--fps",
        runtime.fps_arg,
        "--kf_gap",
        str(runtime.spec.kf_gap),
        "--keyframes_mode",
        str(runtime.args.keyframes_mode),
        "--segment_policy",
        str(runtime.args.segment_policy),
        "--start_idx",
        str(runtime.args.start_idx),
        "--start_sec",
        str(runtime.spec.start_sec),
        "--width",
        str(runtime.spec.w),
        "--height",
        str(runtime.spec.h),
        "--fx",
        str(dataset.fx),
        "--fy",
        str(dataset.fy),
        "--cx",
        str(dataset.cx),
        "--cy",
        str(dataset.cy),
    ] + dist_args

    runtime.step_runner.run_ros(cmd, log_name="segment.log", cwd=runtime.exphub_root)

    ensure_file(contract.artifacts["calib"], "segment calib")
    ensure_file(contract.artifacts["timestamps"], "segment timestamps")
    ensure_file(contract.artifacts["preprocess_meta"], "segment preprocess meta")
    return contract.root
