from __future__ import annotations

from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict
from exphub.common.logging import debug_info, runtime_info
from exphub.contracts import slam as slam_contract


def _run_track(runtime, track_name, segment_path):
    dst_dir = runtime.paths.slam_track_dir(track_name)
    runtime.remove_in_exp(dst_dir)
    runtime_info("slam run={}".format(track_name))

    cmd = [
        "python",
        str(runtime.script_path("slam_droid.py")),
        "--segment_dir",
        str(segment_path),
        "--droid_repo",
        str(runtime.args.droid_repo),
        "--weights",
        str(runtime.args.droid_weights),
        "--out_dir",
        str(runtime.paths.exp_dir),
        "--slam_out_dir",
        str(dst_dir),
        "--fps",
        runtime.fps_arg,
        "--undistort_mode",
        "auto",
        "--resize_interp",
        "linear",
        "--intr_scale_mode",
        "demo",
    ]
    if not runtime.viz_enable:
        cmd.append("--disable_vis")

    runtime.step_runner.run_env_python(cmd, phase_name="slam", log_name="slam_{}.log".format(track_name), cwd=runtime.exphub_root)
    ensure_file(runtime.paths.slam_traj_path(track_name), "slam trajectory")
    ensure_file(runtime.paths.slam_run_meta_path(track_name), "slam run meta")

    meta_obj = read_json_dict(runtime.paths.slam_run_meta_path(track_name))
    tum_meta = Path(str(meta_obj.get("tum_path", ""))).resolve() if meta_obj.get("tum_path") else None
    npz_meta = Path(str(meta_obj.get("npz_path", ""))).resolve() if meta_obj.get("npz_path") else None
    tum_expect = runtime.paths.slam_traj_path(track_name).resolve()
    npz_expect = runtime.paths.slam_npz_path(track_name).resolve()
    if tum_meta != tum_expect or npz_meta != npz_expect:
        raise RuntimeError(
            "slam run_meta path mismatch for track={}: tum={} npz={}".format(track_name, tum_meta, npz_meta)
        )
    debug_info("[OK] slam {} saved: {}".format(track_name, runtime.paths.slam_traj_path(track_name)))


def run(runtime):
    """Formal slam stage entry; still bridges to the existing DROID script."""
    contract = slam_contract.build_contract(runtime.paths)
    seq = runtime.args.droid_seq
    if seq == "auto":
        seq = "both"

    if seq in ("ori", "both"):
        ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")
    if seq in ("gen", "both"):
        ensure_dir(runtime.paths.merge_frames_dir, "merge frames dir")

    if seq == "ori":
        _run_track(runtime, "ori", runtime.paths.segment_dir)
    elif seq == "gen":
        _run_track(runtime, "gen", runtime.paths.merge_dir)
    else:
        _run_track(runtime, "ori", runtime.paths.segment_dir)
        _run_track(runtime, "gen", runtime.paths.merge_dir)

    return contract.root
