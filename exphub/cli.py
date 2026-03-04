from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from .cleanup import apply_keep_level, normalize_keep_level
from .config import ConfigError, load_datasets_cfg, resolve_dataset
from .context import ExperimentContext
from .meta import sanitize_token, write_exp_meta
from .runner import RunnerConfig, StepRunner, conda_exec, detect_conda_base, RunError


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _run(msg: str) -> None:
    print(f"[RUN] {msg}")


def _step(msg: str) -> None:
    print(f"[STEP] {msg}")


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

    # Prompt manifest is a stable payload across keep levels.
    # (segment/clip_prompts.json is optional and may be pruned by cleanup.)
    manifest = ctx.prompt_manifest_path

    ori_n, ori_b = _sum_files(frames_dir, "*.png", follow_symlinks=True)
    kf_n, kf_b = _sum_files(keyframes_dir, "*.png", follow_symlinks=True)

    prompt_files = []
    if manifest.exists() and manifest.is_file():
        prompt_files.append(manifest)

    prompt_n = 0
    prompt_b = 0
    for f in prompt_files:
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


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exphub", add_help=True)

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
    ap.add_argument("--base_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43, dest="seed_base")
    ap.add_argument("--gpus", type=int, default=2)

    ap.add_argument("--infer_extra", default="", help="extra args passed to infer_i2v.py (quoted string)")

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

    ap.add_argument("--videox_root", default=os.environ.get("VIDEOX_ROOT", "/data/hx/VideoX-Fun"))
    ap.add_argument("--droid_repo", default=os.environ.get("DROID_REPO", "/data/hx/DROID-SLAM"))
    ap.add_argument("--droid_weights", default=os.environ.get("DROID_WEIGHTS", "droid.pth"))

    # Prompt generator is now managed under ExpHub/scripts.
    # Qwen2-VL model weights remain under /data/hx/Qwen2-VL-Prompt/models by default.
    ap.add_argument(
        "--qwen_model_dir",
        default=os.environ.get("QWEN_MODEL_DIR", "/data/hx/Qwen2-VL-Prompt/models/Qwen2-VL-7B-Instruct"),
        help="Qwen2-VL model dir used by prompt generator",
    )

    ap.add_argument("--ros_setup", default=os.environ.get("ROS_SETUP", "/opt/ros/noetic/setup.bash"))
    ap.add_argument("--sys_py", default=os.environ.get("SYS_PY", "/usr/bin/python3"), help="python used for segment step")

    # SLAM sequence selection.
    # Default is "both" so that `--mode slam` runs both ori/gen unless explicitly overridden.
    ap.add_argument("--droid_seq", default="both", choices=["auto", "ori", "gen", "both"])
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no_viz", action="store_true")

    args = ap.parse_args(argv)
    args.keep_level = normalize_keep_level(args.keep_level)

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

    # script paths
    scripts_dir = exphub_root / "scripts"
    seg_py = scripts_dir / "segment_make.py"
    infer_py = scripts_dir / "infer_i2v.py"
    merge_py = scripts_dir / "merge_seq.py"
    droid_py = scripts_dir / "slam_droid.py"
    stats_py = scripts_dir / "stats_collect.py"
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

    def _read_log_tail(log_path: Path, n: int) -> List[str]:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if n <= 0:
                return lines
            return lines[-n:]
        except Exception:
            return []

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
        if out_hint:
            _step(f"{step_name} done sec={sec:.2f} out={out_hint}")
        else:
            _step(f"{step_name} done sec={sec:.2f}")

    def step_doctor() -> int:
        _info("STEP doctor: begin")
        _info(f"DOCTOR EXPHUB={exphub_root}")
        _info(f"DOCTOR PYTHON={sys.version.splitlines()[0]}")
        _info(f"DOCTOR NOW={datetime.datetime.now().isoformat(timespec='seconds')}")
        _info(f"DOCTOR EXP_NAME={exp_name}")
        _info(f"DOCTOR EXP_DIR={exp_dir}")
        _info("DOCTOR modes=all,segment,prompt,stats,infer,merge,slam,eval,doctor")
        _info("DOCTOR layout=segment/,prompt/,infer/,merge/,slam/<track>/,eval/,stats/")

        has_critical_missing = False
        has_optional_missing = False

        _info("STEP doctor: check datasets config")
        cfg_ok = cfg_path.is_file()
        _info(f"DOCTOR datasets_cfg={cfg_path} exists={cfg_ok}")
        if not cfg_ok:
            has_critical_missing = True

        ds = None
        if cfg_ok:
            try:
                cfg_obj = load_datasets_cfg(cfg_path)
                ds = resolve_dataset(cfg_obj, exphub_root, dataset, sequence)
            except ConfigError as e:
                has_critical_missing = True
                _warn(f"DOCTOR dataset resolve failed: {e}")

        _info("STEP doctor: check dataset resolved fields")
        if ds is not None:
            _info(f"DOCTOR dataset={ds.dataset} sequence={ds.sequence}")
            _info(f"DOCTOR bag={ds.bag}")
            _info(f"DOCTOR topic={ds.topic}")
            _info(f"DOCTOR intrinsics=fx:{ds.fx} fy:{ds.fy} cx:{ds.cx} cy:{ds.cy}")
            _info(f"DOCTOR dist={ds.dist}")
            bag_ok = ds.bag.exists()
            _info(f"DOCTOR bag_exists={bag_ok}")
            if not bag_ok:
                has_critical_missing = True
        else:
            _warn("DOCTOR dataset/sequence unresolved")

        _info("STEP doctor: check scripts")
        must_scripts = [
            seg_py,
            prompt_gen_py,
            infer_py,
            merge_py,
            droid_py,
            stats_py,
        ]
        for script in must_scripts:
            ok = script.is_file()
            _info(f"DOCTOR script={script} exists={ok}")
            if not ok:
                has_critical_missing = True

        _info("STEP doctor: check external paths")
        optional_dirs = [
            ("videox_root", Path(args.videox_root)),
            ("droid_repo", Path(args.droid_repo)),
            ("qwen_model_dir", Path(args.qwen_model_dir)),
        ]
        for name, raw_path in optional_dirs:
            p = raw_path.expanduser().resolve()
            ok = p.is_dir()
            _info(f"DOCTOR optional_dir={name} path={p} is_dir={ok}")
            if not ok:
                has_optional_missing = True
                _warn(f"DOCTOR optional path missing: {name} ({p})")

        if args.auto_conda:
            _info("STEP doctor: check conda tools")
            if runner_cfg.conda_base is None:
                has_optional_missing = True
                _warn("DOCTOR conda base not found; skip env tool checks")
            else:
                conda_sh = runner_cfg.conda_base / "etc" / "profile.d" / "conda.sh"
                _info(f"DOCTOR conda_sh={conda_sh} exists={conda_sh.exists()}")
                if not conda_sh.exists():
                    has_optional_missing = True
                    _warn("DOCTOR conda.sh missing; skip env tool checks")
                else:
                    tool_checks = [
                        (args.conda_env_vlm, "python"),
                        (args.conda_env_videox, "python"),
                        (args.conda_env_droid, "evo_traj"),
                        (args.conda_env_droid, "evo_ape"),
                    ]
                    for env_name, tool_name in tool_checks:
                        _info(f"STEP doctor: env={env_name} which {tool_name}")
                        try:
                            rc = conda_exec(
                                ["which", tool_name],
                                env_name=env_name,
                                cfg=runner_cfg,
                                cwd=exphub_root,
                                check=False,
                            )
                            if rc == 0:
                                _info(f"DOCTOR env={env_name} tool={tool_name} ok")
                            else:
                                has_optional_missing = True
                                _warn(f"DOCTOR env={env_name} tool={tool_name} missing (rc={rc})")
                        except Exception as e:
                            has_optional_missing = True
                            _warn(f"DOCTOR env/tool check exception: env={env_name} tool={tool_name} ({e})")
        else:
            _info("STEP doctor: skip conda checks (--no_auto_conda)")

        if has_critical_missing:
            _warn("DOCTOR result=FAIL (critical missing)")
            return 2
        if has_optional_missing:
            _warn("DOCTOR result=PASS_WITH_WARN (optional missing)")
            return 0
        _info("DOCTOR result=PASS")
        return 0

    if args.mode == "doctor":
        _run(f"mode=doctor dataset={dataset} seq={sequence} exp_dir={exp_dir} log_level={args.log_level}")
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

    for p in [seg_py, infer_py, merge_py, droid_py, stats_py]:
        _ensure(p, "file")

    _ensure(prompt_gen_py, "file")

    _run(f"mode={args.mode} dataset={dataset} seq={sequence} exp_dir={exp_dir} log_level={args.log_level}")
    _info(f"EXPHUB={exphub_root}")
    _info(f"BAG={ds.bag}")
    _info(f"TOPIC={ds.topic}")

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
            _info(f"overwrite enabled: rm -rf {exp_dir}")
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
                "base_idx": args.base_idx,
                "seed_base": args.seed_base,
                "gpus": args.gpus,
                "droid_seq": args.droid_seq,
                "viz_enable": viz_enable,
                "keep_level": args.keep_level,
            },
            "paths": {
                "segment_dir": str(segment_dir),
                "videox_root": args.videox_root,
                "droid_repo": args.droid_repo,
            },
        }
        write_exp_meta(ctx.exp_meta_path, meta)

    def step_segment() -> None:
        ensure_clean_exp_dir()
        write_meta_snapshot()

        dist_args: List[str] = []
        if ds.dist:
            dist_args = ["--dist", *[str(x) for x in ds.dist]]

        cmd = [
            args.sys_py,
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
        num_segments = ctx.segment_count(base_idx=args.base_idx, requested_segments=0)

        cmd = [
            "python",
            str(prompt_gen_py),
            "--frames_dir",
            str(frames_dir),
            "--exp_dir",
            str(exp_dir),
            "--fps",
            fps_arg,
            "--kf_gap",
            str(kf_gap),
            "--base_idx",
            str(args.base_idx),
            "--num_segments",
            str(num_segments),
            "--model_dir",
            str(args.qwen_model_dir),
            "--out_json",
            str(ctx.segment_clip_prompts_path),
            "--out_manifest",
            str(ctx.prompt_manifest_path),
        ]

        step_runner.run_conda(cmd, env_name=args.conda_env_vlm, log_name="prompt.log", cwd=exphub_root)
        _ensure(ctx.prompt_manifest_path, "file")

    def step_stats() -> None:
        cmd = [
            "python",
            str(stats_py),
            "--exp_dir",
            str(exp_dir),
        ]
        step_runner.run_conda(cmd, env_name=args.conda_env_vlm, log_name="stats.log", cwd=exphub_root)
        _ensure(ctx.stats_report_path, "file")

    def step_infer() -> None:
        manifest = ctx.prompt_manifest_path
        if not manifest.is_file():
            _die(
                'missing prompt/manifest.json. Run "--mode prompt" first or provide a valid prompt manifest.'
            )

        exp_dir.mkdir(parents=True, exist_ok=True)
        _rm_in_exp(infer_dir)

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
            "--prompt_manifest",
            str(manifest),
        ]
        if args.infer_extra:
            import shlex as _sh
            cmd_infer.extend(_sh.split(args.infer_extra))
        step_runner.run_conda(cmd_infer, env_name=args.conda_env_videox, log_name="infer.log", cwd=exphub_root)
        _ensure(ctx.infer_runs_dir, "dir")
        _ensure(ctx.infer_runs_plan_path, "file")

    def step_merge() -> None:
        _ensure(segment_dir, "dir")
        _ensure(ctx.infer_runs_dir, "dir")
        _ensure(ctx.infer_runs_plan_path, "file")

        _rm_in_exp(merge_dir)

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
        step_runner.run_conda(cmd_merge, env_name=args.conda_env_videox, log_name="merge.log", cwd=exphub_root)

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
            _info(f"STEP slam: run={tag_name} segment_dir={seg_path}")

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

            step_runner.run_conda(cmd, env_name=args.conda_env_droid, log_name=f"slam_{tag_name}.log", cwd=exphub_root)
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
            _info(f"[OK] slam {tag_name} saved: {ctx.slam_traj_path(tag_name)}")

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
        # Run evo evaluation inside the DROID conda environment.
        # We must not use shutil.which() from the orchestrator process,
        # because the orchestrator may be running in a different Python/conda env.

        _info(f"STEP eval: env={args.conda_env_droid} viz={viz_enable}")

        _rm_tree(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        def _has_tool(tool: str) -> bool:
            # Check tool availability inside the droid env
            rc = step_runner.run_conda(
                ["bash", "-lc", f"command -v {tool} >/dev/null 2>&1"],
                env_name=args.conda_env_droid,
                check=False,
                log_name="eval.log",
                cwd=exphub_root,
            )
            return rc == 0

        if not _has_tool("evo_traj"):
            _warn("evo_traj not found in droid env; skip eval")
            return

        tum_ori = ctx.slam_traj_path("ori")
        tum_gen = ctx.slam_traj_path("gen")

        def _run_traj(name: str, tum: Path) -> None:
            _info(f"STEP eval: evo_traj {name} ({tum})")
            out_txt = ctx.eval_artifact_path("evo_traj_" + name + ".txt")
            if viz_enable:
                out_png = ctx.eval_artifact_path("traj_" + name + ".png")
                cmd = f"MPLBACKEND=Agg evo_traj tum {tum} -p --save_plot {out_png} 2>&1 | tee {out_txt}"
            else:
                cmd = f"evo_traj tum {tum} 2>&1 | tee {out_txt}"

            step_runner.run_conda(
                ["bash", "-lc", cmd],
                env_name=args.conda_env_droid,
                check=False,
                log_name="eval.log",
                cwd=exphub_root,
            )

        if tum_ori.exists():
            _run_traj("ori", tum_ori)
        else:
            _warn(f"missing ORI traj: {tum_ori}")

        if tum_gen.exists():
            _run_traj("gen", tum_gen)
        else:
            _warn(f"missing GEN traj: {tum_gen}")

        if tum_ori.exists() and tum_gen.exists() and _has_tool("evo_ape"):
            _info(f"STEP eval: evo_ape gen_vs_ori ({tum_gen} vs {tum_ori})")
            out_txt = ctx.eval_artifact_path("evo_ape_gen_vs_ori.txt")
            if viz_enable:
                out_png = ctx.eval_artifact_path("ape_gen_vs_ori.png")
                cmd = f"MPLBACKEND=Agg evo_ape tum {tum_ori} {tum_gen} -a -p --save_plot {out_png} 2>&1 | tee {out_txt}"
            else:
                cmd = f"evo_ape tum {tum_ori} {tum_gen} -a 2>&1 | tee {out_txt}"

            step_runner.run_conda(
                ["bash", "-lc", cmd],
                env_name=args.conda_env_droid,
                check=False,
                log_name="eval.log",
                cwd=exphub_root,
            )


    # Execute mode
    try:
        if args.mode == "segment":
            _run_step("segment", step_segment, str(segment_dir))
        elif args.mode == "prompt":
            _run_step("prompt", step_prompt, str(ctx.prompt_manifest_path))
        elif args.mode == "infer":
            _run_step("infer", step_infer, str(ctx.infer_runs_plan_path))
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
            _run_step("prompt", step_prompt, str(ctx.prompt_manifest_path))
            _run_step("infer", step_infer, str(ctx.infer_runs_plan_path))
            _run_step("merge", step_merge, str(merge_dir))
            _run_step("slam", step_slam, str(slam_root))
            _run_step("eval", step_eval, str(eval_dir))
            _run_step("stats", step_stats, str(ctx.stats_report_path))

        apply_keep_level(exp_dir, args.keep_level)

    except (ConfigError, RunError) as e:
        _die(str(e))

    _info(f"DONE. EXP_DIR={exp_dir}")


if __name__ == "__main__":
    main()
