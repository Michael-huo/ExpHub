from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_prog, log_warn
from exphub.contracts import merge as merge_contract


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _sha1_bytes(payload):
    return hashlib.sha1(payload).hexdigest()


def _guard_safe_out_dir(out_dir, exp_dir, runs_root):
    merge_root = (Path(exp_dir).resolve() / "merge").resolve()
    out_dir_resolved = Path(out_dir).resolve()
    runs_root_resolved = Path(runs_root).resolve()

    try:
        out_dir_resolved.relative_to(merge_root)
    except ValueError:
        raise RuntimeError(
            "unsafe merge out_dir outside expected merge scope: {} (merge_root={})".format(
                out_dir_resolved,
                merge_root,
            )
        )

    try:
        out_dir_resolved.relative_to(runs_root_resolved)
    except ValueError:
        return
    raise RuntimeError(
        "unsafe merge out_dir overlaps infer runs root: {} (runs_root={})".format(
            out_dir_resolved,
            runs_root_resolved,
        )
    )


def _load_runs_plan(plan_path):
    payload = json.loads(ensure_file(plan_path, "infer runs plan").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("invalid infer runs plan payload: {}".format(plan_path))

    segments = list(payload.get("segments") or [])
    if not segments:
        raise RuntimeError("infer runs plan contains zero segments: {}".format(plan_path))

    for idx, item in enumerate(segments):
        if not isinstance(item, dict):
            raise RuntimeError("invalid infer runs plan segment at index {}: not an object".format(idx))
        if not str(item.get("run_name", "") or "").strip():
            raise RuntimeError("invalid infer runs plan segment at index {}: missing run_name".format(idx))
        if item.get("start_idx") is None or item.get("end_idx") is None:
            raise RuntimeError("invalid infer runs plan segment at index {}: missing start_idx/end_idx".format(idx))
    return payload


def _resolve_output_fps(plan_obj, runs_root, fps_override):
    override = _safe_int(fps_override, 0)
    if override > 0:
        return override

    fps_plan = _safe_int(plan_obj.get("fps"), 0)
    if fps_plan > 0:
        return fps_plan

    segments = list(plan_obj.get("segments") or [])
    if segments:
        params_path = Path(runs_root).resolve() / str(segments[0].get("run_name")) / "params.json"
        if params_path.is_file():
            try:
                params_obj = json.loads(params_path.read_text(encoding="utf-8"))
                fps_run = _safe_int(params_obj.get("target_fps"), 0)
                if fps_run > 0:
                    return fps_run
            except Exception:
                pass
    return 25


def _try_ffmpeg_make_video(frames_dir, fps, out_mp4):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        str(int(fps)),
        "-i",
        str(Path(frames_dir).resolve() / "%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(Path(out_mp4).resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
        )
    except Exception:
        return False

    if proc.returncode != 0:
        details = ""
        if proc.stdout:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                details = " | ".join(lines[-3:])
        if details:
            log_warn("merge preview ffmpeg failed rc={} details={}".format(proc.returncode, details))
        else:
            log_warn("merge preview ffmpeg failed rc={}".format(proc.returncode))
        return False
    return Path(out_mp4).is_file()


def _python_make_video(frames_dir, fps, out_mp4):
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        return False

    frames = list_frames_sorted(frames_dir)
    if not frames:
        return False

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(str(out_mp4), fps=int(fps))
        try:
            for frame_path in frames:
                writer.append_data(imageio.imread(str(frame_path)))
        finally:
            writer.close()
    except Exception:
        return False
    return Path(out_mp4).is_file()


def _write_report(report_path, report_obj):
    payload = dict(report_obj or {})
    payload["report_path"] = str(Path(report_path).resolve())
    last_size = None
    for _ in range(3):
        write_json_atomic(report_path, payload, indent=2)
        report_bytes = Path(report_path).read_bytes()
        report_size = int(len(report_bytes))
        outputs = dict(payload.get("outputs", {}) or {})
        outputs["report_bytes_sum"] = report_size
        outputs["bytes_sum"] = int(
            int(outputs.get("report_bytes_sum", 0) or 0)
            + int(outputs.get("manifest_bytes_sum", 0) or 0)
        )
        payload["outputs"] = outputs
        payload["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, payload, indent=2)
    return Path(report_path).resolve()


def _run_formal_mainline(args):
    exp_dir = Path(args.exp_dir).resolve()
    segment_dir = ensure_dir(args.segment_dir, "segment dir")
    runs_root = ensure_dir(args.runs_root, "infer runs dir")
    plan_path = ensure_file(args.plan, "infer runs plan")
    out_dir = Path(args.out_dir).resolve()

    plan_obj = _load_runs_plan(plan_path)
    _guard_safe_out_dir(out_dir, exp_dir, runs_root)

    if out_dir.exists():
        shutil.rmtree(str(out_dir), ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frames = (out_dir / "frames").resolve()
    out_frames.mkdir(parents=True, exist_ok=True)

    segments = list(plan_obj.get("segments") or [])
    merged_start_idx = _safe_int(plan_obj.get("base_idx"), _safe_int(segments[0].get("start_idx"), 0))
    merged_end_idx = None
    merged_frame_count = 0
    merged_frame_bytes = 0
    prev_end = None
    merged_segments = []

    for seg_idx, item in enumerate(segments):
        run_name = str(item.get("run_name"))
        cur_start = _safe_int(item.get("start_idx"), 0)
        cur_end = _safe_int(item.get("end_idx"), 0)
        merged_end_idx = cur_end

        run_dir = (runs_root / run_name).resolve()
        frames_dir = ensure_dir(run_dir / "frames", "infer run frames")
        frames = list_frames_sorted(frames_dir)
        if not frames:
            raise RuntimeError("infer run contains zero frames: {}".format(frames_dir))

        skip = 0
        if prev_end is not None:
            overlap = int(prev_end - cur_start + 1)
            if overlap > 0:
                skip = overlap
        if skip >= len(frames):
            raise RuntimeError(
                "merge overlap invalid for run={}: skip={} frames={}".format(
                    run_name,
                    skip,
                    len(frames),
                )
            )

        out_start = merged_frame_count
        for src_path in frames[skip:]:
            dst_path = out_frames / "{:06d}.png".format(int(merged_frame_count))
            shutil.copy2(str(src_path), str(dst_path))
            try:
                merged_frame_bytes += int(dst_path.stat().st_size)
            except Exception:
                pass
            merged_frame_count += 1
        out_end = merged_frame_count - 1 if merged_frame_count > out_start else out_start - 1

        merged_segments.append(
            {
                "segment_index": int(seg_idx),
                "seg": item.get("seg"),
                "run_name": run_name,
                "start_idx": int(cur_start),
                "end_idx": int(cur_end),
                "source_frame_count": int(len(frames)),
                "skipped_overlap_frames": int(skip),
                "merged_frame_count": int(len(frames) - skip),
                "output_start_frame": int(out_start),
                "output_end_frame": int(out_end),
            }
        )
        prev_end = cur_end

    if merged_end_idx is None:
        raise RuntimeError("failed to resolve merged end index from infer runs plan")

    src_calib = ensure_file(segment_dir / "calib.txt", "segment calib")
    src_timestamps = ensure_file(segment_dir / "timestamps.txt", "segment timestamps")
    calib_path = (out_dir / "calib.txt").resolve()
    timestamps_path = (out_dir / "timestamps.txt").resolve()
    write_text_atomic(calib_path, src_calib.read_text(encoding="utf-8", errors="ignore"))

    timestamp_lines = [line for line in src_timestamps.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    expected_count = int(merged_end_idx - merged_start_idx + 1)
    if expected_count != merged_frame_count:
        raise RuntimeError(
            "merge frame count mismatch: merged={} expected={} start_idx={} end_idx={}".format(
                merged_frame_count,
                expected_count,
                merged_start_idx,
                merged_end_idx,
            )
        )
    if len(timestamp_lines) < merged_start_idx + merged_frame_count:
        raise RuntimeError(
            "segment timestamps too short for merged range: have={} need={}".format(
                len(timestamp_lines),
                merged_start_idx + merged_frame_count,
            )
        )

    sliced = timestamp_lines[merged_start_idx : merged_start_idx + merged_frame_count]
    try:
        t0 = float(sliced[0])
        out_lines = ["{:.9f}".format(float(item) - t0) for item in sliced]
    except Exception:
        out_lines = list(sliced)
    write_text_atomic(timestamps_path, "\n".join(out_lines) + "\n")

    output_fps = _resolve_output_fps(plan_obj, runs_root, args.fps)
    preview_path = (out_dir / "preview.mp4").resolve()
    preview_ok = False
    if not bool(args.no_preview) and merged_frame_count > 0:
        preview_ok = _try_ffmpeg_make_video(out_frames, output_fps, preview_path)
        if not preview_ok:
            preview_ok = _python_make_video(out_frames, output_fps, preview_path)

    manifest = {
        "version": 1,
        "schema": "merge_manifest.v1",
        "stage": "merge",
        "contract": "formal_merge_mainline",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "segment_dir": _relative_path(exp_dir, segment_dir),
            "runs_root": _relative_path(exp_dir, runs_root),
            "runs_plan": _relative_path(exp_dir, plan_path),
            "upstream_contract": "infer/runs_plan.json",
        },
        "artifacts": {
            "frames_dir": _relative_path(exp_dir, out_frames),
            "timestamps": _relative_path(exp_dir, timestamps_path),
            "calib": _relative_path(exp_dir, calib_path),
            "report": _relative_path(exp_dir, out_dir / "report.json"),
            "preview": _relative_path(exp_dir, preview_path) if preview_ok else "",
        },
        "summary": {
            "segment_count": int(len(segments)),
            "merged_frame_count": int(merged_frame_count),
            "merged_frame_bytes_sum": int(merged_frame_bytes),
            "merged_start_idx": int(merged_start_idx),
            "merged_end_idx": int(merged_end_idx),
            "fps": int(output_fps),
            "schedule_source": str(plan_obj.get("schedule_source", "") or ""),
            "execution_backend": str(plan_obj.get("execution_backend", "") or ""),
        },
        "segments": merged_segments,
    }
    manifest_path = (out_dir / "merge_manifest.json").resolve()
    write_json_atomic(manifest_path, manifest, indent=2)
    manifest_bytes = manifest_path.read_bytes()

    report = {
        "report_schema_version": "merge_report.v1",
        "step": "merge",
        "merge_status": "success",
        "created_at": manifest["created_at"],
        "schedule_source": str(plan_obj.get("schedule_source", "") or ""),
        "execution_backend": str(plan_obj.get("execution_backend", "") or ""),
        "fps": int(output_fps),
        "segment_count": int(len(segments)),
        "merged_start_idx": int(merged_start_idx),
        "merged_end_idx": int(merged_end_idx),
        "inputs": dict(manifest.get("inputs") or {}),
        "outputs": {
            "frames_dir": _relative_path(exp_dir, out_frames),
            "frame_count": int(merged_frame_count),
            "frames_bytes_sum": int(merged_frame_bytes),
            "timestamps_path": _relative_path(exp_dir, timestamps_path),
            "calib_path": _relative_path(exp_dir, calib_path),
            "preview_path": _relative_path(exp_dir, preview_path) if preview_ok else "",
            "manifest_bytes_sum": int(len(manifest_bytes)),
            "report_bytes_sum": 0,
            "manifest_file_count": 1,
            "report_file_count": 1,
            "bytes_sum": 0,
        },
        "segments_summary": {
            "count": int(len(segments)),
            "preview": list(merged_segments[:5]),
        },
        "manifest_path": _relative_path(exp_dir, manifest_path),
        "manifest_sha1": _sha1_bytes(manifest_bytes),
        "artifact_contract": {
            "formal_files": [
                "merge_manifest.json",
                "report.json",
                "frames/",
                "timestamps.txt",
                "calib.txt",
            ],
            "legacy_outputs_removed": [
                "merge_meta.json",
                "step_meta.json",
            ],
        },
        "warnings": [],
    }
    report_path = _write_report(out_dir / "report.json", report)

    log_prog(
        "merge summary: merged_frames={} segments={}".format(
            int(merged_frame_count),
            int(len(segments)),
        )
    )
    log_info("merge manifest: {}".format(manifest_path))
    log_info("merge report: {}".format(report_path))
    return report_path


def run(runtime):
    contract = merge_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.infer_runs_dir, "infer runs dir")
    ensure_file(runtime.paths.infer_runs_plan_path, "infer runs plan")

    runtime.remove_in_exp(runtime.paths.merge_dir)
    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "merge" / "service.py").resolve()

    cmd = [
        "python",
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--segment_dir",
        str(runtime.paths.segment_dir),
        "--runs_root",
        str(runtime.paths.infer_runs_dir),
        "--plan",
        str(runtime.paths.infer_runs_plan_path),
        "--out_dir",
        str(runtime.paths.merge_dir),
        "--fps",
        runtime.fps_arg,
    ]
    runtime.step_runner.run_env_python(
        cmd,
        phase_name=runtime.infer_phase_name(),
        log_name="merge.log",
        cwd=runtime.exphub_root,
    )

    ensure_dir(contract.artifacts[merge_contract.FRAMES_DIR], "merge frames dir")
    ensure_file(contract.artifacts[merge_contract.MERGE_MANIFEST], "merge manifest")
    ensure_file(contract.artifacts[merge_contract.REPORT], "merge report")
    ensure_file(contract.artifacts[merge_contract.CALIB], "merge calib")
    ensure_file(contract.artifacts[merge_contract.TIMESTAMPS], "merge timestamps")
    return contract.root


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", default="0")
    parser.add_argument("--no_preview", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("merge service helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
