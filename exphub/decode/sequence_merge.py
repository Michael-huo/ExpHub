from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, remove_path, write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_prog, log_warn
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
    merge_root = (Path(exp_dir).resolve() / "decode").resolve()
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
    payload = json.loads(ensure_file(plan_path, "image gen runs plan").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("invalid image gen runs plan payload: {}".format(plan_path))

    segments = list(payload.get("segments") or [])
    if not segments:
        raise RuntimeError("image gen runs plan contains zero segments: {}".format(plan_path))

    for idx, item in enumerate(segments):
        if not isinstance(item, dict):
            raise RuntimeError("invalid image gen runs plan segment at index {}: not an object".format(idx))
        if not str(item.get("run_name", "") or "").strip():
            raise RuntimeError("invalid image gen runs plan segment at index {}: missing run_name".format(idx))
        required_fields = (
            "raw_start_idx",
            "raw_end_idx",
            "desired_start_idx",
            "desired_end_idx",
            "desired_num_frames",
            "aligned_start_idx",
            "aligned_end_idx",
            "aligned_num_frames",
            "actual_saved_start_idx",
            "actual_saved_end_idx",
            "actual_saved_frames",
        )
        missing = [name for name in required_fields if item.get(name) is None]
        if missing:
            raise RuntimeError(
                "invalid image gen runs plan segment at index {}: missing {}".format(
                    idx,
                    ",".join(missing),
                )
            )
    return payload


def _load_input_report(input_dir, segment_manifest=None):
    if segment_manifest:
        report_path = ensure_file(segment_manifest, "segment manifest")
    else:
        report_path = ensure_file(Path(input_dir).resolve() / "input_report.json", "input report")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("invalid input report payload: {}".format(report_path))
    return payload, report_path


def _float_list(values):
    out = []
    for item in list(values or []):
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _format_calib_lines(calib_values):
    values = _float_list(calib_values)
    if not values:
        return ""
    return " ".join("{:.10f}".format(float(item)) for item in values) + "\n"


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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
    except Exception:
        return False

    if proc.returncode != 0:
        details = ""
        if proc.stdout:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                details = " | ".join(lines[-3:])
        if details:
            log_warn("sequence merge preview ffmpeg failed rc={} details={}".format(proc.returncode, details))
        else:
            log_warn("sequence merge preview ffmpeg failed rc={}".format(proc.returncode))
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
        outputs["bytes_sum"] = int(int(outputs.get("report_bytes_sum", 0) or 0) + int(outputs.get("manifest_bytes_sum", 0) or 0))
        payload["outputs"] = outputs
        payload["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, payload, indent=2)
    return Path(report_path).resolve()


def _run_formal_mainline(args):
    exp_dir = Path(args.exp_dir).resolve()
    input_dir = ensure_dir(args.segment_dir, "input dir")
    runs_root = ensure_dir(args.runs_root, "image gen runs dir")
    plan_path = ensure_file(args.plan, "image gen runs plan")
    out_dir = Path(args.out_dir).resolve()

    plan_obj = _load_runs_plan(plan_path)
    _guard_safe_out_dir(out_dir, exp_dir, runs_root)

    for stale_path in (
        out_dir / "frames",
        out_dir / "decode_merge_report.json",
        out_dir / "calib.txt",
        out_dir / "timestamps.txt",
        out_dir / "preview.mp4",
    ):
        remove_path(stale_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_frames = (out_dir / "frames").resolve()
    out_frames.mkdir(parents=True, exist_ok=True)

    segments = list(plan_obj.get("segments") or [])
    input_report, input_report_path = _load_input_report(input_dir, getattr(args, "segment_manifest", ""))
    camera_meta = dict(input_report.get("camera") or {})
    source_timestamp_values = _float_list(camera_meta.get("timestamps"))
    source_calib_values = _float_list(camera_meta.get("calib"))
    if not source_timestamp_values:
        max_source_idx = max((int(item.get("actual_saved_end_idx", 0) or 0) for item in segments), default=-1)
        if max_source_idx >= 0:
            fps_fallback = float(_safe_int(args.fps, 0) or 0)
            if fps_fallback > 0.0:
                source_timestamp_values = [float(idx) / fps_fallback for idx in range(max_source_idx + 1)]
    merged_start_idx = None
    merged_end_idx = None
    merged_frame_count = 0
    merged_frame_bytes = 0
    expected_merged_frame_count = 0
    execution_frame_count = 0
    shared_boundary_count = 0
    merged_timestamp_lines = []
    prev_end = None
    merged_segments = []
    source_unit_ids = set()
    source_span_ids = set()
    if not source_timestamp_values:
        raise RuntimeError(
            "input report missing camera timestamps and unable to synthesize timestamps: {}".format(
                input_report_path,
            )
        )

    for seg_idx, item in enumerate(segments):
        run_name = str(item.get("run_name"))
        cur_start = _safe_int(item.get("actual_saved_start_idx"), 0)
        cur_end = _safe_int(item.get("actual_saved_end_idx"), 0)
        actual_saved_frames = _safe_int(item.get("actual_saved_frames"), 0)
        aligned_num_frames = _safe_int(item.get("aligned_num_frames"), 0)
        if actual_saved_frames <= 0 or aligned_num_frames <= 0:
            raise RuntimeError("sequence merge invalid aligned/actual frame count for run={}".format(run_name))
        if actual_saved_frames != aligned_num_frames:
            raise RuntimeError(
                "sequence merge aligned/actual mismatch for run={}: aligned_num_frames={} actual_saved_frames={}".format(
                    run_name,
                    aligned_num_frames,
                    actual_saved_frames,
                )
            )
        if cur_end - cur_start + 1 != actual_saved_frames:
            raise RuntimeError(
                "sequence merge actual range mismatch for run={}: start={} end={} actual_saved_frames={}".format(
                    run_name,
                    cur_start,
                    cur_end,
                    actual_saved_frames,
                )
            )
        drop_leading_frames = 0
        if prev_end is not None and cur_start != prev_end:
            raise RuntimeError(
                "sequence merge aligned boundaries must stay globally snapped: prev_end={} current_start={} run={}".format(
                    int(prev_end),
                    int(cur_start),
                    run_name,
                )
            )
        if prev_end is not None:
            drop_leading_frames = 1
            shared_boundary_count += 1
        if actual_saved_frames <= int(drop_leading_frames):
            raise RuntimeError("sequence merge segment is too short after boundary sharing: run={}".format(run_name))
        merged_start_idx = cur_start if merged_start_idx is None else min(int(merged_start_idx), int(cur_start))
        merged_end_idx = cur_end if merged_end_idx is None else max(int(merged_end_idx), int(cur_end))
        expected_merged_frame_count += int(actual_saved_frames - int(drop_leading_frames))
        execution_frame_count += int(actual_saved_frames)

        run_dir = (runs_root / run_name).resolve()
        frames_dir = ensure_dir(run_dir / "frames", "image gen run frames")
        frames = list_frames_sorted(frames_dir)
        if not frames:
            raise RuntimeError("image gen run contains zero frames: {}".format(frames_dir))
        if len(frames) != actual_saved_frames:
            raise RuntimeError(
                "sequence merge saved frame mismatch for run={}: actual_saved_frames={} files={}".format(
                    run_name,
                    actual_saved_frames,
                    len(frames),
                )
            )
        unique_frames = list(frames[int(drop_leading_frames) :])

        out_start = merged_frame_count
        for src_path in unique_frames:
            dst_path = out_frames / "{:06d}.png".format(int(merged_frame_count))
            shutil.copy2(str(src_path), str(dst_path))
            try:
                merged_frame_bytes += int(dst_path.stat().st_size)
            except Exception:
                pass
            merged_frame_count += 1
        out_end = merged_frame_count - 1 if merged_frame_count > out_start else out_start - 1

        if len(source_timestamp_values) < cur_end + 1:
            raise RuntimeError(
                "segment timestamps too short for aligned range: have={} need={}".format(
                    len(source_timestamp_values),
                    cur_end + 1,
                )
            )
        current_timestamps = source_timestamp_values[cur_start : cur_end + 1]
        if len(current_timestamps) != actual_saved_frames:
            raise RuntimeError(
                "sequence merge timestamp slice mismatch for run={}: slice={} actual_saved_frames={}".format(
                    run_name,
                    len(current_timestamps),
                    actual_saved_frames,
                )
            )
        merged_timestamp_lines.extend(current_timestamps[int(drop_leading_frames) :])

        merged_segments.append(
            {
                "segment_index": int(seg_idx),
                "seg": item.get("seg"),
                "segment_id": item.get("segment_id"),
                "state_segment_id": item.get("state_segment_id"),
                "state_label": item.get("state_label"),
                "run_name": run_name,
                "raw_start_idx": int(_safe_int(item.get("raw_start_idx"), cur_start)),
                "raw_end_idx": int(_safe_int(item.get("raw_end_idx"), cur_end)),
                "desired_start_idx": int(_safe_int(item.get("desired_start_idx"), cur_start)),
                "desired_end_idx": int(_safe_int(item.get("desired_end_idx"), cur_end)),
                "desired_num_frames": int(_safe_int(item.get("desired_num_frames"), actual_saved_frames)),
                "aligned_start_idx": int(_safe_int(item.get("aligned_start_idx"), cur_start)),
                "aligned_end_idx": int(_safe_int(item.get("aligned_end_idx"), cur_end)),
                "aligned_num_frames": int(aligned_num_frames),
                "actual_saved_start_idx": int(cur_start),
                "actual_saved_end_idx": int(cur_end),
                "actual_saved_frames": int(actual_saved_frames),
                "shared_boundary_with_previous": bool(drop_leading_frames > 0),
                "merge_drop_leading_frames": int(drop_leading_frames),
                "align_reason": str(item.get("align_reason", "") or ""),
                "left_shift": int(_safe_int(item.get("left_shift"), 0)),
                "right_shift": int(_safe_int(item.get("right_shift"), 0)),
                "run_id": str(item.get("run_id", "") or ""),
                "source_unit_id": str(item.get("source_unit_id", "") or ""),
                "source_span_id": str(item.get("source_span_id", "") or ""),
                "source_prompt_ref": dict(item.get("source_prompt_ref", {}) or {}),
                "source_frame_count": int(len(frames)),
                "merged_frame_count": int(len(unique_frames)),
                "output_start_frame": int(out_start),
                "output_end_frame": int(out_end),
            }
        )
        source_unit_id = str(item.get("source_unit_id", "") or "").strip()
        source_span_id = str(item.get("source_span_id", "") or "").strip()
        if source_unit_id:
            source_unit_ids.add(source_unit_id)
        if source_span_id:
            source_span_ids.add(source_span_id)
        prev_end = cur_end

    if merged_end_idx is None:
        raise RuntimeError("failed to resolve merged end index from image gen runs plan")

    calib_path = (out_dir / "calib.txt").resolve()
    timestamps_path = (out_dir / "timestamps.txt").resolve()
    if not source_calib_values:
        raise RuntimeError("input report missing camera calib: {}".format(input_report_path))
    write_text_atomic(calib_path, _format_calib_lines(source_calib_values))

    if int(expected_merged_frame_count) != int(merged_frame_count):
        raise RuntimeError(
            "sequence merge frame count mismatch: merged={} expected_by_contract={}".format(
                merged_frame_count,
                int(expected_merged_frame_count),
            )
        )
    sliced = list(merged_timestamp_lines)
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

    report = {
        "version": 1,
        "schema": "decode_merge_report.v1",
        "stage": "decode",
        "substage": "sequence_merge",
        "contract": "decode_sequence_merge_mainline",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "planner": "generation_units",
            "prompt_strategy": "prompt_spans",
            "input_dir": _relative_path(exp_dir, input_dir),
            "runs_root": _relative_path(exp_dir, runs_root),
            "runs_plan": _relative_path(exp_dir, plan_path),
            "upstream_contract": "decode.image_gen/decode_plan.json",
        },
        "artifacts": {
            "frames_dir": _relative_path(exp_dir, out_frames),
            "timestamps": _relative_path(exp_dir, timestamps_path),
            "calib": _relative_path(exp_dir, calib_path),
            "report": _relative_path(exp_dir, out_dir / "decode_merge_report.json"),
            "preview": _relative_path(exp_dir, preview_path) if preview_ok else "",
        },
        "summary": {
            "segment_count": int(len(segments)),
            "source_unit_count": int(len(source_unit_ids)),
            "source_span_count": int(len(source_span_ids)),
            "merged_frame_count": int(merged_frame_count),
            "expected_merged_frame_count": int(expected_merged_frame_count),
            "execution_frame_count": int(execution_frame_count),
            "shared_boundary_count": int(shared_boundary_count),
            "shared_anchor_count": int(shared_boundary_count),
            "merged_frame_bytes_sum": int(merged_frame_bytes),
            "merged_start_idx": int(merged_start_idx),
            "merged_end_idx": int(merged_end_idx),
            "fps": int(output_fps),
            "skipped_unit_count": int(len(list(plan_obj.get("skipped_units") or []))),
        },
        "segments": merged_segments,
        "merge_status": "success",
        "fps": int(output_fps),
        "segment_count": int(len(segments)),
        "source_unit_count": int(len(source_unit_ids)),
        "source_span_count": int(len(source_span_ids)),
        "shared_anchor_count": int(shared_boundary_count),
        "merged_start_idx": int(merged_start_idx),
        "merged_end_idx": int(merged_end_idx),
        "outputs": {
            "frames_dir": _relative_path(exp_dir, out_frames),
            "frame_count": int(merged_frame_count),
            "frames_bytes_sum": int(merged_frame_bytes),
            "timestamps_path": _relative_path(exp_dir, timestamps_path),
            "calib_path": _relative_path(exp_dir, calib_path),
            "preview_path": _relative_path(exp_dir, preview_path) if preview_ok else "",
            "report_bytes_sum": 0,
            "report_file_count": 1,
            "bytes_sum": 0,
        },
        "segments_summary": {
            "count": int(len(segments)),
            "preview": list(merged_segments[:5]),
        },
        "artifact_contract": {
            "formal_files": ["decode_merge_report.json", "frames/", "timestamps.txt", "calib.txt"],
            "transitional_files": [],
        },
        "warnings": [],
    }
    report_path = _write_report(out_dir / "decode_merge_report.json", report)

    log_prog(
        "sequence merge summary: merged_frames={} segments={}".format(
            int(merged_frame_count),
            int(len(segments)),
        )
    )
    log_info("sequence merge report: {}".format(report_path))
    return report_path


def run(runtime):
    ensure_dir(runtime.paths.input_dir, "input dir")
    ensure_dir(runtime.paths.decode_runs_dir, "image gen runs dir")
    ensure_file(runtime.paths.decode_plan_path, "decode plan")

    for path in (
        runtime.paths.decode_frames_dir,
        runtime.paths.decode_merge_report_path,
        runtime.paths.decode_calib_path,
        runtime.paths.decode_timestamps_path,
        runtime.paths.decode_preview_path,
    ):
        runtime.remove_in_exp(path)
    cmd = [
        "-m",
        "exphub.decode.sequence_merge",
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--segment_dir",
        str(runtime.paths.prepare_dir),
        "--segment_manifest",
        str(runtime.paths.input_report_path),
        "--runs_root",
        str(runtime.paths.decode_runs_dir),
        "--plan",
        str(runtime.paths.decode_plan_path),
        "--out_dir",
        str(runtime.paths.decode_dir),
        "--fps",
        runtime.fps_arg,
    ]
    runtime.step_runner.run_env_python(
        cmd,
        phase_name=runtime.infer_phase_name(),
        log_name="merge.log",
        cwd=runtime.exphub_root,
    )

    ensure_dir(runtime.paths.decode_frames_dir, "decode frames dir")
    ensure_file(runtime.paths.decode_merge_report_path, "decode merge report")
    ensure_file(runtime.paths.decode_calib_path, "decode calib")
    ensure_file(runtime.paths.decode_timestamps_path, "decode timestamps")
    return runtime.paths.decode_dir


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--segment_manifest", default="")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", default="0")
    parser.add_argument("--no_preview", action="store_true")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("sequence merge helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
