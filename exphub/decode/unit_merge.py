from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from exphub.common.io import ensure_dir, list_frames_sorted, read_json_dict, remove_path, write_json_atomic, write_text_atomic
from exphub.common.logging import log_info, log_prog, log_warn


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _float_list(values):
    out = []
    for item in list(values or []):
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _format_calib(prepare_result):
    intrinsics = _as_dict(prepare_result.get("normalized_intrinsics"))
    values = [
        intrinsics.get("fx"),
        intrinsics.get("fy"),
        intrinsics.get("cx"),
        intrinsics.get("cy"),
    ] + list(intrinsics.get("dist") or [])
    floats = _float_list(values)
    if len(floats) < 4:
        raise RuntimeError("prepare_result.normalized_intrinsics missing calib values")
    return " ".join("{:.10f}".format(float(item)) for item in floats) + "\n"


def _timestamp_values(prepare_result):
    frame_index_map = _as_dict(prepare_result.get("frame_index_map"))
    rel_values = _float_list(frame_index_map.get("prepared_to_rel_time_sec") or [])
    if rel_values:
        return rel_values, "prepare_result.frame_index_map.prepared_to_rel_time_sec"
    abs_values = _float_list(frame_index_map.get("prepared_to_abs_time_sec") or frame_index_map.get("prepared_to_time_sec") or [])
    if abs_values:
        base = float(abs_values[0])
        return [float(item) - base for item in abs_values], "prepare_result.frame_index_map.abs_time_normalized"
    raise RuntimeError("prepare_result.frame_index_map missing timestamps")


def _write_preview(frames_dir, fps, out_mp4):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
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
            if proc.returncode == 0 and Path(out_mp4).is_file():
                return True
            if proc.stdout:
                log_warn("unit merge preview ffmpeg failed: {}".format(" | ".join(proc.stdout.splitlines()[-3:])))
        except Exception:
            pass

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        return False
    frames = list_frames_sorted(frames_dir)
    if not frames:
        return False
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


def _report_unit_map(decode_report):
    out = {}
    for item in list(_as_dict(decode_report).get("units") or []):
        unit_id = str(_as_dict(item).get("unit_id", "") or "")
        if unit_id:
            out[unit_id] = _as_dict(item)
    return out


def merge_units(runtime, tasks_payload, decode_report):
    prepare_result = _as_dict(tasks_payload.get("_raw", {}).get("prepare_result"))
    if not prepare_result:
        prepare_result = read_json_dict(runtime.paths.prepare_result_path)
    timestamps, timestamp_source = _timestamp_values(prepare_result)
    report_units = _report_unit_map(decode_report)
    tasks = list(tasks_payload.get("tasks") or [])
    if not tasks:
        raise RuntimeError("unit merge received zero decode tasks")

    for stale_path in (
        runtime.paths.decode_frames_dir,
        runtime.paths.decode_merge_report_path,
        runtime.paths.decode_calib_path,
        runtime.paths.decode_timestamps_path,
        runtime.paths.decode_preview_path,
    ):
        remove_path(stale_path)

    out_frames = runtime.paths.decode_frames_dir
    out_frames.mkdir(parents=True, exist_ok=True)
    merged_segments = []
    merged_timestamp_values = []
    merged_count = 0
    merged_bytes = 0
    shared_endpoint_count = 0
    expected_count = 0
    execution_frame_count = 0
    prev_end = None

    for idx, task in enumerate(tasks):
        unit_id = str(task["unit_id"])
        start_idx = int(task["start_idx"])
        end_idx = int(task["end_idx"])
        length = int(task["length"])
        if prev_end is not None and start_idx != prev_end:
            raise RuntimeError(
                "unit merge shared endpoint mismatch before {}: prev_end={} current_start={}".format(
                    unit_id,
                    prev_end,
                    start_idx,
                )
            )
        unit_report = report_units.get(unit_id)
        if not unit_report:
            raise RuntimeError("decode_report missing unit {}".format(unit_id))
        frames_dir = ensure_dir(runtime.paths.exp_dir / str(unit_report.get("frames_dir", "")), "decode unit frames dir")
        unit_frames = list_frames_sorted(frames_dir)
        if len(unit_frames) != int(unit_report.get("num_frames", 0) or 0):
            raise RuntimeError("decode_report frame count mismatch for unit {}".format(unit_id))
        if len(unit_frames) != length:
            raise RuntimeError("unit {} generated {} frames, expected {}".format(unit_id, len(unit_frames), length))
        if end_idx >= len(timestamps):
            raise RuntimeError("timestamps too short for unit {} end_idx={}".format(unit_id, end_idx))

        drop_leading = 1 if idx > 0 else 0
        if drop_leading:
            shared_endpoint_count += 1
        unique_frames = unit_frames[drop_leading:]
        expected_count += len(unique_frames)
        execution_frame_count += len(unit_frames)
        out_start = merged_count
        for frame_path in unique_frames:
            dst = out_frames / "{:06d}.png".format(merged_count)
            shutil.copy2(str(frame_path), str(dst))
            try:
                merged_bytes += int(dst.stat().st_size)
            except Exception:
                pass
            merged_count += 1
        merged_timestamp_values.extend(timestamps[start_idx + drop_leading : end_idx + 1])
        out_end = merged_count - 1
        merged_segments.append(
            {
                "unit_index": int(idx),
                "unit_id": unit_id,
                "seg_id": str(task.get("seg_id", "") or ""),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "length": int(length),
                "source_frame_count": int(len(unit_frames)),
                "merged_frame_count": int(len(unique_frames)),
                "shared_boundary_with_previous": bool(drop_leading),
                "merge_drop_leading_frames": int(drop_leading),
                "output_start_frame": int(out_start),
                "output_end_frame": int(out_end),
                "frames_dir": str(unit_report.get("frames_dir", "") or ""),
                "output_dir": str(unit_report.get("output_dir", "") or ""),
            }
        )
        prev_end = end_idx

    if expected_count != merged_count:
        raise RuntimeError("unit merge frame count mismatch: expected={} actual={}".format(expected_count, merged_count))
    if len(merged_timestamp_values) != merged_count:
        raise RuntimeError(
            "unit merge timestamp count mismatch: timestamps={} frames={}".format(
                len(merged_timestamp_values),
                merged_count,
            )
        )

    write_text_atomic(runtime.paths.decode_calib_path, _format_calib(prepare_result))
    t0 = float(merged_timestamp_values[0]) if merged_timestamp_values else 0.0
    write_text_atomic(
        runtime.paths.decode_timestamps_path,
        "\n".join("{:.9f}".format(float(item) - t0) for item in merged_timestamp_values) + "\n",
    )
    preview_ok = _write_preview(runtime.paths.decode_frames_dir, int(float(runtime.fps_arg)), runtime.paths.decode_preview_path)

    source_inputs = dict(tasks_payload.get("source_inputs") or {})
    report = {
        "version": 1,
        "schema": "decode_merge_report.v1",
        "stage": "decode",
        "substage": "unit_merge",
        "contract": "decode_unit_merge_native",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "planner": "generation_units",
            "prompt_strategy": "prompts",
            "source_inputs": source_inputs,
            "decode_report": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_report_path),
            "generation_tasks": "in_memory",
        },
        "artifacts": {
            "frames_dir": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_frames_dir),
            "timestamps": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_timestamps_path),
            "calib": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_calib_path),
            "report": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_merge_report_path),
            "preview": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_preview_path) if preview_ok else "",
        },
        "summary": {
            "unit_count": int(len(tasks)),
            "merged_frame_count": int(merged_count),
            "expected_merged_frame_count": int(expected_count),
            "execution_frame_count": int(execution_frame_count),
            "shared_endpoint_count": int(shared_endpoint_count),
            "shared_anchor_count": int(shared_endpoint_count),
            "merged_start_idx": int(tasks[0]["start_idx"]),
            "merged_end_idx": int(tasks[-1]["end_idx"]),
            "merged_frame_bytes_sum": int(merged_bytes),
            "timestamp_source": str(timestamp_source),
            "calib_source": "prepare_result.normalized_intrinsics",
            "fps": int(float(runtime.fps_arg)),
        },
        "segments": merged_segments,
        "merge_status": "success",
        "outputs": {
            "frames_dir": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_frames_dir),
            "frame_count": int(merged_count),
            "timestamps_path": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_timestamps_path),
            "calib_path": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_calib_path),
            "preview_path": _relative_path(runtime.paths.exp_dir, runtime.paths.decode_preview_path) if preview_ok else "",
        },
        "warnings": [],
    }
    write_json_atomic(runtime.paths.decode_merge_report_path, report, indent=2)
    log_prog("unit merge summary: merged_frames={} units={}".format(merged_count, len(tasks)))
    log_info("unit merge report: {}".format(runtime.paths.decode_merge_report_path))
    return report
