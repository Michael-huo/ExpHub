from __future__ import annotations

import shutil
from pathlib import Path

from exphub.common.io import list_frames_sorted, write_json_atomic, write_text_atomic
from exphub.meta import sanitize_token


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _resolution(prepare_result):
    item = _as_dict(prepare_result.get("normalized_resolution"))
    return int(item.get("width", 0) or 0), int(item.get("height", 0) or 0)


def _prompt_by_unit_id(prompts):
    out = {}
    for item in list(_as_dict(prompts).get("units") or []):
        unit_id = str(_as_dict(item).get("unit_id", "") or "")
        if unit_id:
            out[unit_id] = _as_dict(item)
    return out


def _copy_unit_frames(source_frames, start_idx, end_idx, out_dir):
    start = int(start_idx)
    end = int(end_idx)
    if start < 0 or end < start or end >= len(source_frames):
        raise RuntimeError(
            "unit frame range outside prepared frames: start_idx={} end_idx={} frame_count={}".format(
                start,
                end,
                len(source_frames),
            )
        )
    frames_dir = Path(out_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for out_idx, source_idx in enumerate(range(start, end + 1)):
        source_path = Path(source_frames[source_idx]).resolve()
        if not source_path.is_file():
            raise RuntimeError("missing source prepared frame: {}".format(source_path))
        shutil.copy2(str(source_path), str(frames_dir / "{:06d}{}".format(out_idx, source_path.suffix.lower())))
        count += 1
    return count


def export_sequence_trainset(
    runtime,
    sequence,
    prepare_result,
    prepare_frames_dir,
    generation_units,
    prompts,
    generation_units_path,
    prompts_path,
    clip_start_index,
):
    units = list(_as_dict(generation_units).get("units") or [])
    prompt_units = list(_as_dict(prompts).get("units") or [])
    if len(units) != len(prompt_units):
        raise RuntimeError(
            "generation unit count does not match prompt count: sequence={} units={} prompts={}".format(
                sequence,
                len(units),
                len(prompt_units),
            )
        )

    prompt_map = _prompt_by_unit_id(prompts)
    source_frames = list_frames_sorted(prepare_frames_dir)
    width, height = _resolution(prepare_result)
    fps = int(prepare_result.get("target_fps", runtime.spec.fps) or runtime.spec.fps)
    records = []
    total_frames = 0
    motion_histogram = {}

    for local_idx, raw_unit in enumerate(units):
        unit = _as_dict(raw_unit)
        unit_id = str(unit.get("unit_id", "") or "")
        if not unit_id:
            raise RuntimeError("generation unit missing unit_id: sequence={} index={}".format(sequence, local_idx))
        prompt_item = prompt_map.get(unit_id)
        if not prompt_item:
            raise RuntimeError("prompt missing for generation unit: sequence={} unit_id={}".format(sequence, unit_id))
        prompt = str(prompt_item.get("assembled_prompt", "") or "")
        if not prompt:
            raise RuntimeError("assembled_prompt is empty: sequence={} unit_id={}".format(sequence, unit_id))

        start_idx = int(unit.get("start_idx"))
        end_idx = int(unit.get("end_idx"))
        expected_length = int(end_idx - start_idx + 1)
        unit_length = int(unit.get("length", expected_length) or expected_length)
        if unit_length != expected_length:
            raise RuntimeError(
                "generation unit length mismatch: sequence={} unit_id={} length={} range_length={}".format(
                    sequence,
                    unit_id,
                    unit_length,
                    expected_length,
                )
            )

        clip_index = int(clip_start_index) + int(local_idx)
        clip_id = "{}_{}_{:06d}".format(
            sanitize_token(runtime.spec.dataset),
            sanitize_token(str(sequence)),
            clip_index,
        )
        clip_dir = runtime.paths.trainset_clips_dir / clip_id
        frames_dir = clip_dir / "frames"
        written_count = _copy_unit_frames(source_frames, start_idx, end_idx, frames_dir)
        if written_count != expected_length:
            raise RuntimeError(
                "exported frame count mismatch: sequence={} unit_id={} expected={} got={}".format(
                    sequence,
                    unit_id,
                    expected_length,
                    written_count,
                )
            )

        motion_label = str(unit.get("motion_label", prompt_item.get("motion_label", "mixed")) or "mixed")
        meta = {
            "clip_id": str(clip_id),
            "dataset": str(runtime.spec.dataset),
            "sequence": str(sequence),
            "unit_id": str(unit_id),
            "seg_id": str(unit.get("seg_id", "") or ""),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "length": int(expected_length),
            "fps": int(fps),
            "width": int(width),
            "height": int(height),
            "prompt": str(prompt),
            "motion_label": str(motion_label),
            "source_prepare_result": str(Path(prepare_result.get("frame_dir", "")).resolve().parent / "prepare_result.json"),
            "source_generation_units": str(Path(generation_units_path).resolve()),
            "source_prompts": str(Path(prompts_path).resolve()),
        }
        write_text_atomic(clip_dir / "prompt.txt", prompt)
        write_json_atomic(clip_dir / "meta.json", meta, indent=2)

        rel_clip = Path("clips") / clip_id
        records.append(
            {
                "clip_id": str(clip_id),
                "frames_dir": str((rel_clip / "frames").as_posix()),
                "prompt_file": str((rel_clip / "prompt.txt").as_posix()),
                "meta_file": str((rel_clip / "meta.json").as_posix()),
                "num_frames": int(expected_length),
                "width": int(width),
                "height": int(height),
                "dataset": str(runtime.spec.dataset),
                "sequence": str(sequence),
                "prompt": str(prompt),
                "motion_label": str(motion_label),
                "unit_id": str(unit_id),
            }
        )
        total_frames += int(expected_length)
        motion_histogram[motion_label] = int(motion_histogram.get(motion_label, 0)) + 1

    return {
        "records": records,
        "clip_count": int(len(records)),
        "total_frames": int(total_frames),
        "motion_label_histogram": motion_histogram,
        "unit_lengths": [int(item.get("num_frames", 0) or 0) for item in records],
        "resolution": [int(width), int(height)],
        "fps": int(fps),
    }


def write_trainset_indexes(runtime, records, sequence_count, fps, resolution, motion_histogram, unit_lengths):
    records_out = list(records or [])
    total_frames = sum(int(item.get("num_frames", 0) or 0) for item in records_out)
    unit_length_values = [int(item) for item in list(unit_lengths or [])]
    avg_unit_length = float(sum(unit_length_values)) / float(len(unit_length_values)) if unit_length_values else 0.0
    stats = {
        "dataset": str(runtime.spec.dataset),
        "sequence_count": int(sequence_count),
        "clip_count": int(len(records_out)),
        "total_frames": int(total_frames),
        "fps": int(fps),
        "resolution": [int(resolution[0]), int(resolution[1])] if resolution else [0, 0],
        "motion_label_histogram": dict(motion_histogram or {}),
        "avg_unit_length": float(avg_unit_length),
    }
    write_json_atomic(runtime.paths.trainset_metadata_path, records_out, indent=2)
    write_json_atomic(runtime.paths.trainset_stats_path, stats, indent=2)
    return stats
