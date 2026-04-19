from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from exphub.common.io import ensure_dir, ensure_file, read_json_dict


_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_int(value, label):
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError("{} must be an integer, got {!r}".format(label, value)) from exc


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _sha1_hex(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _load_json_object(path, label):
    resolved = ensure_file(path, label)
    payload = read_json_dict(resolved)
    if not payload:
        raise RuntimeError("invalid {} JSON object: {}".format(label, resolved))
    return payload, resolved


def _frame_path(frames_dir, idx):
    frame_root = ensure_dir(frames_dir, "prepare frames dir")
    stem = "{:06d}".format(int(idx))
    for ext in _FRAME_EXTS:
        candidate = frame_root / "{}{}".format(stem, ext)
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("prepare frame not found for index {} under {}".format(int(idx), frame_root))


def _time_values(prepare_result):
    frame_index_map = _as_dict(prepare_result.get("frame_index_map"))
    for key in ("prepared_to_abs_time_sec", "prepared_to_time_sec", "prepared_to_rel_time_sec"):
        values = frame_index_map.get(key)
        if isinstance(values, list) and values:
            return [float(item) for item in values], key
    return [], ""


def _time_at(values, idx, label):
    if int(idx) < 0 or int(idx) >= len(values):
        raise RuntimeError("{} missing timestamp for frame {}".format(label, int(idx)))
    return float(values[int(idx)])


def _prompt_by_unit(prompts_payload):
    out = {}
    for idx, raw_item in enumerate(list(prompts_payload.get("units") or [])):
        item = _as_dict(raw_item)
        unit_id = _collapse_ws(item.get("unit_id", ""))
        if not unit_id:
            raise RuntimeError("prompts.units[{}] missing unit_id".format(idx))
        if unit_id in out:
            raise RuntimeError("duplicate prompt unit_id: {}".format(unit_id))
        out[unit_id] = item
    if not out:
        raise RuntimeError("prompts.json must contain units")
    return out


def _validate_native_sources(prepare_result, generation_units, prompts_payload, encode_result):
    prepare_num_frames = _as_int(prepare_result.get("num_frames"), "prepare_result.num_frames")
    if prepare_num_frames <= 0:
        raise RuntimeError("prepare_result.num_frames must be > 0")

    units = list(generation_units.get("units") or [])
    if not units:
        raise RuntimeError("generation_units.json must contain units")

    prompt_map = _prompt_by_unit(prompts_payload)
    expected_count = encode_result.get("num_generation_units")
    if expected_count is not None and int(expected_count) != len(units):
        raise RuntimeError(
            "encode_result.num_generation_units mismatch: encode_result={} generation_units={}".format(
                int(expected_count),
                len(units),
            )
        )
    return units, prompt_map, prepare_num_frames


def build_generation_tasks(runtime):
    paths = runtime.paths
    return build_generation_tasks_from_paths(
        exp_dir=paths.exp_dir,
        prepare_result_path=paths.prepare_result_path,
        prepare_frames_dir=paths.prepare_frames_dir,
        generation_units_path=paths.encode_generation_units_path,
        prompts_path=paths.encode_prompts_path,
        encode_result_path=paths.encode_result_path,
        decode_runs_dir=paths.decode_runs_dir,
        seed_base=runtime.args.seed_base,
    )


def build_generation_tasks_from_paths(
    exp_dir,
    prepare_result_path,
    prepare_frames_dir,
    generation_units_path,
    prompts_path,
    encode_result_path,
    decode_runs_dir,
    seed_base=43,
):
    exp_root = Path(exp_dir).resolve()
    prepare_result, prepare_path = _load_json_object(prepare_result_path, "prepare result")
    generation_units, generation_units_path = _load_json_object(generation_units_path, "generation units")
    prompts_payload, prompts_path = _load_json_object(prompts_path, "prompts")
    encode_result, encode_result_path = _load_json_object(encode_result_path, "encode result")
    frames_dir = ensure_dir(prepare_frames_dir, "prepare frames dir")

    units, prompt_map, prepare_num_frames = _validate_native_sources(
        prepare_result,
        generation_units,
        prompts_payload,
        encode_result,
    )
    timestamps, timestamp_key = _time_values(prepare_result)
    if not timestamps:
        raise RuntimeError("prepare_result.frame_index_map must contain prepared timestamps")

    tasks = []
    prev_end = None
    for idx, raw_unit in enumerate(units):
        unit = _as_dict(raw_unit)
        unit_id = _collapse_ws(unit.get("unit_id", ""))
        if not unit_id:
            raise RuntimeError("generation unit at index {} missing unit_id".format(idx))
        expected_unit_id = "unit_{:04d}".format(idx)
        if unit_id != expected_unit_id:
            raise RuntimeError("generation unit order/id mismatch at index {}: {}".format(idx, unit_id))
        if not bool(unit.get("is_valid_for_decode", False)):
            raise RuntimeError("generation unit is not valid for decode: {}".format(unit_id))

        start_idx = _as_int(unit.get("anchor_start_idx", unit.get("start_idx")), "{}.start_idx".format(unit_id))
        end_idx = _as_int(unit.get("anchor_end_idx", unit.get("end_idx")), "{}.end_idx".format(unit_id))
        if end_idx < start_idx:
            raise RuntimeError("generation unit {} has invalid range {}..{}".format(unit_id, start_idx, end_idx))
        if end_idx >= prepare_num_frames:
            raise RuntimeError(
                "generation unit {} exceeds prepare frame count: end_idx={} num_frames={}".format(
                    unit_id,
                    end_idx,
                    prepare_num_frames,
                )
            )
        length = _as_int(unit.get("length", unit.get("duration_frames")), "{}.length".format(unit_id))
        if length != end_idx - start_idx + 1:
            raise RuntimeError(
                "generation unit {} length mismatch: length={} range={}..{}".format(
                    unit_id,
                    length,
                    start_idx,
                    end_idx,
                )
            )
        if prev_end is not None and start_idx != prev_end:
            raise RuntimeError(
                "generation units must use shared endpoints: prev_end={} current_start={} unit={}".format(
                    prev_end,
                    start_idx,
                    unit_id,
                )
            )

        prompt_item = prompt_map.get(unit_id)
        if not prompt_item:
            raise RuntimeError("missing prompt for generation unit {}".format(unit_id))
        prompt = str(prompt_item.get("assembled_prompt", "") or "").strip()
        if not prompt:
            raise RuntimeError("prompt for generation unit {} missing assembled_prompt".format(unit_id))
        negative_prompt = str(prompt_item.get("negative_prompt", prompts_payload.get("negative_prompt", "")) or "")

        start_frame_path = _frame_path(frames_dir, start_idx)
        end_frame_path = _frame_path(frames_dir, end_idx)
        start_abs_time = float(unit.get("start_abs_time_sec", _time_at(timestamps, start_idx, "prepare_result")))
        end_abs_time = float(unit.get("end_abs_time_sec", _time_at(timestamps, end_idx, "prepare_result")))
        seed = int(seed_base) + int(idx)

        task = {
            "task_index": int(idx),
            "unit_id": unit_id,
            "seg_id": str(unit.get("seg_id", "") or ""),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "length": int(length),
            "start_abs_time_sec": float(start_abs_time),
            "end_abs_time_sec": float(end_abs_time),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "base_prompt": str(prompt_item.get("base_prompt", prompts_payload.get("base_prompt", "")) or ""),
            "prompt_source": "prompts.assembled_prompt",
            "prompt_hash8": _sha1_hex(prompt + "\n||NEG||\n" + negative_prompt)[:8],
            "motion_label": str(unit.get("motion_label", prompt_item.get("motion_label", "")) or ""),
            "scene_label": str(unit.get("scene_label", "") or ""),
            "start_frame_path": str(start_frame_path),
            "end_frame_path": str(end_frame_path),
            "output_dir": str((Path(decode_runs_dir).resolve() / unit_id).resolve()),
            "run_name": unit_id,
            "seed": int(seed),
            "source_prompt_ref": dict(_as_dict(unit.get("prompt_ref"))),
            "source_segment_ids": list(unit.get("source_segment_ids") or []),
            "is_valid_for_decode": True,
            "align_reason": "generation_unit_shared_anchor",
            "num_inference_steps": prompt_item.get("num_inference_steps"),
            "guidance_scale": prompt_item.get("guidance_scale"),
        }
        tasks.append(task)
        prev_end = end_idx

    source_inputs = {
        "prepare_result": _relative_path(exp_root, prepare_path),
        "prepare_frames": _relative_path(exp_root, frames_dir),
        "generation_units": _relative_path(exp_root, generation_units_path),
        "prompts": _relative_path(exp_root, prompts_path),
        "encode_result": _relative_path(exp_root, encode_result_path),
    }
    return {
        "schema": "generation_tasks.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_strategy": "prompts",
        "source_inputs": source_inputs,
        "timestamp_source": str(timestamp_key),
        "prepare_num_frames": int(prepare_num_frames),
        "sequence_range": dict(_as_dict(generation_units.get("sequence_range"))),
        "tasks": tasks,
        "summary": {
            "task_count": int(len(tasks)),
            "shared_endpoint_count": int(max(0, len(tasks) - 1)),
            "start_idx": int(tasks[0]["start_idx"]),
            "end_idx": int(tasks[-1]["end_idx"]),
        },
        "_raw": {
            "prepare_result": prepare_result,
            "generation_units": generation_units,
            "prompts": prompts_payload,
            "encode_result": encode_result,
        },
    }
