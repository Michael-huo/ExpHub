from __future__ import annotations

import hashlib
from pathlib import Path

from exphub.common.io import write_json_atomic


REPORT_FILENAME = "report.json"


def _sha1_bytes(payload_bytes):
    # type: (bytes) -> str
    return hashlib.sha1(payload_bytes).hexdigest()


def _count_by_key(items, key):
    # type: (list, str) -> dict
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get(key, "") or "").strip() or "unknown"
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def _relative_path(base_dir, target_path):
    # type: (Path, Path) -> str
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _collapse_ws(text):
    # type: (object) -> str
    return " ".join(str(text or "").strip().split()).strip()


def _word_count(text):
    # type: (object) -> int
    collapsed = _collapse_ws(text)
    if not collapsed:
        return 0
    return int(len([part for part in collapsed.split(" ") if part]))


def _numeric_summary(values):
    # type: (list) -> dict
    nums = []
    for value in list(values or []):
        try:
            nums.append(int(value))
        except Exception:
            continue
    if not nums:
        return {"min": 0, "max": 0, "avg": 0.0}
    return {
        "min": int(min(nums)),
        "max": int(max(nums)),
        "avg": float(sum(nums) / float(len(nums))),
    }


def _state_control_statistics(state_prompt_manifest, runtime_prompt_plan):
    # type: (dict, dict) -> dict
    state_segments = list((state_prompt_manifest or {}).get("state_segments") or [])
    runtime_segments = list((runtime_prompt_plan or {}).get("segments") or [])
    state_controls = [dict(item.get("state_control") or {}) for item in state_segments if isinstance(item, dict)]
    return {
        "state_segment_count": int(len(state_segments)),
        "deploy_segment_count": int(len(runtime_segments)),
        "state_label_counts": dict(_count_by_key(state_segments, "state_label")),
        "continuity_emphasis_counts": dict(_count_by_key(state_controls, "continuity_emphasis")),
        "negative_prompt_delta_segment_count": int(
            len([item for item in state_controls if _collapse_ws(item.get("negative_prompt_delta", ""))])
        ),
        "deploy_match_source_counts": dict(_count_by_key(runtime_segments, "match_source")),
        "runtime_state_label_counts": dict(_count_by_key(runtime_segments, "state_label")),
    }


def _scene_prompt_examples(values, limit=6):
    # type: (list, int) -> list
    out = []
    seen = set()
    for raw_value in list(values or []):
        value = _collapse_ws(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= int(limit):
            break
    return out


def _scene_prompt_statistics(state_scene_encoding, runtime_prompt_plan):
    # type: (dict, dict) -> dict
    state_scene_segments = list((state_scene_encoding or {}).get("state_segments") or [])
    runtime_segments = list((runtime_prompt_plan or {}).get("segments") or [])
    state_scene_prompts = []
    runtime_scene_prompts = []
    nonempty = 0
    for item in runtime_segments:
        if not isinstance(item, dict):
            continue
        prompt = _collapse_ws(item.get("scene_prompt", ""))
        if prompt:
            nonempty += 1
            runtime_scene_prompts.append(prompt)

    raw_word_counts = []
    normalized_word_counts = []
    changed_count = 0
    for item in state_scene_segments:
        if not isinstance(item, dict):
            continue
        prompt = _collapse_ws(item.get("scene_prompt", ""))
        if prompt:
            state_scene_prompts.append(prompt)
        raw_count = item.get("raw_scene_prompt_word_count")
        normalized_count = item.get("scene_prompt_word_count")
        if raw_count is not None:
            raw_word_counts.append(raw_count)
        if normalized_count is not None:
            normalized_word_counts.append(normalized_count)
        if bool(dict(item.get("scene_prompt_normalization") or {}).get("changed")):
            changed_count += 1

    return {
        "state_scene_segment_count": int(len(state_scene_segments)),
        "deploy_segment_count": int(len(runtime_segments)),
        "scene_prompt_segment_count": int(nonempty),
        "empty_scene_prompt_segment_count": int(max(0, len(runtime_segments) - nonempty)),
        "unique_state_scene_prompt_count": int(len(set(state_scene_prompts))),
        "unique_runtime_scene_prompt_count": int(len(set(runtime_scene_prompts))),
        "scene_prompt_source_counts": dict(_count_by_key(runtime_segments, "scene_prompt_source")),
        "runtime_scene_prompt_word_count": dict(_numeric_summary([_word_count(value) for value in runtime_scene_prompts])),
        "state_scene_normalization": {
            "changed_segment_count": int(changed_count),
            "raw_word_count": dict(_numeric_summary(raw_word_counts)),
            "normalized_word_count": dict(_numeric_summary(normalized_word_counts)),
        },
        "scene_prompt_examples": list(_scene_prompt_examples(state_scene_prompts or runtime_scene_prompts)),
    }


def build_prompt_report(
    prompt_dir,
    base_prompt_payload,
    state_prompt_manifest,
    state_scene_encoding,
    runtime_prompt_plan,
    frame_files_count,
    total_sec,
    assembly_notes,
):
    # type: (...) -> dict
    prompt_dir = Path(prompt_dir).resolve()
    base_prompt_path = (prompt_dir / "base_prompt.json").resolve()
    state_prompt_manifest_path = (prompt_dir / "state_prompt_manifest.json").resolve()
    runtime_prompt_plan_path = (prompt_dir / "runtime_prompt_plan.json").resolve()

    base_prompt_bytes = base_prompt_path.read_bytes()
    state_prompt_manifest_bytes = state_prompt_manifest_path.read_bytes()
    runtime_prompt_plan_bytes = runtime_prompt_plan_path.read_bytes()

    return {
        "report_schema_version": "prompt_report.v3",
        "step": "prompt",
        "prompt_status": "success",
        "prompt_strategy": "formal_prompt_final",
        "prompt_assembly_mode": "base_scene_control",
        "scene_prompt_mode": str((runtime_prompt_plan or {}).get("scene_prompt_mode", "") or "state_v2t_primary_frame"),
        "scene_prompt_style": str((runtime_prompt_plan or {}).get("scene_prompt_style", "") or "compact_canonical_phrase_v1"),
        "prompt_total_sec": float(total_sec),
        "frames_count": int(frame_files_count),
        "assembly_notes": dict(assembly_notes or {}),
        "base_prompt_path": str(base_prompt_path),
        "base_prompt_size": int(len(base_prompt_bytes)),
        "base_prompt_sha1": _sha1_bytes(base_prompt_bytes),
        "state_prompt_manifest_path": str(state_prompt_manifest_path),
        "state_prompt_manifest_size": int(len(state_prompt_manifest_bytes)),
        "state_prompt_manifest_sha1": _sha1_bytes(state_prompt_manifest_bytes),
        "runtime_prompt_plan_path": str(runtime_prompt_plan_path),
        "runtime_prompt_plan_size": int(len(runtime_prompt_plan_bytes)),
        "runtime_prompt_plan_sha1": _sha1_bytes(runtime_prompt_plan_bytes),
        "outputs": {
            "bytes_sum": 0,
            "prompt_bytes": 0,
            "report_bytes": 0,
            "base_prompt_bytes": int(len(base_prompt_bytes)),
            "state_prompt_manifest_bytes": int(len(state_prompt_manifest_bytes)),
            "runtime_prompt_plan_bytes": int(len(runtime_prompt_plan_bytes)),
        },
        "base_prompt_preview": str((base_prompt_payload or {}).get("base_prompt", "") or ""),
        "negative_prompt_preview": str((base_prompt_payload or {}).get("negative_prompt", "") or ""),
        "state_control_summary": dict((state_prompt_manifest or {}).get("summary", {}) or {}),
        "state_control_statistics": _state_control_statistics(state_prompt_manifest, runtime_prompt_plan),
        "scene_prompt_statistics": _scene_prompt_statistics(state_scene_encoding, runtime_prompt_plan),
        "source_files": {
            "base_prompt": _relative_path(prompt_dir.parent, base_prompt_path),
            "state_prompt_manifest": _relative_path(prompt_dir.parent, state_prompt_manifest_path),
            "runtime_prompt_plan": _relative_path(prompt_dir.parent, runtime_prompt_plan_path),
        },
        "artifact_contract": {
            "formal_files": ["runtime_prompt_plan.json", REPORT_FILENAME],
            "internal_support_files": ["base_prompt.json", "state_prompt_manifest.json"],
        },
    }


def write_prompt_report(prompt_dir, report):
    # type: (Path, dict) -> Path
    prompt_dir = Path(prompt_dir).resolve()
    report_path = prompt_dir / REPORT_FILENAME
    report_obj = dict(report or {})
    report_obj["report_path"] = str(report_path)
    last_size = None
    for _ in range(3):
        write_json_atomic(report_path, report_obj, indent=2)
        report_bytes = report_path.read_bytes()
        report_size = int(len(report_bytes))
        outputs = dict(report_obj.get("outputs", {}) or {})
        outputs["report_bytes"] = report_size
        outputs["prompt_bytes"] = int(
            int(outputs.get("report_bytes", 0) or 0)
            + int(outputs.get("base_prompt_bytes", 0) or 0)
            + int(outputs.get("state_prompt_manifest_bytes", 0) or 0)
            + int(outputs.get("runtime_prompt_plan_bytes", 0) or 0)
        )
        outputs["bytes_sum"] = int(outputs.get("prompt_bytes", 0) or 0)
        report_obj["outputs"] = outputs
        report_obj["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, report_obj, indent=2)
    return report_path
