from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from exphub.common.io import write_json_atomic


def _relative_to_root(root_dir, target_path):
    root = Path(root_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(root))
    except Exception:
        return str(target)


def build_dataset_report(
    export_root,
    scope,
    focus_name,
    training_spec,
    targets,
    exported_clips,
    skipped_clips,
    metadata_paths,
):
    root = Path(export_root).resolve()
    counts_by_split = defaultdict(int)
    counts_by_dataset = defaultdict(int)
    split_files = defaultdict(list)
    prompt_span_ids = set()
    total_units_consumed = 0
    shared_prompt_unit_count = 0
    skipped_short_span_count = 0

    for item in list(exported_clips or []):
        split = str(item.get("split", "train") or "train")
        dataset = str(item.get("dataset", "unknown") or "unknown")
        counts_by_split[split] += 1
        counts_by_dataset[dataset] += 1
        split_files[split].append(str(item.get("file_path", "") or ""))
        span_id = str(item.get("source_span_id", "") or "")
        if span_id:
            prompt_span_ids.add(span_id)
        unit_ids = list(item.get("source_unit_ids") or [])
        total_units_consumed += int(len(unit_ids))
        shared_prompt_unit_count += int(max(0, len(unit_ids) - 1))

    for item in list(skipped_clips or []):
        if str(item.get("reason", "") or "").strip() == "span_length_insufficient":
            skipped_short_span_count += 1

    prompt_span_count = int(len(prompt_span_ids))
    clip_count = int(len(list(exported_clips or [])))

    return {
        "report_schema_version": "export_dataset_report.v1",
        "step": "export",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planner": "generation_units",
        "prompt_strategy": "prompt_spans",
        "workflow": "input -> encode -> export",
        "export_scope": str(scope),
        "export_focus": str(focus_name),
        "training_spec": dict(training_spec or {}),
        "dataset_priority": ["ncd", "scand"],
        "sources": {
            "targets": list(targets or []),
        },
        "summary": {
            "bag_count": int(len(list(targets or []))),
            "clip_count": int(clip_count),
            "exported_clip_count": int(len(list(exported_clips or []))),
            "skipped_clip_count": int(len(list(skipped_clips or []))),
            "total_units_consumed": int(total_units_consumed),
            "prompt_span_count": int(prompt_span_count),
            "shared_prompt_unit_count": int(shared_prompt_unit_count),
            "prompt_reuse_ratio": float(float(shared_prompt_unit_count) / float(total_units_consumed)) if total_units_consumed > 0 else 0.0,
            "mean_clips_per_span": float(float(clip_count) / float(prompt_span_count)) if prompt_span_count > 0 else 0.0,
            "skipped_short_span_count": int(skipped_short_span_count),
            "split_counts": dict(counts_by_split),
            "dataset_counts": dict(counts_by_dataset),
        },
        "outputs": {
            "clips_dir": _relative_to_root(root, root / "clips"),
            "metadata_dir": _relative_to_root(root, root / "metadata"),
            "clip_manifests_dir": _relative_to_root(root, root / "clip_manifests"),
            "metadata_files": {
                split: _relative_to_root(root, path)
                for split, path in dict(metadata_paths or {}).items()
            },
            "split_files": dict(split_files),
        },
        "exported_clips": list(exported_clips or []),
        "skipped_clips": list(skipped_clips or []),
    }


def write_dataset_report(export_root, report_obj):
    report_path = (Path(export_root).resolve() / "dataset_report.json").resolve()
    write_json_atomic(report_path, dict(report_obj or {}), indent=2)
    return report_path
