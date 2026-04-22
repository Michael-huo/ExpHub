from __future__ import annotations

from pathlib import Path

from exphub.common.io import write_json_atomic


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _motion_label_counts(motion_segments):
    counts = {}
    for item in list(_as_dict(motion_segments).get("segments") or []):
        label = str(_as_dict(item).get("motion_label", "") or "unknown")
        counts[label] = int(counts.get(label, 0)) + 1
    return counts


def _artifact_rel(paths, attr_name):
    if paths is None:
        defaults = {
            "encode_motion_segments_path": "encode/motion_segments.json",
            "encode_semantic_anchors_path": "encode/semantic_anchors.json",
            "encode_generation_units_path": "encode/generation_units.json",
            "encode_prompts_path": "encode/prompts.json",
            "encode_overview_path": "encode/encode_overview.png",
        }
        return defaults.get(str(attr_name), str(attr_name))
    path = Path(getattr(paths, attr_name)).resolve()
    exp_dir = Path(getattr(paths, "exp_dir", path.parent.parent)).resolve()
    try:
        return path.relative_to(exp_dir).as_posix()
    except Exception:
        return path.name


def _build_encode_result(motion_segments, semantic_anchors, generation_units, prompts, paths=None):
    units = list(_as_dict(generation_units).get("units") or [])
    prompt_units = list(_as_dict(prompts).get("units") or [])
    return {
        "version": 1,
        "source": "encode.result.v1",
        "num_motion_segments": int(len(list(_as_dict(motion_segments).get("segments") or []))),
        "num_semantic_anchor_groups": int(len(list(_as_dict(semantic_anchors).get("segments") or []))),
        "num_generation_units": int(len(units)),
        "num_prompt_units": int(len(prompt_units)),
        "motion_labels": _motion_label_counts(motion_segments),
        "unit_lengths": [int(item.get("length", item.get("duration_frames", 0)) or 0) for item in units],
        "prompt_mode": "base+motion+semantic",
        "artifacts": {
            "motion_segments": _artifact_rel(paths, "encode_motion_segments_path"),
            "semantic_anchors": _artifact_rel(paths, "encode_semantic_anchors_path"),
            "generation_units": _artifact_rel(paths, "encode_generation_units_path"),
            "prompts": _artifact_rel(paths, "encode_prompts_path"),
            "overview": _artifact_rel(paths, "encode_overview_path"),
        },
    }


def write_encode_overview(output_path, motion_segments, semantic_anchors, generation_units):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion_colors = {
        "stop": "#7f8c8d",
        "forward": "#2ca02c",
        "left_turn": "#1f77b4",
        "right_turn": "#d62728",
        "mixed": "#9467bd",
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    ax_motion, ax_semantic, ax_units = axes

    max_idx = 0
    for segment in list(_as_dict(motion_segments).get("segments") or []):
        start = int(segment.get("start_idx", 0) or 0)
        end = int(segment.get("end_idx", start) or start)
        label = str(segment.get("motion_label", "mixed") or "mixed")
        max_idx = max(max_idx, end)
        ax_motion.axvspan(start, end, color=motion_colors.get(label, "#8c564b"), alpha=0.35)
        ax_motion.text((start + end) / 2.0, 0.5, label, ha="center", va="center", fontsize=9)
    ax_motion.set_yticks([])
    ax_motion.set_ylabel("Motion")
    ax_motion.set_title("Motion segmentation")

    gap_rows = list(_as_dict(semantic_anchors).get("gap_rows") or [])
    if gap_rows:
        xs = [int(item.get("frame_idx", 0) or 0) for item in gap_rows]
        ys = [float(item.get("score", 0.0) or 0.0) for item in gap_rows]
        ax_semantic.plot(xs, ys, color="#444444", linewidth=1.2, label="semantic gap")
    reason_colors = {"segment_boundary": "#333333", "semantic_gain": "#e377c2", "duration_fallback": "#ff7f0e"}
    reason_markers = {"segment_boundary": "|", "semantic_gain": "o", "duration_fallback": "s"}
    seen_reasons = set()
    for group in list(_as_dict(semantic_anchors).get("segments") or []):
        for item in list(_as_dict(group).get("anchor_items") or []):
            idx = int(item.get("frame_idx", 0) or 0)
            reason = str(item.get("reason", "") or "segment_boundary")
            score = float(item.get("score", 0.0) or 0.0)
            max_idx = max(max_idx, idx)
            label = reason if reason not in seen_reasons else None
            seen_reasons.add(reason)
            ax_semantic.axvline(idx, color=reason_colors.get(reason, "#17becf"), alpha=0.45, linewidth=1)
            ax_semantic.scatter(
                [idx],
                [score],
                color=reason_colors.get(reason, "#17becf"),
                marker=reason_markers.get(reason, "o"),
                s=45 if reason != "segment_boundary" else 70,
                label=label,
            )
    ax_semantic.set_ylabel("Anchor score")
    ax_semantic.set_title("Semantic anchors")
    if seen_reasons:
        ax_semantic.legend(loc="upper right", fontsize=8)

    for unit_idx, unit in enumerate(list(_as_dict(generation_units).get("units") or [])):
        start = int(unit.get("start_idx", 0) or 0)
        end = int(unit.get("end_idx", start) or start)
        label = "{}\n{}-{} len={}\n{}".format(
            str(unit.get("unit_id", "") or ""),
            start,
            end,
            int(unit.get("length", 0) or 0),
            str(unit.get("motion_label", "") or ""),
        )
        max_idx = max(max_idx, end)
        color = "#17becf" if unit_idx % 2 == 0 else "#bcbd22"
        ax_units.axvspan(start, end, color=color, alpha=0.35)
        ax_units.text((start + end) / 2.0, 0.5, label, ha="center", va="center", fontsize=8)
    ax_units.set_yticks([])
    ax_units.set_ylabel("Units")
    ax_units.set_title("Generation units")
    ax_units.set_xlabel("frame_idx")
    ax_units.set_xlim(0, max(1, int(max_idx)))
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=140)
    plt.close(fig)


def write_encode_outputs(
    runtime,
    prepare_result,
    motion_segments,
    semantic_anchors,
    generation_units,
    prompts,
    elapsed_sec=0.0,
    paths=None,
):
    del prepare_result, elapsed_sec
    out_paths = paths or runtime.paths
    out_paths.encode_dir.mkdir(parents=True, exist_ok=True)

    encode_result = _build_encode_result(motion_segments, semantic_anchors, generation_units, prompts, paths=out_paths)

    write_json_atomic(out_paths.encode_motion_segments_path, motion_segments, indent=2)
    write_json_atomic(out_paths.encode_semantic_anchors_path, semantic_anchors, indent=2)
    write_json_atomic(out_paths.encode_generation_units_path, generation_units, indent=2)
    write_json_atomic(out_paths.encode_prompts_path, prompts, indent=2)
    write_json_atomic(out_paths.encode_result_path, encode_result, indent=2)
    write_encode_overview(out_paths.encode_overview_path, motion_segments, semantic_anchors, generation_units)
    return out_paths.encode_result_path
