from __future__ import annotations

from pathlib import Path

from exphub.common.io import write_json_atomic


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _motion_label_counts(motion_segments):
    counts = {}
    for item in list(_as_dict(motion_segments).get("motion_states") or []):
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
            "encode_result_path": "encode/encode_result.json",
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
    semantic_summary = _as_dict(_as_dict(semantic_anchors).get("summary"))
    semantic_policy = _as_dict(_as_dict(semantic_anchors).get("policy"))
    unit_summary = _as_dict(_as_dict(generation_units).get("summary"))
    return {
        "version": 1,
        "source": "encode.result.v1",
        "num_motion_states": int(len(list(_as_dict(motion_segments).get("motion_states") or []))),
        "num_semantic_states": int(semantic_summary.get("semantic_state_count", 0) or 0),
        "visual_anchor_count": int(semantic_summary.get("visual_anchor_count", 0) or 0),
        "coverage_gap_count": int(semantic_summary.get("coverage_gap_count", 0) or 0),
        "num_generation_units": int(len(units)),
        "num_prompt_units": int(len(prompt_units)),
        "motion_labels": _motion_label_counts(motion_segments),
        "unit_lengths": [int(item.get("length", item.get("duration_frames", 0)) or 0) for item in units],
        "unit_length_guard_count": int(unit_summary.get("unit_length_guard_count", 0) or 0),
        "max_unit_span_frames": int(semantic_policy.get("max_unit_span_frames", 0) or 0),
        "prompt_schema": "prompts.v3",
        "prompt_strategy": str(_as_dict(prompts).get("prompt_strategy", "base_motion_fixed_prompt_v1") or ""),
        "prompt_source": "prompts.prompt_positive",
        "semantic_state_source": str(_as_dict(semantic_anchors).get("source", "") or ""),
        "anchor_backend": "image_embedding_visual_anchor",
        "artifacts": {
            "motion_segments": _artifact_rel(paths, "encode_motion_segments_path"),
            "semantic_anchors": _artifact_rel(paths, "encode_semantic_anchors_path"),
            "generation_units": _artifact_rel(paths, "encode_generation_units_path"),
            "prompts": _artifact_rel(paths, "encode_prompts_path"),
            "encode_result": _artifact_rel(paths, "encode_result_path"),
            "overview": _artifact_rel(paths, "encode_overview_path"),
        },
    }


def write_encode_overview(output_path, motion_segments, semantic_anchors, generation_units):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    for motion_state in list(_as_dict(motion_segments).get("motion_states") or []):
        start = int(motion_state.get("start_idx", 0) or 0)
        end = int(motion_state.get("end_idx", start) or start)
        label = str(motion_state.get("motion_label", "mixed") or "mixed")
        max_idx = max(max_idx, end)
        ax_motion.axvspan(start, end, color=motion_colors.get(label, "#8c564b"), alpha=0.35)
        ax_motion.text((start + end) / 2.0, 0.5, label, ha="center", va="center", fontsize=9)
        ax_motion.axvline(start, color="#333333", alpha=0.35, linewidth=1)
    if list(_as_dict(motion_segments).get("motion_states") or []):
        final_motion_state = _as_dict(list(_as_dict(motion_segments).get("motion_states") or [])[-1])
        ax_motion.axvline(int(final_motion_state.get("end_idx", 0) or 0), color="#333333", alpha=0.35, linewidth=1)
    ax_motion.set_yticks([])
    ax_motion.set_ylabel("Motion")
    ax_motion.set_title("Motion states")
    ax_motion.legend(
        handles=[Line2D([0], [0], color="#333333", linewidth=1, label="motion boundary")],
        loc="upper right",
        fontsize=8,
    )

    semantic_colors = {
        "semantic_state_start": "#2ca02c",
        "semantic_update": "#e377c2",
        "coverage_gap": "#8c564b",
    }
    semantic_markers = {
        "semantic_state_start": "^",
        "semantic_update": "o",
        "coverage_gap": "s",
    }
    seen_semantic_events = set()
    for group in list(_as_dict(semantic_anchors).get("motion_states") or []):
        for item in list(_as_dict(group).get("semantic_events") or []):
            idx = int(item.get("frame_idx", 0) or 0)
            event_type = str(item.get("event_type", "") or "semantic_state_start")
            reason = str(item.get("reason", "") or "")
            event_key = reason if reason in semantic_colors else event_type
            score = float(item.get("image_novelty", 0.0) or 0.0)
            max_idx = max(max_idx, idx)
            label = event_key if event_key not in seen_semantic_events else None
            seen_semantic_events.add(event_key)
            ax_semantic.axvline(idx, color=semantic_colors.get(event_key, "#17becf"), alpha=0.45, linewidth=1)
            ax_semantic.scatter(
                [idx],
                [score],
                color=semantic_colors.get(event_key, "#17becf"),
                marker=semantic_markers.get(event_key, "o"),
                s=45,
                label=label,
            )
    ax_semantic.set_ylabel("Image novelty")
    ax_semantic.set_title("Visual anchor states")
    semantic_handles = []
    for label, marker in [
        ("semantic state start", "^"),
        ("semantic update", "o"),
        ("coverage gap", "s"),
    ]:
        semantic_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=semantic_colors[label.replace(" ", "_").replace("-", "_")],
                markersize=7,
                label=label,
            )
        )
    ax_semantic.legend(handles=semantic_handles, loc="upper right", fontsize=8)

    unit_boundary_colors = {
        "unit_boundary": "#17becf",
        "motion_boundary": "#333333",
        "semantic_boundary": "#2ca02c",
        "unit_length_guard": "#ff7f0e",
    }
    seen_unit_sources = set()
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
        for source in list(unit.get("unit_boundary_sources") or ["unit_boundary"]):
            source = str(source or "unit_boundary")
            unit_color = unit_boundary_colors.get(source, unit_boundary_colors["unit_boundary"])
            ax_units.axvline(end, color=unit_color, alpha=0.65, linewidth=1.2)
            seen_unit_sources.add(source)
    ax_units.set_yticks([])
    ax_units.set_ylabel("Units")
    ax_units.set_title("Generation units")
    ax_units.set_xlabel("frame_idx")
    ax_units.set_xlim(0, max(1, int(max_idx)))
    unit_handles = [
        Line2D([0], [0], color=unit_boundary_colors["unit_boundary"], linewidth=1.2, label="unit boundary"),
        Line2D([0], [0], color=unit_boundary_colors["motion_boundary"], linewidth=1.2, label="motion boundary"),
        Line2D([0], [0], color=unit_boundary_colors["semantic_boundary"], linewidth=1.2, label="semantic boundary"),
        Line2D([0], [0], color=unit_boundary_colors["unit_length_guard"], linewidth=1.2, label="unit length guard"),
    ]
    ax_units.legend(handles=unit_handles, loc="upper right", fontsize=8)
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
