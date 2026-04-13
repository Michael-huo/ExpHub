from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_file, read_json_dict, write_json_atomic
from exphub.common.logging import log_info
from ._scene_prompt import build_scene_prompt_payload


INVARIANT_BASE_POSITIVE_LINES = [
    "Maintain first-person viewpoint continuity across the full sequence.",
    "Preserve stable scene geometry, perspective, and camera alignment.",
    "Keep exposure and white balance stable over time.",
    "Preserve temporal coherence without flicker or drifting structure.",
]

INVARIANT_BASE_NEGATIVE_ITEMS = [
    "flickering",
    "warping",
    "ghosting",
    "geometry drift",
    "inconsistent perspective",
    "exposure instability",
    "white balance shifts",
    "rolling shutter wobble",
    "texture swimming",
    "motion tearing",
    "double edges",
    "heavy blur",
    "low quality",
]

DEFAULT_MOTION_PROMPT = {
    "motion_prompt": "steady egomotion, smooth viewpoint progression.",
    "negative_prompt_delta": "",
    "continuity_emphasis": "balanced",
}

MOTION_PROMPT_BY_STATE = {
    "low_state": {
        "motion_prompt": "steady egomotion, smooth viewpoint progression.",
        "negative_prompt_delta": "",
        "continuity_emphasis": "steady",
    },
    "high_state": {
        "motion_prompt": "elevated motion change, preserve transition continuity and camera stability.",
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "continuity_emphasis": "reinforced",
    },
}

MOTION_PROMPT_BY_PLANNER_LABEL = {
    "steady": {
        "motion_prompt": "steady egomotion, smooth viewpoint progression.",
        "negative_prompt_delta": "",
        "continuity_emphasis": "steady",
    },
    "mixed": {
        "motion_prompt": "moderate motion change, keep transitions coherent and camera movement readable.",
        "negative_prompt_delta": "jerky motion, inconsistent transition timing",
        "continuity_emphasis": "balanced",
    },
    "dynamic": {
        "motion_prompt": "dynamic motion change, preserve transition continuity and camera stability.",
        "negative_prompt_delta": "abrupt perspective jumps, transition discontinuity, motion tearing",
        "continuity_emphasis": "reinforced",
    },
}


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _relative_to_exp(exp_dir, target_path):
    exp_root = Path(exp_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(exp_root))
    except Exception:
        return str(target)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _join_prompt_parts(*parts):
    values = [_collapse_ws(part) for part in list(parts or [])]
    values = [item for item in values if item]
    return _collapse_ws(" ".join(values))


def _join_negative_prompt(base_negative_prompt, negative_delta):
    base = _collapse_ws(base_negative_prompt)
    delta = _collapse_ws(negative_delta)
    if not base:
        return delta
    if not delta:
        return base
    return "{}, {}".format(base, delta)


def _labeled_prompt_clause(label, text):
    body = _collapse_ws(text).rstrip(" ,;:.")
    if not body:
        return ""
    return "{}: {}.".format(str(label), body)


def _count_by_key(items, key):
    counts = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        name = _collapse_ws(item.get(key, "")) or "unknown"
        counts[name] = int(counts.get(name, 0) or 0) + 1
    return counts


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def get_base_prompt():
    return _collapse_ws(" ".join([str(item).strip() for item in INVARIANT_BASE_POSITIVE_LINES if str(item).strip()]))


def get_negative_prompt():
    return ", ".join([str(item).strip() for item in INVARIANT_BASE_NEGATIVE_ITEMS if str(item).strip()]).strip()


def build_base_prompt_payload():
    return {
        "version": 1,
        "schema": "base_prompt.v1",
        "base_prompt": get_base_prompt(),
        "negative_prompt": get_negative_prompt(),
        "source": "encode.text_gen.base_prompt",
        "geometry_constraints_included": True,
    }


def _motion_payload_for_state_label(state_label):
    motion_cfg = dict(DEFAULT_MOTION_PROMPT)
    motion_cfg.update(_as_dict(MOTION_PROMPT_BY_STATE.get(str(state_label))))
    return motion_cfg


def build_motion_prompt_payload(segment_inputs):
    state_payload = _as_dict(segment_inputs.get("state_segments_payload"))
    state_rows = list(state_payload.get("segments") or [])
    if not state_rows:
        raise RuntimeError("segment manifest has no state segments for motion prompts")

    segments = []
    for idx, raw_item in enumerate(state_rows):
        item = _as_dict(raw_item)
        state_label = str(item.get("state_label", "state_unlabeled") or "state_unlabeled")
        motion_cfg = _motion_payload_for_state_label(state_label)
        state_segment_id = int(item.get("segment_id", idx) or idx)
        segments.append(
            {
                "state_segment_id": int(state_segment_id),
                "state_label": str(state_label),
                "start_frame": int(item.get("start_frame", 0) or 0),
                "end_frame": int(item.get("end_frame", 0) or 0),
                "motion_prompt": str(motion_cfg.get("motion_prompt", "") or ""),
                "negative_prompt_delta": str(motion_cfg.get("negative_prompt_delta", "") or ""),
                "continuity_emphasis": str(motion_cfg.get("continuity_emphasis", "balanced") or "balanced"),
                "motion_prompt_source": "scene_split.state_segments",
            }
        )

    return {
        "version": 1,
        "schema": "motion_prompt_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "encode.text_gen.motion_prompt",
        "motion_prompt_mode": "state_label_mapping_v1",
        "segments": segments,
        "summary": {
            "state_segment_count": int(len(segments)),
        },
    }


def resolve_motion_prompt_from_planner_label(motion_label):
    payload = dict(DEFAULT_MOTION_PROMPT)
    payload.update(_as_dict(MOTION_PROMPT_BY_PLANNER_LABEL.get(str(motion_label or "").strip().lower())))
    return payload


def _join_prompt_parts(*parts):
    values = [_collapse_ws(part) for part in list(parts or [])]
    values = [item for item in values if item]
    return _collapse_ws(" ".join(values))


def _join_negative_prompt(base_negative_prompt, negative_delta):
    base = _collapse_ws(base_negative_prompt)
    delta = _collapse_ws(negative_delta)
    if not base:
        return delta
    if not delta:
        return base
    return "{}, {}".format(base, delta)


def _labeled_prompt_clause(label, text):
    body = _collapse_ws(text).rstrip(" ,;:.")
    if not body:
        return ""
    return "{}: {}.".format(str(label), body)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _overlap_frames(start_a, end_a, start_b, end_b):
    left = max(int(start_a), int(start_b))
    right = min(int(end_a), int(end_b))
    return max(0, int(right) - int(left) + 1)


def _segment_prompt_map(prompt_manifest):
    mapping = {}
    for idx, raw_item in enumerate(list(_as_dict(prompt_manifest).get("segments") or [])):
        item = _as_dict(raw_item)
        mapping[_safe_int(item.get("state_segment_id"), idx)] = item
    return mapping


def _resolve_scene_prompt(span_units, prompt_segment_map):
    weight_by_prompt = {}
    for unit in list(span_units or []):
        unit_start = _safe_int(unit.get("anchor_start_idx"), 0)
        unit_end = _safe_int(unit.get("anchor_end_idx"), 0)
        for segment_id in list(unit.get("source_segment_ids") or []):
            prompt_item = _as_dict(prompt_segment_map.get(_safe_int(segment_id), {}))
            if not prompt_item:
                continue
            scene_prompt = _collapse_ws(prompt_item.get("scene_prompt", ""))
            if not scene_prompt:
                continue
            overlap = _overlap_frames(
                unit_start,
                unit_end,
                _safe_int(prompt_item.get("start_frame"), 0),
                _safe_int(prompt_item.get("end_frame"), 0),
            )
            if overlap <= 0:
                overlap = max(
                    1,
                    _safe_int(prompt_item.get("end_frame"), 0) - _safe_int(prompt_item.get("start_frame"), 0) + 1,
                )
            weight_by_prompt[scene_prompt] = int(weight_by_prompt.get(scene_prompt, 0) + overlap)
    if not weight_by_prompt:
        return ""
    return max(weight_by_prompt.items(), key=lambda item: (int(item[1]), item[0]))[0]


def build_prompt_spans_payload(prompt_manifest, generation_units_payload):
    prompt_payload = _as_dict(prompt_manifest)
    units_payload = _as_dict(generation_units_payload)
    units = list(units_payload.get("units") or [])
    if not units:
        raise RuntimeError("prompt spans require generation units")

    base_prompt = str(prompt_payload.get("base_prompt", "") or "")
    negative_prompt = str(prompt_payload.get("negative_prompt", "") or "")
    prompt_segment_map = _segment_prompt_map(prompt_payload)

    grouped = {}
    for raw_unit in units:
        unit = _as_dict(raw_unit)
        prompt_ref = _as_dict(unit.get("prompt_ref"))
        span_id = str(prompt_ref.get("span_id", "") or "")
        if not span_id:
            raise RuntimeError("generation unit missing prompt_ref.span_id")
        grouped.setdefault(span_id, []).append(unit)

    ordered_span_ids = sorted(grouped.keys())
    spans = []
    for span_id in ordered_span_ids:
        span_units = list(grouped.get(span_id) or [])
        motion_label = str(span_units[0].get("motion_label", "steady") or "steady")
        scene_label = str(span_units[0].get("scene_label", "scene_group_000") or "scene_group_000")
        motion_prompt_payload = resolve_motion_prompt_from_planner_label(motion_label)
        scene_prompt = _resolve_scene_prompt(span_units, prompt_segment_map)
        resolved_prompt = _join_prompt_parts(
            base_prompt,
            _labeled_prompt_clause("Scene", scene_prompt),
            _labeled_prompt_clause("Motion", motion_prompt_payload.get("motion_prompt", "")),
        )
        spans.append(
            {
                "span_id": str(span_id),
                "scene_label": str(scene_label),
                "motion_label": str(motion_label),
                "anchor_start_idx": int(min([_safe_int(unit.get("anchor_start_idx"), 0) for unit in span_units])),
                "anchor_end_idx": int(max([_safe_int(unit.get("anchor_end_idx"), 0) for unit in span_units])),
                "unit_ids": [str(unit.get("unit_id", "") or "") for unit in span_units],
                "source_segment_ids": sorted(
                    set([int(segment_id) for unit in span_units for segment_id in list(unit.get("source_segment_ids") or [])])
                ),
                "shared_unit_count": int(len(span_units)),
                "base_prompt": str(base_prompt),
                "scene_prompt": str(scene_prompt),
                "scene_prompt_source": "prompt_manifest.scene_prompt_majority_overlap",
                "motion_prompt": str(motion_prompt_payload.get("motion_prompt", "") or ""),
                "motion_prompt_source": "planner.motion_label_mapping",
                "negative_prompt_delta": str(motion_prompt_payload.get("negative_prompt_delta", "") or ""),
                "continuity_emphasis": str(motion_prompt_payload.get("continuity_emphasis", "balanced") or "balanced"),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": _join_negative_prompt(negative_prompt, motion_prompt_payload.get("negative_prompt_delta", "")),
            }
        )

    return {
        "version": 1,
        "schema": "prompt_spans.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "text_gen",
        "source": "encode.text_gen.prompt_spans",
        "prompt_structure": "base_scene_motion",
        "grouping_policy": "contiguous_generation_units_share_prompt_when_prompt_ref.span_id_matches",
        "base_prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "spans": spans,
        "summary": {
            "span_count": int(len(spans)),
            "shared_prompt_unit_count": int(sum([int(item.get("shared_unit_count", 0) or 0) for item in spans])),
            "multi_unit_span_count": int(len([item for item in spans if int(item.get("shared_unit_count", 0) or 0) > 1])),
        },
        "artifact_paths": {
            "encode_plan": "encode/encode_plan.json",
            "prompt_spans": "encode/prompt_spans.json",
        },
    }


def load_input_text_inputs(input_report_path):
    report_path = ensure_file(input_report_path, "input report")
    report = read_json_dict(report_path)
    if not report:
        raise RuntimeError("invalid input report: {}".format(report_path))

    exp_dir = report_path.parent.parent.resolve()
    state_segments_payload = _as_dict(report.get("state_segments"))
    if not state_segments_payload:
        raise RuntimeError("input report missing state_segments payload: {}".format(report_path))

    frames_meta = _as_dict(report.get("frames"))
    return {
        "input_report_path": report_path,
        "input_report": report,
        "state_segments_payload": state_segments_payload,
        "frame_count": int(frames_meta.get("frame_count", 0) or 0),
        "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
        "exp_dir": exp_dir,
        "source_files": {
            "input_report": _relative_to_exp(exp_dir, report_path),
            "state_segments": "{}#state_segments".format(_relative_to_exp(exp_dir, report_path)),
        },
    }


def _build_prompt_segment_map(prompt_segments):
    mapping = {}
    for item in list(prompt_segments or []):
        if not isinstance(item, dict):
            continue
        state_segment_id = _safe_int(item.get("state_segment_id"), -1)
        if state_segment_id < 0:
            continue
        mapping[int(state_segment_id)] = dict(item)
    return mapping


def build_prompt_manifest(segment_inputs, frames_dir, prompt_model_dir=""):
    base_prompt_payload = build_base_prompt_payload()
    scene_prompt_payload = build_scene_prompt_payload(
        segment_inputs=segment_inputs,
        frames_dir=frames_dir,
        prompt_model_dir=prompt_model_dir,
    )
    motion_prompt_payload = build_motion_prompt_payload(segment_inputs)

    scene_map = _build_prompt_segment_map(scene_prompt_payload.get("segments"))
    motion_map = _build_prompt_segment_map(motion_prompt_payload.get("segments"))
    state_rows = list(_as_dict(segment_inputs.get("state_segments_payload")).get("segments") or [])
    exp_dir = Path(segment_inputs.get("exp_dir")).resolve()

    base_prompt = str(base_prompt_payload.get("base_prompt", "") or "")
    negative_prompt = str(base_prompt_payload.get("negative_prompt", "") or "")
    segments = []
    for idx, raw_item in enumerate(state_rows):
        state_row = _as_dict(raw_item)
        state_segment_id = _safe_int(state_row.get("segment_id"), idx)
        scene_item = _as_dict(scene_map.get(state_segment_id))
        motion_item = _as_dict(motion_map.get(state_segment_id))
        if not scene_item:
            raise RuntimeError("missing scene prompt for state segment {}".format(state_segment_id))
        if not motion_item:
            raise RuntimeError("missing motion prompt for state segment {}".format(state_segment_id))

        scene_prompt = _collapse_ws(scene_item.get("scene_prompt", ""))
        motion_prompt = _collapse_ws(motion_item.get("motion_prompt", ""))
        negative_prompt_delta = _collapse_ws(motion_item.get("negative_prompt_delta", ""))
        resolved_prompt = _join_prompt_parts(
            base_prompt,
            _labeled_prompt_clause("Scene", scene_prompt),
            _labeled_prompt_clause("Motion", motion_prompt),
        )
        segments.append(
            {
                "prompt_segment_id": int(state_segment_id),
                "state_segment_id": int(state_segment_id),
                "source_segment_id": int(state_segment_id),
                "state_label": str(state_row.get("state_label", "state_unlabeled") or "state_unlabeled"),
                "start_frame": int(state_row.get("start_frame", 0) or 0),
                "end_frame": int(state_row.get("end_frame", 0) or 0),
                "scene_prompt": str(scene_prompt),
                "motion_prompt": str(motion_prompt),
                "negative_prompt_delta": str(negative_prompt_delta),
                "continuity_emphasis": str(motion_item.get("continuity_emphasis", "balanced") or "balanced"),
                "scene_prompt_source": str(scene_item.get("scene_prompt_source", "") or ""),
                "motion_prompt_source": str(motion_item.get("motion_prompt_source", "") or ""),
                "scene_encoding_backend": str(scene_item.get("scene_encoding_backend", "") or ""),
                "representative_frame": dict(scene_item.get("representative_frame") or {}),
                "resolved_prompt": str(resolved_prompt),
                "negative_prompt": _join_negative_prompt(negative_prompt, negative_prompt_delta),
            }
        )

    return {
        "version": 1,
        "schema": "prompt_manifest.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "encode",
        "substage": "text_gen",
        "source": "encode.text_gen",
        "prompt_structure": "base_scene_motion",
        "base_prompt": str(base_prompt),
        "negative_prompt": str(negative_prompt),
        "scene_prompt_mode": str(scene_prompt_payload.get("scene_prompt_mode", "") or "smolvlm2_primary_frame"),
        "scene_prompt_style": str(scene_prompt_payload.get("scene_prompt_style", "") or "compact_canonical_phrase_v1"),
        "motion_prompt_mode": str(motion_prompt_payload.get("motion_prompt_mode", "") or "state_label_mapping_v1"),
        "scene_encoding_backend": str(scene_prompt_payload.get("backend", "") or ""),
        "backend_meta": dict(scene_prompt_payload.get("backend_meta") or {}),
        "segments": segments,
        "source_files": {
            "input_report": str((segment_inputs.get("source_files") or {}).get("input_report", "") or ""),
        },
        "summary": {
            "prompt_segment_count": int(len(segments)),
            "state_label_counts": dict(_count_by_key(segments, "state_label")),
            "scene_prompt_segment_count": int(len([item for item in segments if _collapse_ws(item.get("scene_prompt", ""))])),
        },
        "artifact_paths": {
            "input_report": str((segment_inputs.get("source_files") or {}).get("input_report", "") or ""),
        },
    }


def _run_formal_mainline(args):
    exp_dir = Path(args.exp_dir).resolve()
    input_report_path = ensure_file(args.input_report, "input report")
    out_path = Path(args.out_path).resolve()
    frames_dir = (exp_dir / "input" / "frames").resolve()
    started = time.time()

    segment_inputs = load_input_text_inputs(input_report_path)
    prompt_manifest = build_prompt_manifest(
        segment_inputs=segment_inputs,
        frames_dir=frames_dir,
        prompt_model_dir=str(args.prompt_model_dir or ""),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, prompt_manifest, indent=2)

    log_info(
        "prompt manifest generated: count={} sec={:.2f}".format(
            int((prompt_manifest.get("summary") or {}).get("prompt_segment_count", 0) or 0),
            float(time.time() - started),
        )
    )
    log_info("wrote: {}".format(out_path))
    return out_path


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--input_report", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--prompt_model_dir", default="")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("prompt helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
