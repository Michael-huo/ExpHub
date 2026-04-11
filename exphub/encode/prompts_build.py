from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog
from ._base_prompt import build_base_prompt_payload
from ._motion_prompt import build_motion_prompt_payload
from ._scene_prompt import build_scene_prompt_payload


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
        counts[name] = int(counts.get(name, 0)) + 1
    return counts


def load_segment_text_inputs(segment_manifest_path):
    manifest_path = ensure_file(segment_manifest_path, "segment manifest")
    manifest = read_json_dict(manifest_path)
    if not manifest:
        raise RuntimeError("invalid segment manifest: {}".format(manifest_path))

    exp_dir = manifest_path.parent.parent.resolve()
    state_segments_payload = _as_dict(manifest.get("state_segments"))
    if not state_segments_payload:
        raise RuntimeError("segment manifest missing state_segments payload: {}".format(manifest_path))

    frames_meta = _as_dict(manifest.get("frames"))
    return {
        "segment_manifest_path": manifest_path,
        "segment_manifest": manifest,
        "state_segments_payload": state_segments_payload,
        "frame_count": int(frames_meta.get("frame_count", 0) or 0),
        "frame_count_used": int(frames_meta.get("frame_count_used", 0) or 0),
        "exp_dir": exp_dir,
        "source_files": {
            "segment_manifest": _relative_to_exp(exp_dir, manifest_path),
            "state_segments": "{}#state_segments".format(_relative_to_exp(exp_dir, manifest_path)),
        },
    }


def _build_prompt_segment_map(prompt_segments):
    mapping = {}
    for item in list(prompt_segments or []):
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
            "segment_manifest": str((segment_inputs.get("source_files") or {}).get("segment_manifest", "") or ""),
        },
        "summary": {
            "prompt_segment_count": int(len(segments)),
            "state_label_counts": dict(_count_by_key(segments, "state_label")),
            "scene_prompt_segment_count": int(len([item for item in segments if _collapse_ws(item.get("scene_prompt", ""))])),
        },
        "artifact_paths": {
            "segment_manifest": str((segment_inputs.get("source_files") or {}).get("segment_manifest", "") or ""),
            "prompt_manifest": _relative_to_exp(exp_dir, Path(exp_dir) / "prompt" / "prompt_manifest.json"),
        },
    }


def _sha1_bytes(payload_bytes):
    import hashlib

    return hashlib.sha1(payload_bytes).hexdigest()


def build_prompt_report(prompt_dir, prompt_manifest, frame_files_count, total_sec):
    prompt_dir = Path(prompt_dir).resolve()
    prompt_manifest_path = (prompt_dir / "prompt_manifest.json").resolve()
    prompt_manifest_bytes = prompt_manifest_path.read_bytes()
    segments = list(_as_dict(prompt_manifest).get("segments") or [])
    return {
        "report_schema_version": "prompt_report.v5",
        "step": "encode",
        "substage": "text_gen",
        "prompt_status": "success",
        "prompt_structure": "base_scene_motion",
        "prompt_total_sec": float(total_sec),
        "frames_count": int(frame_files_count),
        "prompt_manifest_path": str(prompt_manifest_path),
        "prompt_manifest_size": int(len(prompt_manifest_bytes)),
        "prompt_manifest_sha1": _sha1_bytes(prompt_manifest_bytes),
        "outputs": {
            "bytes_sum": 0,
            "prompt_bytes": 0,
            "report_bytes": 0,
            "prompt_manifest_bytes": int(len(prompt_manifest_bytes)),
        },
        "base_prompt_preview": str(prompt_manifest.get("base_prompt", "") or ""),
        "negative_prompt_preview": str(prompt_manifest.get("negative_prompt", "") or ""),
        "assembly_notes": {
            "scene_prompt_mode": str(prompt_manifest.get("scene_prompt_mode", "") or ""),
            "scene_prompt_style": str(prompt_manifest.get("scene_prompt_style", "") or ""),
            "motion_prompt_mode": str(prompt_manifest.get("motion_prompt_mode", "") or ""),
            "scene_encoding_backend": str(prompt_manifest.get("scene_encoding_backend", "") or ""),
        },
        "prompt_statistics": {
            "prompt_segment_count": int(len(segments)),
            "state_label_counts": dict(_count_by_key(segments, "state_label")),
            "continuity_emphasis_counts": dict(_count_by_key(segments, "continuity_emphasis")),
        },
        "source_files": {
            "prompt_manifest": _relative_to_exp(prompt_dir.parent, prompt_manifest_path),
        },
        "artifact_contract": {
            "formal_files": ["prompt_manifest.json", "report.json"],
            "transitional_files": [],
        },
    }


def write_prompt_report(prompt_dir, report):
    prompt_dir = Path(prompt_dir).resolve()
    report_path = prompt_dir / "report.json"
    report_obj = dict(report or {})
    report_obj["report_path"] = str(report_path)
    last_size = None
    for _ in range(3):
        write_json_atomic(report_path, report_obj, indent=2)
        report_bytes = report_path.read_bytes()
        report_size = int(len(report_bytes))
        outputs = dict(report_obj.get("outputs", {}) or {})
        outputs["report_bytes"] = report_size
        outputs["prompt_bytes"] = int(int(outputs.get("report_bytes", 0) or 0) + int(outputs.get("prompt_manifest_bytes", 0) or 0))
        outputs["bytes_sum"] = int(outputs.get("prompt_bytes", 0) or 0)
        report_obj["outputs"] = outputs
        report_obj["report_size"] = report_size
        if report_size == last_size:
            break
        last_size = report_size
    write_json_atomic(report_path, report_obj, indent=2)
    return report_path


def run_formal_mainline(args):
    total_t0 = time.time()
    exp_dir = Path(args.exp_dir).resolve()
    prompt_dir = (exp_dir / "prompt").resolve()
    prompt_dir.mkdir(parents=True, exist_ok=True)

    segment_inputs = load_segment_text_inputs(args.segment_manifest)
    frames_dir = ensure_dir(exp_dir / "segment" / "frames", "segment frames dir")
    frame_files = [item for item in sorted(frames_dir.iterdir()) if item.is_file()]
    if not frame_files:
        raise RuntimeError("no frame files found in {}".format(frames_dir))

    prompt_manifest = build_prompt_manifest(
        segment_inputs=segment_inputs,
        frames_dir=frames_dir,
        prompt_model_dir=str(args.prompt_model_dir or ""),
    )
    backend_meta = dict(prompt_manifest.get("backend_meta") or {})
    if backend_meta:
        log_info("processor loaded in {:.2f}s".format(float(backend_meta.get("processor_load_sec", 0.0) or 0.0)))
        log_info("model weights loaded in {:.2f}s".format(float(backend_meta.get("model_load_sec", 0.0) or 0.0)))

    write_json_atomic(prompt_dir / "prompt_manifest.json", prompt_manifest, indent=2)
    prompt_report = build_prompt_report(
        prompt_dir=prompt_dir,
        prompt_manifest=prompt_manifest,
        frame_files_count=len(frame_files),
        total_sec=float(time.time() - total_t0),
    )
    prompt_report["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    prompt_report["frames_dir"] = str(frames_dir)
    report_path = write_prompt_report(prompt_dir, prompt_report)

    log_prog("prompt manifest assembled from base + scene + motion")
    log_info("prompt manifest generated: count={}".format(int((prompt_manifest.get("summary") or {}).get("prompt_segment_count", 0) or 0)))
    log_info("scene prompt backend: {}".format(str(prompt_manifest.get("scene_encoding_backend", "") or "<missing>")))
    log_info("wrote: {}".format(prompt_dir / "prompt_manifest.json"))
    log_info("wrote: {}".format(report_path))
    return report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub encode.text_gen mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--segment_manifest", required=True, help="scene_split segment_manifest.json path")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--prompt_model_dir", default="")
    parser.add_argument("--backend_python_phase", default="prompt_smol")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    run_formal_mainline(args)


if __name__ == "__main__":
    main()
