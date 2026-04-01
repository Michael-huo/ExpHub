from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import ensure_dir, ensure_file, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_prompt, log_warn
from exphub.contracts import prompt as prompt_contract
from exphub.pipeline.prompt.base_prompt import build_base_prompt_payload
from exphub.pipeline.prompt.backends import create_backend
from exphub.pipeline.prompt.backends.smolvlm2 import DEFAULT_SMOLVLM2_MODEL_ID
from exphub.pipeline.prompt.reporting import (
    cleanup_legacy_prompt_outputs,
    build_prompt_report,
    write_prompt_report,
)
from exphub.pipeline.prompt.runtime_plan import build_runtime_prompt_plan
from exphub.pipeline.prompt.state_manifest import build_state_prompt_manifest, load_segment_prompt_inputs


_BAR_FORMAT = "[BAR] {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_NATURAL_RE = re.compile(r"(\d+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9_]+")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_PROFILE_VERSION = 1
SCENE_TYPES = [
    "park_walkway",
    "campus_path",
    "orchard_row",
    "field_path",
    "road_edge",
    "corridor",
    "indoor_walkway",
    "unknown",
]  # type: List[str]
SURFACE_TYPES = ["pavement", "concrete", "brick", "dirt", "mixed", "unknown"]  # type: List[str]
SIDE_STRUCTURES = [
    "grass",
    "trees",
    "shrubs",
    "walls",
    "columns",
    "fence",
    "buildings",
    "crops",
    "soil_edges",
    "hedges",
]  # type: List[str]
LIGHTING_TYPES = ["daylight", "overcast", "indoor_even", "unknown"]  # type: List[str]
DYNAMIC_RISKS = ["low", "people", "vehicles", "animals", "mixed"]  # type: List[str]
REPETITION_RISKS = ["low", "medium", "high"]  # type: List[str]
PROFILE_CONFIDENCES = ["low", "medium", "high"]  # type: List[str]
SIDE_STRUCTURE_PRIORITY = list(SIDE_STRUCTURES)
CONFIDENCE_WEIGHTS = {"low": 1, "medium": 2, "high": 3}  # type: Dict[str, int]


def run(runtime):
    contract = prompt_contract.build_contract(runtime.paths)
    ensure_dir(runtime.paths.segment_dir, "segment dir")
    ensure_dir(runtime.paths.segment_frames_dir, "segment frames dir")
    ensure_file(runtime.paths.segment_manifest_path, "segment manifest")

    runtime.paths.exp_dir.mkdir(parents=True, exist_ok=True)
    runtime.remove_in_exp(runtime.paths.prompt_dir)

    prompt_phase = runtime.prompt_phase_name()
    prompt_python = runtime.phase_python(prompt_phase)
    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "prompt" / "service.py").resolve()

    cmd = [
        prompt_python,
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
        "--segment_manifest",
        str(runtime.paths.segment_manifest_path),
        "--fps",
        runtime.fps_arg,
        "--backend",
        str(runtime.args.prompt_backend),
        "--model_ref",
        str(runtime.prompt_model_ref() or DEFAULT_SMOLVLM2_MODEL_ID),
        "--dtype",
        str(runtime.args.prompt_dtype),
        "--sample_mode",
        str(runtime.args.prompt_sample_mode),
        "--num_images",
        str(runtime.args.prompt_num_images),
        "--backend_python_phase",
        str(prompt_phase),
    ]
    runtime.step_runner.run_env_python(cmd, phase_name=prompt_phase, log_name="prompt.log", cwd=runtime.exphub_root)

    ensure_file(contract.artifacts[prompt_contract.REPORT], "prompt report")
    ensure_file(contract.artifacts[prompt_contract.BASE_PROMPT], "prompt base prompt")
    ensure_file(contract.artifacts[prompt_contract.STATE_PROMPT_MANIFEST], "state prompt manifest")
    ensure_file(contract.artifacts[prompt_contract.RUNTIME_PROMPT_PLAN], "runtime prompt plan")
    return contract.artifacts[prompt_contract.REPORT]


def _maybe_progress(iterable):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc="Prompt Profile", bar_format=_BAR_FORMAT)


def _natural_sort_key(path_text):
    # type: (str) -> List[object]
    text = str(path_text)
    name = Path(text).name
    parts = _NATURAL_RE.split(name)
    out = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def _list_images(image_dir):
    # type: (Path) -> List[str]
    image_root = Path(image_dir).resolve()
    items = []
    for fp in image_root.iterdir():
        if fp.is_file() and fp.suffix.lower() in _IMG_EXTS:
            items.append(str(fp.resolve()))
    items.sort(key=_natural_sort_key)
    return items


def _dedupe_keep_order(items):
    # type: (List[str]) -> List[str]
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(str(item))
    return out


def _select_positions(total, count):
    # type: (int, int) -> List[int]
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    out = []
    seen = set()
    last = max(0, total - 1)
    for idx in range(count):
        pos = int(round(float(last) * float(idx) / float(count - 1)))
        pos = max(0, min(last, pos))
        if pos in seen:
            continue
        seen.add(pos)
        out.append(pos)
    if not out:
        return [0]
    return out


def _sample_images(files, sample_mode, num_images):
    # type: (List[str], str, int) -> List[str]
    ordered = [str(item) for item in files]
    if not ordered:
        return []
    mode = str(sample_mode or "even").strip().lower()
    count = int(num_images)
    if count <= 0:
        raise RuntimeError("num_images must be > 0")
    if mode == "first":
        return list(ordered[:count])
    if mode == "last":
        return list(ordered[-count:]) if count < len(ordered) else list(ordered)
    if mode in ("quartiles", "even"):
        positions = _select_positions(len(ordered), count)
        return _dedupe_keep_order([ordered[pos] for pos in positions])
    raise RuntimeError("unsupported prompt sample_mode: {}".format(sample_mode))


def _normalize_key(text):
    # type: (object) -> str
    value = str(text or "").strip().lower()
    value = value.replace("soil edges", "soil_edges")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    return _NON_ALNUM_RE.sub("_", value).strip("_")


def _coerce_choice(value, allowed, default):
    # type: (object, List[str], str) -> str
    key = _normalize_key(value)
    if key in allowed:
        return key
    return str(default)


def _sort_side_structures(items):
    # type: (List[str]) -> List[str]
    seen = set()
    out = []
    for key in SIDE_STRUCTURE_PRIORITY:
        if key in list(items or []) and key not in seen:
            seen.add(key)
            out.append(str(key))
    return out


def _coerce_side_structures(value):
    # type: (object) -> List[str]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        text = str(value or "").strip()
        raw_items = re.split(r"[\n,;/]+", text) if text else []
    seen = set()
    items = []
    for item in raw_items:
        key = _normalize_key(item)
        if key not in SIDE_STRUCTURES or key in seen:
            continue
        seen.add(key)
        items.append(key)
    return _sort_side_structures(items)[:2]


def default_prompt_profile():
    # type: () -> Dict[str, object]
    return {
        "version": int(PROMPT_PROFILE_VERSION),
        "scene_type": "unknown",
        "surface_type": "unknown",
        "side_structures": [],
        "lighting_type": "unknown",
        "dynamic_risk": "low",
        "repetition_risk": "medium",
        "profile_confidence": "low",
    }


def build_profile_instruction():
    # type: () -> str
    return (
        "Classify this frame into a closed-set PromptProfile for first-person walkway video generation.\n"
        "Return JSON only with keys: scene_type, surface_type, side_structures, lighting_type, dynamic_risk, repetition_risk, profile_confidence.\n"
        "scene_type must be one of: park_walkway, campus_path, orchard_row, field_path, road_edge, corridor, indoor_walkway, unknown.\n"
        "surface_type must be one of: pavement, concrete, brick, dirt, mixed, unknown.\n"
        "side_structures must be an array with up to 2 items chosen only from: grass, trees, shrubs, walls, columns, fence, buildings, crops, soil_edges, hedges.\n"
        "lighting_type must be one of: daylight, overcast, indoor_even, unknown.\n"
        "dynamic_risk must be one of: low, people, vehicles, animals, mixed.\n"
        "repetition_risk must be one of: low, medium, high.\n"
        "profile_confidence must be one of: low, medium, high.\n"
        "Do not output descriptions, explanations, colors, local decorations, or any free text outside the JSON object.\n"
        "If uncertain, use unknown and low."
    )


def normalize_prompt_profile(payload):
    # type: (object) -> Dict[str, object]
    data = payload if isinstance(payload, dict) else {}
    base = default_prompt_profile()
    base["scene_type"] = _coerce_choice(data.get("scene_type", ""), SCENE_TYPES, "unknown")
    base["surface_type"] = _coerce_choice(data.get("surface_type", ""), SURFACE_TYPES, "unknown")
    base["side_structures"] = _sort_side_structures(_coerce_side_structures(data.get("side_structures", [])))
    base["lighting_type"] = _coerce_choice(data.get("lighting_type", ""), LIGHTING_TYPES, "unknown")
    base["dynamic_risk"] = _coerce_choice(data.get("dynamic_risk", ""), DYNAMIC_RISKS, "low")
    base["repetition_risk"] = _coerce_choice(data.get("repetition_risk", ""), REPETITION_RISKS, "medium")
    base["profile_confidence"] = _coerce_choice(data.get("profile_confidence", ""), PROFILE_CONFIDENCES, "low")
    return base


def _extract_json_dict(text):
    # type: (str) -> Dict[str, object]
    raw = str(text or "").strip()
    if not raw:
        return {}
    candidates = [raw]
    match = _JSON_BLOCK_RE.search(raw)
    if match is not None:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = __import__("json").loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    return {}


def _scan_choice(raw_text, allowed, default):
    # type: (str, List[str], str) -> str
    text = " " + _normalize_key(raw_text).replace("_", " ") + " "
    for item in allowed:
        token = " " + str(item).replace("_", " ") + " "
        if token in text:
            return str(item)
    return str(default)


def _scan_side_structures(raw_text):
    # type: (str) -> List[str]
    text = " " + _normalize_key(raw_text).replace("_", " ") + " "
    found = []
    for item in SIDE_STRUCTURES:
        token = " " + str(item).replace("_", " ") + " "
        if token in text:
            found.append(str(item))
        if len(found) >= 2:
            break
    return found


def parse_profile_response(raw_text):
    # type: (object) -> Dict[str, object]
    raw = str(raw_text or "").strip()
    parsed = _extract_json_dict(raw)
    if parsed:
        return normalize_prompt_profile(parsed)
    return normalize_prompt_profile(
        {
            "scene_type": _scan_choice(raw, SCENE_TYPES, "unknown"),
            "surface_type": _scan_choice(raw, SURFACE_TYPES, "unknown"),
            "side_structures": _scan_side_structures(raw),
            "lighting_type": _scan_choice(raw, LIGHTING_TYPES, "unknown"),
            "dynamic_risk": _scan_choice(raw, DYNAMIC_RISKS, "low"),
            "repetition_risk": _scan_choice(raw, REPETITION_RISKS, "medium"),
            "profile_confidence": _scan_choice(raw, PROFILE_CONFIDENCES, "low"),
        }
    )


def _count_values(candidates, field_name, default):
    # type: (List[Dict[str, object]], str, str) -> Dict[str, int]
    counts = {}
    for candidate in candidates:
        value = str(candidate.get(field_name, default) or default)
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _count_side_presence(candidates):
    # type: (List[Dict[str, object]]) -> Dict[str, int]
    counts = {}
    for candidate in candidates:
        seen = set()
        for value in list(candidate.get("side_structures", []) or []):
            key = str(value)
            if key not in SIDE_STRUCTURES or key in seen:
                continue
            seen.add(key)
            counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _vote_scalar(candidates, field_name, allowed, default, ignore_default=False):
    # type: (List[Dict[str, object]], str, List[str], str, bool) -> str
    scores = {}
    first_seen = {}
    for idx, candidate in enumerate(candidates):
        value = str(candidate.get(field_name, default) or default)
        if value not in allowed:
            value = str(default)
        if ignore_default and value == default:
            continue
        if value not in first_seen:
            first_seen[value] = int(idx)
        weight = int(CONFIDENCE_WEIGHTS.get(str(candidate.get("profile_confidence", "low")), 1))
        scores[value] = int(scores.get(value, 0)) + weight
    if not scores:
        return str(default)
    best_value = str(default)
    best_rank = None
    for value in allowed:
        score = int(scores.get(value, 0))
        rank = (score, 0 if value == default else 1, -int(first_seen.get(value, 999999)))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_value = str(value)
    return best_value


def _vote_side_structures(candidates):
    # type: (List[Dict[str, object]]) -> List[str]
    scores = {}
    first_seen = {}
    for idx, candidate in enumerate(candidates):
        weight = int(CONFIDENCE_WEIGHTS.get(str(candidate.get("profile_confidence", "low")), 1))
        for item in list(candidate.get("side_structures", []) or []):
            value = str(item)
            if value not in SIDE_STRUCTURES:
                continue
            if value not in first_seen:
                first_seen[value] = int(idx)
            scores[value] = int(scores.get(value, 0)) + weight
    ranked = sorted(
        list(scores.keys()),
        key=lambda key: (-int(scores.get(key, 0)), SIDE_STRUCTURE_PRIORITY.index(key), int(first_seen.get(key, 999999))),
    )
    return _sort_side_structures([str(item) for item in ranked[:2]])


def _has_obvious_conflict(counts, default, total):
    # type: (Dict[str, int], str, int) -> bool
    active = []
    for key, value in counts.items():
        if str(key) == str(default):
            continue
        if int(value) <= 0:
            continue
        active.append((str(key), int(value)))
    if len(active) <= 1:
        return False
    active.sort(key=lambda item: (-int(item[1]), item[0]))
    return int(active[1][1]) >= max(2, (int(total) + 2) // 3)


def aggregate_prompt_profiles(candidates):
    # type: (List[Dict[str, object]]) -> Dict[str, object]
    cleaned = [normalize_prompt_profile(item) for item in list(candidates or []) if isinstance(item, dict)]
    if not cleaned:
        return default_prompt_profile()

    total = int(len(cleaned))
    majority = int(total // 2) + 1
    profile = default_prompt_profile()
    profile["scene_type"] = _vote_scalar(cleaned, "scene_type", SCENE_TYPES, "unknown", ignore_default=True)
    profile["surface_type"] = _vote_scalar(cleaned, "surface_type", SURFACE_TYPES, "unknown", ignore_default=True)
    profile["side_structures"] = _vote_side_structures(cleaned)
    profile["lighting_type"] = _vote_scalar(cleaned, "lighting_type", LIGHTING_TYPES, "unknown", ignore_default=True)
    profile["dynamic_risk"] = _vote_scalar(cleaned, "dynamic_risk", DYNAMIC_RISKS, "low")
    profile["repetition_risk"] = _vote_scalar(cleaned, "repetition_risk", REPETITION_RISKS, "medium")

    scene_counts = _count_values(cleaned, "scene_type", "unknown")
    surface_counts = _count_values(cleaned, "surface_type", "unknown")
    side_counts = _count_side_presence(cleaned)
    confidence_counts = _count_values(cleaned, "profile_confidence", "low")

    scene_majority = int(scene_counts.get(str(profile["scene_type"]), 0)) >= majority and str(profile["scene_type"]) != "unknown"
    surface_majority = int(surface_counts.get(str(profile["surface_type"]), 0)) >= majority and str(profile["surface_type"]) != "unknown"
    top_side_hits = 0
    if list(profile.get("side_structures", []) or []):
        top_side_hits = int(max([side_counts.get(item, 0) for item in list(profile.get("side_structures", []) or [])]))
    side_consistent = (not list(profile.get("side_structures", []) or [])) or top_side_hits >= majority
    low_majority = int(confidence_counts.get("low", 0)) >= majority
    scene_conflict = _has_obvious_conflict(scene_counts, "unknown", total)
    surface_conflict = _has_obvious_conflict(surface_counts, "unknown", total)
    side_conflict = len([key for key, value in side_counts.items() if int(value) >= max(2, majority - 1)]) > 2 and not side_consistent
    matching_core_frames = 0
    for item in cleaned:
        if str(item.get("scene_type", "")) == str(profile["scene_type"]) and str(item.get("surface_type", "")) == str(profile["surface_type"]):
            matching_core_frames += 1

    if scene_conflict or surface_conflict or side_conflict:
        profile["profile_confidence"] = "low"
    elif scene_majority and surface_majority and side_consistent and matching_core_frames >= majority and not low_majority:
        profile["profile_confidence"] = "high"
    elif scene_majority or surface_majority or matching_core_frames >= majority:
        profile["profile_confidence"] = "medium"
    else:
        profile["profile_confidence"] = "low"
    return profile


def _extract_frame_index(path_text):
    # type: (str) -> int
    stem = Path(path_text).stem
    if stem.isdigit():
        return int(stem)
    match = _NATURAL_RE.search(stem)
    if match is None:
        return -1
    return int(match.group(1))


def _run_formal_mainline(args):
    # type: (argparse.Namespace) -> Path
    total_t0 = time.time()
    exp_dir = Path(args.exp_dir).resolve()
    prompt_dir = (exp_dir / "prompt").resolve()
    prompt_dir.mkdir(parents=True, exist_ok=True)

    segment_manifest_path = ensure_file(args.segment_manifest, "segment manifest")
    segment_inputs = load_segment_prompt_inputs(segment_manifest_path)
    frames_dir = ensure_dir(exp_dir / "segment" / "frames", "segment frames dir")

    frame_files = _list_images(frames_dir)
    if not frame_files:
        raise RuntimeError("no image files found in {}".format(frames_dir))

    rep_count = max(3, min(5, int(args.num_images)))
    if int(args.num_images) != rep_count:
        log_warn("prompt representative frame count clamped from {} to {}".format(int(args.num_images), rep_count))

    selected_paths = _sample_images(frame_files, str(args.sample_mode), rep_count)
    if len(selected_paths) < 3:
        log_warn("only {} representative frames available; using all available samples".format(len(selected_paths)))

    backend = create_backend(
        backend_name=str(args.backend),
        model_ref=str(args.model_ref or DEFAULT_SMOLVLM2_MODEL_ID),
        dtype=str(args.dtype),
        max_new_tokens=int(args.max_new_tokens),
    )
    log_info(
        "initializing prompt backend={} model={} sample_mode={} rep_frames={}".format(
            str(args.backend),
            str(args.model_ref or DEFAULT_SMOLVLM2_MODEL_ID),
            str(args.sample_mode),
            len(selected_paths),
        )
    )
    log_info("loading prompt backend resources...")
    backend.load()
    backend_meta = dict(backend.meta() or {})
    log_info("processor loaded in {:.2f}s".format(float(backend_meta.get("processor_load_sec", 0.0) or 0.0)))
    log_info("model weights loaded in {:.2f}s".format(float(backend_meta.get("model_load_sec", 0.0) or 0.0)))

    instruction = build_profile_instruction()
    candidates = []
    frame_records = []
    errors = []
    prompt_times = []

    for idx, frame_path in enumerate(_maybe_progress(selected_paths)):
        frame_t0 = time.time()
        frame_name = Path(frame_path).name
        try:
            raw_output = str(backend.generate([frame_path], instruction) or "").strip()
            candidate = parse_profile_response(raw_output)
        except Exception as exc:
            candidate = default_prompt_profile()
            errors.append("frame {} classification failed: {}".format(frame_name, exc))
            log_warn("frame classification failed: {} ({})".format(frame_name, exc))

        elapsed = float(time.time() - frame_t0)
        prompt_times.append(elapsed)
        candidates.append(candidate)
        frame_records.append(
            {
                "frame": frame_name,
                "frame_idx": _extract_frame_index(frame_path),
                "profile": dict(candidate),
            }
        )
        log_prompt(
            "prompt detail: frame {}/{} classified in {:.2f}s: scene={} surface={} lighting={} risk={}".format(
                int(idx + 1),
                int(len(selected_paths)),
                elapsed,
                str(candidate.get("scene_type", "")),
                str(candidate.get("surface_type", "")),
                str(candidate.get("lighting_type", "")),
                str(candidate.get("dynamic_risk", "")),
            )
        )

    aggregated_profile = aggregate_prompt_profiles(candidates)
    aggregated_profile["version"] = int(PROMPT_PROFILE_VERSION)
    base_prompt_payload = build_base_prompt_payload(aggregated_profile)
    state_prompt_manifest = build_state_prompt_manifest(
        segment_inputs=segment_inputs,
        prompt_dir=prompt_dir,
        base_prompt_path=prompt_dir / "base_prompt.json",
    )
    runtime_prompt_plan = build_runtime_prompt_plan(
        segment_inputs=segment_inputs,
        state_prompt_manifest=state_prompt_manifest,
        base_prompt_payload=base_prompt_payload,
        prompt_dir=prompt_dir,
    )

    write_json_atomic(prompt_dir / "base_prompt.json", base_prompt_payload, indent=2)
    write_json_atomic(prompt_dir / "state_prompt_manifest.json", state_prompt_manifest, indent=2)
    write_json_atomic(prompt_dir / "runtime_prompt_plan.json", runtime_prompt_plan, indent=2)

    avg_prompt_sec = float(sum(prompt_times) / float(len(prompt_times))) if prompt_times else 0.0
    model_record = str(backend_meta.get("model_dir", "") or backend_meta.get("model_id", "") or args.model_ref)
    prompt_report = build_prompt_report(
        prompt_dir=prompt_dir,
        aggregated_profile=aggregated_profile,
        base_prompt_payload=base_prompt_payload,
        state_prompt_manifest=state_prompt_manifest,
        runtime_prompt_plan=runtime_prompt_plan,
        backend_meta=backend_meta,
        backend_name=str(args.backend),
        backend_python_phase=str(args.backend_python_phase or ""),
        model_record=model_record,
        dtype=str(args.dtype),
        sample_mode=str(args.sample_mode),
        num_images_requested=int(args.num_images),
        selected_paths=selected_paths,
        frame_files_count=len(frame_files),
        fps=(int(float(args.fps)) if float(args.fps) > 0 else None),
        frame_records=frame_records,
        errors=errors,
        total_sec=float(time.time() - total_t0),
        avg_prompt_sec=avg_prompt_sec,
    )
    prompt_report["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    prompt_report["profile_version"] = int(PROMPT_PROFILE_VERSION)
    prompt_report["frames_dir"] = str(frames_dir)
    report_path = write_prompt_report(prompt_dir, prompt_report)
    cleanup_legacy_prompt_outputs(prompt_dir)

    log_prog("prompt profile generated from {} representative frames".format(int(len(selected_paths))))
    log_info(
        "state prompt sources: segment_manifest={} state_segments={} deploy_schedule={}".format(
            str(((segment_inputs.get("source_files") or {}).get("segment_manifest", "")) or "<missing>"),
            str(((segment_inputs.get("source_files") or {}).get("state_segments", "")) or "<missing>"),
            str(((segment_inputs.get("source_files") or {}).get("deploy_schedule", "")) or "<missing>"),
        )
    )
    log_info("state prompt manifest generated: count={}".format(int(state_prompt_manifest.get("state_segment_count", 0) or 0)))
    log_info("runtime prompt plan generated: count={}".format(int(runtime_prompt_plan.get("deploy_segment_count", 0) or 0)))
    log_info("wrote: {}".format(prompt_dir / "base_prompt.json"))
    log_info("wrote: {}".format(prompt_dir / "state_prompt_manifest.json"))
    log_info("wrote: {}".format(prompt_dir / "runtime_prompt_plan.json"))
    log_info("wrote: {}".format(report_path))
    if errors:
        log_warn("{} representative frames fell back to the safe default profile".format(int(len(errors))))
    return report_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Formal ExpHub prompt mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exp_dir", required=True, help="ExpHub experiment dir")
    parser.add_argument("--segment_manifest", required=True, help="formal segment_manifest.json path")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--backend", default="smolvlm2", choices=["smolvlm2"])
    parser.add_argument("--model_ref", default=DEFAULT_SMOLVLM2_MODEL_ID)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--sample_mode", default="even", choices=["quartiles", "even", "first", "last"])
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--backend_python_phase", default="prompt_smol")
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
