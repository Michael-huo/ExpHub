from __future__ import annotations

import copy
import shutil
import zipfile
from pathlib import Path

from exphub.common.io import write_json_atomic


_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_ALLOWED_TOP_LEVEL = {"frames", "prompts.json", "motion_params.json"}


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _frame_path(frames_dir, idx):
    frame_root = Path(frames_dir).resolve()
    stem = "{:06d}".format(int(idx))
    for ext in _FRAME_EXTS:
        candidate = frame_root / "{}{}".format(stem, ext)
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("payload boundary frame not found for index {} under {}".format(int(idx), frame_root))


def _boundary_indices(generation_units):
    seen = set()
    out = []
    for raw_unit in list(_as_dict(generation_units).get("units") or []):
        unit = _as_dict(raw_unit)
        for key in ("start_idx", "end_idx"):
            try:
                idx = int(unit.get(key))
            except Exception:
                continue
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
    out.sort()
    if not out:
        raise RuntimeError("hvm payload requires at least one generation unit boundary frame")
    return out


def _payload_motion_params(generation_units):
    payload = copy.deepcopy(_as_dict(generation_units))
    for raw_unit in list(payload.get("units") or []):
        unit = _as_dict(raw_unit)
        prompt_ref = unit.get("prompt_ref")
        if isinstance(prompt_ref, dict):
            prompt_ref["artifact_path"] = "prompts.json"
    return payload


def _validate_payload_dir(payload_dir, expected_frame_names=None):
    root = Path(payload_dir).resolve()
    if not root.is_dir():
        raise RuntimeError("hvm payload directory missing: {}".format(root))

    names = {item.name for item in root.iterdir()}
    unexpected = sorted(names - _ALLOWED_TOP_LEVEL)
    missing = sorted(_ALLOWED_TOP_LEVEL - names)
    if unexpected or missing:
        raise RuntimeError(
            "hvm payload purity violation: missing={} unexpected={} dir={}".format(
                missing,
                unexpected,
                root,
            )
        )

    frames_dir = root / "frames"
    if not frames_dir.is_dir() or frames_dir.is_symlink():
        raise RuntimeError("hvm payload frames must be a real directory: {}".format(frames_dir))
    frame_files = sorted(frames_dir.iterdir(), key=lambda item: item.name)
    if not frame_files:
        raise RuntimeError("hvm payload frames directory is empty: {}".format(frames_dir))
    if expected_frame_names is not None:
        actual_names = [item.name for item in frame_files]
        expected_names = sorted(str(item) for item in expected_frame_names)
        if actual_names != expected_names:
            raise RuntimeError(
                "hvm payload frames purity violation: expected={} actual={}".format(
                    expected_names,
                    actual_names,
                )
            )
    for frame in frame_files:
        if frame.is_symlink() or not frame.is_file():
            raise RuntimeError("hvm payload frame must be a regular deep-copied file: {}".format(frame))
        if frame.suffix.lower() not in _FRAME_EXTS:
            raise RuntimeError("hvm payload frame has unsupported extension: {}".format(frame))

    for name in ("prompts.json", "motion_params.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("hvm payload metadata must be a regular file: {}".format(path))


def write_hvm_payload_zip(payload_dir, zip_path):
    root = Path(payload_dir).resolve()
    _validate_payload_dir(root)
    out_path = Path(zip_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with zipfile.ZipFile(str(tmp_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            if not path.is_file():
                continue
            arcname = path.relative_to(root).as_posix()
            if arcname.startswith("/") or ".." in Path(arcname).parts:
                raise RuntimeError("unsafe hvm payload zip member: {}".format(arcname))
            zf.write(str(path), arcname)
    tmp_path.replace(out_path)
    return out_path


def write_hvm_payload(frames_dir, generation_units, prompts, payload_dir):
    payload_root = Path(payload_dir).resolve()
    payload_root.mkdir(parents=True, exist_ok=True)
    payload_frames = payload_root / "frames"
    payload_frames.mkdir(parents=True, exist_ok=True)

    boundary_indices = _boundary_indices(generation_units)
    copied_frames = []
    for frame_idx in boundary_indices:
        src = _frame_path(frames_dir, frame_idx)
        dst = payload_frames / src.name
        shutil.copy2(str(src), str(dst), follow_symlinks=True)
        if dst.is_symlink() or not dst.is_file():
            raise RuntimeError("failed to deep-copy hvm payload frame: {}".format(dst))
        copied_frames.append(dst.resolve())

    write_json_atomic(payload_root / "prompts.json", prompts, indent=2)
    write_json_atomic(payload_root / "motion_params.json", _payload_motion_params(generation_units), indent=2)

    _validate_payload_dir(payload_root, expected_frame_names=[item.name for item in copied_frames])

    return {
        "payload_dir": payload_root,
        "frame_count": int(len(copied_frames)),
        "boundary_indices": list(boundary_indices),
    }
