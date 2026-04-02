from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

_FRAME_RE = re.compile(r"(?:^|_)(\d+)$")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def ensure_dir(path, name="directory"):
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise RuntimeError("missing required {}: {}".format(name, resolved))
    if not resolved.is_dir():
        raise RuntimeError("expected directory for {}, got: {}".format(name, resolved))
    return resolved


def ensure_file(path, name="file"):
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise RuntimeError("missing required {}: {}".format(name, resolved))
    if not resolved.is_file():
        raise RuntimeError("expected file for {}, got: {}".format(name, resolved))
    return resolved


def write_json_atomic(path, obj, indent=2):
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
    os.replace(str(tmp_path), str(resolved))


def read_json_dict(path):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {}
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def read_step_meta(path):
    return read_json_dict(path)


def remove_path(path):
    resolved = Path(path)
    try:
        if resolved.is_symlink() or resolved.is_file():
            resolved.unlink()
        elif resolved.is_dir():
            shutil.rmtree(str(resolved), ignore_errors=True)
    except FileNotFoundError:
        return


def write_text_atomic(path, text):
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    tmp_path.write_text(str(text), encoding="utf-8")
    os.replace(str(tmp_path), str(resolved))


def frame_sort_key(path):
    item = Path(path)
    stem = item.stem
    if stem.isdigit():
        return (int(stem), item.name)
    match = _FRAME_RE.search(stem)
    if match:
        return (int(match.group(1)), item.name)
    return (10 ** 12, item.name)


def list_frames_sorted(frames_dir):
    resolved = Path(frames_dir).resolve()
    items = []
    for path in resolved.iterdir():
        if path.is_file() and path.suffix.lower() in _IMG_EXTS:
            items.append(path)
    items.sort(key=frame_sort_key)
    return items
