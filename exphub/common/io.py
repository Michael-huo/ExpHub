from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import csv
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


def unique_sibling_temp_path(path):
    resolved = Path(path).resolve()
    suffix = resolved.suffix
    stem = resolved.name[: -len(suffix)] if suffix else resolved.name
    return resolved.with_name("{}.{:s}{}".format(stem, uuid.uuid4().hex, suffix))


def replace_nonempty_file(temp_path, final_path, name="artifact"):
    temp = Path(temp_path).resolve()
    final = Path(final_path).resolve()
    if not temp.is_file():
        raise RuntimeError("failed to create required {}: {}".format(name, temp))
    try:
        size = int(temp.stat().st_size)
    except Exception:
        size = 0
    if size <= 0:
        try:
            temp.unlink()
        except Exception:
            pass
        raise RuntimeError("required {} is empty: {}".format(name, temp))
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(temp), str(final))
    return final


def write_csv_atomic(path, fieldnames, rows):
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow(dict(row))
    os.replace(str(tmp_path), str(resolved))


def write_yaml_atomic(path, obj):
    import yaml

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(resolved.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=True, allow_unicode=False)
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
