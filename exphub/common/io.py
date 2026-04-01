from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


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
