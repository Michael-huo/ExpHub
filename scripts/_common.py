#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import sys
from pathlib import Path


_FRAME_RE = re.compile(r"(?:^|_)(\d+)$")
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def log_info(msg):
    print("[INFO] {}".format(msg))


def log_warn(msg):
    print("[WARN] {}".format(msg))


def log_err(msg):
    print("[ERR] {}".format(msg), file=sys.stderr)


def ensure_dir(path, name="directory"):
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit("[ERR] missing required {}: {}".format(name, p))
    if not p.is_dir():
        raise SystemExit("[ERR] expected directory for {}, got: {}".format(name, p))
    return p


def ensure_file(path, name="file"):
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit("[ERR] missing required {}: {}".format(name, p))
    if not p.is_file():
        raise SystemExit("[ERR] expected file for {}, got: {}".format(name, p))
    return p


def ensure_cmd(cmd, name=None, required=True):
    found = shutil.which(str(cmd))
    if found:
        return found
    if required:
        nm = name or str(cmd)
        raise SystemExit("[ERR] required command not found: {}".format(nm))
    return None


def _frame_sort_key(p):
    stem = p.stem
    if stem.isdigit():
        return (int(stem), p.name)
    m = _FRAME_RE.search(stem)
    if m:
        return (int(m.group(1)), p.name)
    return (10 ** 12, p.name)


def list_frames_sorted(frames_dir):
    p = Path(frames_dir).resolve()
    items = []
    for fp in p.iterdir():
        if fp.is_file() and fp.suffix.lower() in _IMG_EXTS:
            items.append(fp)
    items.sort(key=_frame_sort_key)
    return items


def write_json_atomic(path, obj, indent=2):
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(str(tmp), str(p))

