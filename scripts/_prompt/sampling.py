from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_NATURAL_RE = re.compile(r"(\d+)")


def _natural_sort_key(path_text):  # type: (str) -> List[object]
    text = str(path_text)
    name = Path(text).name
    parts = _NATURAL_RE.split(name)
    out = []  # type: List[object]
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part.lower())
    return out


def list_images(image_dir):  # type: (str) -> List[str]
    image_root = Path(image_dir).resolve()
    items = []  # type: List[str]
    for fp in image_root.iterdir():
        if fp.is_file() and fp.suffix.lower() in _IMG_EXTS:
            items.append(str(fp.resolve()))
    items.sort(key=_natural_sort_key)
    return items


def _dedupe_keep_order(items):  # type: (Sequence[str]) -> List[str]
    out = []  # type: List[str]
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(str(item))
    return out


def _select_positions(total, count):  # type: (int, int) -> List[int]
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    out = []  # type: List[int]
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


def sample_images(files, sample_mode, num_images):  # type: (Sequence[str], str, int) -> List[str]
    ordered = [str(item) for item in files]
    if not ordered:
        return []
    mode = str(sample_mode or "quartiles").strip().lower()
    count = int(num_images)
    if mode == "all":
        return list(ordered)
    if count <= 0:
        raise ValueError("num_images must be > 0 when sample_mode != all")
    if mode == "first":
        return list(ordered[:count])
    if mode == "last":
        return list(ordered[-count:]) if count < len(ordered) else list(ordered)
    if mode in ("quartiles", "even"):
        positions = _select_positions(len(ordered), count)
        sampled = [ordered[pos] for pos in positions]
        return _dedupe_keep_order(sampled)
    raise ValueError("unsupported sample_mode: {}".format(sample_mode))
