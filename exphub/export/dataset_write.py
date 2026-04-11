from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exphub.common.io import write_json_atomic, write_text_atomic


def assign_split(sample_key, seed=13):
    payload = "{}::{}".format(int(seed), str(sample_key))
    ratio = int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def ensure_layout(export_root):
    root = Path(export_root).resolve()
    clips_dir = (root / "clips").resolve()
    metadata_dir = (root / "metadata").resolve()
    clip_manifests_dir = (root / "clip_manifests").resolve()
    split_dirs = {}
    for split in ("train", "val", "test"):
        split_dir = (clips_dir / split).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)
    clip_manifests_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "clips_dir": clips_dir,
        "metadata_dir": metadata_dir,
        "clip_manifests_dir": clip_manifests_dir,
        "split_dirs": split_dirs,
    }


def relative_to_root(root_dir, target_path):
    root = Path(root_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(root))
    except Exception:
        return str(target)


def write_metadata_files(export_root, entries_by_split):
    layout = ensure_layout(export_root)
    metadata_paths = {}
    for split in ("train", "val", "test"):
        path = (layout["metadata_dir"] / "{}.jsonl".format(split)).resolve()
        lines = []
        for entry in list((entries_by_split or {}).get(split) or []):
            payload = {
                "file_path": str(entry.get("file_path", "") or ""),
                "text": str(entry.get("text", "") or ""),
                "type": "video",
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        write_text_atomic(path, ("\n".join(lines) + "\n") if lines else "")
        metadata_paths[split] = path
    return metadata_paths


def write_clip_manifest(export_root, clip_stem, clip_manifest):
    layout = ensure_layout(export_root)
    path = (layout["clip_manifests_dir"] / "{}.json".format(str(clip_stem))).resolve()
    write_json_atomic(path, dict(clip_manifest or {}), indent=2)
    return path
