from __future__ import annotations

from pathlib import Path

def materialize_formal_visuals(paths, detector_result):
    raw_source_path = detector_result.get("state_overview_path")
    source_path = Path(raw_source_path).resolve() if raw_source_path else None
    return {
        "state_overview_path": source_path if source_path is not None and source_path.is_file() else None,
        "source_path": source_path,
    }
