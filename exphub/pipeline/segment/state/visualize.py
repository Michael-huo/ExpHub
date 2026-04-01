from __future__ import annotations

from pathlib import Path

from .. import artifacts as segment_artifacts


def materialize_formal_visuals(paths, detector_result):
    source_path = Path(detector_result.get("state_overview_path")).resolve()
    copied_path = segment_artifacts.copy_overview_to_formal_visuals(paths, source_path)
    return {
        "state_overview_path": copied_path,
        "source_path": source_path,
    }
