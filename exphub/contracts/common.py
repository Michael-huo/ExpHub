from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from exphub.common.types import STAGE_ORDER


CONTRACT_STAGE_ORDER = STAGE_ORDER


@dataclass(frozen=True)
class StageContract:
    stage: str
    root: Path
    artifacts: Dict[str, Path]
