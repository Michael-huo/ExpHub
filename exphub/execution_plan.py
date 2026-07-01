from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


INFER_STAGES: Tuple[str, ...] = ("prepare", "encode", "decode", "eval")
TRAIN_STAGES: Tuple[str, ...] = ("prepare", "encode", "lora")
ALL_STAGES: Tuple[str, ...] = ("prepare", "encode", "decode", "eval", "lora", "all")
EXPERIMENT_CHOICES: Tuple[str, ...] = ("motion-benchmark", "compression-benchmark", "image-quality")


class ExecutionPlanError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str
    requested_step: Optional[str]
    resolved_step: str
    stages: Tuple[str, ...]
    experiments: Tuple[str, ...]
    droid_live_viewer: bool


def _normalize_text(value) -> str:
    return str(value or "").strip().lower()


def _validate_seed(seed_value) -> None:
    try:
        seed = int(seed_value)
    except Exception as exc:
        raise ExecutionPlanError("--seed must be a non-negative integer") from exc
    if seed < 0:
        raise ExecutionPlanError("--seed must be a non-negative integer")


def _validate_experiments(mode: str, resolved_step: str, experiments: Tuple[str, ...]) -> None:
    seen = set()
    for item in experiments:
        if item in seen:
            raise ExecutionPlanError("duplicate experiment: {}".format(item))
        seen.add(item)
    if experiments and not (mode == "infer" and resolved_step == "all"):
        raise ExecutionPlanError("experiments are only allowed for infer + all")


def build_execution_plan(
    *,
    mode: str,
    requested_step: Optional[str],
    experiments: Sequence[str] = (),
    seed: int = 12345,
) -> ExecutionPlan:
    mode_norm = _normalize_text(mode)
    requested_norm = _normalize_text(requested_step) if requested_step is not None else None
    resolved_step = requested_norm or "all"
    experiments_norm = tuple(str(item or "").strip() for item in (experiments or ()))

    _validate_seed(seed)

    if mode_norm == "infer":
        if resolved_step == "lora":
            raise ExecutionPlanError("infer + lora is invalid")
        if resolved_step == "all":
            stages = INFER_STAGES
        elif resolved_step in INFER_STAGES:
            stages = (resolved_step,)
        else:
            raise ExecutionPlanError("infer + {} is invalid".format(resolved_step))
    elif mode_norm == "train":
        if resolved_step in ("decode", "eval"):
            raise ExecutionPlanError("train + {} is invalid".format(resolved_step))
        if resolved_step == "all":
            stages = TRAIN_STAGES
        elif resolved_step in TRAIN_STAGES:
            stages = (resolved_step,)
        else:
            raise ExecutionPlanError("train + {} is invalid".format(resolved_step))
    else:
        raise ExecutionPlanError("unsupported mode: {}".format(mode_norm or "<empty>"))

    _validate_experiments(mode_norm, resolved_step, experiments_norm)

    return ExecutionPlan(
        mode=mode_norm,
        requested_step=requested_norm,
        resolved_step=resolved_step,
        stages=tuple(stages),
        experiments=experiments_norm,
        droid_live_viewer=(mode_norm == "infer" and requested_norm == "eval"),
    )


__all__ = [
    "ALL_STAGES",
    "EXPERIMENT_CHOICES",
    "ExecutionPlan",
    "ExecutionPlanError",
    "INFER_STAGES",
    "TRAIN_STAGES",
    "build_execution_plan",
]
