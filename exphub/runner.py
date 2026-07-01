from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from exphub.common.io import remove_path
from exphub.common.logging import log_step, log_warn, runtime_info
from exphub.common.paths import ExperimentPaths
from exphub.common.subprocess import RunError, StepRunner
from exphub.execution_plan import ExecutionPlan
from exphub.meta import ExperimentSpec

from .decode import decode as decode_pipeline
from .encode import encode as encode_pipeline
from .eval import eval as eval_pipeline
from .lora import lora as lora_pipeline
from .prepare import prepare as prepare_pipeline
from .provenance import update_run_status, write_run_start


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    sequence: str
    tag: str
    fps: int
    start: str
    dur: str
    seed: int
    decode_profile: str
    log_level: str


@dataclass
class OrchestrationResult:
    mode: str
    exp_dir: Path
    step_times: Dict[str, float]
    experiment_times: Dict[str, float]
    result_root: Path
    main_pipeline_wall_time_s: Optional[float] = None
    selected_stages_wall_time_s: Optional[float] = None
    optional_total_time_s: Optional[float] = None
    full_command_wall_time_s: Optional[float] = None


@dataclass
class PipelineRuntime:
    config: RunConfig
    execution_plan: ExecutionPlan
    spec: ExperimentSpec
    paths: ExperimentPaths
    cfg_path: Path
    step_runner: StepRunner
    command_argv: tuple[str, ...] = ()
    step_times: Dict[str, float] = field(default_factory=dict)

    @property
    def exphub_root(self) -> Path:
        return self.spec.exphub_root

    @property
    def mode(self) -> str:
        return self.execution_plan.mode

    @property
    def fps_arg(self) -> str:
        return self.spec.fps_text

    def ensure_clean_exp_dir(self) -> None:
        self.paths.exp_dir.mkdir(parents=True, exist_ok=True)

    def assert_under_exp(self, path) -> None:
        base = self.paths.exp_dir.resolve()
        target = Path(path).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise RuntimeError("unsafe path outside exp_dir: {} (exp_dir={})".format(target, base)) from exc

    def remove_in_exp(self, path) -> None:
        self.assert_under_exp(path)
        remove_path(path)

_SERVICE_BY_STAGE = {
    "prepare": prepare_pipeline,
    "encode": encode_pipeline,
    "lora": lora_pipeline,
    "decode": decode_pipeline,
    "eval": eval_pipeline,
}


def _read_log_tail(log_path: Path, n: int):
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    if n <= 0:
        return lines
    return lines[-n:]


def _format_out_hint(exp_dir: Path, out_hint) -> str:
    text = str(out_hint or "").strip()
    if not text:
        return ""
    try:
        target = Path(text).resolve()
        short_text = target.relative_to(exp_dir.resolve()).as_posix()
    except Exception:
        short_text = text
    if Path(text).is_dir() and short_text and not short_text.endswith("/"):
        short_text += "/"
    return short_text or "."


def _run_step(runtime: PipelineRuntime, step_name: str, service_module, *, droid_live_viewer: bool = False):
    started = time.time()
    log_step("{} start mode={} step={}".format(step_name, runtime.execution_plan.mode, runtime.execution_plan.resolved_step))
    try:
        if step_name == "eval":
            out_hint = service_module.run(runtime, droid_live_viewer=bool(droid_live_viewer))
        else:
            out_hint = service_module.run(runtime)
    except RunError as exc:
        elapsed = time.time() - started
        return_code = exc.returncode if exc.returncode is not None else -1
        log_path = str(exc.log_path) if exc.log_path else "-"
        log_step("{} FAIL sec={:.2f} rc={} log={}".format(step_name, elapsed, return_code, log_path))
        tail_lines = []
        if exc.log_path and Path(exc.log_path).is_file():
            tail_lines = _read_log_tail(Path(exc.log_path), runtime.step_runner.fail_tail_lines)
        if not tail_lines:
            tail_lines = list(exc.tail_lines)
        if tail_lines:
            log_warn("{} last {} lines:".format(step_name, len(tail_lines)))
            for line in tail_lines:
                print("[TAIL] {}".format(line))
        raise
    except Exception:
        elapsed = time.time() - started
        log_step("{} FAIL sec={:.2f}".format(step_name, elapsed))
        raise

    elapsed = time.time() - started
    runtime.step_times[step_name] = float(elapsed)
    out_hint_short = _format_out_hint(runtime.paths.exp_dir, out_hint)
    if out_hint_short:
        log_step("{} done sec={:.2f} out={}".format(step_name, elapsed, out_hint_short))
    else:
        log_step("{} done sec={:.2f}".format(step_name, elapsed))
    return out_hint


def _repo_root_from_package() -> Path:
    return Path(__file__).resolve().parents[1]


def build_runtime(config: RunConfig, execution_plan: ExecutionPlan, command_argv=()) -> PipelineRuntime:
    exphub_root = _repo_root_from_package()
    cfg_path = (exphub_root / "config" / "datasets.json").resolve()

    spec = ExperimentSpec(
        exphub_root=exphub_root,
        mode=execution_plan.mode,
        dataset=config.dataset,
        sequence=config.sequence,
        tag=config.tag,
        start=config.start,
        dur=config.dur,
        fps=config.fps,
    )
    paths = ExperimentPaths.from_spec(spec)
    step_runner = StepRunner(
        logs_dir=paths.logs_dir,
        log_level=config.log_level,
        pass_prefixes=("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]"),
        fail_tail_lines=30,
    )

    runtime = PipelineRuntime(
        config=config,
        execution_plan=execution_plan,
        spec=spec,
        paths=paths,
        cfg_path=cfg_path,
        step_runner=step_runner,
        command_argv=tuple(str(item) for item in tuple(command_argv or ())),
    )
    if spec.kf_gap % 4 != 0:
        log_warn("kf_gap={} not divisible by 4 (r=4). model may truncate length.".format(spec.kf_gap))
    return runtime


def _validate_scripts_for_stages(runtime: PipelineRuntime, stages) -> None:
    script_by_stage = {
        "prepare": [runtime.exphub_root / "exphub" / "prepare" / "prepare.py"],
        "encode": [runtime.exphub_root / "exphub" / "encode" / "encode.py"],
        "decode": [
            runtime.exphub_root / "exphub" / "decode" / "decode.py",
            runtime.exphub_root / "exphub" / "decode" / "comfyui_client.py",
        ],
        "eval": [runtime.exphub_root / "exphub" / "eval" / "eval.py"],
        "lora": [runtime.exphub_root / "exphub" / "lora" / "lora.py"],
    }
    required = []
    for stage in stages:
        required.extend(script_by_stage.get(stage, []))
    for path in required:
        script_path = Path(path).resolve()
        if not script_path.is_file():
            raise RuntimeError("file not found: {}".format(script_path))


def run_runtime(runtime: PipelineRuntime, execution_plan: ExecutionPlan) -> OrchestrationResult:
    command_started = time.perf_counter()
    mode = str(execution_plan.mode or "").strip().lower()
    if mode not in ("infer", "train"):
        raise RuntimeError("unsupported mode: {}".format(mode))
    _validate_scripts_for_stages(runtime, execution_plan.stages)

    stages = list(execution_plan.stages)
    full_infer_main = mode == "infer" and tuple(stages) == ("prepare", "encode", "decode", "eval")
    stage_window_started = None
    stage_window_elapsed = None
    main_pipeline_wall_time = None
    experiment_times = {}
    provenance_start = write_run_start(runtime, runtime.command_argv)
    try:
        last_out_hint = runtime.paths.exp_dir
        stage_window_started = time.perf_counter()
        for stage_name in stages:
            service_module = _SERVICE_BY_STAGE.get(stage_name)
            if service_module is None:
                raise RuntimeError("unsupported stage mode: {}".format(stage_name))
            last_out_hint = (
                _run_step(
                    runtime,
                    stage_name,
                    service_module,
                    droid_live_viewer=execution_plan.droid_live_viewer,
                )
                or last_out_hint
            )
        stage_window_elapsed = float(time.perf_counter() - stage_window_started)
        if full_infer_main:
            main_pipeline_wall_time = float(stage_window_elapsed)

        if execution_plan.experiments:
            from exphub.experiments import run_requested_experiments

            last_out_hint, experiment_times = run_requested_experiments(runtime, execution_plan) or (last_out_hint, {})
    except Exception as exc:
        update_run_status(runtime, status="failed", start_time=provenance_start, error=exc)
        raise

    update_run_status(runtime, status="success", start_time=provenance_start)
    full_command_wall = float(time.perf_counter() - command_started)
    optional_total = float(sum(float(value) for value in dict(experiment_times or {}).values())) if experiment_times else None
    runtime_info("DONE.")
    return OrchestrationResult(
        mode=mode,
        exp_dir=runtime.paths.exp_dir,
        step_times=dict(runtime.step_times),
        experiment_times=dict(experiment_times or {}),
        result_root=Path(last_out_hint).resolve() if last_out_hint else runtime.paths.exp_dir,
        main_pipeline_wall_time_s=main_pipeline_wall_time,
        selected_stages_wall_time_s=stage_window_elapsed,
        optional_total_time_s=optional_total,
        full_command_wall_time_s=full_command_wall,
    )


def run(config: RunConfig, execution_plan: ExecutionPlan) -> OrchestrationResult:
    return run_runtime(build_runtime(config, execution_plan), execution_plan)


__all__ = [
    "OrchestrationResult",
    "PipelineRuntime",
    "RunConfig",
    "RunError",
    "StepRunner",
    "build_runtime",
    "run",
    "run_runtime",
]
