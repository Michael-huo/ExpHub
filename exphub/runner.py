from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from exphub.cleanup import apply_keep_level
from exphub.common.io import read_json_dict, remove_path, write_json_atomic
from exphub.common.logging import log_info, log_step, log_warn, runtime_info
from exphub.common.paths import ExperimentPaths
from exphub.common.subprocess import RunError, RunnerConfig, StepRunner, detect_conda_base, resolve_phase_python
from exphub.config import get_phase_python_config, load_datasets_cfg, resolve_dataset
from exphub.meta import ExperimentSpec, STAGE_ORDER

from .decode import pipeline_run as decode_pipeline
from .encode import encode as encode_pipeline
from .eval import pipeline_run as eval_pipeline
from .export import pipeline_run as export_pipeline
from .prepare import prepare as prepare_pipeline


@dataclass
class OrchestrationResult:
    mode: str
    exp_dir: Path
    step_times: Dict[str, float]
    result_root: Path


@dataclass
class PipelineRuntime:
    args: object
    spec: ExperimentSpec
    paths: ExperimentPaths
    cfg_path: Path
    runner_cfg: RunnerConfig
    step_runner: StepRunner
    viz_enable: bool
    step_times: Dict[str, float] = field(default_factory=dict)
    _dataset_resolved: Optional[object] = None
    _phase_python_cache: Dict[str, str] = field(default_factory=dict)
    _prepare_result_cache: Optional[Dict[str, object]] = None

    @property
    def exphub_root(self) -> Path:
        return self.spec.exphub_root

    @property
    def fps_arg(self) -> str:
        return self.spec.fps_text

    @property
    def start_arg(self) -> str:
        return str(self.spec.start)

    def dataset(self):
        if self._dataset_resolved is None:
            cfg = load_datasets_cfg(self.cfg_path)
            self._dataset_resolved = resolve_dataset(
                cfg,
                self.exphub_root,
                self.spec.dataset,
                self.spec.sequence,
            )
            if not self._dataset_resolved.bag.exists():
                raise RuntimeError("bag not found: {}".format(self._dataset_resolved.bag))
        return self._dataset_resolved

    def phase_python(self, phase_name: str) -> str:
        phase_key = str(phase_name)
        if phase_key not in self._phase_python_cache:
            self._phase_python_cache[phase_key] = resolve_phase_python(phase_key)
        return self._phase_python_cache[phase_key]

    def infer_phase_name(self) -> str:
        backend = str(self.args.infer_backend or "wan_fun_5b_inp").strip().lower()
        if backend != "wan_fun_5b_inp":
            raise RuntimeError("decode backend supports only wan_fun_5b_inp: {}".format(backend or "<empty>"))
        return "infer_fun_5b"

    def ensure_clean_exp_dir(self) -> None:
        if self.paths.exp_dir.exists():
            remove_path(self.paths.exp_dir)
        self.paths.exp_dir.mkdir(parents=True, exist_ok=True)

    def prepare_result(self) -> Dict[str, object]:
        if self._prepare_result_cache is None:
            if not self.paths.prepare_result_path.is_file():
                raise RuntimeError("prepare_result.json not found: {}".format(self.paths.prepare_result_path))
            payload = read_json_dict(self.paths.prepare_result_path)
            if not payload:
                raise RuntimeError("invalid prepare_result.json: {}".format(self.paths.prepare_result_path))
            self._prepare_result_cache = payload
        return dict(self._prepare_result_cache)

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

    def write_meta_snapshot(self) -> None:
        prepare_result_path = self.paths.prepare_result_path
        prepare_result = read_json_dict(prepare_result_path) if prepare_result_path.is_file() else {}
        meta = {
            "mode": str(self.args.mode),
            "step": str(self.args.step),
            "dataset": self.spec.dataset,
            "sequence": self.spec.sequence,
            "tag": self.spec.tag,
            "fps": self.spec.fps,
            "dur": self.spec.dur,
            "start": self.spec.start,
            "run_id": self.spec.exp_name,
            "artifact_root": str(self.paths.exp_dir),
            "prepare_result_path": str(prepare_result_path),
            "params": {
                "fps": self.spec.fps,
                "dur": self.spec.dur,
                "start": self.spec.start,
                "kf_gap": self.spec.kf_gap,
                "segment_policy": self.args.segment_policy,
                "seed_base": self.args.seed_base,
                "gpus": self.args.gpus,
                "planner": "generation_units",
                "prompt_strategy": "prompt_spans",
                "workflow": "prepare -> encode -> decode -> eval",
                "prompt_model_dir": self.args.prompt_model_dir,
                "infer_backend": self.args.infer_backend,
                "infer_model_dir": self.args.infer_model_dir,
                "droid_seq": self.args.droid_seq,
                "viz_enable": self.viz_enable,
                "keep_level": self.args.keep_level,
            },
            "paths": {
                "prepare_dir": str(self.paths.prepare_dir),
                "prepare_frames_dir": str(self.paths.prepare_frames_dir),
                "encode_dir": str(self.paths.encode_dir),
                "decode_dir": str(self.paths.decode_dir),
                "eval_dir": str(self.paths.eval_dir),
                "logs_dir": str(self.paths.logs_dir),
                "semantic_openclip_python": get_phase_python_config("semantic_openclip"),
                "videox_root": self.args.videox_root,
                "droid_repo": self.args.droid_repo,
            },
            "prepare": {
                "num_frames": prepare_result.get("num_frames"),
                "normalized_resolution": prepare_result.get("normalized_resolution"),
                "normalized_intrinsics": prepare_result.get("normalized_intrinsics"),
                "legal_grid": prepare_result.get("legal_grid"),
            },
        }
        write_json_atomic(self.paths.exp_meta_path, meta, indent=2)


_SERVICE_BY_STAGE = {
    "prepare": prepare_pipeline,
    "encode": encode_pipeline,
    "decode": decode_pipeline,
    "eval": eval_pipeline,
    "export": export_pipeline,
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


def _run_step(runtime: PipelineRuntime, step_name: str, service_module):
    started = time.time()
    log_step("{} start mode={} step={}".format(step_name, runtime.args.mode, runtime.args.step))
    try:
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


def _doctor(runtime: PipelineRuntime) -> None:
    log_info("STEP doctor: begin")
    has_critical_missing = False
    phase_names = ["segment", "prompt_smol", runtime.infer_phase_name(), "slam"]
    for phase_name in phase_names:
        python_bin = get_phase_python_config(phase_name)
        exists = False
        if python_bin:
            phase_path = Path(str(python_bin)).expanduser()
            exists = phase_path.is_file() and os.access(str(phase_path), os.X_OK)
        log_info(
            "DOCTOR phase={} python={} exists={}".format(
                phase_name,
                python_bin or "<missing>",
                exists,
            )
        )
        if not python_bin or not exists:
            has_critical_missing = True
    if has_critical_missing:
        log_warn("DOCTOR result=FAIL")
        raise SystemExit(2)
    log_info("DOCTOR result=PASS")
    runtime_info("DONE. MODE=doctor")


def build_runtime(args) -> PipelineRuntime:
    exphub_root = Path(args.exphub).resolve() if args.exphub else Path.cwd().resolve()
    if not ((exphub_root / "exphub").exists() and (exphub_root / "config").exists()):
        current = Path.cwd().resolve()
        found = None
        for path in [current] + list(current.parents):
            if (path / "exphub").exists() and (path / "config").exists():
                found = path
                break
        if found is not None:
            exphub_root = found
        else:
            log_warn("Cannot verify ExpHub root at {}; continuing".format(exphub_root))

    cfg_path = Path(args.datasets_cfg) if args.datasets_cfg else (exphub_root / "config" / "datasets.json")
    if not cfg_path.is_absolute():
        cfg_path = (exphub_root / cfg_path).resolve()

    spec = ExperimentSpec(
        exphub_root=exphub_root,
        dataset=args.dataset,
        sequence=args.sequence,
        tag=args.tag,
        start=args.start,
        dur=args.dur,
        fps=args.fps,
        kf_gap_input=args.kf_gap,
        exp_root_override=Path(args.exp_root).resolve() if args.exp_root else None,
    )
    paths = ExperimentPaths.from_spec(spec)
    runner_cfg = RunnerConfig(
        auto_conda=bool(args.auto_conda),
        conda_base=detect_conda_base() if args.auto_conda else None,
        ros_setup=Path(args.ros_setup) if args.ros_setup else None,
    )
    step_runner = StepRunner(
        logs_dir=paths.logs_dir,
        log_level=args.log_level,
        runner_cfg=runner_cfg,
        pass_prefixes=("[INFO]", "[WARN]", "[ERR]", "[PROG]", "[STEP]"),
        fail_tail_lines=30,
    )

    if args.viz and args.no_viz:
        raise RuntimeError("--viz and --no_viz are mutually exclusive")
    if args.viz:
        viz_enable = True
    elif args.no_viz:
        viz_enable = False
    else:
        viz_enable = args.step == "eval"

    runtime = PipelineRuntime(
        args=args,
        spec=spec,
        paths=paths,
        cfg_path=cfg_path,
        runner_cfg=runner_cfg,
        step_runner=step_runner,
        viz_enable=viz_enable,
    )
    if spec.kf_gap % 4 != 0:
        log_warn("kf_gap={} not divisible by 4 (r=4). model may truncate length.".format(spec.kf_gap))
    return runtime


def _validate_scripts(runtime: PipelineRuntime) -> None:
    required = [
        (runtime.exphub_root / "exphub" / "encode" / "encode.py").resolve(),
        (runtime.exphub_root / "exphub" / "decode" / "pipeline_run.py").resolve(),
        (runtime.exphub_root / "exphub" / "eval" / "pipeline_run.py").resolve(),
        (runtime.exphub_root / "exphub" / "export" / "pipeline_run.py").resolve(),
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("file not found: {}".format(path))


def run_runtime(runtime: PipelineRuntime) -> OrchestrationResult:
    _validate_scripts(runtime)
    mode = str(runtime.args.mode or "").strip().lower()
    step = str(runtime.args.step or "").strip().lower()
    if mode != "infer":
        raise RuntimeError("only infer mode is connected in this pass: {}".format(mode))

    runtime.dataset()

    if step == "all":
        stages = list(STAGE_ORDER)
    else:
        stages = [step]

    last_out_hint = runtime.paths.exp_dir
    for stage_name in stages:
        service_module = _SERVICE_BY_STAGE.get(stage_name)
        if service_module is None:
            raise RuntimeError("unsupported stage mode: {}".format(stage_name))
        last_out_hint = _run_step(runtime, stage_name, service_module) or last_out_hint

    apply_keep_level(runtime.paths.exp_dir, runtime.args.keep_level)
    runtime_info("DONE.")
    return OrchestrationResult(
        mode=mode,
        exp_dir=runtime.paths.exp_dir,
        step_times=dict(runtime.step_times),
        result_root=Path(last_out_hint).resolve() if last_out_hint else runtime.paths.exp_dir,
    )


def run(args) -> OrchestrationResult:
    return run_runtime(build_runtime(args))


__all__ = [
    "OrchestrationResult",
    "PipelineRuntime",
    "RunError",
    "RunnerConfig",
    "StepRunner",
    "build_runtime",
    "detect_conda_base",
    "resolve_phase_python",
    "run",
    "run_runtime",
]
