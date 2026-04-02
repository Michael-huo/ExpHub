from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from exphub.cleanup import apply_keep_level
from exphub.common.config import get_phase_python_config, load_datasets_cfg, resolve_dataset
from exphub.common.io import remove_path, write_json_atomic
from exphub.common.logging import debug_info, log_info, log_step, log_warn, runtime_info
from exphub.common.paths import ExperimentPaths
from exphub.common.subprocess import RunError, RunnerConfig, StepRunner, detect_conda_base, resolve_phase_python
from exphub.common.types import ExperimentSpec, STAGE_ORDER

from .eval import service as eval_service
from .infer import service as infer_service
from .merge import service as merge_service
from .prompt import service as prompt_service
from .segment import service as segment_service
from .slam import service as slam_service
from .stats import service as stats_service


@dataclass
class OrchestrationResult:
    mode: str
    exp_dir: Path
    step_times: Dict[str, float]


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

    @property
    def exphub_root(self):
        return self.spec.exphub_root

    @property
    def fps_arg(self):
        return self.spec.fps_text

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

    def phase_python(self, phase_name):
        phase_key = str(phase_name)
        if phase_key not in self._phase_python_cache:
            self._phase_python_cache[phase_key] = resolve_phase_python(phase_key)
        return self._phase_python_cache[phase_key]

    def prompt_phase_name(self):
        backend = str(self.args.prompt_backend or "smolvlm2").strip().lower()
        if backend == "smolvlm2":
            return "prompt_smol"
        return "prompt"

    def prompt_model_ref(self):
        if str(self.args.prompt_model_dir or "").strip():
            return str(self.args.prompt_model_dir).strip()
        if str(self.args.prompt_backend or "smolvlm2").strip().lower() == "qwen":
            return str(self.args.qwen_model_dir or "").strip()
        return ""

    def infer_phase_name(self):
        backend = str(self.args.infer_backend or "wan_fun_5b_inp").strip().lower()
        if backend == "wan_fun_5b_inp":
            return "infer_fun_5b"
        return "infer"

    def ensure_clean_exp_dir(self):
        if self.paths.exp_dir.exists():
            debug_info("overwrite enabled: rm -rf {}".format(self.paths.exp_dir))
            remove_path(self.paths.exp_dir)
        self.paths.exp_dir.mkdir(parents=True, exist_ok=True)

    def assert_under_exp(self, path):
        base = self.paths.exp_dir.resolve()
        target = Path(path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise RuntimeError("unsafe path outside EXP_DIR: {} (exp_dir={})".format(target, base))

    def remove_in_exp(self, path):
        self.assert_under_exp(path)
        remove_path(path)

    def write_meta_snapshot(self):
        ds = self.dataset()
        meta = {
            "dataset": self.spec.dataset,
            "sequence": self.spec.sequence,
            "tag": self.spec.tag,
            "exp_name": self.spec.exp_name,
            "exp_dir": str(self.paths.exp_dir),
            "inputs": {
                "bag": str(ds.bag),
                "topic": ds.topic,
                "intrinsics": {
                    "fx": ds.fx,
                    "fy": ds.fy,
                    "cx": ds.cx,
                    "cy": ds.cy,
                    "dist": ds.dist,
                },
            },
            "params": {
                "w": self.spec.w,
                "h": self.spec.h,
                "fps": self.spec.fps,
                "dur": self.spec.dur,
                "start_sec": self.spec.start_sec,
                "start_idx": self.args.start_idx,
                "kf_gap": self.spec.kf_gap,
                "segment_policy": self.args.segment_policy,
                "base_idx": self.args.base_idx,
                "seed_base": self.args.seed_base,
                "gpus": self.args.gpus,
                "prompt_backend": self.args.prompt_backend,
                "prompt_model_dir": self.args.prompt_model_dir,
                "infer_backend": self.args.infer_backend,
                "infer_model_dir": self.args.infer_model_dir,
                "prompt_sample_mode": self.args.prompt_sample_mode,
                "prompt_num_images": self.args.prompt_num_images,
                "droid_seq": self.args.droid_seq,
                "viz_enable": self.viz_enable,
                "keep_level": self.args.keep_level,
            },
            "paths": {
                "segment_dir": str(self.paths.segment_dir),
                "segment_python": self.phase_python("segment"),
                "videox_root": self.args.videox_root,
                "droid_repo": self.args.droid_repo,
            },
        }
        write_json_atomic(self.paths.exp_meta_path, meta, indent=2)


_SERVICE_BY_STAGE = {
    "segment": segment_service,
    "prompt": prompt_service,
    "infer": infer_service,
    "merge": merge_service,
    "slam": slam_service,
    "eval": eval_service,
    "stats": stats_service,
}


def _read_log_tail(log_path, n):
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if n <= 0:
            return lines
        return lines[-n:]
    except Exception:
        return []


def _format_out_hint(exp_dir, out_hint):
    text = str(out_hint or "").strip()
    if not text:
        return ""
    try:
        target = Path(text).resolve()
        relative = target.relative_to(exp_dir.resolve())
        short_text = relative.as_posix()
    except Exception:
        short_text = text
    if Path(text).is_dir() and short_text and not short_text.endswith("/"):
        short_text += "/"
    return short_text or "."


def _run_step(runtime, step_name, service_module):
    started = time.time()
    log_step("{} start mode={}".format(step_name, runtime.args.mode))
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


def _doctor(runtime):
    log_info("STEP doctor: begin")
    has_critical_missing = False
    phase_names = ["segment", "prompt", runtime.infer_phase_name(), "slam"]
    if str(runtime.args.prompt_backend or "smolvlm2").strip().lower() == "smolvlm2":
        phase_names.append("prompt_smol")

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


def build_runtime(args):
    exphub_root = Path(args.exphub).resolve() if args.exphub else Path.cwd().resolve()
    if not (exphub_root / "scripts").exists():
        current = Path.cwd().resolve()
        found = None
        for path in [current] + list(current.parents):
            if (path / "scripts").exists() and (path / "config").exists():
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
        w=args.w,
        h=args.h,
        start_sec=args.start_sec,
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
        viz_enable = args.mode in ("slam", "eval")

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


def _validate_scripts(runtime):
    required = [
        (runtime.exphub_root / "exphub" / "pipeline" / "segment" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "prompt" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "infer" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "merge" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "slam" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "eval" / "service.py").resolve(),
        (runtime.exphub_root / "exphub" / "pipeline" / "stats" / "service.py").resolve(),
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("file not found: {}".format(path))


def run_runtime(runtime):
    _validate_scripts(runtime)

    mode = str(runtime.args.mode or "all").strip().lower()
    if mode == "doctor":
        _doctor(runtime)
        return OrchestrationResult(mode=mode, exp_dir=runtime.paths.exp_dir, step_times=dict(runtime.step_times))

    runtime.dataset()

    if mode in ("all", "workflow"):
        stages = list(STAGE_ORDER)
    else:
        stages = [mode]

    for stage_name in stages:
        service_module = _SERVICE_BY_STAGE.get(stage_name)
        if service_module is None:
            raise RuntimeError("unsupported stage mode: {}".format(stage_name))
        _run_step(runtime, stage_name, service_module)

    apply_keep_level(runtime.paths.exp_dir, runtime.args.keep_level)
    runtime_info("DONE.")
    return OrchestrationResult(mode=mode, exp_dir=runtime.paths.exp_dir, step_times=dict(runtime.step_times))


def run(args):
    return run_runtime(build_runtime(args))
