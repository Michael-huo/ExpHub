from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import ConfigError
from .common.logging import set_cli_log_level
from .execution_plan import ALL_STAGES, EXPERIMENT_CHOICES, ExecutionPlanError, build_execution_plan
from .meta import sanitize_token
from .runner import RunConfig, build_runtime, run_runtime
from .runner import RunError


_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_STEP = "\033[1;36m"
_STEP_SEPARATOR = "=" * 70
_CLI_LOG_LEVEL = "info"


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _runtime_info(msg: str) -> None:
    if _CLI_LOG_LEVEL != "quiet":
        _info(msg)


def _die(msg: str) -> None:
    raise SystemExit(f"[ERR] {msg}")


def _print_experiment_summary(
    mode: str,
    step: str,
    dataset: str,
    sequence: str,
    tag: str,
    fps_text: str,
    dur_text: str,
    start_text: str,
    exp_dir: Path,
) -> None:
    _info(
        "run start mode={} step={} dataset={} sequence={} tag={} fps={} dur={} start={} root={}".format(
            mode,
            step,
            dataset,
            sequence,
            tag,
            fps_text,
            dur_text,
            start_text,
            exp_dir,
        )
    )


def _read_json_dict(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def _print_rows(rows: List[tuple]) -> None:
    if not rows:
        return
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        _info("{:<{w}} : {}".format(str(key), value, w=width))


def _as_dict(value: object) -> Dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_number(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_optional_number(value: object, digits: int = 3, suffix: str = "") -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    return "{:.{digits}f}{}".format(number, suffix, digits=int(digits))


def _format_signed_number(value: object, digits: int = 3, suffix: str = "") -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    return "{:+.{digits}f}{}".format(number, suffix, digits=int(digits))


def _format_mib(value: object) -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    return "{:.2f} MiB".format(number / (1024.0 * 1024.0))


def _format_count(value: object) -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    return str(int(number))


def _delta(after: object, before: object) -> Optional[float]:
    after_value = _as_number(after)
    before_value = _as_number(before)
    if after_value is None or before_value is None:
        return None
    return float(after_value) - float(before_value)


def _format_time(value: object) -> str:
    return _format_optional_number(value, 2, "s")


def _print_runtime_times(result, execution_plan) -> None:
    step_times = getattr(result, "step_times", {}) if isinstance(getattr(result, "step_times", {}), dict) else {}
    rows = []
    for stage in tuple(execution_plan.stages or ()):
        value = step_times.get(stage)
        if value is not None:
            rows.append(("{} time".format(stage), _format_time(value)))
    main_wall = getattr(result, "main_pipeline_wall_time_s", None)
    selected_wall = getattr(result, "selected_stages_wall_time_s", None)
    if _as_number(main_wall) is not None:
        rows.append(("main pipeline wall time", _format_time(main_wall)))
        title = "[Main Pipeline Times]"
    else:
        if _as_number(selected_wall) is not None:
            rows.append(("selected stages wall time", _format_time(selected_wall)))
        title = "[Selected Stage Times]"
    if rows:
        _info("-" * 70)
        _info(title)
        _print_rows(rows)

    experiment_times = getattr(result, "experiment_times", {}) if isinstance(getattr(result, "experiment_times", {}), dict) else {}
    if experiment_times:
        rows = []
        for name in tuple(execution_plan.experiments or ()):
            if name not in experiment_times:
                continue
            value = experiment_times.get(name)
            rows.append((str(name).replace("-", " ") + " time", _format_time(value)))
        optional_total = getattr(result, "optional_total_time_s", None)
        if _as_number(optional_total) is not None:
            rows.append(("optional total time", _format_time(optional_total)))
        _info("-" * 70)
        _info("[Optional Experiment Times]")
        _print_rows(rows)

    full_wall = getattr(result, "full_command_wall_time_s", None)
    if _as_number(full_wall) is not None:
        _info("-" * 70)
        _info("[Full Command]")
        _print_rows([("wall time", _format_time(full_wall))])


def _print_eval_summary(summary: Dict[str, object]) -> None:
    payload = _as_dict(summary.get("payload"))
    vslam = _as_dict(summary.get("vslam"))

    _info("-" * 70)
    _info("[Payload]")
    _print_rows(
        [
            ("Raw size", _format_mib(payload.get("raw_bytes"))),
            ("Payload size", _format_mib(payload.get("payload_bytes"))),
            ("Transmission ratio", _format_optional_number(payload.get("ratio"), 4)),
            ("Reduction", _format_optional_number(payload.get("reduction_pct"), 2, "%")),
            ("Raw frames", _format_count(payload.get("raw_frame_count"))),
            ("Transmitted frames", _format_count(payload.get("transmitted_frame_count"))),
            ("Generation units", _format_count(payload.get("unit_count"))),
        ]
    )

    ori_ape = vslam.get("ori_ape_rmse_m")
    rec_ape = vslam.get("rec_ape_rmse_m")
    _info("-" * 70)
    _info("[VSLAM]")
    _print_rows(
        [
            ("APE RMSE ORI", _format_optional_number(ori_ape, 4, " m")),
            ("APE RMSE REC", _format_optional_number(rec_ape, 4, " m")),
            (
                "APE REC-ORI",
                _format_signed_number(vslam.get("ape_delta_rec_minus_ori_m", _delta(rec_ape, ori_ape)), 4, " m"),
            ),
        ]
    )

    warnings = list(summary.get("warnings") or [])
    if warnings:
        _info("-" * 70)
        _info("[Warnings]")
        for warning in warnings:
            _info(str(warning))


def _print_requested_experiment_summaries(exp_dir: Path, experiments: object) -> None:
    for name in tuple(experiments or ()):
        if name == "motion-benchmark":
            summary = _read_json_dict(exp_dir / "encode" / "motion_benchmark" / "summary.json")
            _info("-" * 70)
            _info("[Motion Benchmark]")
            methods = _as_dict(summary.get("methods"))
            if not methods:
                _info("summary : unavailable")
                continue
            rows = []
            for method, payload in sorted(methods.items()):
                item = _as_dict(payload)
                rows.append(
                    (
                        method,
                        "valid={} time={} avg={}".format(
                            _format_optional_number(item.get("valid_rate"), 3),
                            _format_optional_number(item.get("total_time_s"), 2, "s"),
                            _format_optional_number(item.get("avg_time_ms_per_pair"), 2, " ms/pair"),
                        ),
                    )
                )
            _print_rows(rows)
        elif name == "compression-benchmark":
            summary = _read_json_dict(exp_dir / "eval" / "compression_benchmark" / "summary.json")
            _info("-" * 70)
            _info("[Compression Benchmark]")
            methods = _as_dict(summary.get("methods"))
            if not methods:
                _info("summary : unavailable")
                continue
            rows = []
            for method, payload in sorted(methods.items()):
                item = _as_dict(payload)
                parts = [
                    "status={}".format(item.get("status") or "n/a"),
                    "payload={}".format(_format_mib(item.get("payload_bytes"))),
                    "reduction={}".format(_format_optional_number(item.get("reduction_pct"), 2, "%")),
                    "APE={}".format(_format_optional_number(item.get("ape_rmse_m"), 4, " m")),
                ]
                error = str(item.get("error_message") or "").strip()
                if error:
                    parts.append("error={}".format(error))
                rows.append(
                    (
                        method,
                        " ".join(parts),
                    )
                )
            _print_rows(rows)
        elif name == "image-quality":
            summary = _read_json_dict(exp_dir / "decode" / "image_quality" / "summary.json")
            _info("-" * 70)
            _info("[Image Quality]")
            row = _as_dict(summary.get("row"))
            if not row:
                _info("summary : unavailable")
                continue
            _print_rows(
                [
                    ("Matched frames", _format_count(row.get("matched_frame_count"))),
                    ("Evaluated frames", _format_count(row.get("evaluated_frame_count"))),
                    ("LPIPS", _format_optional_number(row.get("lpips"), 6)),
                    ("SSIM", _format_optional_number(row.get("ssim"), 6)),
                    ("FID", _format_optional_number(row.get("fid"), 6)),
                ]
            )


def _print_completion_report(result, execution_plan) -> None:
    sep = "=" * 70
    _info(sep)
    _info("RUN COMPLETE")
    _info(sep)
    _print_rows(
        [
            ("Stages", ", ".join(str(stage) for stage in execution_plan.stages)),
            ("Experiments", ", ".join(str(item) for item in execution_plan.experiments) or "<none>"),
            ("Artifact Root", str(result.exp_dir)),
        ]
    )

    if "eval" not in tuple(execution_plan.stages):
        _print_runtime_times(result, execution_plan)
        _info(sep)
        return

    summary = _read_json_dict(Path(result.exp_dir) / "eval" / "summary.json")
    if not summary:
        _info("Eval summary : unavailable")
        _info(sep)
        return

    _print_runtime_times(result, execution_plan)
    _print_eval_summary(summary)
    _print_requested_experiment_summaries(Path(result.exp_dir), execution_plan.experiments)
    _info(sep)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m exphub", add_help=True, allow_abbrev=False)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["infer", "train"],
        help="execution mode",
    )
    ap.add_argument(
        "--step",
        default=None,
        choices=list(ALL_STAGES),
        help="pipeline step",
    )

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sequence", default="")
    ap.add_argument("--tag", required=True)

    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--dur", type=str, default=None)
    ap.add_argument("--start", type=str, default=None)

    ap.add_argument("--seed", type=int, default=12345, dest="seed_base")
    ap.add_argument("--decode-profile", default="", dest="decode_profile", help="ComfyUI decode workflow profile override")
    ap.add_argument("--experiments", nargs="+", default=(), choices=list(EXPERIMENT_CHOICES))
    ap.add_argument("--log-level", default="info", choices=["info", "quiet"], dest="log_level", help="child process terminal verbosity")
    return ap


def _validate_mode_inputs(ap: argparse.ArgumentParser, args) -> None:
    if args.mode == "infer":
        if not args.sequence:
            ap.error("--sequence is required for --mode infer")
        if args.start is None or str(args.start).strip() == "":
            ap.error("--start is required for --mode infer")
        if args.dur is None or str(args.dur).strip() == "":
            ap.error("--dur is required for --mode infer")
    elif args.mode == "train":
        if args.start is not None and str(args.start).strip() != "":
            ap.error("train mode does not accept --start")
        if args.dur is not None and str(args.dur).strip() != "":
            ap.error("train mode does not accept --dur")
        args.start = ""
        args.dur = ""
    else:
        ap.error("unsupported mode: {}".format(args.mode))

    args.dataset = sanitize_token(args.dataset)
    args.sequence = sanitize_token(args.sequence) if str(args.sequence or "").strip() else ""
    args.tag = sanitize_token(args.tag)
    if not args.dataset or not args.tag:
        ap.error("dataset/tag becomes empty after sanitize")
    if args.mode == "infer" and not args.sequence:
        ap.error("dataset/sequence/tag becomes empty after sanitize")


def _parse_args_and_plan(argv: Optional[List[str]] = None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    args.mode = str(args.mode or "").strip().lower()
    requested_step = args.step
    try:
        execution_plan = build_execution_plan(
            mode=args.mode,
            requested_step=requested_step,
            experiments=tuple(args.experiments or ()),
            seed=args.seed_base,
        )
    except ExecutionPlanError as exc:
        ap.error(str(exc))

    args.mode = execution_plan.mode
    args.requested_step = execution_plan.requested_step
    args.step = execution_plan.resolved_step
    args.experiments = execution_plan.experiments
    _validate_mode_inputs(ap, args)
    run_config = RunConfig(
        dataset=args.dataset,
        sequence=args.sequence,
        tag=args.tag,
        fps=int(args.fps),
        start=str(args.start),
        dur=str(args.dur),
        seed=int(args.seed_base),
        decode_profile=str(args.decode_profile or ""),
        log_level=str(args.log_level or "info").strip().lower(),
    )
    return run_config, execution_plan


def main(argv: Optional[List[str]] = None) -> None:
    command_argv = list(sys.argv[1:] if argv is None else argv)
    run_config, execution_plan = _parse_args_and_plan(argv)
    global _CLI_LOG_LEVEL
    _CLI_LOG_LEVEL = str(run_config.log_level or "info").strip().lower()
    set_cli_log_level(_CLI_LOG_LEVEL)

    runtime = build_runtime(run_config, execution_plan, command_argv=command_argv)

    if _CLI_LOG_LEVEL != "quiet":
        _print_experiment_summary(
            mode=execution_plan.mode,
            step=execution_plan.resolved_step,
            dataset=runtime.spec.dataset,
            sequence=runtime.spec.sequence or "<all>",
            tag=runtime.spec.tag,
            fps_text=runtime.fps_arg,
            dur_text=str(run_config.dur),
            start_text=str(run_config.start),
            exp_dir=runtime.paths.exp_dir,
        )

    try:
        result = run_runtime(runtime, execution_plan)
        _print_completion_report(result, execution_plan)
    except (ConfigError, RunError, RuntimeError) as e:
        _die(str(e))


if __name__ == "__main__":
    main()
