from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_warn
from exphub.contracts import stats as stats_contract


_PROMPT_PHASE = "prompt_smol"


def _as_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _as_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return str(target.relative_to(base))
    except Exception:
        return str(target)


def _get_nested(obj, path):
    current = obj
    for key in list(path or []):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _pick_int(meta, candidate_paths):
    for path in list(candidate_paths or []):
        value = _as_int_or_none(_get_nested(meta, path))
        if value is not None:
            return value
    return None


def _pick_float(meta, candidate_paths):
    for path in list(candidate_paths or []):
        value = _as_float_or_none(_get_nested(meta, path))
        if value is not None:
            return value
    return None


def _dir_png_stats(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        return 0, 0
    count = 0
    bytes_sum = 0
    for item in root.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            continue
        count += 1
        try:
            bytes_sum += int(item.stat().st_size)
        except Exception:
            pass
    return int(count), int(bytes_sum)


def _stage_status(stage_name, report_obj):
    candidates = [
        "{}_status".format(stage_name),
        "eval_status",
        "prompt_status",
        "infer_status",
        "merge_status",
    ]
    for key in candidates:
        value = str(report_obj.get(key, "") or "").strip()
        if value:
            return value
    return "missing" if not report_obj else "success"


def _read_stage_report(path_obj, warnings):
    path = Path(path_obj).resolve()
    if not path.is_file():
        msg = "missing stage report: {}".format(path)
        warnings.append(msg)
        log_warn(msg)
        return {}
    payload = read_json_dict(path)
    if not payload:
        msg = "invalid or empty stage report: {}".format(path)
        warnings.append(msg)
        log_warn(msg)
    return payload


def _build_compression_summary(exp_dir, segment_report, prompt_report, warnings):
    segment_frames_dir = Path(exp_dir).resolve() / "segment" / "frames"
    ori_frames, ori_bytes = _dir_png_stats(segment_frames_dir)

    keyframes_frames = _pick_int(
        segment_report,
        [
            ("keyframes", "count"),
            ("outputs", "keyframes", "frame_count"),
        ],
    )

    keyframes_bytes = _pick_int(
        segment_report,
        [
            ("keyframes", "bytes_sum"),
        ],
    )

    prompt_bytes = _pick_int(
        prompt_report,
        [
            ("outputs", "bytes_sum"),
            ("outputs", "prompt_bytes"),
        ],
    )
    if prompt_bytes is None:
        msg = "missing prompt bytes_sum in prompt/report.json; prompt_bytes set to null"
        warnings.append(msg)
        log_warn(msg)

    if ori_frames <= 0:
        msg = "segment/frames unavailable for stats compression scan; ori_frames and ori_bytes set to null"
        warnings.append(msg)
        log_warn(msg)
        ori_frames = None
        ori_bytes = None

    ratio_bytes = None
    if ori_bytes and ori_bytes > 0 and keyframes_bytes is not None and prompt_bytes is not None:
        ratio_bytes = float(keyframes_bytes + prompt_bytes) / float(ori_bytes)

    ratio_frames = None
    if ori_frames and ori_frames > 0 and keyframes_frames is not None:
        ratio_frames = float(keyframes_frames) / float(ori_frames)

    return {
        "ori_frames": ori_frames,
        "ori_bytes": ori_bytes,
        "keyframes_frames": keyframes_frames,
        "keyframes_bytes": keyframes_bytes,
        "prompt_bytes": prompt_bytes,
        "ratio_bytes": ratio_bytes,
        "ratio_frames": ratio_frames,
    }


def _build_quality(eval_report, slam_report):
    traj_metrics = dict(eval_report.get("traj_eval") or {}) if isinstance(eval_report.get("traj_eval"), dict) else {}

    primary_track = str(_get_nested(slam_report, ("primary_track",)) or "")
    return {
        "eval_status": str(eval_report.get("eval_status", "") or ""),
        "traj_status": str(traj_metrics.get("eval_status", "") or ""),
        "ape_rmse": _pick_float(traj_metrics, [("ape_trans", "rmse")]),
        "rpe_trans_rmse": _pick_float(traj_metrics, [("rpe_trans", "rmse")]),
        "rpe_rot_rmse": _pick_float(traj_metrics, [("rpe_rot", "rmse")]),
        "matched_pose_count": _pick_int(traj_metrics, [("matched_pose_count",)]),
        "ori_path_length_m": _pick_float(traj_metrics, [("ori_path_length_m",)]),
        "gen_path_length_m": _pick_float(traj_metrics, [("gen_path_length_m",)]),
        "primary_track": primary_track,
    }


def _build_stage_table(exp_dir, reports):
    stage_names = ["segment", "prompt", "infer", "merge", "slam", "eval"]
    table = {}
    for stage_name in stage_names:
        report_path = Path(exp_dir).resolve() / stage_name / "report.json"
        if stage_name == "stats":
            continue
        report_obj = dict(reports.get(stage_name) or {})
        table[stage_name] = {
            "report_path": _relative_path(exp_dir, report_path),
            "status": _stage_status(stage_name, report_obj),
            "created_at": str(report_obj.get("created_at", "") or ""),
        }
    return table


def _build_cli_compression_snapshot(exp_dir, compression_summary):
    ratio = compression_summary.get("ratio_bytes")
    reduction = None if ratio is None else 1.0 - float(ratio)
    comp_size = None
    if compression_summary.get("keyframes_bytes") is not None and compression_summary.get("prompt_bytes") is not None:
        comp_size = int(compression_summary.get("keyframes_bytes")) + int(compression_summary.get("prompt_bytes"))
    return {
        "ratio": ratio,
        "reduction": reduction,
        "orig_size": compression_summary.get("ori_bytes"),
        "comp_size": comp_size,
        "keyframes": compression_summary.get("keyframes_frames"),
    }


def _run_formal_mainline(args):
    exp_dir = Path(args.exp_dir).resolve()
    stats_dir = (exp_dir / "stats").resolve()
    stats_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    contract = stats_contract.build_contract(type("PathsProxy", (), {
        "segment_report_path": exp_dir / "segment" / "report.json",
        "prompt_report_path": exp_dir / "prompt" / "report.json",
        "infer_report_path": exp_dir / "infer" / "report.json",
        "merge_report_path": exp_dir / "merge" / "report.json",
        "slam_report_path": exp_dir / "slam" / "report.json",
        "eval_report_path": exp_dir / "eval" / "report.json",
        "stats_dir": stats_dir,
        "stats_report_path": stats_dir / "final_report.json",
        "stats_compression_path": stats_dir / "compression.json",
    })())

    stage_reports = {
        "segment": _read_stage_report(contract.artifacts[stats_contract.SEGMENT_REPORT], warnings),
        "prompt": _read_stage_report(contract.artifacts[stats_contract.PROMPT_REPORT], warnings),
        "infer": _read_stage_report(contract.artifacts[stats_contract.INFER_REPORT], warnings),
        "merge": _read_stage_report(contract.artifacts[stats_contract.MERGE_REPORT], warnings),
        "slam": _read_stage_report(contract.artifacts[stats_contract.SLAM_REPORT], warnings),
        "eval": _read_stage_report(contract.artifacts[stats_contract.EVAL_REPORT], warnings),
    }

    compression_summary = _build_compression_summary(exp_dir, stage_reports["segment"], stage_reports["prompt"], warnings)
    quality = _build_quality(stage_reports["eval"], stage_reports["slam"])

    final_report = {
        "report_schema_version": "stats_final_report.v1",
        "step": "stats",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "workflow": "segment -> prompt -> infer -> merge -> slam -> eval -> stats",
        "inputs": {
            "segment_report": _relative_path(exp_dir, contract.artifacts[stats_contract.SEGMENT_REPORT]),
            "prompt_report": _relative_path(exp_dir, contract.artifacts[stats_contract.PROMPT_REPORT]),
            "infer_report": _relative_path(exp_dir, contract.artifacts[stats_contract.INFER_REPORT]),
            "merge_report": _relative_path(exp_dir, contract.artifacts[stats_contract.MERGE_REPORT]),
            "slam_report": _relative_path(exp_dir, contract.artifacts[stats_contract.SLAM_REPORT]),
            "eval_report": _relative_path(exp_dir, contract.artifacts[stats_contract.EVAL_REPORT]),
        },
        "stages": _build_stage_table(exp_dir, stage_reports),
        "compression": compression_summary,
        "quality": quality,
        "warnings": warnings,
        "artifact_contract": {
            "formal_files": [
                "final_report.json",
                "compression.json",
            ],
        },
    }

    compression_snapshot = _build_cli_compression_snapshot(exp_dir, compression_summary)
    write_json_atomic(contract.artifacts[stats_contract.FINAL_REPORT], final_report, indent=2)
    write_json_atomic(contract.artifacts[stats_contract.COMPRESSION], compression_snapshot, indent=2)

    log_prog("stats summary: final report generated")
    log_info("stats final report: {}".format(contract.artifacts[stats_contract.FINAL_REPORT]))
    log_info("stats compression snapshot: {}".format(contract.artifacts[stats_contract.COMPRESSION]))
    return Path(contract.artifacts[stats_contract.FINAL_REPORT]).resolve()


def run(runtime):
    helper_path = (runtime.exphub_root / "exphub" / "pipeline" / "stats" / "service.py").resolve()
    cmd = [
        "python",
        str(helper_path),
        "--run-formal-mainline",
        "--exp_dir",
        str(runtime.paths.exp_dir),
    ]
    runtime.step_runner.run_env_python(
        cmd,
        phase_name=_PROMPT_PHASE,
        log_name="stats.log",
        cwd=runtime.exphub_root,
    )

    contract = stats_contract.build_contract(runtime.paths)
    if not Path(contract.artifacts[stats_contract.FINAL_REPORT]).is_file():
        raise RuntimeError("missing stats final report: {}".format(contract.artifacts[stats_contract.FINAL_REPORT]))
    return contract.artifacts[stats_contract.FINAL_REPORT]


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("stats service helper requires --run-formal-mainline")
    _run_formal_mainline(args)


if __name__ == "__main__":
    main()
