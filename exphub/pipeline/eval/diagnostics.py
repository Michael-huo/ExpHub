from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exphub.common.io import read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_prog, log_warn


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
    keyframes_bytes = _pick_int(segment_report, [("keyframes", "bytes_sum")])
    prompt_bytes = _pick_int(prompt_report, [("outputs", "bytes_sum"), ("outputs", "prompt_bytes")])

    if prompt_bytes is None:
        msg = "missing prompt bytes_sum in prompt/report.json; prompt_bytes set to null"
        warnings.append(msg)
        log_warn(msg)
    if ori_frames <= 0:
        msg = "segment/frames unavailable for eval compression scan; ori_frames and ori_bytes set to null"
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


def _build_quality(traj_metrics, slam_report):
    primary_track = str(_get_nested(slam_report, ("primary_track",)) or "")
    return {
        "eval_status": str((traj_metrics or {}).get("eval_status", "") or ""),
        "traj_status": str((traj_metrics or {}).get("eval_status", "") or ""),
        "ape_rmse": _pick_float(traj_metrics, [("ape_trans", "rmse")]),
        "rpe_trans_rmse": _pick_float(traj_metrics, [("rpe_trans", "rmse")]),
        "rpe_rot_rmse": _pick_float(traj_metrics, [("rpe_rot", "rmse")]),
        "matched_pose_count": _pick_int(traj_metrics, [("matched_pose_count",)]),
        "ori_path_length_m": _pick_float(traj_metrics, [("ori_path_length_m",)]),
        "gen_path_length_m": _pick_float(traj_metrics, [("gen_path_length_m",)]),
        "primary_track": primary_track,
    }


def _stage_status(*statuses):
    ordered = []
    for value in statuses:
        text = str(value or "").strip()
        if text:
            ordered.append(text)
    if not ordered:
        return "missing"
    if any(item not in ("success", "partial") for item in ordered):
        return ordered[0]
    if any(item == "partial" for item in ordered):
        return "partial"
    return "success"


def _stage_created_at(*reports):
    for report_obj in reversed(list(reports)):
        value = str((report_obj or {}).get("created_at", "") or "")
        if value:
            return value
    return ""


def _build_source_summary(infer_report, merge_report, merge_manifest, eval_source, decode_source):
    merge_summary = dict((merge_manifest or {}).get("summary") or {})
    return {
        "eval_source": str(eval_source or "aligned"),
        "decode_source": str(
            (merge_report or {}).get("decode_source")
            or (infer_report or {}).get("decode_source")
            or merge_summary.get("decode_source")
            or decode_source
            or "aligned"
        ),
        "source_unit_count": int(merge_summary.get("source_unit_count", 0) or 0),
        "source_span_count": int(merge_summary.get("source_span_count", 0) or 0),
        "shared_anchor_count": int(merge_summary.get("shared_anchor_count", 0) or 0),
    }


def _build_stage_table(exp_dir, stage_reports, traj_metrics, inputs, eval_dir, eval_source):
    return {
        "encode": {
            "status": _stage_status(
                stage_reports["segment"].get("segment_status"),
                stage_reports["prompt"].get("prompt_status"),
            ),
            "created_at": _stage_created_at(stage_reports["segment"], stage_reports["prompt"]),
            "artifacts": {
                "segment_report": _relative_path(exp_dir, Path(exp_dir) / "segment" / "report.json"),
                "prompt_report": _relative_path(exp_dir, Path(exp_dir) / "prompt" / "report.json"),
            },
        },
        "decode": {
            "status": _stage_status(
                stage_reports["infer"].get("infer_status"),
                stage_reports["merge"].get("merge_status"),
            ),
            "created_at": _stage_created_at(stage_reports["infer"], stage_reports["merge"]),
            "artifacts": {
                "decode_source": str(inputs.get("decode_source", "aligned") or "aligned"),
                "infer_report": _relative_path(exp_dir, Path(inputs.get("infer_report")).resolve()),
                "merge_report": _relative_path(exp_dir, Path(inputs.get("merge_report")).resolve()),
                "merge_manifest": _relative_path(exp_dir, Path(inputs.get("merge_manifest")).resolve()),
            },
        },
        "eval": {
            "status": _stage_status(stage_reports["slam"].get("slam_status"), traj_metrics.get("eval_status")),
            "created_at": _stage_created_at(stage_reports["slam"], {"created_at": traj_metrics.get("created_at")}),
            "artifacts": {
                "eval_source": str(eval_source or "aligned"),
                "slam_report": _relative_path(exp_dir, Path(eval_dir).resolve() / "slam" / "report.json"),
                "traj_metrics": _relative_path(exp_dir, Path(eval_dir).resolve() / "metrics" / "traj_eval.json"),
                "report": _relative_path(exp_dir, Path(eval_dir).resolve() / "report.json"),
            },
        },
    }


def _build_compression_snapshot(compression_summary):
    ratio = compression_summary.get("ratio_bytes")
    reduction = None if ratio is None else 1.0 - float(ratio)
    comp_size = None
    if compression_summary.get("keyframes_bytes") is not None and compression_summary.get("prompt_bytes") is not None:
        comp_size = int(compression_summary["keyframes_bytes"]) + int(compression_summary["prompt_bytes"])
    return {
        "ratio": ratio,
        "reduction": reduction,
        "orig_size": compression_summary.get("ori_bytes"),
        "comp_size": comp_size,
        "keyframes": compression_summary.get("keyframes_frames"),
    }


def run_diagnostics_substage(args):
    exp_dir = Path(args.exp_dir).resolve()
    eval_dir = Path(args.out_dir).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_source = str(args.eval_source or "aligned").strip().lower() or "aligned"
    decode_source = str(args.decode_source or eval_source).strip().lower() or "aligned"

    warnings = []
    stage_reports = {
        "segment": _read_stage_report(exp_dir / "segment" / "report.json", warnings),
        "prompt": _read_stage_report(exp_dir / "prompt" / "report.json", warnings),
        "infer": _read_stage_report(Path(args.infer_report), warnings),
        "merge": _read_stage_report(Path(args.merge_report), warnings),
        "slam": _read_stage_report(Path(args.slam_report), warnings),
    }
    merge_manifest = dict(read_json_dict(Path(args.merge_manifest)) or {})
    if not merge_manifest:
        msg = "missing or invalid merge manifest: {}".format(Path(args.merge_manifest).resolve())
        warnings.append(msg)
        log_warn(msg)
    traj_metrics = dict(read_json_dict(Path(args.traj_metrics)) or {})
    if not traj_metrics:
        msg = "missing eval trajectory metrics: {}".format(Path(args.traj_metrics).resolve())
        warnings.append(msg)
        log_warn(msg)

    compression_summary = _build_compression_summary(exp_dir, stage_reports["segment"], stage_reports["prompt"], warnings)
    quality = _build_quality(traj_metrics, stage_reports["slam"])
    compression_snapshot = _build_compression_snapshot(compression_summary)
    source_summary = _build_source_summary(
        infer_report=stage_reports["infer"],
        merge_report=stage_reports["merge"],
        merge_manifest=merge_manifest,
        eval_source=eval_source,
        decode_source=decode_source,
    )

    final_report = {
        "report_schema_version": "eval_report.v2",
        "step": "eval",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_source": str(eval_source),
        "decode_source": str(source_summary["decode_source"]),
        "source_unit_count": int(source_summary["source_unit_count"]),
        "source_span_count": int(source_summary["source_span_count"]),
        "shared_anchor_count": int(source_summary["shared_anchor_count"]),
        "workflow": "encode -> decode -> eval",
        "inputs": {
            "segment_report": _relative_path(exp_dir, exp_dir / "segment" / "report.json"),
            "prompt_report": _relative_path(exp_dir, exp_dir / "prompt" / "report.json"),
            "infer_report": _relative_path(exp_dir, Path(args.infer_report).resolve()),
            "merge_report": _relative_path(exp_dir, Path(args.merge_report).resolve()),
            "merge_manifest": _relative_path(exp_dir, Path(args.merge_manifest).resolve()),
            "slam_report": _relative_path(exp_dir, Path(args.slam_report).resolve()),
            "traj_metrics": _relative_path(exp_dir, Path(args.traj_metrics).resolve()),
            "summary": _relative_path(exp_dir, Path(args.summary).resolve()),
            "details": _relative_path(exp_dir, Path(args.details).resolve()),
        },
        "stages": _build_stage_table(
            exp_dir,
            stage_reports,
            traj_metrics,
            inputs={
                "infer_report": args.infer_report,
                "merge_report": args.merge_report,
                "merge_manifest": args.merge_manifest,
                "decode_source": source_summary["decode_source"],
            },
            eval_dir=eval_dir,
            eval_source=eval_source,
        ),
        "compression": compression_summary,
        "quality": quality,
        "traj_eval": traj_metrics,
        "source_summary": source_summary,
        "slam": {
            "slam_status": str(stage_reports["slam"].get("slam_status", "") or ""),
            "primary_track": str(stage_reports["slam"].get("primary_track", "") or ""),
            "primary_trajectory_path": str(stage_reports["slam"].get("primary_trajectory_path", "") or ""),
            "reference_track": str(stage_reports["slam"].get("reference_track", "") or ""),
            "reference_trajectory_path": str(stage_reports["slam"].get("reference_trajectory_path", "") or ""),
        },
        "warnings": warnings,
        "artifact_contract": {
            "formal_files": [
                "report.json",
                "compression.json",
                "summary.txt",
                "details.csv",
                "metrics/traj_eval.json",
                "plots/traj_xy.png",
                "plots/metrics_overview.png",
                "slam/report.json",
                "slam/traj_est.txt",
            ],
            "formal_track_files": [
                "slam/<track>/traj_est.tum",
                "slam/<track>/traj_est.npz",
                "slam/<track>/run_meta.json",
            ],
        },
    }

    report_path = eval_dir / "report.json"
    compression_path = eval_dir / "compression.json"
    write_json_atomic(report_path, final_report, indent=2)
    write_json_atomic(compression_path, compression_snapshot, indent=2)
    log_prog("eval diagnostics: final report generated")
    log_info("eval report: {}".format(report_path))
    log_info("eval compression snapshot: {}".format(compression_path))
    return {
        "report_path": report_path,
        "compression_path": compression_path,
        "report": final_report,
        "compression": compression_snapshot,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-formal-mainline", action="store_true")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--slam_report", required=True)
    parser.add_argument("--infer_report", required=True)
    parser.add_argument("--merge_report", required=True)
    parser.add_argument("--merge_manifest", required=True)
    parser.add_argument("--eval_source", default="aligned")
    parser.add_argument("--decode_source", default="aligned")
    parser.add_argument("--traj_metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--details", required=True)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("eval diagnostics helper requires --run-formal-mainline")
    run_diagnostics_substage(args)


if __name__ == "__main__":
    main()
