from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from exphub.common.config import load_datasets_cfg, resolve_dataset
from exphub.common.io import read_json_dict, remove_path
from exphub.common.logging import log_info, log_prog, log_warn
from exphub.common.types import canon_num_str, sanitize_token
from exphub.pipeline.encode import service as encode_service
from exphub.pipeline.export import clip_builder, dataset_writer, report


FOCUS_DATASETS = {
    "ncd_scand": ["ncd", "scand"],
    "ncd": ["ncd"],
    "scand": ["scand"],
}


def _collapse_ws(text):
    return " ".join(str(text or "").strip().split()).strip()


def _load_cfg(runtime):
    return load_datasets_cfg(runtime.cfg_path)


def _focus_name(args):
    if str(args.export_scope or "single") == "focus":
        return str(args.export_focus or "ncd_scand")
    if str(args.export_scope or "single") == "dataset":
        return str(args.dataset)
    return "{}_{}".format(str(args.dataset), str(args.sequence))


def _export_root(runtime):
    base = Path(str(runtime.args.export_root or "").strip()).resolve() if str(runtime.args.export_root or "").strip() else (
        runtime.exphub_root / "exports"
    ).resolve()
    scope = str(runtime.args.export_scope or "single")
    focus_name = _focus_name(runtime.args)
    return (
        base
        / scope
        / sanitize_token(focus_name, max_len=64)
        / sanitize_token(runtime.args.tag, max_len=64)
    ).resolve()


def _export_profile(args):
    return clip_builder.build_export_profile(
        target_fps=int(args.export_target_fps),
        target_num_frames=int(args.export_target_num_frames),
        target_width=int(args.export_target_width),
        target_height=int(args.export_target_height),
        harvest_sec=float(args.export_harvest_sec) if float(args.export_harvest_sec or 0.0) > 0.0 else None,
        stride_sec=float(args.export_stride_sec) if float(args.export_stride_sec or 0.0) > 0.0 else None,
    )


def _iter_dataset_sequences(cfg, dataset_name):
    datasets = dict((cfg or {}).get("datasets") or {})
    dataset_obj = dict(datasets.get(dataset_name) or {})
    sequences = dict(dataset_obj.get("sequences") or {})
    return sorted(sequences.keys())


def _resolve_targets(runtime, cfg):
    args = runtime.args
    scope = str(args.export_scope or "single")
    selected = []

    if scope == "single":
        selected.append({"dataset": str(args.dataset), "sequence": str(args.sequence)})
    elif scope == "dataset":
        for sequence_name in _iter_dataset_sequences(cfg, str(args.dataset)):
            selected.append({"dataset": str(args.dataset), "sequence": str(sequence_name)})
    elif scope == "focus":
        for dataset_name in list(FOCUS_DATASETS.get(str(args.export_focus or "ncd_scand"), [])):
            per_dataset = 0
            for sequence_name in _iter_dataset_sequences(cfg, dataset_name):
                selected.append({"dataset": str(dataset_name), "sequence": str(sequence_name)})
                per_dataset += 1
                if int(args.export_max_bags_per_dataset or 0) > 0 and per_dataset >= int(args.export_max_bags_per_dataset):
                    break
    else:
        raise RuntimeError("unsupported export scope: {}".format(scope))

    if int(args.export_max_bags or 0) > 0:
        selected = selected[: int(args.export_max_bags)]

    resolved = []
    for item in selected:
        ds = resolve_dataset(cfg, runtime.exphub_root, item["dataset"], item["sequence"])
        resolved.append(
            {
                "dataset": str(ds.dataset),
                "sequence": str(ds.sequence),
                "bag": str(ds.bag),
                "topic": str(ds.topic),
                "fx": float(ds.fx),
                "fy": float(ds.fy),
                "cx": float(ds.cx),
                "cy": float(ds.cy),
                "dist": list(ds.dist),
            }
        )
    return resolved


def _inspect_bag_topic(runtime, bag_path, topic):
    python_bin = runtime.phase_python("segment")
    ros_setup = Path(str(runtime.args.ros_setup or "")).resolve() if str(runtime.args.ros_setup or "").strip() else None
    script = r"""
import json
import sys
import rosbag

bag_path = sys.argv[1]
topic = sys.argv[2]
first_stamp = None
last_stamp = None
message_count = 0
with rosbag.Bag(bag_path, "r") as bag:
    for _topic, _msg, stamp in bag.read_messages(topics=[topic]):
        value = float(stamp.to_sec())
        if first_stamp is None:
            first_stamp = value
        last_stamp = value
        message_count += 1

if first_stamp is None or last_stamp is None:
    payload = {"message_count": 0, "duration_sec": 0.0}
else:
    payload = {
        "message_count": int(message_count),
        "duration_sec": max(0.0, float(last_stamp - first_stamp)),
        "first_stamp": float(first_stamp),
        "last_stamp": float(last_stamp),
    }
print(json.dumps(payload, ensure_ascii=False))
""".strip()
    cmd = [shlex.quote(str(python_bin)), "-c", shlex.quote(script), shlex.quote(str(Path(bag_path).resolve())), shlex.quote(str(topic))]
    if ros_setup is not None and ros_setup.exists():
        shell_script = "source {} && {}".format(shlex.quote(str(ros_setup)), " ".join(cmd))
    else:
        shell_script = " ".join(cmd)
    output = subprocess.check_output(["bash", "-lc", shell_script], text=True)
    payload = json.loads(str(output or "").strip() or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("bag inspector returned non-dict payload")
    return payload


def _build_harvest_plan(bag_info, harvest_sec, max_clips_per_bag):
    harvest_duration = float(harvest_sec)
    max_start = float(bag_info.get("duration_sec", 0.0) or 0.0) - harvest_duration
    if max_start < -1e-9:
        return []

    harvests = []
    harvest_idx = 0
    start_sec = 0.0
    while start_sec <= max_start + 1e-9:
        harvests.append(
            {
                "clip_index": int(harvest_idx),
                "start_sec": float(round(start_sec, 6)),
                "duration_sec": float(harvest_duration),
            }
        )
        harvest_idx += 1
        if int(max_clips_per_bag or 0) > 0 and len(harvests) >= int(max_clips_per_bag):
            break
        start_sec += float(harvest_duration)
    return harvests


def _make_encode_args(runtime, target, export_root, clip_plan, profile):
    from types import SimpleNamespace

    cache_root = (Path(export_root).resolve() / "cache" / target["dataset"] / target["sequence"]).resolve()
    return SimpleNamespace(
        mode="encode",
        exphub=str(runtime.exphub_root),
        datasets_cfg=str(runtime.cfg_path),
        dataset=str(target["dataset"]),
        sequence=str(target["sequence"]),
        tag=str(runtime.args.tag),
        w=int(profile["target_width"]),
        h=int(profile["target_height"]),
        fps=float(profile["target_fps"]),
        dur=str(canon_num_str(profile["harvest_sec"])),
        start_sec=str(clip_plan["start_sec"]),
        start_idx=-1,
        kf_gap=0,
        keyframes_mode=str(runtime.args.keyframes_mode),
        segment_policy=str(runtime.args.segment_policy),
        base_idx=0,
        seed_base=int(runtime.args.seed_base),
        gpus=int(runtime.args.gpus),
        infer_extra=str(runtime.args.infer_extra),
        infer_backend=str(runtime.args.infer_backend),
        infer_model_dir=str(runtime.args.infer_model_dir),
        exp_root=str(cache_root),
        keep_level="max",
        log_level=str(runtime.args.log_level),
        auto_conda=bool(runtime.args.auto_conda),
        videox_root=str(runtime.args.videox_root),
        droid_repo=str(runtime.args.droid_repo),
        droid_weights=str(runtime.args.droid_weights),
        prompt_model_dir=str(runtime.args.prompt_model_dir),
        ros_setup=str(runtime.args.ros_setup),
        droid_seq=str(runtime.args.droid_seq),
        viz=False,
        no_viz=True,
        export_root=str(runtime.args.export_root or ""),
        export_scope=str(runtime.args.export_scope),
        export_focus=str(runtime.args.export_focus),
        export_target_fps=int(profile["target_fps"]),
        export_target_num_frames=int(profile["target_num_frames"]),
        export_target_width=int(profile["target_width"]),
        export_target_height=int(profile["target_height"]),
        export_harvest_sec=float(profile["harvest_sec"]),
        export_stride_sec=float(profile["stride_sec"]),
        export_max_bags=int(runtime.args.export_max_bags),
        export_max_bags_per_dataset=int(runtime.args.export_max_bags_per_dataset),
        export_max_clips_per_bag=int(runtime.args.export_max_clips_per_bag),
        export_split_seed=int(runtime.args.export_split_seed),
    )


def _run_encode_for_clip(runtime, target, export_root, clip_plan, profile):
    from exphub.pipeline.orchestrator import build_runtime

    encode_args = _make_encode_args(runtime, target, export_root, clip_plan, profile)
    encode_runtime = build_runtime(encode_args)
    encode_service.run(encode_runtime)
    return encode_runtime.paths.exp_dir.resolve()


def _metadata_entry(export_root, split, clip_path, caption):
    return {
        "file_path": dataset_writer.relative_to_root(export_root, clip_path),
        "text": str(caption),
        "type": "video",
        "split": str(split),
    }


def _candidate_rejections_to_skips(target, clip_plan, exp_dir, rejections):
    skips = []
    for rejection in list(rejections or []):
        item = dict(rejection or {})
        item.update(
            {
                "dataset": str(target["dataset"]),
                "sequence": str(target["sequence"]),
                "clip_index": int(clip_plan["clip_index"]),
                "start_sec": float(clip_plan["start_sec"]),
            }
        )
        if exp_dir is not None:
            item["exp_dir"] = str(exp_dir)
        skips.append(item)
    return skips


def run(runtime):
    args = runtime.args
    cfg = _load_cfg(runtime)
    export_root = _export_root(runtime)
    profile = _export_profile(args)
    if export_root.exists():
        remove_path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    layout = dataset_writer.ensure_layout(export_root)
    targets = _resolve_targets(runtime, cfg)
    if not targets:
        raise RuntimeError("export resolved no dataset targets")

    log_info(
        "export start: planner=generation_units scope={} focus={} targets={} profile={}fps/{}f/{}x{} harvest={:.2f}s stride={:.2f}s".format(
            str(args.export_scope),
            str(args.export_focus),
            int(len(targets)),
            int(profile["target_fps"]),
            int(profile["target_num_frames"]),
            int(profile["target_width"]),
            int(profile["target_height"]),
            float(profile["harvest_sec"]),
            float(profile["stride_sec"]),
        )
    )

    exported_clips = []
    skipped_clips = []
    entries_by_split = {"train": [], "val": [], "test": []}
    total_t0 = time.time()

    for target in targets:
        bag_info = _inspect_bag_topic(runtime, target["bag"], target["topic"])
        harvest_plan_items = _build_harvest_plan(
            bag_info=bag_info,
            harvest_sec=float(profile["harvest_sec"]),
            max_clips_per_bag=int(args.export_max_clips_per_bag or 0),
        )
        if not harvest_plan_items:
            skipped_clips.append(
                {
                    "dataset": str(target["dataset"]),
                    "sequence": str(target["sequence"]),
                    "reason": "bag_too_short_for_harvest_window",
                    "bag_duration_sec": float(bag_info.get("duration_sec", 0.0) or 0.0),
                    "harvest_sec": float(profile["harvest_sec"]),
                }
            )
            continue

        log_info(
            "export bag: dataset={} sequence={} harvest_windows={} duration={:.2f}s".format(
                str(target["dataset"]),
                str(target["sequence"]),
                int(len(harvest_plan_items)),
                float(bag_info.get("duration_sec", 0.0) or 0.0),
            )
        )
        for clip_plan in harvest_plan_items:
            exp_dir = None
            try:
                exp_dir = _run_encode_for_clip(runtime, target, export_root, clip_plan, profile)
                segment_manifest_path = (exp_dir / "segment" / "segment_manifest.json").resolve()
                prompt_manifest_path = (exp_dir / "prompt" / "prompt_manifest.json").resolve()
                segment_manifest = read_json_dict(segment_manifest_path)
                prompt_manifest = read_json_dict(prompt_manifest_path)
                candidate_result = clip_builder.select_training_candidates(
                    segment_manifest=segment_manifest,
                    exp_dir=exp_dir,
                    profile=profile,
                )
                skipped_clips.extend(
                    _candidate_rejections_to_skips(
                        target=target,
                        clip_plan=clip_plan,
                        exp_dir=exp_dir,
                        rejections=candidate_result.get("rejections") or [],
                    )
                )

                candidates = list(candidate_result.get("candidates") or [])
                if not candidates:
                    continue

                for candidate in candidates:
                    clip_key = "{}:{}:{:.6f}:{}".format(
                        target["dataset"],
                        target["sequence"],
                        float(clip_plan["start_sec"]),
                        str(candidate.get("clip_id", "") or "clip"),
                    )
                    split = dataset_writer.assign_split(clip_key, seed=int(args.export_split_seed))
                    clip_filename = clip_builder.build_clip_filename(
                        target["dataset"],
                        target["sequence"],
                        clip_plan["clip_index"],
                        clip_plan["start_sec"],
                        sample_index=candidate.get("sample_index"),
                    )
                    clip_output_path = (layout["split_dirs"][split] / clip_filename).resolve()
                    clip_builder.write_training_clip(
                        exp_dir / "segment" / "frames",
                        clip_output_path,
                        start_idx=int(candidate["clip_start_idx"]),
                        num_frames=int(candidate["clip_num_frames"]),
                        fps=int(profile["target_fps"]),
                    )
                    clip_manifest_path = dataset_writer.write_clip_manifest(
                        export_root,
                        Path(clip_filename).stem,
                        candidate.get("clip_manifest") or {},
                    )
                    metadata_entry = _metadata_entry(export_root, split, clip_output_path, candidate["caption"])
                    entries_by_split[split].append(metadata_entry)

                    clip_record = {
                        "dataset": str(target["dataset"]),
                        "sequence": str(target["sequence"]),
                        "clip_index": int(clip_plan["clip_index"]),
                        "clip_id": str(candidate.get("clip_id", "") or ""),
                        "start_sec": float(clip_plan["start_sec"]),
                        "harvest_sec": float(profile["harvest_sec"]),
                        "split": str(split),
                        "file_path": metadata_entry["file_path"],
                        "text": metadata_entry["text"],
                        "type": "video",
                        "exp_dir": str(exp_dir),
                        "segment_manifest": str(segment_manifest_path),
                        "prompt_manifest": str(prompt_manifest_path),
                        "train_clip_manifest": dataset_writer.relative_to_root(export_root, clip_manifest_path),
                        "train_start_idx": int(candidate["clip_start_idx"]),
                        "train_end_idx": int(candidate["clip_end_idx"]),
                        "target_num_frames": int(candidate["clip_num_frames"]),
                        "target_fps": int(profile["target_fps"]),
                        "selection_reason": str(candidate.get("selection_reason", "") or ""),
                        "candidate_summary": dict(candidate.get("summary") or {}),
                        "source_unit_ids": list(candidate.get("source_unit_ids") or []),
                        "source_span_id": str(candidate.get("source_span_id", "") or ""),
                        "source_prompt_ref": dict(candidate.get("source_prompt_ref") or {}),
                        "resolved_prompt_source": str(candidate.get("resolved_prompt_source", "prompt_spans") or "prompt_spans"),
                    }
                    exported_clips.append(clip_record)
            except Exception as exc:
                skipped_clips.append(
                    {
                        "dataset": str(target["dataset"]),
                        "sequence": str(target["sequence"]),
                        "clip_index": int(clip_plan["clip_index"]),
                        "start_sec": float(clip_plan["start_sec"]),
                        "reason": "export_failed:{}".format(_collapse_ws(exc)),
                    }
                )
                log_warn(
                    "export clip skipped: dataset={} sequence={} start_sec={} reason={}".format(
                        str(target["dataset"]),
                        str(target["sequence"]),
                        str(clip_plan["start_sec"]),
                        _collapse_ws(exc),
                    )
                )

    metadata_paths = dataset_writer.write_metadata_files(export_root, entries_by_split)
    dataset_report = report.build_dataset_report(
        export_root=export_root,
        scope=str(args.export_scope),
        focus_name=str(_focus_name(args)),
        training_spec=clip_builder.training_spec(profile),
        targets=targets,
        exported_clips=exported_clips,
        skipped_clips=skipped_clips,
        metadata_paths=metadata_paths,
    )
    report_path = report.write_dataset_report(export_root, dataset_report)

    log_prog(
        "export summary: clips={} skipped={} elapsed={:.2f}s".format(
            int(len(exported_clips)),
            int(len(skipped_clips)),
            float(time.time() - total_t0),
        )
    )
    log_info("export dataset report: {}".format(report_path))
    if not exported_clips:
        raise RuntimeError("export produced no training clips: {}".format(report_path))
    return export_root


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="ExpHub export mainline.")
    parser.add_argument("--run-formal-mainline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exphub", default="")
    parser.add_argument("--datasets_cfg", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--export_root", default="")
    parser.add_argument("--export_scope", default="single", choices=["single", "dataset", "focus"])
    parser.add_argument("--export_focus", default="ncd_scand", choices=sorted(FOCUS_DATASETS.keys()))
    parser.add_argument("--export_target_fps", type=int, default=clip_builder.DEFAULT_EXPORT_FPS)
    parser.add_argument("--export_target_num_frames", type=int, default=clip_builder.DEFAULT_EXPORT_NUM_FRAMES)
    parser.add_argument("--export_target_width", type=int, default=clip_builder.DEFAULT_EXPORT_WIDTH)
    parser.add_argument("--export_target_height", type=int, default=clip_builder.DEFAULT_EXPORT_HEIGHT)
    parser.add_argument("--export_harvest_sec", type=float, default=0.0)
    parser.add_argument("--export_stride_sec", type=float, default=0.0)
    parser.add_argument("--export_max_bags", type=int, default=0)
    parser.add_argument("--export_max_bags_per_dataset", type=int, default=0)
    parser.add_argument("--export_max_clips_per_bag", type=int, default=0)
    parser.add_argument("--export_split_seed", type=int, default=13)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.run_formal_mainline:
        raise SystemExit("[ERR] use --run-formal-mainline")
    raise SystemExit("export helper should be run through exphub orchestrator")
