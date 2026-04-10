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
from exphub.common.types import sanitize_token
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
    return (base / scope / sanitize_token(focus_name, max_len=64) / sanitize_token(runtime.args.tag, max_len=64)).resolve()


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
        shell_script = "source {} && {}".format(
            shlex.quote(str(ros_setup)),
            " ".join(cmd),
        )
    else:
        shell_script = " ".join(cmd)
    output = subprocess.check_output(["bash", "-lc", shell_script], text=True)
    payload = json.loads(str(output or "").strip() or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("bag inspector returned non-dict payload")
    return payload


def _build_clip_plan(bag_info, stride_sec, max_clips_per_bag):
    clip_duration = float(clip_builder.EXPORT_DURATION_SEC)
    max_start = float(bag_info.get("duration_sec", 0.0) or 0.0) - clip_duration
    if max_start < -1e-9:
        return []

    clip_starts = []
    clip_idx = 0
    start_sec = 0.0
    while start_sec <= max_start + 1e-9:
        clip_starts.append(
            {
                "clip_index": int(clip_idx),
                "start_sec": float(round(start_sec, 6)),
                "duration_sec": float(clip_duration),
            }
        )
        clip_idx += 1
        if int(max_clips_per_bag or 0) > 0 and len(clip_starts) >= int(max_clips_per_bag):
            break
        start_sec += float(stride_sec)
    return clip_starts


def _make_encode_args(runtime, target, export_root, clip_plan):
    from types import SimpleNamespace

    cache_root = (Path(export_root).resolve() / "cache" / target["dataset"] / target["sequence"]).resolve()
    return SimpleNamespace(
        mode="encode",
        exphub=str(runtime.exphub_root),
        datasets_cfg=str(runtime.cfg_path),
        dataset=str(target["dataset"]),
        sequence=str(target["sequence"]),
        tag=str(runtime.args.tag),
        w=int(clip_builder.EXPORT_WIDTH),
        h=int(clip_builder.EXPORT_HEIGHT),
        fps=float(clip_builder.EXPORT_FPS),
        dur=str(int(round(float(clip_builder.EXPORT_DURATION_SEC)))),
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
        export_stride_sec=float(runtime.args.export_stride_sec),
        export_max_bags=int(runtime.args.export_max_bags),
        export_max_bags_per_dataset=int(runtime.args.export_max_bags_per_dataset),
        export_max_clips_per_bag=int(runtime.args.export_max_clips_per_bag),
        export_split_seed=int(runtime.args.export_split_seed),
    )


def _run_encode_for_clip(runtime, target, export_root, clip_plan):
    from exphub.pipeline.orchestrator import build_runtime

    encode_args = _make_encode_args(runtime, target, export_root, clip_plan)
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


def run(runtime):
    args = runtime.args
    cfg = _load_cfg(runtime)
    export_root = _export_root(runtime)
    if export_root.exists():
        remove_path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    layout = dataset_writer.ensure_layout(export_root)
    targets = _resolve_targets(runtime, cfg)
    if not targets:
        raise RuntimeError("export resolved no dataset targets")

    log_info(
        "export start: scope={} focus={} targets={} spec={}fps/{}f/{}x{}".format(
            str(args.export_scope),
            str(args.export_focus),
            int(len(targets)),
            int(clip_builder.EXPORT_FPS),
            int(clip_builder.EXPORT_NUM_FRAMES),
            int(clip_builder.EXPORT_WIDTH),
            int(clip_builder.EXPORT_HEIGHT),
        )
    )

    exported_clips = []
    skipped_clips = []
    entries_by_split = {"train": [], "val": [], "test": []}
    total_t0 = time.time()

    for target in targets:
        bag_info = _inspect_bag_topic(runtime, target["bag"], target["topic"])
        clip_plan_items = _build_clip_plan(
            bag_info=bag_info,
            stride_sec=float(args.export_stride_sec or clip_builder.EXPORT_DURATION_SEC),
            max_clips_per_bag=int(args.export_max_clips_per_bag or 0),
        )
        if not clip_plan_items:
            skipped_clips.append(
                {
                    "dataset": str(target["dataset"]),
                    "sequence": str(target["sequence"]),
                    "reason": "bag_too_short_for_fixed_spec",
                    "bag_duration_sec": float(bag_info.get("duration_sec", 0.0) or 0.0),
                }
            )
            continue

        log_info(
            "export bag: dataset={} sequence={} clips={} duration={:.2f}s".format(
                str(target["dataset"]),
                str(target["sequence"]),
                int(len(clip_plan_items)),
                float(bag_info.get("duration_sec", 0.0) or 0.0),
            )
        )
        for clip_plan in clip_plan_items:
            clip_key = "{}:{}:{:.6f}".format(target["dataset"], target["sequence"], float(clip_plan["start_sec"]))
            try:
                exp_dir = _run_encode_for_clip(runtime, target, export_root, clip_plan)
                segment_manifest_path = (exp_dir / "segment" / "segment_manifest.json").resolve()
                prompt_manifest_path = (exp_dir / "prompt" / "prompt_manifest.json").resolve()
                segment_manifest = read_json_dict(segment_manifest_path)
                prompt_manifest = read_json_dict(prompt_manifest_path)
                candidate = clip_builder.select_training_candidate(segment_manifest, prompt_manifest, exp_dir=exp_dir)
                if not bool(candidate.get("accepted")):
                    skipped_clips.append(
                        {
                            "dataset": str(target["dataset"]),
                            "sequence": str(target["sequence"]),
                            "clip_index": int(clip_plan["clip_index"]),
                            "start_sec": float(clip_plan["start_sec"]),
                            "reason": str(candidate.get("reason", "candidate_rejected")),
                            "exp_dir": str(exp_dir),
                        }
                    )
                    continue

                split = dataset_writer.assign_split(clip_key, seed=int(args.export_split_seed))
                clip_filename = clip_builder.build_clip_filename(
                    target["dataset"],
                    target["sequence"],
                    clip_plan["clip_index"],
                    clip_plan["start_sec"],
                )
                clip_output_path = (layout["split_dirs"][split] / clip_filename).resolve()
                clip_builder.write_training_clip(
                    exp_dir / "segment" / "frames",
                    clip_output_path,
                    start_idx=int(candidate["clip_start_idx"]),
                    num_frames=int(candidate["clip_num_frames"]),
                    fps=clip_builder.EXPORT_FPS,
                )
                clip_manifest_path = dataset_writer.write_clip_manifest(
                    export_root,
                    Path(clip_filename).stem,
                    candidate.get("clip_manifest") or {},
                )
                aligned_segment_plan_path = str(
                    ((candidate.get("clip_manifest") or {}).get("source_files") or {}).get("aligned_segment_plan", "")
                    or ""
                )

                metadata_entry = _metadata_entry(export_root, split, clip_output_path, candidate["caption"])
                entries_by_split[split].append(metadata_entry)
                exported_clips.append(
                    {
                        "dataset": str(target["dataset"]),
                        "sequence": str(target["sequence"]),
                        "clip_index": int(clip_plan["clip_index"]),
                        "start_sec": float(clip_plan["start_sec"]),
                        "split": str(split),
                        "file_path": metadata_entry["file_path"],
                        "text": metadata_entry["text"],
                        "type": "video",
                        "exp_dir": str(exp_dir),
                        "segment_manifest": str(segment_manifest_path),
                        "prompt_manifest": str(prompt_manifest_path),
                        "aligned_segment_plan": str(aligned_segment_plan_path),
                        "train_clip_manifest": dataset_writer.relative_to_root(export_root, clip_manifest_path),
                        "clip_start_idx": int(candidate["clip_start_idx"]),
                        "clip_end_idx": int(candidate["clip_end_idx"]),
                        "clip_num_frames": int(candidate["clip_num_frames"]),
                        "selection_reason": str(candidate.get("selection_reason", "") or ""),
                        "candidate_summary": dict(candidate.get("summary") or {}),
                    }
                )
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
        training_spec=clip_builder.training_spec(),
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
    parser.add_argument("--export_stride_sec", type=float, default=clip_builder.EXPORT_DURATION_SEC)
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
