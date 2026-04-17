from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from exphub.common.io import write_json_atomic
from exphub.prepare.geometry_adapter import adapt_frames_and_intrinsics
from exphub.prepare.legal_grid import build_legal_grid
from exphub.prepare.ros_reader import (
    load_dataset_config,
    maybe_write_frames,
    read_ros_frames,
    sample_frames_to_target_fps,
)


@dataclass(frozen=True)
class PrepareResult:
    mode: str
    dataset: str
    sequence: str
    bag_path: str
    topic: str
    frame_dir: str
    target_fps: int
    num_frames: int
    time_range: Dict[str, Optional[float]]
    original_resolution: Dict[str, int]
    normalized_resolution: Dict[str, int]
    original_intrinsics: Dict[str, object]
    normalized_intrinsics: Dict[str, object]
    transform_meta: Dict[str, object]
    legal_grid: Dict[str, object]
    frame_index_map: Dict[str, List[Union[int, float]]]

    def to_dict(self):
        return asdict(self)


def _repo_root_from_config(config_path):
    path = Path(config_path).resolve()
    if path.parent.name == "config":
        return path.parent.parent.resolve()
    return path.parent.resolve()


def _default_output_root(config_path):
    return (_repo_root_from_config(config_path) / "artifacts" / "prepare").resolve()


def _clean_token(value):
    text = str(value or "").strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._") or "prepare"


def _make_run_id(mode, run_id=None):
    if run_id:
        return _clean_token(run_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{}_prepare_{}".format(_clean_token(mode), stamp)


def _prepare_run_dir(output_root, dataset, sequence, run_id):
    root = Path(output_root).resolve()
    return (root / _clean_token(dataset) / _clean_token(sequence) / _clean_token(run_id)).resolve()


def _reset_frame_dir(frame_dir):
    path = Path(frame_dir).resolve()
    if path.is_dir():
        shutil.rmtree(str(path), ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def save_prepare_result(result, output_path=None):
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    path = Path(output_path) if output_path is not None else Path(payload["frame_dir"]).resolve().parent / "prepare_result.json"
    write_json_atomic(path, payload, indent=2)
    return path.resolve()


def _run_single_prepare(
    mode,
    config_path,
    dataset_name,
    sequence_name,
    target_fps,
    start_sec=None,
    dur_sec=None,
    run_id=None,
    output_root=None,
    output_dir=None,
):
    cfg = load_dataset_config(config_path, dataset_name, sequence_name)
    fps = int(target_fps)
    if fps <= 0:
        raise RuntimeError("target_fps must be > 0, got {}".format(target_fps))

    if output_dir is not None:
        run_dir = Path(output_dir).resolve()
    else:
        output_root_path = Path(output_root).resolve() if output_root else _default_output_root(config_path)
        run_dir = _prepare_run_dir(output_root_path, cfg.dataset, cfg.sequence, _make_run_id(mode, run_id))
    frame_dir = (run_dir / "frames").resolve()
    result_path = (run_dir / "prepare_result.json").resolve()

    print(
        "[INFO] prepare start: mode={} dataset={} sequence={} fps={} bag={}".format(
            mode,
            cfg.dataset,
            cfg.sequence,
            fps,
            cfg.bag_path,
        )
    )
    ros_frames = read_ros_frames(cfg, start_sec=start_sec, dur_sec=dur_sec)
    sampled = sample_frames_to_target_fps(ros_frames, target_fps=fps, start_sec=start_sec, dur_sec=dur_sec)
    geometry = adapt_frames_and_intrinsics(
        sampled.frames,
        cfg.intrinsics,
        image_size=cfg.image_size,
        multiple=32,
    )

    _reset_frame_dir(frame_dir)
    maybe_write_frames(geometry.frames, frame_dir)

    legal_grid = build_legal_grid(num_frames=len(geometry.frames), fps=fps)
    result = PrepareResult(
        mode=str(mode),
        dataset=str(cfg.dataset),
        sequence=str(cfg.sequence),
        bag_path=str(cfg.bag_path),
        topic=str(cfg.topic),
        frame_dir=str(frame_dir),
        target_fps=int(fps),
        num_frames=int(len(geometry.frames)),
        time_range={
            "start_sec": float(sampled.start_sec) if sampled.start_sec is not None else None,
            "end_sec": float(sampled.end_sec) if sampled.end_sec is not None else None,
            "dur_sec": float(sampled.dur_sec) if sampled.dur_sec is not None else None,
        },
        original_resolution=geometry.original_resolution,
        normalized_resolution=geometry.normalized_resolution,
        original_intrinsics=geometry.original_intrinsics,
        normalized_intrinsics=geometry.normalized_intrinsics,
        transform_meta=geometry.transform_meta,
        legal_grid=legal_grid,
        frame_index_map={
            "prepared_to_source": [int(item) for item in sampled.prepared_to_source],
            "prepared_to_time_sec": [float(item) for item in sampled.prepared_to_time_sec],
            "prepared_to_rel_time_sec": [float(item) for item in sampled.prepared_to_time_sec],
            "prepared_to_abs_time_sec": [float(item) for item in sampled.prepared_to_abs_time_sec],
        },
    )
    save_prepare_result(result, result_path)
    print(
        "[INFO] prepare done: mode={} frames={} resolution={}x{} result={}".format(
            mode,
            result.num_frames,
            result.normalized_resolution["width"],
            result.normalized_resolution["height"],
            result_path,
        )
    )
    return result


def infer_prepare(
    config_path,
    dataset_name,
    sequence_name,
    target_fps,
    start_sec=None,
    dur_sec=None,
    run_id=None,
    output_root=None,
    output_dir=None,
):
    return _run_single_prepare(
        mode="infer",
        config_path=config_path,
        dataset_name=dataset_name,
        sequence_name=sequence_name,
        target_fps=target_fps,
        start_sec=start_sec,
        dur_sec=dur_sec,
        run_id=run_id,
        output_root=output_root,
        output_dir=output_dir,
    )


def _all_sequences(config_path, dataset_name):
    path = Path(config_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") or {}
    dataset_cfg = datasets.get(dataset_name) or {}
    sequences = dataset_cfg.get("sequences") or {}
    return list(sequences.keys())


def train_prepare(
    config_path,
    dataset_name,
    sequence_name=None,
    target_fps=24,
    run_id=None,
    output_root=None,
    output_dir=None,
):
    if sequence_name:
        return _run_single_prepare(
            mode="train",
            config_path=config_path,
            dataset_name=dataset_name,
            sequence_name=sequence_name,
            target_fps=target_fps,
            start_sec=None,
            dur_sec=None,
            run_id=run_id,
            output_root=output_root,
            output_dir=output_dir,
        )

    results = []
    for item_sequence in _all_sequences(config_path, dataset_name):
        results.append(
            _run_single_prepare(
                mode="train",
                config_path=config_path,
                dataset_name=dataset_name,
                sequence_name=item_sequence,
                target_fps=target_fps,
                start_sec=None,
                dur_sec=None,
                run_id=run_id,
                output_root=output_root,
                output_dir=output_dir,
            )
        )
    if not results:
        raise RuntimeError("no sequences found for train_prepare dataset={}".format(dataset_name))
    return results


def run_prepare(
    mode,
    config_path,
    dataset_name,
    sequence_name=None,
    target_fps=24,
    start_sec=None,
    dur_sec=None,
    run_id=None,
    output_root=None,
    output_dir=None,
):
    mode_value = str(mode or "").strip().lower()
    if mode_value == "infer":
        if not sequence_name:
            raise RuntimeError("infer_prepare requires --sequence")
        return infer_prepare(
            config_path=config_path,
            dataset_name=dataset_name,
            sequence_name=sequence_name,
            target_fps=target_fps,
            start_sec=start_sec,
            dur_sec=dur_sec,
            run_id=run_id,
            output_root=output_root,
            output_dir=output_dir,
        )
    if mode_value == "train":
        return train_prepare(
            config_path=config_path,
            dataset_name=dataset_name,
            sequence_name=sequence_name,
            target_fps=target_fps,
            run_id=run_id,
            output_root=output_root,
            output_dir=output_dir,
        )
    raise RuntimeError("unsupported prepare mode: {}".format(mode))


def run(runtime):
    if str(runtime.args.mode).strip().lower() != "infer":
        raise RuntimeError("prepare runner currently supports infer mode only")
    runtime.ensure_clean_exp_dir()
    runtime.paths.prepare_dir.mkdir(parents=True, exist_ok=True)
    result = infer_prepare(
        config_path=runtime.cfg_path,
        dataset_name=runtime.spec.dataset,
        sequence_name=runtime.spec.sequence,
        target_fps=int(float(runtime.spec.fps)),
        start_sec=float(runtime.spec.start),
        dur_sec=float(runtime.spec.dur),
        run_id=runtime.spec.exp_name,
        output_dir=runtime.paths.prepare_dir,
    )
    runtime._prepare_result_cache = result.to_dict()
    runtime.write_meta_snapshot()
    return runtime.paths.prepare_dir


def _build_parser():
    parser = argparse.ArgumentParser(description="ExpHub prepare foundation entrypoint")
    parser.add_argument("--mode", choices=["infer", "train"], required=True)
    parser.add_argument("--config", required=True, help="Path to datasets.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--dur-sec", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "infer" and not args.sequence:
        parser.error("--sequence is required for --mode infer")
    result = run_prepare(
        mode=args.mode,
        config_path=args.config,
        dataset_name=args.dataset,
        sequence_name=args.sequence,
        target_fps=args.fps,
        start_sec=args.start_sec,
        dur_sec=args.dur_sec,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    if isinstance(result, list):
        print("[INFO] prepare batch results={}".format(len(result)))
    return result


if __name__ == "__main__":
    main()
