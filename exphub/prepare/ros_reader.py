from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class DatasetPrepareConfig:
    dataset: str
    sequence: str
    format: str
    root: str
    bag_path: str
    topic: str
    image_size: List[int]
    intrinsics: Dict[str, object]


@dataclass(frozen=True)
class RosFrame:
    source_index: int
    time_sec: float
    image_bgr: Any


@dataclass(frozen=True)
class SampledFrames:
    frames: List[Any]
    prepared_to_source: List[int]
    prepared_to_time_sec: List[float]
    prepared_to_abs_time_sec: List[float]
    start_sec: float
    end_sec: float
    dur_sec: float


def _repo_root_from_config(config_path):
    path = Path(config_path).resolve()
    if path.parent.name == "config":
        return path.parent.parent.resolve()
    return path.parent.resolve()


def load_dataset_config(config_path, dataset_name, sequence_name):
    path = Path(config_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("failed to read datasets.json: {} ({})".format(path, exc))

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError("datasets.json missing top-level datasets object: {}".format(path))
    if dataset_name not in datasets:
        raise RuntimeError("dataset not found in datasets.json: dataset={}".format(dataset_name))

    dataset_cfg = datasets[dataset_name] or {}
    format_name = str(dataset_cfg.get("format", "") or "").strip().lower()
    if format_name != "rosbag":
        raise RuntimeError(
            "unsupported dataset format for prepare: dataset={} format={} (expected rosbag)".format(
                dataset_name,
                format_name or "<missing>",
            )
        )

    root_text = str(dataset_cfg.get("root", "") or "").strip()
    topic = str(dataset_cfg.get("topic", "") or "").strip()
    if not root_text:
        raise RuntimeError("datasets.{}.root is required".format(dataset_name))
    if not topic:
        raise RuntimeError("datasets.{}.topic is required".format(dataset_name))

    intrinsics_cfg = dataset_cfg.get("intrinsics") or {}
    image_size = intrinsics_cfg.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or int(image_size[0]) <= 0
        or int(image_size[1]) <= 0
    ):
        raise RuntimeError("datasets.{}.intrinsics.image_size must be [width, height]".format(dataset_name))

    try:
        intrinsics = {
            "fx": float(intrinsics_cfg["fx"]),
            "fy": float(intrinsics_cfg["fy"]),
            "cx": float(intrinsics_cfg["cx"]),
            "cy": float(intrinsics_cfg["cy"]),
            "dist": [float(item) for item in list(intrinsics_cfg.get("dist") or [])],
        }
    except Exception as exc:
        raise RuntimeError("datasets.{}.intrinsics has invalid fx/fy/cx/cy/dist".format(dataset_name)) from exc

    sequences = dataset_cfg.get("sequences") or {}
    if sequence_name not in sequences:
        raise RuntimeError(
            "sequence not found in datasets.json: dataset={} sequence={}".format(dataset_name, sequence_name)
        )
    sequence_cfg = sequences[sequence_name] or {}
    bag_name = str(sequence_cfg.get("bag", "") or "").strip()
    if not bag_name:
        raise RuntimeError("datasets.{}.sequences.{}.bag is required".format(dataset_name, sequence_name))

    root_path = Path(root_text)
    if not root_path.is_absolute():
        root_path = (_repo_root_from_config(path) / root_path).resolve()
    bag_path = (root_path / bag_name).resolve()

    return DatasetPrepareConfig(
        dataset=str(dataset_name),
        sequence=str(sequence_name),
        format=str(format_name),
        root=str(root_path),
        bag_path=str(bag_path),
        topic=str(topic),
        image_size=[int(image_size[0]), int(image_size[1])],
        intrinsics=intrinsics,
    )


def _require_rosbag_dependencies():
    try:
        import rosbag
    except Exception as exc:
        raise RuntimeError("failed to import rosbag while reading prepare frames: {}".format(exc))
    return rosbag


def _decode_compressed_image(msg):
    import cv2
    import numpy as np

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2.imdecode returned None for compressed ROS image")
    return image


def _decode_raw_image(msg):
    import cv2
    import numpy as np

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step) if hasattr(msg, "step") else width
    encoding = str(msg.encoding or "").lower()

    if encoding in ("bgr8", "rgb8", "bgra8", "rgba8"):
        channels = 4 if encoding in ("bgra8", "rgba8") else 3
        need = int(width) * int(channels)
        if int(step) < need:
            raise RuntimeError("step too small for encoding {}: step={} need={}".format(encoding, step, need))
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr[: int(step) * int(height)].reshape((int(height), int(step)))
        arr = arr[:, :need].reshape((int(height), int(width), int(channels)))
        if encoding == "bgr8":
            return arr.copy()
        if encoding == "rgb8":
            return arr[:, :, ::-1].copy()
        if encoding == "bgra8":
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    if encoding in ("mono8", "8uc1"):
        if int(step) < int(width):
            raise RuntimeError("step too small for encoding {}: step={} width={}".format(encoding, step, width))
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr[: int(step) * int(height)].reshape((int(height), int(step)))
        gray = arr[:, : int(width)]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in ("mono16", "16uc1"):
        need = int(width) * 2
        if int(step) < need:
            raise RuntimeError("step too small for encoding {}: step={} need={}".format(encoding, step, need))
        dtype = ">u2" if int(getattr(msg, "is_bigendian", 0) or 0) else "<u2"
        arr = np.frombuffer(msg.data, dtype=dtype)
        row_values = int(step) // 2
        arr = arr[: row_values * int(height)].reshape((int(height), int(row_values)))
        gray16 = arr[:, : int(width)]
        gray8 = np.right_shift(gray16, 8).astype(np.uint8)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    raise RuntimeError("unsupported sensor_msgs/Image encoding: {}".format(msg.encoding))


def _decode_ros_image_to_bgr(msg):
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "encoding"):
        return _decode_compressed_image(msg)
    if hasattr(msg, "encoding") and hasattr(msg, "data") and hasattr(msg, "height") and hasattr(msg, "width"):
        return _decode_raw_image(msg)
    raise RuntimeError("unknown ROS image message type: {}".format(getattr(msg, "_type", type(msg))))


def _topic_names(bag):
    try:
        return set((bag.get_type_and_topic_info().topics or {}).keys())
    except Exception:
        return set()


def read_ros_frames(dataset_config, start_sec=None, dur_sec=None):
    cfg = dataset_config
    bag_path = Path(cfg.bag_path).resolve()
    topic = str(cfg.topic)
    if not bag_path.is_file():
        raise RuntimeError(
            "bag not found for prepare: dataset={} sequence={} bag_path={}".format(
                cfg.dataset,
                cfg.sequence,
                bag_path,
            )
        )

    start_value = 0.0 if start_sec is None else float(start_sec)
    if start_value < 0.0:
        raise RuntimeError("start_sec must be >= 0 for prepare, got {}".format(start_sec))
    dur_value = None if dur_sec is None else float(dur_sec)
    if dur_value is not None and dur_value < 0.0:
        raise RuntimeError("dur_sec must be >= 0 for prepare, got {}".format(dur_sec))

    rosbag = _require_rosbag_dependencies()
    frames = []
    first_stamp = None
    source_index = -1
    end_value = None if dur_value is None else start_value + dur_value

    with rosbag.Bag(str(bag_path), "r") as bag:
        topics = _topic_names(bag)
        if topics and topic not in topics:
            raise RuntimeError(
                "topic not found in bag for prepare: dataset={} sequence={} topic={} bag_path={}".format(
                    cfg.dataset,
                    cfg.sequence,
                    topic,
                    bag_path,
                )
            )

        for _topic, msg, stamp in bag.read_messages(topics=[topic]):
            source_index += 1
            stamp_sec = float(stamp.to_sec())
            if first_stamp is None:
                first_stamp = stamp_sec
            rel_sec = float(stamp_sec - first_stamp)
            if rel_sec + 1e-9 < start_value:
                continue
            if end_value is not None and rel_sec > end_value + 1e-9:
                break
            image_bgr = _decode_ros_image_to_bgr(msg)
            frames.append(RosFrame(source_index=int(source_index), time_sec=float(rel_sec), image_bgr=image_bgr))

    if first_stamp is None:
        raise RuntimeError(
            "no messages found for prepare topic: dataset={} sequence={} topic={} bag_path={}".format(
                cfg.dataset,
                cfg.sequence,
                topic,
                bag_path,
            )
        )
    if not frames:
        raise RuntimeError(
            "no frames decoded for prepare window: dataset={} sequence={} topic={} bag_path={} start_sec={} dur_sec={}".format(
                cfg.dataset,
                cfg.sequence,
                topic,
                bag_path,
                start_sec,
                dur_sec,
            )
        )
    return frames


def sample_frames_to_target_fps(frames, target_fps, start_sec=None, dur_sec=None):
    frame_list = list(frames or [])
    fps = int(target_fps)
    if fps <= 0:
        raise RuntimeError("target_fps must be > 0, got {}".format(target_fps))
    if not frame_list:
        raise RuntimeError("cannot sample prepare frames: no source frames")

    times = [float(frame.time_sec) for frame in frame_list]
    source_start = 0.0 if start_sec is None else float(start_sec)
    if source_start < 0.0:
        raise RuntimeError("start_sec must be >= 0, got {}".format(start_sec))
    available_end = float(times[-1])
    requested_end = available_end if dur_sec is None else source_start + float(dur_sec)
    end_sec = float(requested_end)
    if end_sec + 1e-9 < source_start:
        raise RuntimeError(
            "prepare sampling window has no duration: start_sec={} end_sec={} available_end={}".format(
                source_start,
                end_sec,
                available_end,
            )
        )

    duration = max(0.0, float(end_sec - source_start))
    out_count = int(math.floor(duration * float(fps) + 1e-9)) + 1
    target_times = [source_start + float(idx) / float(fps) for idx in range(out_count)]

    sampled_frames = []
    prepared_to_source = []
    prepared_to_time_sec = []
    prepared_to_abs_time_sec = []
    for prepared_idx, target_time in enumerate(target_times):
        pos = bisect_left(times, float(target_time))
        if pos <= 0:
            chosen_index = 0
        elif pos >= len(times):
            chosen_index = len(times) - 1
        else:
            before = pos - 1
            after = pos
            if abs(float(target_time) - times[before]) <= abs(times[after] - float(target_time)):
                chosen_index = before
            else:
                chosen_index = after
        chosen_frame = frame_list[chosen_index]
        sampled_frames.append(chosen_frame.image_bgr)
        prepared_to_source.append(int(chosen_frame.source_index))
        prepared_to_time_sec.append(float(prepared_idx) / float(fps))
        prepared_to_abs_time_sec.append(float(source_start) + float(prepared_idx) / float(fps))

    actual_dur = float(prepared_to_time_sec[-1]) if prepared_to_time_sec else 0.0
    return SampledFrames(
        frames=sampled_frames,
        prepared_to_source=prepared_to_source,
        prepared_to_time_sec=prepared_to_time_sec,
        prepared_to_abs_time_sec=prepared_to_abs_time_sec,
        start_sec=float(source_start),
        end_sec=float(source_start + actual_dur),
        dur_sec=float(actual_dur),
    )


def maybe_write_frames(frames, frame_dir):
    import cv2

    frame_dir_path = Path(frame_dir).resolve()
    frame_dir_path.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for idx, frame in enumerate(list(frames or [])):
        out_path = frame_dir_path / "{:06d}.png".format(int(idx))
        ok = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("failed to write prepared frame: {}".format(out_path))
        written_paths.append(str(out_path))
    return written_paths
