from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from exphub.common.io import list_frames_sorted, remove_path, write_json_atomic, write_text_atomic
from exphub.meta import sanitize_token


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _positive_int(value, name):
    out = int(value)
    if out <= 0:
        raise RuntimeError("{} must be > 0, got {}".format(name, value))
    return out


def _resolution(prepare_result):
    item = _as_dict(prepare_result.get("normalized_resolution"))
    width = int(item.get("width", 0) or 0)
    height = int(item.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("prepare_result.normalized_resolution is invalid: {}".format(item))
    return width, height


def _prompt_by_unit_id(prompts):
    out = {}
    for item in list(_as_dict(prompts).get("units") or []):
        prompt_item = _as_dict(item)
        unit_id = str(prompt_item.get("unit_id", "") or "")
        if unit_id:
            out[unit_id] = prompt_item
    return out


def _parse_rate(value):
    text = str(value or "").strip()
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(text)


def _int_or_none(value):
    try:
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        return int(float(text))
    except Exception:
        return None


def _probe_video(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found; cannot validate mp4: {}".format(path))
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(Path(path).resolve()),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed for {}: {}".format(path, proc.stdout.strip()))
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        raise RuntimeError("ffprobe returned invalid json for {}: {}".format(path, proc.stdout.strip())) from exc
    streams = list(payload.get("streams") or [])
    if not streams:
        raise RuntimeError("ffprobe found no video stream: {}".format(path))
    stream = _as_dict(streams[0])
    frames = _int_or_none(stream.get("nb_read_frames"))
    if frames is None:
        frames = _int_or_none(stream.get("nb_frames"))
    if frames is None:
        rate = _parse_rate(stream.get("r_frame_rate"))
        duration = float(stream.get("duration") or 0.0)
        frames = int(round(duration * rate)) if rate > 0.0 and duration > 0.0 else None
    return {
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "frames": frames,
    }


def _quote_concat_path(path):
    return "file '{}'\n".format(str(Path(path).resolve()).replace("'", "'\\''"))


class TrainExportSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.target_num_frames = _positive_int(
            getattr(runtime.args, "train_clip_num_frames", 73),
            "train_clip_num_frames",
        )
        self.window_stride = _positive_int(
            getattr(runtime.args, "train_clip_stride", 36),
            "train_clip_stride",
        )
        self.records = []
        self.clip_index = 0
        self.skipped_short_unit_count = 0
        self.motion_label_histogram = {}
        self.unit_lengths = []
        self.fps = int(float(runtime.spec.fps))
        self.resolution = [0, 0]
        self._tmp_dir = runtime.paths.trainset_dir / ".tmp"

    def prepare_output_dirs(self):
        remove_path(self.runtime.paths.trainset_dir)
        self.runtime.paths.trainset_videos_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def _next_clip_id(self, sequence):
        clip_id = "{}_{}_{:06d}".format(
            sanitize_token(self.runtime.spec.dataset),
            sanitize_token(str(sequence)),
            int(self.clip_index),
        )
        self.clip_index += 1
        return clip_id

    def _validate_window(self, source_frames, clip_start, clip_end):
        if clip_start < 0 or clip_end < clip_start or clip_end >= len(source_frames):
            raise RuntimeError(
                "clip frame range outside prepared frames: start_idx={} end_idx={} frame_count={}".format(
                    int(clip_start),
                    int(clip_end),
                    len(source_frames),
                )
            )
        for source_idx in range(int(clip_start), int(clip_end) + 1):
            source_path = Path(source_frames[source_idx]).resolve()
            if not source_path.is_file():
                raise RuntimeError("missing source prepared frame: {}".format(source_path))

    def _encode_window(self, source_frames, clip_start, clip_end, out_mp4, fps, width, height):
        self._validate_window(source_frames, clip_start, clip_end)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found; cannot encode train clip")

        list_path = self._tmp_dir / "{}.txt".format(Path(out_mp4).stem)
        lines = [_quote_concat_path(source_frames[idx]) for idx in range(int(clip_start), int(clip_end) + 1)]
        write_text_atomic(list_path, "".join(lines))

        tmp_mp4 = Path(out_mp4).with_name(Path(out_mp4).name + ".tmp.mp4")
        remove_path(tmp_mp4)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(int(fps)),
            "-i",
            str(list_path.resolve()),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale={}:{}".format(int(width), int(height)),
            "-movflags",
            "+faststart",
            str(tmp_mp4.resolve()),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
        remove_path(list_path)
        if proc.returncode != 0:
            remove_path(tmp_mp4)
            raise RuntimeError("ffmpeg failed for {}: {}".format(out_mp4, proc.stdout.strip()))
        if not tmp_mp4.is_file():
            raise RuntimeError("ffmpeg did not produce output: {}".format(tmp_mp4))
        tmp_mp4.replace(Path(out_mp4).resolve())

        probed = _probe_video(out_mp4)
        expected_frames = int(clip_end) - int(clip_start) + 1
        if int(probed["width"]) != int(width) or int(probed["height"]) != int(height):
            raise RuntimeError(
                "encoded mp4 resolution mismatch: path={} expected={}x{} got={}x{}".format(
                    out_mp4,
                    int(width),
                    int(height),
                    int(probed["width"]),
                    int(probed["height"]),
                )
            )
        if probed["frames"] is not None and int(probed["frames"]) != int(expected_frames):
            raise RuntimeError(
                "encoded mp4 frame count mismatch: path={} expected={} got={}".format(
                    out_mp4,
                    int(expected_frames),
                    int(probed["frames"]),
                )
            )

    def export_sequence(
        self,
        sequence,
        prepare_result,
        prepare_frames_dir,
        generation_units,
        prompts,
        generation_units_path,
        prompts_path,
    ):
        units = list(_as_dict(generation_units).get("units") or [])
        prompt_units = list(_as_dict(prompts).get("units") or [])
        if len(units) != len(prompt_units):
            raise RuntimeError(
                "generation unit count does not match prompt count: sequence={} units={} prompts={}".format(
                    sequence,
                    len(units),
                    len(prompt_units),
                )
            )

        prompt_map = _prompt_by_unit_id(prompts)
        source_frames = list_frames_sorted(prepare_frames_dir)
        width, height = _resolution(prepare_result)
        fps = int(prepare_result.get("target_fps", self.runtime.spec.fps) or self.runtime.spec.fps)
        self.fps = int(fps)
        if self.resolution == [0, 0]:
            self.resolution = [int(width), int(height)]
        elif self.resolution != [int(width), int(height)]:
            raise RuntimeError(
                "train sequence resolution mismatch: sequence={} expected={} got={}".format(
                    sequence,
                    self.resolution,
                    [int(width), int(height)],
                )
            )

        for local_idx, raw_unit in enumerate(units):
            unit = _as_dict(raw_unit)
            unit_id = str(unit.get("unit_id", "") or "")
            if not unit_id:
                raise RuntimeError("generation unit missing unit_id: sequence={} index={}".format(sequence, local_idx))
            prompt_item = prompt_map.get(unit_id)
            if not prompt_item:
                raise RuntimeError("prompt missing for generation unit: sequence={} unit_id={}".format(sequence, unit_id))
            prompt = str(prompt_item.get("prompt_positive", "") or "")
            if not prompt:
                raise RuntimeError("prompt_positive is empty: sequence={} unit_id={}".format(sequence, unit_id))

            start_idx = int(unit.get("start_idx"))
            end_idx = int(unit.get("end_idx"))
            expected_length = int(end_idx - start_idx + 1)
            unit_length = int(unit.get("length", expected_length) or expected_length)
            if unit_length != expected_length:
                raise RuntimeError(
                    "generation unit length mismatch: sequence={} unit_id={} length={} range_length={}".format(
                        sequence,
                        unit_id,
                        unit_length,
                        expected_length,
                    )
                )
            self.unit_lengths.append(int(unit_length))
            if unit_length < self.target_num_frames:
                self.skipped_short_unit_count += 1
                continue

            motion_label = str(unit.get("motion_label", prompt_item.get("motion_label", "mixed")) or "mixed")
            clip_start = int(start_idx)
            while clip_start + int(self.target_num_frames) - 1 <= int(end_idx):
                clip_end = int(clip_start) + int(self.target_num_frames) - 1
                clip_id = self._next_clip_id(sequence)
                rel_video = Path("videos") / "{}.mp4".format(clip_id)
                out_mp4 = self.runtime.paths.trainset_dir / rel_video
                self._encode_window(source_frames, clip_start, clip_end, out_mp4, fps, width, height)
                self.records.append(
                    {
                        "clip_id": str(clip_id),
                        "file_path": str(rel_video.as_posix()),
                        "text": str(prompt),
                        "type": "video",
                        "dataset": str(self.runtime.spec.dataset),
                        "sequence": str(sequence),
                        "unit_id": str(unit_id),
                        "seg_id": str(unit.get("seg_id", "") or ""),
                        "start_idx": int(clip_start),
                        "end_idx": int(clip_end),
                        "num_frames": int(self.target_num_frames),
                        "width": int(width),
                        "height": int(height),
                        "fps": int(fps),
                        "motion_label": str(motion_label),
                        "source_generation_units": str(Path(generation_units_path).resolve()),
                        "source_prompts": str(Path(prompts_path).resolve()),
                    }
                )
                self.motion_label_histogram[motion_label] = int(self.motion_label_histogram.get(motion_label, 0)) + 1
                clip_start += int(self.window_stride)

    def write_indexes(self, sequence_count):
        metadata = [
            {
                "file_path": str(item["file_path"]),
                "text": str(item["text"]),
                "type": "video",
            }
            for item in self.records
        ]
        unit_length_values = [int(item) for item in self.unit_lengths]
        avg_unit_length = (
            float(sum(unit_length_values)) / float(len(unit_length_values)) if unit_length_values else 0.0
        )
        stats = {
            "dataset": str(self.runtime.spec.dataset),
            "sequence_count": int(sequence_count),
            "clip_count": int(len(self.records)),
            "total_frames": int(len(self.records) * int(self.target_num_frames)),
            "fps": int(self.fps),
            "resolution": [int(self.resolution[0]), int(self.resolution[1])] if self.resolution else [0, 0],
            "target_num_frames": int(self.target_num_frames),
            "window_stride": int(self.window_stride),
            "min_unit_frames": int(self.target_num_frames),
            "skipped_short_unit_count": int(self.skipped_short_unit_count),
            "motion_label_histogram": dict(self.motion_label_histogram),
            "avg_unit_length": float(avg_unit_length),
            "train_metadata_path": "trainset/train_metadata.json",
        }
        write_json_atomic(self.runtime.paths.trainset_metadata_path, metadata, indent=2)
        write_json_atomic(self.runtime.paths.trainset_stats_path, stats, indent=2)
        remove_path(self._tmp_dir)
        return stats
