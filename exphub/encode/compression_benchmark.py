from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from exphub.common.io import ensure_dir, frame_sort_key, list_frames_sorted
from exphub.common.logging import log_info
from exphub.encode.payload_writer import write_hvm_payload_zip


def _safe_token(value):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return token or "bitrate"


class CompressionBenchmark:
    def __init__(
        self,
        frames_dir,
        output_dir,
        fps,
        bitrate="10M",
        hvm_payload_dir=None,
        hvm_algorithmic_time=0.0,
        ffmpeg_bin=None,
    ):
        self.frames_dir = Path(frames_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.fps = self._validate_fps(fps)
        self.bitrate = self._validate_bitrate(bitrate)
        self.hvm_payload_dir = Path(hvm_payload_dir).resolve() if hvm_payload_dir is not None else None
        self.hvm_algorithmic_time = self._validate_nonnegative_float(
            hvm_algorithmic_time,
            "hvm_algorithmic_time",
        )
        self.ffmpeg_bin = str(ffmpeg_bin or shutil.which("ffmpeg") or "")

    @staticmethod
    def _validate_fps(value):
        try:
            fps = int(value)
        except Exception as exc:
            raise RuntimeError("compression benchmark fps must be an integer, got {!r}".format(value)) from exc
        if fps <= 0:
            raise RuntimeError("compression benchmark fps must be > 0, got {}".format(fps))
        return fps

    @staticmethod
    def _validate_bitrate(value):
        bitrate = str(value or "").strip()
        if not bitrate:
            raise RuntimeError("compression benchmark video bitrate must be non-empty")
        return bitrate

    @staticmethod
    def _validate_nonnegative_float(value, label):
        try:
            parsed = float(value)
        except Exception as exc:
            raise RuntimeError("{} must be a non-negative number, got {!r}".format(label, value)) from exc
        if parsed < 0.0:
            raise RuntimeError("{} must be >= 0, got {}".format(label, parsed))
        return float(parsed)

    @property
    def raw_zip_path(self):
        return self.output_dir / "raw_images.zip"

    @property
    def h265_path(self):
        return self.output_dir / "h265_video_{}.mp4".format(_safe_token(self.bitrate))

    @property
    def hvm_payload_zip_path(self):
        return self.output_dir / "hvm_payload.zip"

    def collect_frames(self):
        frame_dir = ensure_dir(self.frames_dir, "prepare frames dir")
        frames = [Path(item).resolve() for item in list_frames_sorted(frame_dir)]
        if not frames:
            raise RuntimeError("compression benchmark requires at least one prepared frame: {}".format(frame_dir))
        return frames

    def _write_raw_zip(self, frames):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seen_names = set()
        tmp_path = self.raw_zip_path.with_name(self.raw_zip_path.name + ".tmp")
        with zipfile.ZipFile(str(tmp_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for frame in list(frames):
                path = Path(frame).resolve()
                name = path.name
                if name in seen_names:
                    raise RuntimeError("duplicate frame basename for raw zip: {}".format(name))
                seen_names.add(name)
                zf.write(str(path), name)
        tmp_path.replace(self.raw_zip_path)
        return self.raw_zip_path

    def _write_hvm_payload_zip(self):
        if self.hvm_payload_dir is None:
            raise RuntimeError("compression benchmark requires hvm_payload_dir")
        return write_hvm_payload_zip(self.hvm_payload_dir, self.hvm_payload_zip_path)

    @staticmethod
    def _concat_line(path):
        resolved = Path(path).resolve()
        text = str(resolved)
        if not resolved.is_absolute():
            raise RuntimeError("ffmpeg concat path must be absolute: {}".format(text))
        if "\n" in text or "\r" in text or "'" in text:
            raise RuntimeError("ffmpeg concat path cannot contain newline or single quote: {}".format(text))
        return "file '{}'".format(text)

    def _write_concat_list(self, frames, concat_path):
        path = Path(concat_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered_frames = [Path(frame).resolve() for frame in list(frames)]
        sorted_frames = sorted(ordered_frames, key=frame_sort_key)
        if ordered_frames != sorted_frames:
            raise RuntimeError("ffmpeg concat frames must be sorted chronologically")
        lines = [self._concat_line(frame) for frame in ordered_frames]
        if not lines:
            raise RuntimeError("ffmpeg concat list cannot be empty")
        text = "\n".join(lines) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def _build_ffmpeg_cmd(self, concat_path):
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for compression benchmark")
        return [
            self.ffmpeg_bin,
            "-y",
            "-r",
            str(int(self.fps)),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(Path(concat_path).resolve()),
            "-c:v",
            "libx265",
            "-b:v",
            str(self.bitrate),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(self.h265_path),
        ]

    def _run_ffmpeg(self, concat_path):
        cmd = self._build_ffmpeg_cmd(concat_path)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            output = str(proc.stdout or "")
            tail = "\n".join(output.splitlines()[-12:])
            raise RuntimeError(
                "ffmpeg H.265 compression benchmark failed rc={} cmd={} tail={}".format(
                    int(proc.returncode),
                    " ".join(cmd),
                    tail,
                )
            )
        if not self.h265_path.is_file() or self.h265_path.stat().st_size <= 0:
            raise RuntimeError("ffmpeg did not create H.265 benchmark video: {}".format(self.h265_path))
        return cmd

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.collect_frames()

        started = time.perf_counter()
        self._write_raw_zip(frames)
        raw_zip_sec = float(time.perf_counter() - started)

        started = time.perf_counter()
        hvm_payload_zip = self._write_hvm_payload_zip()
        hvm_zip_sec = float(time.perf_counter() - started)
        # HVM Total = formal encode/payload algorithm time + payload zip I/O time.
        hvm_total_sec = float(self.hvm_algorithmic_time) + float(hvm_zip_sec)

        concat_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.output_dir),
                prefix="ffmpeg_concat_",
                suffix=".txt",
                delete=False,
            ) as handle:
                concat_file = Path(handle.name).resolve()
            self._write_concat_list(frames, concat_file)
            started = time.perf_counter()
            command = self._run_ffmpeg(concat_file)
            h265_sec = float(time.perf_counter() - started)
        finally:
            if concat_file is not None:
                try:
                    Path(concat_file).unlink()
                except FileNotFoundError:
                    pass

        log_info(
            "[Compression Benchmark] Time Stats -> Raw ZIP: {:.2f}s | H.265 MP4: {:.2f}s | HVM Total: {:.2f}s".format(
                float(raw_zip_sec),
                float(h265_sec),
                float(hvm_total_sec),
            )
        )

        return {
            "output_dir": self.output_dir,
            "raw_zip": self.raw_zip_path,
            "h265_video": self.h265_path,
            "hvm_payload_zip": hvm_payload_zip,
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "bitrate": str(self.bitrate),
            "raw_zip_sec": float(raw_zip_sec),
            "h265_sec": float(h265_sec),
            "hvm_algorithmic_sec": float(self.hvm_algorithmic_time),
            "hvm_zip_sec": float(hvm_zip_sec),
            "hvm_total_sec": float(hvm_total_sec),
            "ffmpeg_command": list(command),
        }
