from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

from exphub.common.compression_benchmark import (
    METHOD_ORDER,
    canonical_method_report,
    file_size,
    method_summary_lines,
    relative_path,
    safe_token,
    sum_file_sizes,
    bytes_to_mib,
)
from exphub.common.io import ensure_dir, frame_sort_key, list_frames_sorted, write_json_atomic
from exphub.common.logging import log_info, log_warn
from exphub.encode.dcvc_fm_adapter import DcvcFmAdapter
from exphub.encode.payload_writer import write_hvm_payload_zip


def _log_encode_summary(report):
    try:
        log_info("[Compression Benchmark: encode] Method Summary:")
        for line in method_summary_lines(report):
            log_info(line)
    except Exception as exc:
        log_warn("compression benchmark encode summary logging skipped: {}".format(exc))


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
        exphub_root=None,
        exp_dir=None,
    ):
        self.frames_dir = Path(frames_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.exp_dir = Path(exp_dir).resolve() if exp_dir is not None else self.output_dir.parent.parent.resolve()
        self.exphub_root = Path(exphub_root).resolve() if exphub_root is not None else Path.cwd().resolve()
        self.fps = self._validate_fps(fps)
        self.bitrate = self._validate_bitrate(bitrate)
        self.hvm_payload_dir = Path(hvm_payload_dir).resolve() if hvm_payload_dir is not None else None
        self.hvm_algorithmic_time = self._validate_nonnegative_float(
            hvm_algorithmic_time,
            "vlmem_algorithmic_time",
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
    def zip_dir(self):
        return self.output_dir / "zip"

    @property
    def raw_zip_path(self):
        return self.zip_dir / "raw_images.zip"

    @property
    def h265_dir(self):
        return self.output_dir / "h265"

    @property
    def h265_path(self):
        return self.h265_dir / "h265_video_{}.mp4".format(safe_token(self.bitrate))

    @property
    def dcvc_dir(self):
        return self.output_dir / "dcvc_fm_q21"

    @property
    def vlmem_dir(self):
        return self.output_dir / "vlmem"

    @property
    def hvm_payload_zip_path(self):
        return self.vlmem_dir / "hvm_payload.zip"

    @property
    def benchmark_report_path(self):
        return self.output_dir / "compression_benchmark_encode_report.json"

    def collect_frames(self):
        frame_dir = ensure_dir(self.frames_dir, "prepare frames dir")
        frames = [Path(item).resolve() for item in list_frames_sorted(frame_dir)]
        if not frames:
            raise RuntimeError("compression benchmark requires at least one prepared frame: {}".format(frame_dir))
        return frames

    def _write_raw_zip(self, frames):
        self.zip_dir.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError("compression benchmark requires VLMem payload dir")
        self.vlmem_dir.mkdir(parents=True, exist_ok=True)
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
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _build_ffmpeg_cmd(self, concat_path):
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for compression benchmark")
        self.h265_dir.mkdir(parents=True, exist_ok=True)
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

    def _write_report(self, frames, raw_frame_bytes, method_reports):
        report = {
            "version": 2,
            "stage": "encode",
            "source": "exphub.encode.compression_benchmark",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": relative_path(self.exp_dir, self.output_dir),
            "source_frames_dir": relative_path(self.exp_dir, self.frames_dir),
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "bitrate": str(self.bitrate),
            "raw_frame_bytes": int(raw_frame_bytes),
            "raw_frame_mib": bytes_to_mib(raw_frame_bytes),
            "reference_zip_path": relative_path(self.exp_dir, self.raw_zip_path),
            "reference_zip_bytes": file_size(self.raw_zip_path),
            "methods_order": list(METHOD_ORDER),
            "methods": {str(item.get("method_key")): dict(item) for item in list(method_reports or [])},
        }
        write_json_atomic(self.benchmark_report_path, report, indent=2)
        return report

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.collect_frames()
        raw_frame_bytes = sum_file_sizes(frames)

        started = time.perf_counter()
        self._write_raw_zip(frames)
        raw_zip_sec = float(time.perf_counter() - started)
        raw_zip_bytes = file_size(self.raw_zip_path)

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
            h265_command = self._run_ffmpeg(concat_file)
            h265_sec = float(time.perf_counter() - started)
        finally:
            if concat_file is not None:
                try:
                    Path(concat_file).unlink()
                except FileNotFoundError:
                    pass

        started = time.perf_counter()
        hvm_payload_zip = self._write_hvm_payload_zip()
        hvm_zip_sec = float(time.perf_counter() - started)
        hvm_wall_with_archive_sec = float(self.hvm_algorithmic_time) + float(hvm_zip_sec)
        hvm_payload_bytes = file_size(hvm_payload_zip)
        hvm_payload_frames = list_frames_sorted(self.hvm_payload_dir / "frames") if self.hvm_payload_dir is not None else []

        zip_report = canonical_method_report(
            exp_dir=self.exp_dir,
            method_key="zip",
            display_name="ZIP/ORI",
            status="ok",
            source_frames_dir=self.frames_dir,
            encoded_artifact_path=self.raw_zip_path,
            payload_bytes=raw_zip_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            enc_time_sec=raw_zip_sec,
            decode_time_sec=None,
            codec_wall_time_sec=raw_zip_sec,
            time_semantics="zip_archive_wall_time",
            frame_count=len(frames),
            fps=self.fps,
            extra={"transmitted_frame_count": int(len(frames))},
        )
        h265_report = canonical_method_report(
            exp_dir=self.exp_dir,
            method_key="h265",
            display_name="H.265",
            status="ok",
            source_frames_dir=self.frames_dir,
            encoded_artifact_path=self.h265_path,
            payload_bytes=file_size(self.h265_path),
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            enc_time_sec=h265_sec,
            decode_time_sec=None,
            codec_wall_time_sec=h265_sec,
            time_semantics="h265_video_encode_wall_time_only",
            frame_count=len(frames),
            fps=self.fps,
            command=h265_command,
            extra={"transmitted_frame_count": int(len(frames))},
        )
        dcvc_report = DcvcFmAdapter(
            exphub_root=self.exphub_root,
            frames_dir=self.frames_dir,
            output_dir=self.dcvc_dir,
            exp_dir=self.exp_dir,
            fps=self.fps,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            save_decoded_frame=False,
        ).run()
        vlmem_report = canonical_method_report(
            exp_dir=self.exp_dir,
            method_key="vlmem",
            display_name="VLMem/REC",
            status="ok",
            source_frames_dir=self.frames_dir,
            encoded_artifact_path=hvm_payload_zip,
            encoded_artifact_dir=self.vlmem_dir,
            payload_bytes=hvm_payload_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            enc_time_sec=self.hvm_algorithmic_time,
            decode_time_sec=None,
            codec_wall_time_sec=hvm_wall_with_archive_sec,
            time_semantics="vlmem_payload_construction_wall_time; payload archive time excluded from Table-II enc_time_sec",
            frame_count=len(frames),
            fps=self.fps,
            extra={
                "transmitted_frame_count": int(len(hvm_payload_frames)),
                "payload_archive_sec": float(hvm_zip_sec),
            },
        )
        report = self._write_report(
            frames=frames,
            raw_frame_bytes=raw_frame_bytes,
            method_reports=[zip_report, h265_report, dcvc_report, vlmem_report],
        )

        log_info(
            "[Compression Benchmark: encode] Time Stats -> ZIP: {:.2f}s | H.265 encode: {:.2f}s | VLMem payload: {:.2f}s".format(
                float(raw_zip_sec),
                float(h265_sec),
                float(self.hvm_algorithmic_time),
            )
        )
        _log_encode_summary(report)

        dcvc_required = bool(dict(dcvc_report.get("config") or {}).get("required", False))
        if dcvc_required and str(dcvc_report.get("status") or "") != "ok":
            raise RuntimeError("required DCVC-FM benchmark failed: {}".format(dcvc_report.get("error_message") or "unknown error"))

        return {
            "output_dir": self.output_dir,
            "benchmark_report": self.benchmark_report_path,
            "benchmark": report,
            "raw_zip": self.raw_zip_path,
            "h265_video": self.h265_path,
            "hvm_payload_zip": hvm_payload_zip,
            "vlmem_payload_zip": hvm_payload_zip,
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "bitrate": str(self.bitrate),
            "raw_zip_sec": float(raw_zip_sec),
            "h265_sec": float(h265_sec),
            "hvm_algorithmic_sec": float(self.hvm_algorithmic_time),
            "vlmem_algorithmic_sec": float(self.hvm_algorithmic_time),
            "hvm_zip_sec": float(hvm_zip_sec),
            "vlmem_zip_sec": float(hvm_zip_sec),
            "hvm_total_sec": float(hvm_wall_with_archive_sec),
            "vlmem_total_sec": float(hvm_wall_with_archive_sec),
        }
