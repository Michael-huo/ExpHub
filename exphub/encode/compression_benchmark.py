from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from exphub.common.compression_benchmark import (
    METHOD_ORDER,
    canonical_method_report,
    file_size,
    relative_path,
    safe_token,
    sum_file_sizes,
    bytes_to_mib,
)
from exphub.common.io import ensure_dir, frame_sort_key, list_frames_sorted, write_json_atomic
from exphub.common.logging import log_warn
from exphub.encode.dcvc_fm_adapter import DcvcFmAdapter


_REQUIRED_ENCODE_PAYLOAD_FIELDS = (
    "raw_bytes",
    "payload_bytes",
    "reduction_pct",
    "transmitted_frame_count",
    "generation_unit_count",
)


class CompressionBenchmark:
    def __init__(
        self,
        frames_dir,
        output_dir,
        fps,
        bitrate="10M",
        encode_result=None,
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
        self.encode_result = dict(encode_result or {})
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
    def raw_dir(self):
        return self.output_dir / "raw"

    @property
    def raw_frames_dir(self):
        return self.raw_dir / "frames"

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
    def benchmark_report_path(self):
        return self.output_dir / "report.json"

    def _canonical_payload(self):
        missing = [
            field
            for field in _REQUIRED_ENCODE_PAYLOAD_FIELDS
            if field not in self.encode_result or self.encode_result.get(field) is None
        ]
        if missing:
            raise RuntimeError("compression benchmark requires encode_result canonical payload fields: {}".format(", ".join(missing)))
        raw_bytes = int(self.encode_result["raw_bytes"])
        payload_bytes = int(self.encode_result["payload_bytes"])
        transmitted = int(self.encode_result["transmitted_frame_count"])
        if raw_bytes <= 0 or payload_bytes < 0 or transmitted <= 0:
            raise RuntimeError("invalid encode_result canonical payload values for compression benchmark")
        return {
            "raw_bytes": raw_bytes,
            "payload_bytes": payload_bytes,
            "reduction_pct": float(self.encode_result["reduction_pct"]),
            "transmitted_frame_count": transmitted,
            "generation_unit_count": int(self.encode_result["generation_unit_count"]),
        }

    def collect_frames(self):
        frame_dir = ensure_dir(self.frames_dir, "prepare frames dir")
        frames = [Path(item).resolve() for item in list_frames_sorted(frame_dir)]
        if not frames:
            raise RuntimeError("compression benchmark requires at least one prepared frame: {}".format(frame_dir))
        return frames

    def _remove_output_child_dir(self, name):
        output_root = self.output_dir.resolve()
        target = (self.output_dir / str(name)).resolve()
        if target.parent != output_root or target.name != str(name):
            log_warn("stale compression benchmark cleanup ignored unsafe path: {}".format(target))
            return
        if not target.exists() and not target.is_symlink():
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(str(target))
        else:
            log_warn("stale compression benchmark cleanup ignored non-file directory entry: {}".format(target))
            return

    def _cleanup_stale_raw_artifacts(self):
        for name in ["zip", "raw_zip_artifact", "raw"]:
            self._remove_output_child_dir(name)

    @staticmethod
    def _ensure_unique_frame_names(frames):
        seen_names = set()
        for frame in list(frames or []):
            name = Path(frame).name
            if name in seen_names:
                raise RuntimeError("duplicate frame basename for raw frames: {}".format(name))
            seen_names.add(name)

    def _materialize_raw_frames_with_strategy(self, frames, strategy):
        self.raw_frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in list(frames or []):
            source = Path(frame).resolve()
            target = self.raw_frames_dir / source.name
            if strategy == "hardlink":
                os.link(str(source), str(target))
            elif strategy == "symlink":
                link_target = os.path.relpath(str(source), start=str(target.parent.resolve()))
                os.symlink(link_target, str(target))
            elif strategy == "copy":
                shutil.copy2(str(source), str(target), follow_symlinks=True)
            else:
                raise RuntimeError("unknown raw frame materialization strategy: {}".format(strategy))
            if not target.is_file():
                raise RuntimeError("raw frame materialization did not create a readable file: {}".format(target))

    def _validate_raw_frames_dir(self, frames):
        for frame in list(frames or []):
            source = Path(frame).resolve()
            target = self.raw_frames_dir / source.name
            if not target.is_file():
                raise RuntimeError("raw frames artifact missing frame: {}".format(target))
            if int(target.stat().st_size) != int(source.stat().st_size):
                raise RuntimeError(
                    "raw frames artifact size mismatch for {}: materialized={} source={}".format(
                        target,
                        int(target.stat().st_size),
                        int(source.stat().st_size),
                    )
                )

    def _materialize_raw_frames(self, frames):
        ordered_frames = [Path(frame).resolve() for frame in list(frames or [])]
        self._ensure_unique_frame_names(ordered_frames)
        last_error = None
        for strategy in ["hardlink", "symlink", "copy"]:
            self._remove_output_child_dir("raw")
            try:
                self._materialize_raw_frames_with_strategy(ordered_frames, strategy)
                self._validate_raw_frames_dir(ordered_frames)
                return self.raw_frames_dir, strategy
            except Exception as exc:
                last_error = exc
                self._remove_output_child_dir("raw")
        raise RuntimeError("failed to materialize raw frames: {}".format(last_error))

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
            "methods_order": list(METHOD_ORDER),
            "methods": {str(item.get("method_key")): dict(item) for item in list(method_reports or [])},
        }
        write_json_atomic(self.benchmark_report_path, report, indent=2)
        return report

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_raw_artifacts()
        frames = self.collect_frames()
        measured_raw_frame_bytes = sum_file_sizes(frames)
        if measured_raw_frame_bytes <= 0:
            raise RuntimeError("compression benchmark selected frames have zero total byte size: {}".format(self.frames_dir))
        canonical_payload = self._canonical_payload()
        raw_frame_bytes = int(canonical_payload["raw_bytes"])

        raw_frames_dir, raw_frame_strategy = self._materialize_raw_frames(frames)
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

        self.vlmem_dir.mkdir(parents=True, exist_ok=True)
        hvm_payload_bytes = int(canonical_payload["payload_bytes"])

        raw_report = canonical_method_report(
            exp_dir=self.exp_dir,
            method_key="raw",
            display_name="Raw",
            status="ok",
            source_frames_dir=self.frames_dir,
            encoded_artifact_path=None,
            encoded_artifact_dir=raw_frames_dir,
            payload_bytes=raw_frame_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=None,
            enc_time_sec=None,
            decode_time_sec=None,
            codec_wall_time_sec=None,
            time_semantics="direct_original_frame_transmission_no_encoder",
            frame_count=len(frames),
            fps=self.fps,
            extra={
                "transmitted_frame_count": int(len(frames)),
                "raw_frame_materialize_strategy": str(raw_frame_strategy),
                "raw_frame_count": int(len(frames)),
                "raw_frame_payload_bytes": int(raw_frame_bytes),
                "measured_raw_frame_bytes": int(measured_raw_frame_bytes),
                "note": "direct original-frame transmission; no encoder-side construction time",
            },
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
            zip_reference_bytes=None,
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
            zip_reference_bytes=None,
            save_decoded_frame=False,
        ).run()
        vlmem_report = canonical_method_report(
            exp_dir=self.exp_dir,
            method_key="vlmem",
            display_name="VLMem/REC",
            status="ok",
            source_frames_dir=self.frames_dir,
            encoded_artifact_path=None,
            encoded_artifact_dir=self.vlmem_dir,
            payload_bytes=hvm_payload_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=None,
            enc_time_sec=self.hvm_algorithmic_time,
            decode_time_sec=None,
            codec_wall_time_sec=self.hvm_algorithmic_time,
            time_semantics="vlmem_payload_from_main_encode_result_no_rescan",
            frame_count=len(frames),
            fps=self.fps,
            extra={
                "transmitted_frame_count": int(canonical_payload["transmitted_frame_count"]),
                "generation_unit_count": int(canonical_payload["generation_unit_count"]),
            },
        )
        report = self._write_report(
            frames=frames,
            raw_frame_bytes=raw_frame_bytes,
            method_reports=[raw_report, h265_report, dcvc_report, vlmem_report],
        )

        dcvc_required = bool(dict(dcvc_report.get("config") or {}).get("required", False))
        if dcvc_required and str(dcvc_report.get("status") or "") != "ok":
            raise RuntimeError("required DCVC-FM benchmark failed: {}".format(dcvc_report.get("error_message") or "unknown error"))

        return {
            "output_dir": self.output_dir,
            "benchmark_report": self.benchmark_report_path,
            "benchmark": report,
            "raw_frames": raw_frames_dir,
            "raw_frame_materialize_strategy": str(raw_frame_strategy),
            "h265_video": self.h265_path,
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "bitrate": str(self.bitrate),
            "h265_sec": float(h265_sec),
            "hvm_algorithmic_sec": float(self.hvm_algorithmic_time),
            "vlmem_algorithmic_sec": float(self.hvm_algorithmic_time),
            "hvm_total_sec": float(self.hvm_algorithmic_time),
            "vlmem_total_sec": float(self.hvm_algorithmic_time),
        }
