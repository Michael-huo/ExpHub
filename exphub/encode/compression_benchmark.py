from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

from exphub.common.io import ensure_dir, frame_sort_key, list_frames_sorted, write_json_atomic
from exphub.common.logging import log_info, log_warn
from exphub.encode.dcvc_fm_adapter import DcvcFmAdapter
from exphub.encode.payload_writer import write_hvm_payload_zip


METHOD_ORDER = ("zip", "h265", "dcvc_fm_q21", "vlmem")
DISPLAY_NAMES = {
    "zip": "ZIP/ORI",
    "h265": "H.265",
    "dcvc_fm_q21": "DCVC-FM q21",
    "vlmem": "VLMem/REC",
}


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _safe_token(value):
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return token or "bitrate"


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return target.relative_to(base).as_posix()
    except Exception:
        return str(target)


def _file_size(path_obj):
    try:
        path = Path(path_obj).resolve()
        if path.is_file():
            return int(path.stat().st_size)
    except Exception:
        pass
    return 0


def _sum_file_sizes(paths):
    return int(sum(_file_size(path) for path in list(paths or [])))


def _bytes_to_mib(value):
    try:
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return None


def _reduction_pct(reference_bytes, payload_bytes):
    try:
        ref = float(reference_bytes)
        val = float(payload_bytes)
    except Exception:
        return None
    if ref <= 0.0:
        return None
    return float((1.0 - val / ref) * 100.0)


def _path_or_none(exp_dir, path_obj):
    if path_obj is None:
        return None
    return _relative_path(exp_dir, path_obj)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _format_seconds(value):
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return "{:.2f}s".format(parsed)


def _format_mib_from_report(item):
    report = _as_dict(item)
    mib = _safe_float(report.get("payload_mib"))
    if mib is None:
        payload_bytes = _safe_float(report.get("payload_bytes"))
        if payload_bytes is not None:
            mib = payload_bytes / (1024.0 * 1024.0)
    if mib is None:
        return "n/a"
    return "{:.2f}MiB".format(mib)


def _short_reason(value, max_len=120):
    try:
        text = str(value or "").splitlines()[0].strip()
    except Exception:
        text = ""
    if not text:
        return ""
    if len(text) <= int(max_len):
        return text
    return text[: max(0, int(max_len) - 3)].rstrip() + "..."


def _log_method_summary(benchmark_report):
    try:
        report = _as_dict(benchmark_report)
        methods = _as_dict(report.get("methods"))
        log_info("[Compression Benchmark] Method Summary:")
        for method_key in METHOD_ORDER:
            item = _as_dict(methods.get(method_key))
            display_name = str(item.get("display_name") or DISPLAY_NAMES.get(method_key, method_key))
            status = str(item.get("status") or ("missing" if not item else "n/a"))
            line = (
                "  - {}: status={} payload={} enc={} dec={}".format(
                    display_name,
                    status,
                    _format_mib_from_report(item),
                    _format_seconds(item.get("encode_time_sec")),
                    _format_seconds(item.get("decode_time_sec")),
                )
            )

            semantics = str(item.get("codec_time_semantics") or "")
            wall_time = item.get("codec_wall_time_sec")
            if semantics == "combined_encode_decode_wall_time" and wall_time is not None:
                line += " wall={}".format(_format_seconds(wall_time))
            if semantics == "combined_encode_decode_wall_time":
                line += " semantics={}".format(semantics)

            if status in ("missing", "skipped", "failed"):
                reason = _short_reason(item.get("error_message"))
                if not reason and status == "missing":
                    reason = "method report missing"
                if reason:
                    line += " reason={}".format(reason)
            log_info(line)
    except Exception as exc:
        log_warn("compression benchmark method summary logging skipped: {}".format(exc))


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
            "ours_algorithmic_time",
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
    def h265_decoded_frames_dir(self):
        return self.output_dir / "h265" / "decoded_frames"

    @property
    def hvm_payload_zip_path(self):
        return self.output_dir / "hvm_payload.zip"

    @property
    def benchmark_report_path(self):
        return self.output_dir / "benchmark_report.json"

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
            raise RuntimeError("compression benchmark requires Ours payload dir")
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

    def _decode_h265_frames(self):
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for H.265 decode")
        if self.h265_decoded_frames_dir.exists():
            shutil.rmtree(str(self.h265_decoded_frames_dir), ignore_errors=True)
        self.h265_decoded_frames_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.h265_path),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(self.h265_decoded_frames_dir / "%06d.png"),
        ]
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
                "ffmpeg H.265 decode failed rc={} cmd={} tail={}".format(
                    int(proc.returncode),
                    " ".join(cmd),
                    tail,
                )
            )
        return cmd

    def _validate_decoded_frames(self, decoded_dir, expected_count, label):
        frames = list_frames_sorted(decoded_dir)
        if len(frames) != int(expected_count):
            raise RuntimeError(
                "{} decoded frame count mismatch: decoded={} expected={} dir={}".format(
                    str(label),
                    int(len(frames)),
                    int(expected_count),
                    Path(decoded_dir).resolve(),
                )
            )
        expected_names = ["{:06d}.png".format(int(idx)) for idx in range(int(expected_count))]
        actual_names = [Path(item).name for item in frames]
        if actual_names != expected_names:
            raise RuntimeError(
                "{} decoded frames must be normalized to ExpHub names; expected first/last={} actual first/last={}".format(
                    str(label),
                    (expected_names[:2], expected_names[-2:]),
                    (actual_names[:2], actual_names[-2:]),
                )
            )
        return frames

    def _method_report(
        self,
        method_key,
        display_name,
        status,
        source_frames_dir,
        decoded_frames_dir=None,
        payload_path=None,
        bitstream_dir=None,
        payload_bytes=None,
        raw_reference_bytes=None,
        zip_reference_bytes=None,
        encode_time_sec=None,
        decode_time_sec=None,
        codec_wall_time_sec=None,
        codec_time_semantics="separate_encode_decode_wall_time",
        frame_count=0,
        transmitted_frame_count=None,
        error_message="",
        command=None,
        trajectory_role="codec_decoded",
    ):
        return {
            "method_key": str(method_key),
            "display_name": str(display_name),
            "status": str(status),
            "error_message": str(error_message or ""),
            "source_frames_dir": _relative_path(self.exp_dir, source_frames_dir),
            "decoded_frames_dir": _path_or_none(self.exp_dir, decoded_frames_dir),
            "payload_path": _path_or_none(self.exp_dir, payload_path),
            "bitstream_dir": _path_or_none(self.exp_dir, bitstream_dir),
            "payload_bytes": int(payload_bytes) if payload_bytes is not None else None,
            "payload_mib": _bytes_to_mib(payload_bytes) if payload_bytes is not None else None,
            "reduction_pct_vs_zip": _reduction_pct(zip_reference_bytes, payload_bytes),
            "reduction_pct_vs_raw_frames": _reduction_pct(raw_reference_bytes, payload_bytes),
            "encode_time_sec": float(encode_time_sec) if encode_time_sec is not None else None,
            "decode_time_sec": float(decode_time_sec) if decode_time_sec is not None else None,
            "codec_wall_time_sec": float(codec_wall_time_sec) if codec_wall_time_sec is not None else None,
            "codec_time_semantics": str(codec_time_semantics),
            "frame_count": int(frame_count),
            "transmitted_frame_count": int(transmitted_frame_count) if transmitted_frame_count is not None else None,
            "fps": int(self.fps),
            "trajectory_role": str(trajectory_role),
            "command": list(command or []),
        }

    def _write_benchmark_report(self, frames, raw_frame_bytes, method_reports):
        report = {
            "version": 1,
            "source": "exphub.encode.compression_benchmark",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": _relative_path(self.exp_dir, self.output_dir),
            "source_frames_dir": _relative_path(self.exp_dir, self.frames_dir),
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "raw_frame_bytes": int(raw_frame_bytes),
            "raw_frame_mib": _bytes_to_mib(raw_frame_bytes),
            "reference_zip_path": _relative_path(self.exp_dir, self.raw_zip_path),
            "reference_zip_bytes": _file_size(self.raw_zip_path),
            "methods_order": ["zip", "h265", "dcvc_fm_q21", "vlmem"],
            "methods": {str(item.get("method_key")): dict(item) for item in list(method_reports or [])},
        }
        write_json_atomic(self.benchmark_report_path, report, indent=2)
        return report

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.collect_frames()
        raw_frame_bytes = _sum_file_sizes(frames)

        started = time.perf_counter()
        self._write_raw_zip(frames)
        raw_zip_sec = float(time.perf_counter() - started)
        raw_zip_bytes = _file_size(self.raw_zip_path)

        started = time.perf_counter()
        hvm_payload_zip = self._write_hvm_payload_zip()
        hvm_zip_sec = float(time.perf_counter() - started)
        # Ours Total = formal encode/Ours payload algorithm time + payload zip I/O time.
        hvm_total_sec = float(self.hvm_algorithmic_time) + float(hvm_zip_sec)
        hvm_payload_bytes = _file_size(hvm_payload_zip)
        hvm_payload_frames = list_frames_sorted(self.hvm_payload_dir / "frames") if self.hvm_payload_dir is not None else []

        concat_file = None
        h265_decode_sec = None
        h265_decode_command = []
        h265_status = "ok"
        h265_error = ""
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
            try:
                started = time.perf_counter()
                h265_decode_command = self._decode_h265_frames()
                h265_decode_sec = float(time.perf_counter() - started)
                self._validate_decoded_frames(self.h265_decoded_frames_dir, len(frames), "H.265")
            except Exception as exc:
                h265_status = "failed"
                h265_error = str(exc)
                log_warn("H.265 decoded frame preservation failed: {}".format(h265_error))
        finally:
            if concat_file is not None:
                try:
                    Path(concat_file).unlink()
                except FileNotFoundError:
                    pass

        zip_report = self._method_report(
            method_key="zip",
            display_name="ZIP/ORI",
            status="ok",
            source_frames_dir=self.frames_dir,
            decoded_frames_dir=self.frames_dir,
            payload_path=self.raw_zip_path,
            payload_bytes=raw_zip_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            encode_time_sec=raw_zip_sec,
            decode_time_sec=0.0,
            codec_wall_time_sec=raw_zip_sec,
            codec_time_semantics="zip_archive_wall_time",
            frame_count=len(frames),
            transmitted_frame_count=len(frames),
            trajectory_role="ORI",
        )
        h265_report = self._method_report(
            method_key="h265",
            display_name="H.265",
            status=h265_status,
            source_frames_dir=self.frames_dir,
            decoded_frames_dir=self.h265_decoded_frames_dir if h265_status == "ok" else None,
            payload_path=self.h265_path,
            payload_bytes=_file_size(self.h265_path),
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            encode_time_sec=h265_sec,
            decode_time_sec=h265_decode_sec,
            codec_wall_time_sec=(float(h265_sec) + float(h265_decode_sec)) if h265_decode_sec is not None else h265_sec,
            codec_time_semantics="separate_encode_decode_wall_time" if h265_decode_sec is not None else "encode_only_plus_decode_failed",
            frame_count=len(frames),
            transmitted_frame_count=len(frames),
            error_message=h265_error,
            command=list(command) + ([";"] + list(h265_decode_command) if h265_decode_command else []),
            trajectory_role="codec_decoded",
        )
        dcvc_report = DcvcFmAdapter(
            exphub_root=self.exphub_root,
            frames_dir=self.frames_dir,
            output_dir=self.output_dir / "dcvc_fm_q21",
            exp_dir=self.exp_dir,
            fps=self.fps,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
        ).run()
        vlmem_report = self._method_report(
            method_key="vlmem",
            display_name="VLMem/REC",
            status="ok",
            source_frames_dir=self.frames_dir,
            decoded_frames_dir=None,
            payload_path=hvm_payload_zip,
            payload_bytes=hvm_payload_bytes,
            raw_reference_bytes=raw_frame_bytes,
            zip_reference_bytes=raw_zip_bytes,
            encode_time_sec=hvm_total_sec,
            decode_time_sec=None,
            codec_wall_time_sec=hvm_total_sec,
            codec_time_semantics="vlmem_encode_payload_zip_wall_time",
            frame_count=len(frames),
            transmitted_frame_count=len(hvm_payload_frames),
            trajectory_role="REC",
        )
        benchmark_report = self._write_benchmark_report(
            frames=frames,
            raw_frame_bytes=raw_frame_bytes,
            method_reports=[zip_report, h265_report, dcvc_report, vlmem_report],
        )

        log_info(
            "[Compression Benchmark] Time Stats -> Raw ZIP: {:.2f}s | H.265 MP4: {:.2f}s | H.265 Decode: {} | Ours Total: {:.2f}s".format(
                float(raw_zip_sec),
                float(h265_sec),
                "{:.2f}s".format(float(h265_decode_sec)) if h265_decode_sec is not None else "failed",
                float(hvm_total_sec),
            )
        )
        _log_method_summary(benchmark_report)

        dcvc_required = bool(dict(dcvc_report.get("config") or {}).get("required", False))
        if dcvc_required and str(dcvc_report.get("status") or "") != "ok":
            raise RuntimeError("required DCVC-FM benchmark failed: {}".format(dcvc_report.get("error_message") or "unknown error"))

        return {
            "output_dir": self.output_dir,
            "benchmark_report": self.benchmark_report_path,
            "benchmark": benchmark_report,
            "raw_zip": self.raw_zip_path,
            "h265_video": self.h265_path,
            "h265_decoded_frames_dir": self.h265_decoded_frames_dir if h265_status == "ok" else None,
            "hvm_payload_zip": hvm_payload_zip,
            "ours_payload_zip": hvm_payload_zip,
            "frame_count": int(len(frames)),
            "fps": int(self.fps),
            "bitrate": str(self.bitrate),
            "raw_zip_sec": float(raw_zip_sec),
            "h265_sec": float(h265_sec),
            "h265_decode_sec": h265_decode_sec,
            "hvm_algorithmic_sec": float(self.hvm_algorithmic_time),
            "ours_algorithmic_sec": float(self.hvm_algorithmic_time),
            "hvm_zip_sec": float(hvm_zip_sec),
            "ours_zip_sec": float(hvm_zip_sec),
            "hvm_total_sec": float(hvm_total_sec),
            "ours_total_sec": float(hvm_total_sec),
            "ffmpeg_command": list(command),
            "ffmpeg_decode_command": list(h265_decode_command),
        }
