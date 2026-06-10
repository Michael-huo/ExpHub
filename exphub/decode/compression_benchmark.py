from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from exphub.common.compression_benchmark import (
    METHOD_ORDER,
    as_dict,
    canonical_method_report,
    file_size,
    method_summary_lines,
    relative_path,
    resolve_path,
    safe_float,
)
from exphub.common.io import ensure_dir, ensure_file, list_frames_sorted, read_json_dict, write_json_atomic
from exphub.common.logging import log_info, log_warn
from exphub.encode.dcvc_fm_adapter import DcvcFmAdapter


def _remove_dir(path_obj):
    path = Path(path_obj)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(str(path), ignore_errors=True)


def _copy_or_link_frames(source_dir, target_dir):
    source = ensure_dir(source_dir, "benchmark source frames")
    target = Path(target_dir).resolve()
    _remove_dir(target)
    target.mkdir(parents=True, exist_ok=True)
    strategy = "symlinked_frame_files"
    for frame in list_frames_sorted(source):
        dst = target / Path(frame).name
        try:
            os.symlink(str(Path(frame).resolve()), str(dst))
        except Exception:
            strategy = "copied_frame_files"
            shutil.copy2(str(Path(frame).resolve()), str(dst), follow_symlinks=True)
    return strategy


def _copy_frames(source_dir, target_dir):
    source = ensure_dir(source_dir, "benchmark source frames")
    target = Path(target_dir).resolve()
    _remove_dir(target)
    target.mkdir(parents=True, exist_ok=True)
    for frame in list_frames_sorted(source):
        shutil.copy2(str(Path(frame).resolve()), str(target / Path(frame).name), follow_symlinks=True)
    return target


def _validate_decoded_frames(decoded_dir, expected_count, label):
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


def _format_summary(report):
    try:
        log_info("[Compression Benchmark: decode] Method Summary:")
        for line in method_summary_lines(report):
            log_info(line)
    except Exception as exc:
        log_warn("compression benchmark decode summary logging skipped: {}".format(exc))


class CompressionBenchmarkDecode:
    def __init__(
        self,
        exp_dir,
        output_dir,
        encode_report_path,
        prepare_frames_dir,
        native_decode_frames_dir,
        native_decode_report_path,
        fps,
        exphub_root,
        ffmpeg_bin=None,
    ):
        self.exp_dir = Path(exp_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.encode_report_path = Path(encode_report_path).resolve()
        self.prepare_frames_dir = Path(prepare_frames_dir).resolve()
        self.native_decode_frames_dir = Path(native_decode_frames_dir).resolve()
        self.native_decode_report_path = Path(native_decode_report_path).resolve()
        self.fps = int(fps)
        self.exphub_root = Path(exphub_root).resolve()
        self.ffmpeg_bin = str(ffmpeg_bin or shutil.which("ffmpeg") or "")

    @property
    def report_path(self):
        return self.output_dir / "compression_benchmark_decode_report.json"

    def _method_dir(self, method_key):
        return self.output_dir / str(method_key)

    @staticmethod
    def _get_nested(obj, keys):
        cur = obj
        for key in list(keys or []):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    def _native_decode_generation_time(self):
        report = read_json_dict(self.native_decode_report_path)
        candidates = [
            ("wall_generate_sec", report.get("wall_generate_sec")),
            ("generate_sec", report.get("generate_sec")),
            ("decode_generate_sec", report.get("decode_generate_sec")),
            ("backend_result.wall_generate_sec", self._get_nested(report, ["backend_result", "wall_generate_sec"])),
            ("backend_result.generate_sec", self._get_nested(report, ["backend_result", "generate_sec"])),
            ("backend_result.decode_generate_sec", self._get_nested(report, ["backend_result", "decode_generate_sec"])),
            ("total_runtime_sec", report.get("total_runtime_sec")),
            ("backend_result.total_runtime_sec", self._get_nested(report, ["backend_result", "total_runtime_sec"])),
        ]
        for source, value in candidates:
            parsed = safe_float(value)
            if parsed is not None:
                return float(parsed), source
        return None, ""

    def _failure_report(self, method_key, encode_method, message, status="failed"):
        source_dir = resolve_path(self.exp_dir, encode_method.get("source_frames_dir")) or self.prepare_frames_dir
        return canonical_method_report(
            exp_dir=self.exp_dir,
            method_key=method_key,
            status=status,
            source_frames_dir=source_dir,
            fps=self.fps,
            frame_count=int(encode_method.get("frame_count") or 0),
            display_name=encode_method.get("display_name"),
            error_message=message,
            payload_bytes=encode_method.get("payload_bytes"),
            enc_time_sec=encode_method.get("enc_time_sec"),
            decode_time_sec=None,
            codec_wall_time_sec=encode_method.get("codec_wall_time_sec"),
            time_semantics=str(encode_method.get("time_semantics") or ""),
            encoded_artifact_path=resolve_path(self.exp_dir, encode_method.get("encoded_artifact_path")),
            encoded_artifact_dir=resolve_path(self.exp_dir, encode_method.get("encoded_artifact_dir")),
        )

    def _report_from_decode(
        self,
        method_key,
        encode_method,
        decoded_frames_dir,
        decode_time_sec,
        decode_strategy,
        extra=None,
    ):
        source_dir = resolve_path(self.exp_dir, encode_method.get("source_frames_dir")) or self.prepare_frames_dir
        raw_reference_bytes = as_dict(self.encode_report).get("raw_frame_bytes")
        zip_reference_bytes = as_dict(self.encode_report).get("reference_zip_bytes")
        merged_extra = {
            "decode_strategy": str(decode_strategy),
            "transmitted_frame_count": encode_method.get("transmitted_frame_count"),
        }
        if extra:
            merged_extra.update(dict(extra))
        return canonical_method_report(
            exp_dir=self.exp_dir,
            method_key=method_key,
            status="ok",
            source_frames_dir=source_dir,
            fps=self.fps,
            frame_count=int(encode_method.get("frame_count") or 0),
            display_name=encode_method.get("display_name"),
            payload_bytes=encode_method.get("payload_bytes"),
            raw_reference_bytes=raw_reference_bytes,
            zip_reference_bytes=zip_reference_bytes,
            enc_time_sec=encode_method.get("enc_time_sec"),
            decode_time_sec=decode_time_sec,
            codec_wall_time_sec=encode_method.get("codec_wall_time_sec"),
            time_semantics=str(encode_method.get("time_semantics") or ""),
            encoded_artifact_path=resolve_path(self.exp_dir, encode_method.get("encoded_artifact_path")),
            encoded_artifact_dir=resolve_path(self.exp_dir, encode_method.get("encoded_artifact_dir")),
            decoded_frames_dir=decoded_frames_dir,
            command=encode_method.get("command"),
            extra=merged_extra,
        )

    def _decode_zip(self, encode_method):
        method_dir = self._method_dir("zip")
        frames_dir = method_dir / "frames"
        started = time.perf_counter()
        strategy = _copy_or_link_frames(self.prepare_frames_dir, frames_dir)
        elapsed = float(time.perf_counter() - started)
        _validate_decoded_frames(frames_dir, int(encode_method.get("frame_count") or 0), "ZIP/ORI")
        return self._report_from_decode(
            "zip",
            encode_method,
            frames_dir,
            elapsed,
            "{}_from_prepare_frames".format(strategy),
        )

    def _decode_h265(self, encode_method):
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for H.265 decode")
        h265_path = resolve_path(self.exp_dir, encode_method.get("encoded_artifact_path"))
        if h265_path is None:
            raise RuntimeError("H.265 encoded_artifact_path missing in encode report")
        ensure_file(h265_path, "H.265 encoded video")
        method_dir = self._method_dir("h265")
        frames_dir = method_dir / "frames"
        _remove_dir(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(h265_path),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(frames_dir / "%06d.png"),
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = float(time.perf_counter() - started)
        if proc.returncode != 0:
            output = str(proc.stdout or "")
            tail = "\n".join(output.splitlines()[-12:])
            raise RuntimeError("ffmpeg H.265 decode failed rc={} cmd={} tail={}".format(int(proc.returncode), " ".join(cmd), tail))
        _validate_decoded_frames(frames_dir, int(encode_method.get("frame_count") or 0), "H.265")
        return self._report_from_decode(
            "h265",
            encode_method,
            frames_dir,
            elapsed,
            "ffmpeg_extract_frames_from_encode_stage_mp4",
            extra={"decode_command": cmd},
        )

    def _decode_dcvc(self, encode_method):
        status = str(encode_method.get("status") or "")
        if status != "ok":
            raise RuntimeError("DCVC-FM encode method status is {}; cannot create downstream frames".format(status or "missing"))
        method_dir = self._method_dir("dcvc_fm_q21")
        frames_dir = method_dir / "frames"
        work_dir = method_dir / "frame_generation_work"
        _remove_dir(frames_dir)
        _remove_dir(work_dir)
        started = time.perf_counter()
        frame_result = DcvcFmAdapter(
            exphub_root=self.exphub_root,
            frames_dir=self.prepare_frames_dir,
            output_dir=work_dir,
            exp_dir=self.exp_dir,
            fps=self.fps,
            raw_reference_bytes=as_dict(self.encode_report).get("raw_frame_bytes"),
            zip_reference_bytes=as_dict(self.encode_report).get("reference_zip_bytes"),
            save_decoded_frame=True,
        ).run()
        elapsed = float(time.perf_counter() - started)
        if str(frame_result.get("status") or "") != "ok":
            raise RuntimeError("DCVC-FM frame-generation fallback failed: {}".format(frame_result.get("error_message") or "unknown error"))
        generated_dir = resolve_path(self.exp_dir, frame_result.get("decoded_frames_dir"))
        if generated_dir is None:
            raise RuntimeError("DCVC-FM frame-generation fallback did not report decoded frames")
        _copy_frames(generated_dir, frames_dir)
        _validate_decoded_frames(frames_dir, int(encode_method.get("frame_count") or 0), "DCVC-FM q21")
        return self._report_from_decode(
            "dcvc_fm_q21",
            encode_method,
            frames_dir,
            elapsed,
            "dcvc_fm_frame_generation_fallback_no_clean_pure_decode_api",
            extra={
                "decoded_frame_generation_sec": elapsed,
                "frame_generation_work_dir": relative_path(self.exp_dir, work_dir),
                "frame_generation_report": frame_result,
                "time_semantics": (
                    str(encode_method.get("time_semantics") or "")
                    + "; decode_time_sec is a separate downstream frame-generation fallback and is not Table-II enc_time_sec"
                ),
            },
        )

    def _decode_vlmem(self, encode_method):
        method_dir = self._method_dir("vlmem")
        frames_dir = method_dir / "frames"
        started = time.perf_counter()
        strategy = _copy_or_link_frames(self.native_decode_frames_dir, frames_dir)
        materialize_sec = float(time.perf_counter() - started)
        _validate_decoded_frames(frames_dir, int(encode_method.get("frame_count") or 0), "VLMem/REC")
        native_decode_sec, source = self._native_decode_generation_time()
        warning = ""
        if native_decode_sec is None:
            native_decode_sec = materialize_sec
            source = "materialization_fallback"
            warning = (
                "native decode generation time missing in {}; VLMem decode_time_sec fell back to "
                "materialization time"
            ).format(relative_path(self.exp_dir, self.native_decode_report_path))
            log_warn(warning)
        return self._report_from_decode(
            "vlmem",
            encode_method,
            frames_dir,
            native_decode_sec,
            "reuse_native_rec_frames",
            extra={
                "materialize_time_sec": materialize_sec,
                "decode_materialize_time_sec": materialize_sec,
                "materialize_strategy": "{}_from_native_rec_frames".format(strategy),
                "decode_time_sec_source": source,
                "decode_time_sec_warning": warning,
                "native_decode_report": relative_path(self.exp_dir, self.native_decode_report_path),
                "time_semantics": (
                    str(encode_method.get("time_semantics") or "")
                    + "; decoded frames reuse native REC frames; decode_time_sec reflects native "
                    "world-model generation time from decode_report.json"
                    if source != "materialization_fallback"
                    else str(encode_method.get("time_semantics") or "")
                    + "; decoded frames reuse native REC frames; decode_time_sec fell back to "
                    "frame materialization time because native generation time was unavailable"
                ),
            },
        )

    def run(self):
        ensure_file(self.encode_report_path, "compression benchmark encode report")
        ensure_dir(self.prepare_frames_dir, "prepare frames dir")
        ensure_dir(self.native_decode_frames_dir, "native decode frames dir")
        ensure_file(self.native_decode_report_path, "native decode report")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encode_report = read_json_dict(self.encode_report_path)
        if not self.encode_report:
            raise RuntimeError("invalid compression benchmark encode report: {}".format(self.encode_report_path))
        methods = as_dict(self.encode_report.get("methods"))

        rows = []
        for method_key, handler in [
            ("zip", self._decode_zip),
            ("h265", self._decode_h265),
            ("dcvc_fm_q21", self._decode_dcvc),
            ("vlmem", self._decode_vlmem),
        ]:
            encode_method = as_dict(methods.get(method_key))
            if not encode_method:
                rows.append(self._failure_report(method_key, {}, "encode report missing method {}".format(method_key), status="failed"))
                continue
            try:
                rows.append(handler(encode_method))
            except Exception as exc:
                log_warn("compression benchmark decode {} failed: {}".format(method_key, exc))
                rows.append(self._failure_report(method_key, encode_method, str(exc), status="failed"))

        report = {
            "version": 2,
            "stage": "decode",
            "source": "exphub.decode.compression_benchmark",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": relative_path(self.exp_dir, self.output_dir),
            "encode_report": relative_path(self.exp_dir, self.encode_report_path),
            "native_decode_report": relative_path(self.exp_dir, self.native_decode_report_path),
            "source_frames_dir": relative_path(self.exp_dir, self.prepare_frames_dir),
            "native_decode_frames_dir": relative_path(self.exp_dir, self.native_decode_frames_dir),
            "frame_count": int(self.encode_report.get("frame_count") or 0),
            "fps": int(self.fps),
            "methods_order": list(METHOD_ORDER),
            "methods": {row["method_key"]: dict(row) for row in rows},
            "rows": rows,
        }
        write_json_atomic(self.report_path, report, indent=2)
        log_info("compression benchmark decode stage report: {}".format(relative_path(self.exp_dir, self.report_path)))
        _format_summary(report)
        return {
            "report_path": self.report_path,
            "summary": report,
            "out_dir": self.output_dir,
        }
