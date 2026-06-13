from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from exphub.common.io import frame_sort_key, list_frames_sorted, write_json_atomic
from exphub.config import get_platform_config


METHOD_KEY = "dcvc_fm_q21"
DISPLAY_NAME = "DCVC-FM q21"
DCVC_DS_NAME = "EXPHUB"
DCVC_BASE_PATH = "EXPHUB"
DCVC_SEQUENCE_NAME = "exphub"


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _relative_path(base_dir, target_path):
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()
    try:
        return target.relative_to(base).as_posix()
    except Exception:
        return str(target)


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


def _resolve_path(text, exphub_root):
    value = str(text or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(exphub_root).resolve() / path
    return path.resolve()


def _resolve_path_under(text, base_dir):
    value = str(text or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir).resolve() / path
    return path.resolve()


def _image_size(path_obj):
    path = Path(path_obj).resolve()
    if path.suffix.lower() == ".png":
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                if width > 0 and height > 0:
                    return int(width), int(height)
        except Exception:
            pass
    try:
        from PIL import Image

        with Image.open(str(path)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise RuntimeError("cannot read image dimensions for {}: {}".format(path, exc)) from exc


def _copy_as_png(src, dst):
    source = Path(src).resolve()
    target = Path(dst).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".png":
        shutil.copy2(str(source), str(target))
        return
    try:
        from PIL import Image

        with Image.open(str(source)) as image:
            image.convert("RGB").save(str(target))
    except Exception as exc:
        raise RuntimeError("failed to stage non-PNG frame {} as {}: {}".format(source, target, exc)) from exc


def _sum_files(root, suffixes):
    root_path = Path(root).resolve()
    total = 0
    paths = []
    suffix_set = {str(item).lower() for item in suffixes}
    if not root_path.exists():
        return 0, []
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        if suffix_set and path.suffix.lower() not in suffix_set:
            continue
        try:
            total += int(path.stat().st_size)
        except Exception:
            pass
        paths.append(path.resolve())
    return int(total), paths


class DcvcFmAdapter:
    def __init__(
        self,
        exphub_root,
        frames_dir,
        output_dir,
        exp_dir,
        fps,
        raw_reference_bytes=None,
        zip_reference_bytes=None,
        save_decoded_frame=False,
    ):
        self.exphub_root = Path(exphub_root).resolve()
        self.frames_dir = Path(frames_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.exp_dir = Path(exp_dir).resolve()
        self.fps = int(fps)
        self.raw_reference_bytes = raw_reference_bytes
        self.zip_reference_bytes = zip_reference_bytes
        self.save_decoded_frame = bool(save_decoded_frame)

    def _base_report(self, status="skipped", error_message=""):
        staged_input_dir = self.output_dir / "staged_input"
        resolved_sequence_path = staged_input_dir / DCVC_BASE_PATH / DCVC_SEQUENCE_NAME
        stream_dir = self.output_dir / "stream"
        dataset_config_path = self.output_dir / "dcvc_dataset_config.json"
        return {
            "method_key": METHOD_KEY,
            "display_name": DISPLAY_NAME,
            "status": str(status),
            "error_message": str(error_message or ""),
            "source_frames_dir": _relative_path(self.exp_dir, self.frames_dir),
            "decoded_frames_dir": _relative_path(self.exp_dir, self.output_dir / "decoded_frames")
            if self.save_decoded_frame
            else None,
            "encoded_artifact_path": None,
            "encoded_artifact_dir": _relative_path(self.exp_dir, stream_dir),
            "payload_bytes": None,
            "payload_mib": None,
            "reduction_pct": None,
            "reduction_pct_vs_zip": None,
            "reduction_pct_vs_raw_frames": None,
            "enc_time_sec": None,
            "decode_time_sec": None,
            "codec_wall_time_sec": None,
            "time_semantics": "unavailable",
            "frame_count": 0,
            "fps": int(self.fps),
            "trajectory_role": "codec_decoded",
            "command": [],
            "command_argv": [],
            "cwd": None,
            "stdout_log": None,
            "stderr_log": None,
            "stdout_log_path": None,
            "stderr_log_path": None,
            "test_config_path": _relative_path(self.exp_dir, dataset_config_path),
            "staged_input_dir": _relative_path(self.exp_dir, staged_input_dir),
            "resolved_sequence_path": str(resolved_sequence_path.resolve()),
            "stream_dir": _relative_path(self.exp_dir, stream_dir),
            "decoded_frame_save_requested": bool(self.save_decoded_frame),
            "config": {},
        }

    def _platform_config(self):
        try:
            cfg = get_platform_config(exphub_root=self.exphub_root)
        except Exception:
            return {}
        return _as_dict(_as_dict(cfg.get("external_codecs")).get("dcvc_fm"))

    def _resolve_runtime_config(self, frames):
        cfg = self._platform_config()
        enabled = _as_bool(cfg.get("enabled"), default=False)
        required = _as_bool(cfg.get("required"), default=False)
        q_i, q_p = self._resolve_q_indexes(cfg)

        root = _resolve_path(cfg.get("root"), self.exphub_root)
        if root is None:
            root = (self.exphub_root / ".." / "DCVC-FM").resolve()

        python_path = _resolve_path(cfg.get("python"), self.exphub_root)
        if python_path is None:
            python_path = _resolve_path(os.environ.get("DCVC_FM_PYTHON"), self.exphub_root)

        script = _resolve_path_under(cfg.get("script"), root)
        if script is None:
            script = root / "test_video.py"

        model_path_i = _resolve_path_under(cfg.get("model_path_i"), root)
        if model_path_i is None:
            model_path_i = root / "checkpoints" / "cvpr2024_image.pth.tar"
        model_path_p = _resolve_path_under(cfg.get("model_path_p"), root)
        if model_path_p is None:
            model_path_p = root / "checkpoints" / "cvpr2024_video.pth.tar"

        width, height = _image_size(frames[0])
        return {
            "raw": cfg,
            "enabled": bool(enabled),
            "required": bool(required),
            "root": root,
            "python": python_path,
            "script": script,
            "model_path_i": model_path_i,
            "model_path_p": model_path_p,
            "q_index_i": int(q_i),
            "q_index_p": int(q_p),
            "width": int(width),
            "height": int(height),
            "worker": max(1, _as_int(cfg.get("worker"), 1)),
            "cuda": _as_bool(cfg.get("cuda"), default=True),
            "force_intra_period": _as_int(cfg.get("force_intra_period"), 9999),
            "command_template": str(cfg.get("command_template") or "").strip(),
        }

    @staticmethod
    def _resolve_q_indexes(cfg):
        q_index = cfg.get("q_index")
        q_i = cfg.get("q_indexes_i", q_index if q_index is not None else 21)
        q_p = cfg.get("q_indexes_p", q_index if q_index is not None else 21)
        if isinstance(q_i, (list, tuple)):
            q_i = q_i[0] if q_i else 21
        if isinstance(q_p, (list, tuple)):
            q_p = q_p[0] if q_p else 21
        return _as_int(q_i, 21), _as_int(q_p, 21)

    def _unavailable_report(self, config, message, stdout_log=None, stderr_log=None):
        status = "failed" if bool(config.get("required")) else "skipped"
        report = self._base_report(status=status, error_message=message)
        report["config"] = self._report_config(config)
        if config.get("root") is not None:
            report["cwd"] = str(Path(config["root"]).resolve())
        if stdout_log is not None:
            report["stdout_log"] = _relative_path(self.exp_dir, stdout_log)
            report["stdout_log_path"] = report["stdout_log"]
        if stderr_log is not None:
            report["stderr_log"] = _relative_path(self.exp_dir, stderr_log)
            report["stderr_log_path"] = report["stderr_log"]
        return report

    def _report_config(self, config):
        return {
            "enabled": bool(config.get("enabled", False)),
            "required": bool(config.get("required", False)),
            "root": str(config.get("root") or ""),
            "python": str(config.get("python") or ""),
            "script": str(config.get("script") or ""),
            "model_path_i": str(config.get("model_path_i") or ""),
            "model_path_p": str(config.get("model_path_p") or ""),
            "q_index_i": int(config.get("q_index_i", 21) or 21),
            "q_index_p": int(config.get("q_index_p", 21) or 21),
            "worker": int(config.get("worker", 1) or 1),
            "cuda": bool(config.get("cuda", True)),
            "force_intra_period": int(config.get("force_intra_period", 9999) or 9999),
        }

    def _validate_config(self, config):
        if not config["enabled"]:
            return "external_codecs.dcvc_fm.enabled is false"
        if int(config["q_index_i"]) != 21 or int(config["q_index_p"]) != 21:
            return (
                "dcvc_fm_q21 requires q_index_i=21 and q_index_p=21; got q_index_i={} q_index_p={}".format(
                    int(config["q_index_i"]),
                    int(config["q_index_p"]),
                )
            )
        for key, label, executable in [
            ("root", "DCVC-FM root", False),
            ("python", "DCVC-FM python executable", True),
            ("script", "DCVC-FM script", False),
            ("model_path_i", "DCVC-FM image model", False),
            ("model_path_p", "DCVC-FM video model", False),
        ]:
            path = Path(config[key]).resolve() if config.get(key) is not None else None
            if path is None or not path.exists():
                return "{} not found: {}".format(label, path or "<missing>")
            if key == "root" and not path.is_dir():
                return "{} is not a directory: {}".format(label, path)
            if key != "root" and not path.is_file():
                return "{} is not a file: {}".format(label, path)
            if executable and not os.access(str(path), os.X_OK):
                return "{} is not executable: {}".format(label, path)
        return ""

    def _inspect_script(self, config):
        stdout_log = self.output_dir / "dcvc_help_stdout.txt"
        stderr_log = self.output_dir / "dcvc_help_stderr.txt"
        cmd = [str(config["python"]), str(config["script"]), "--help"]
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                cmd,
                cwd=str(config["root"]),
                stdout=out,
                stderr=err,
                text=True,
                check=False,
                timeout=60,
            )
        if proc.returncode != 0:
            return (
                "DCVC-FM script inspection failed rc={} stdout={} stderr={}".format(
                    int(proc.returncode),
                    _relative_path(self.exp_dir, stdout_log),
                    _relative_path(self.exp_dir, stderr_log),
                ),
                stdout_log,
                stderr_log,
            )
        text = stdout_log.read_text(encoding="utf-8", errors="ignore")
        required_args = [
            "--test_config",
            "--output_path",
            "--write_stream",
            "--save_decoded_frame",
            "--stream_path",
            "--q_indexes_i",
            "--q_indexes_p",
        ]
        missing = [arg for arg in required_args if arg not in text]
        if missing:
            return (
                "DCVC-FM script does not expose expected arguments: {}".format(", ".join(missing)),
                stdout_log,
                stderr_log,
            )
        return "", stdout_log, stderr_log

    def _stage_frames(self, frames, config):
        stage_root = self.output_dir / "staged_input"
        sequence_dir = stage_root / DCVC_BASE_PATH / DCVC_SEQUENCE_NAME
        if stage_root.exists():
            shutil.rmtree(str(stage_root), ignore_errors=True)
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames, start=1):
            _copy_as_png(frame, sequence_dir / "im{:05d}.png".format(int(idx)))
        dataset_config = {
            "root_path": str(stage_root.resolve()),
            "test_classes": {
                DCVC_DS_NAME: {
                    "test": 1,
                    "base_path": DCVC_BASE_PATH,
                    "src_type": "png",
                    "sequences": {
                        DCVC_SEQUENCE_NAME: {
                            "width": int(config["width"]),
                            "height": int(config["height"]),
                            "frames": int(len(frames)),
                            "intra_period": int(config["force_intra_period"]),
                        }
                    },
                }
            },
        }
        config_path = self.output_dir / "dcvc_dataset_config.json"
        write_json_atomic(config_path, dataset_config, indent=2)
        return config_path, stage_root, sequence_dir

    def _build_command(self, config, dataset_config_path, stream_dir, output_json):
        mapping = {
            "python": str(config["python"]),
            "script": str(config["script"]),
            "model_path_i": str(config["model_path_i"]),
            "model_path_p": str(config["model_path_p"]),
            "test_config": str(Path(dataset_config_path).resolve()),
            "stream_path": str(Path(stream_dir).resolve()),
            "output_path": str(Path(output_json).resolve()),
            "q_index": str(int(config["q_index_i"])),
            "q_index_i": str(int(config["q_index_i"])),
            "q_index_p": str(int(config["q_index_p"])),
            "worker": str(int(config["worker"])),
            "cuda": "true" if bool(config["cuda"]) else "false",
            "force_intra_period": str(int(config["force_intra_period"])),
            "force_frame_num": str(-1),
            "save_decoded_frame": "true" if bool(self.save_decoded_frame) else "false",
        }
        template = str(config.get("command_template") or "").strip()
        if template:
            return shlex.split(template.format(**mapping))
        return [
            str(config["python"]),
            str(config["script"]),
            "--model_path_i",
            str(config["model_path_i"]),
            "--model_path_p",
            str(config["model_path_p"]),
            "--rate_num",
            "1",
            "--q_indexes_i",
            str(int(config["q_index_i"])),
            "--q_indexes_p",
            str(int(config["q_index_p"])),
            "--test_config",
            str(Path(dataset_config_path).resolve()),
            "--worker",
            str(int(config["worker"])),
            "--cuda",
            "true" if bool(config["cuda"]) else "false",
            "--write_stream",
            "true",
            "--stream_path",
            str(Path(stream_dir).resolve()),
            "--save_decoded_frame",
            "true" if bool(self.save_decoded_frame) else "false",
            "--output_path",
            str(Path(output_json).resolve()),
            "--force_intra_period",
            str(int(config["force_intra_period"])),
        ]

    def _normalize_decoded_frames(self, stream_dir, prepared_frames):
        decoded_dir = self.output_dir / "decoded_frames"
        if decoded_dir.exists():
            shutil.rmtree(str(decoded_dir), ignore_errors=True)
        decoded_dir.mkdir(parents=True, exist_ok=True)

        prepared = [Path(item).resolve() for item in prepared_frames]
        frame_count = int(len(prepared))
        source_dir = Path(stream_dir).resolve() / DCVC_DS_NAME
        decoded = [
            item
            for item in source_dir.iterdir()
            if item.is_file() and item.name.startswith("im") and item.suffix.lower() == ".png"
        ] if source_dir.is_dir() else []
        decoded.sort(key=frame_sort_key)
        if len(decoded) != int(frame_count):
            raise RuntimeError(
                "DCVC-FM decoded frame count mismatch: decoded={} expected={} dir={}".format(
                    int(len(decoded)),
                    int(frame_count),
                    source_dir,
                )
            )
        expected_decoded_names = ["im{:05d}.png".format(int(idx)) for idx in range(1, int(frame_count) + 1)]
        actual_decoded_names = [item.name for item in decoded]
        if actual_decoded_names != expected_decoded_names:
            raise RuntimeError(
                "DCVC-FM decoded frame names mismatch: expected first={} last={} actual first={} last={}".format(
                    expected_decoded_names[0] if expected_decoded_names else "",
                    expected_decoded_names[-1] if expected_decoded_names else "",
                    actual_decoded_names[0] if actual_decoded_names else "",
                    actual_decoded_names[-1] if actual_decoded_names else "",
                )
            )
        for idx, frame in enumerate(decoded):
            decoded_size = _image_size(frame)
            prepared_size = _image_size(prepared[idx])
            if decoded_size != prepared_size:
                raise RuntimeError(
                    "DCVC-FM decoded frame dimension mismatch at index {}: decoded={} prepared={}".format(
                        int(idx),
                        decoded_size,
                        prepared_size,
                    )
                )
            _copy_as_png(frame, decoded_dir / "{:06d}.png".format(int(idx)))
        normalized = list_frames_sorted(decoded_dir)
        expected_names = ["{:06d}.png".format(int(idx)) for idx in range(int(frame_count))]
        actual_names = [item.name for item in normalized]
        if actual_names != expected_names:
            raise RuntimeError("DCVC-FM normalized frame names mismatch")
        return decoded_dir

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = [Path(item).resolve() for item in list_frames_sorted(self.frames_dir)]
        if not frames:
            report = self._base_report(status="failed", error_message="no prepared frames for DCVC-FM")
            return report

        config = self._resolve_runtime_config(frames)
        invalid = self._validate_config(config)
        if invalid:
            return self._unavailable_report(config, invalid)

        inspect_error, help_stdout_log, help_stderr_log = self._inspect_script(config)
        if inspect_error:
            return self._unavailable_report(config, inspect_error, help_stdout_log, help_stderr_log)

        dataset_config_path, stage_root, sequence_dir = self._stage_frames(frames, config)
        stream_dir = self.output_dir / "stream"
        output_json = self.output_dir / "dcvc_result.json"
        if stream_dir.exists():
            shutil.rmtree(str(stream_dir), ignore_errors=True)
        stream_dir.mkdir(parents=True, exist_ok=True)

        stdout_log = self.output_dir / "dcvc_stdout.txt"
        stderr_log = self.output_dir / "dcvc_stderr.txt"
        command = self._build_command(config, dataset_config_path, stream_dir, output_json)
        started = time.perf_counter()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(
                command,
                cwd=str(config["root"]),
                stdout=out,
                stderr=err,
                text=True,
                check=False,
            )
        wall_sec = float(time.perf_counter() - started)

        report = self._base_report(status="failed", error_message="")
        report["config"] = self._report_config(config)
        report["command"] = list(command)
        report["command_argv"] = list(command)
        report["cwd"] = str(Path(config["root"]).resolve())
        report["stdout_log"] = _relative_path(self.exp_dir, stdout_log)
        report["stderr_log"] = _relative_path(self.exp_dir, stderr_log)
        report["stdout_log_path"] = report["stdout_log"]
        report["stderr_log_path"] = report["stderr_log"]
        report["test_config_path"] = _relative_path(self.exp_dir, dataset_config_path)
        report["staged_input_dir"] = _relative_path(self.exp_dir, stage_root)
        report["resolved_sequence_path"] = str(Path(sequence_dir).resolve())
        report["stream_dir"] = _relative_path(self.exp_dir, stream_dir)
        report["encoded_artifact_dir"] = _relative_path(self.exp_dir, stream_dir)
        report["codec_wall_time_sec"] = float(wall_sec)
        report["enc_time_sec"] = float(wall_sec)
        report["decode_time_sec"] = None
        report["time_semantics"] = (
            "dcvc_fm_official_stream_writer_wall_time; official runner constructs the transmitted "
            "bitstream and performs internal reconstruction for metrics even when decoded-frame saving is disabled"
        )
        report["frame_count"] = int(len(frames))

        if proc.returncode != 0:
            report["error_message"] = "DCVC-FM command failed rc={} stdout={} stderr={}".format(
                int(proc.returncode),
                report["stdout_log"],
                report["stderr_log"],
            )
            return report

        try:
            decoded_dir = self._normalize_decoded_frames(stream_dir, frames) if self.save_decoded_frame else None
            payload_bytes, bitstreams = _sum_files(stream_dir, suffixes=(".bin",))
            if payload_bytes <= 0:
                raise RuntimeError("DCVC-FM produced no bitstream bytes under {}".format(stream_dir))
            reduction = _reduction_pct(self.raw_reference_bytes, payload_bytes)
            report.update(
                {
                    "status": "ok",
                    "error_message": "",
                    "decoded_frames_dir": _relative_path(self.exp_dir, decoded_dir) if decoded_dir is not None else None,
                    "encoded_artifact_path": _relative_path(self.exp_dir, bitstreams[0]) if len(bitstreams) == 1 else None,
                    "encoded_artifact_dir": _relative_path(self.exp_dir, stream_dir),
                    "payload_bytes": int(payload_bytes),
                    "payload_mib": _bytes_to_mib(payload_bytes),
                    "reduction_pct": reduction,
                    "reduction_pct_vs_zip": reduction,
                    "reduction_pct_vs_raw_frames": reduction,
                }
            )
        except Exception as exc:
            report["status"] = "failed"
            report["error_message"] = str(exc)
        return report
