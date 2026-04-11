from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from exphub.config import get_platform_config
from exphub.common.io import ensure_dir, ensure_file


class InferBackend(object):
    name = ""

    def load(self):
        # type: () -> None
        raise NotImplementedError

    def run(self, request):
        # type: (object) -> dict
        raise NotImplementedError

    def meta(self):
        # type: () -> dict
        return {}


def _split_model_ref(model_ref):
    # type: (str) -> Tuple[Optional[str], Optional[str]]
    ref = str(model_ref or "").strip()
    if not ref:
        return None, None

    if os.path.isabs(ref) or ref.startswith("."):
        return str(Path(ref).expanduser()), None

    candidate = Path(ref).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), None

    return None, ref


def _run_filtered(cmd, cwd, env):
    # type: (List[str], Path, Dict[str, str]) -> int
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None

    tail = deque(maxlen=250)
    for line in proc.stdout:
        tail.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    rc = proc.wait()
    if rc != 0:
        sys.stderr.write("[ERR] infer backend failed (rc={}). Showing last {} lines:\n".format(rc, len(tail)))
        for line in tail:
            sys.stderr.write(line)
    return rc


class ConfiguredInferBackend(InferBackend):
    backend_entry_type = ""
    default_phase = "infer"
    model_config_keys = ()  # type: Tuple[str, ...]

    def __init__(
        self,
        videox_root,  # type: str
        model_ref="",  # type: str
        backend_python_phase="infer",  # type: str
    ):
        # type: (...) -> None
        self.videox_root = Path(str(videox_root)).expanduser().resolve()
        self.model_ref_override = str(model_ref or "").strip()
        self.backend_python_phase = str(backend_python_phase or self.default_phase).strip() or self.default_phase
        self._cfg = get_platform_config()
        self._loaded = False
        self._model_dir = None  # type: Optional[str]
        self._model_id = None  # type: Optional[str]
        self._config_path = None  # type: Optional[str]

    def _get_model_cfg(self):
        # type: () -> Dict[str, object]
        models_cfg = self._cfg.get("models", {})
        if not isinstance(models_cfg, dict):
            models_cfg = {}
        for key in self.model_config_keys:
            item = models_cfg.get(str(key), {})
            if isinstance(item, dict) and item:
                return item
        return {}

    def _resolve_model_ref(self):
        # type: () -> Tuple[Optional[str], Optional[str], Optional[str]]
        model_cfg = self._get_model_cfg()
        config_path = str(model_cfg.get("config", "") or "").strip() or None

        if self.model_ref_override:
            model_dir, model_id = _split_model_ref(self.model_ref_override)
            if model_dir is not None:
                return model_dir, None, config_path
            return None, model_id, config_path

        cfg_path = str(model_cfg.get("path", "") or "").strip()
        cfg_id = str(model_cfg.get("model_id", "") or "").strip()
        if cfg_path:
            return cfg_path, None, config_path
        if cfg_id:
            return None, cfg_id, config_path
        return None, None, config_path

    def load(self):
        # type: () -> None
        ensure_dir(self.videox_root, "videox_root")
        model_dir, model_id, config_path = self._resolve_model_ref()
        if not model_dir and not model_id:
            raise SystemExit(
                "[ERR] infer backend '{}' has no model configured. Set config/platform.yaml or pass --infer_model_dir.".format(
                    self.name
                )
            )

        self._model_dir = model_dir
        self._model_id = model_id
        self._config_path = config_path
        self._loaded = True

    def _build_cmd(self, request):
        # type: (object) -> List[str]
        raise NotImplementedError

    def meta(self):
        # type: () -> dict
        if not self._loaded:
            model_dir, model_id, config_path = self._resolve_model_ref()
        else:
            model_dir = self._model_dir
            model_id = self._model_id
            config_path = self._config_path
        return {
            "infer_backend": str(self.name),
            "model_dir": str(model_dir) if model_dir else None,
            "model_id": str(model_id) if model_id else None,
            "config_path": str(config_path) if config_path else None,
            "backend_python_phase": str(self.backend_python_phase),
            "backend_entry_type": str(self.backend_entry_type),
        }

    @staticmethod
    def _mean_deploy_gap(segments):
        # type: (List[Dict[str, object]]) -> float
        if not segments:
            return 0.0
        total = 0.0
        for item in segments:
            try:
                total += float(int(item.get("deploy_gap", 0)))
            except Exception:
                total += 0.0
        return total / float(len(segments))

    @staticmethod
    def _segment_seconds(request):
        # type: (object) -> float
        fps = max(int(request.fps), 1)
        segments = list(request.execution_segments or [])
        if segments:
            try:
                first_gap = int(segments[0].get("deploy_gap", 0))
                if first_gap > 0:
                    return float(first_gap) / float(fps)
            except Exception:
                pass
        return float(max(int(request.kf_gap), 0)) / float(fps)


class DirectInferBackend(ConfiguredInferBackend):
    backend_entry_type = "direct_backend"

    @staticmethod
    def _runs_parent(request):
        value = getattr(request, "runs_parent", None)
        if value is not None:
            return str(Path(value).resolve())
        return str((Path(request.exp_dir).resolve() / "infer").resolve())

    def _build_cmd(self, request):
        # type: (object) -> List[str]
        cmd = [
            "--gpus",
            str(int(request.gpus)),
            "--batch",
            "--frames_dir",
            str(request.frames_dir),
            "--dataset_fps",
            str(int(request.fps)),
            "--fps",
            str(int(request.fps)),
            "--kf_gap",
            str(int(request.kf_gap)),
            "--segment_seconds",
            "{:.9f}".format(self._segment_seconds(request)),
            "--base_idx",
            str(int(request.base_idx)),
            "--num_segments",
            str(int(request.num_segments)),
            "--seed_base",
            str(int(request.seed_base)),
            "--prompt_file",
            str(request.prompt_file_path),
            "--execution_plan",
            str(request.execution_plan_path),
            "--runs_parent",
            self._runs_parent(request),
            "--exp_name",
            "runs",
        ]
        if self._model_dir:
            cmd += ["--model_name", str(self._model_dir)]
        elif self._model_id:
            cmd += ["--model_name", str(self._model_id)]
        if self._config_path:
            cmd += ["--config_path", str(self._config_path)]

        cmd += list(request.infer_extra or [])
        return cmd

    def _run_direct(self, argv):
        # type: (List[str]) -> None
        raise NotImplementedError

    def run(self, request):
        # type: (object) -> dict
        if not self._loaded:
            self.load()

        old_pp = os.environ.get("PYTHONPATH", "")
        old_sys_path = list(sys.path)
        os.environ["PYTHONPATH"] = str(self.videox_root) + (os.pathsep + old_pp if old_pp else "")
        if str(self.videox_root) not in sys.path:
            sys.path.insert(0, str(self.videox_root))
        try:
            self._run_direct(self._build_cmd(request))
        finally:
            sys.path[:] = old_sys_path
            if old_pp:
                os.environ["PYTHONPATH"] = old_pp
            elif "PYTHONPATH" in os.environ:
                del os.environ["PYTHONPATH"]
        return self.meta()


class SubprocessInferBackend(ConfiguredInferBackend):
    backend_entry_type = "subprocess_wrapper"

    def __init__(
        self,
        videox_root,  # type: str
        model_ref="",  # type: str
        backend_python_phase="infer",  # type: str
    ):
        # type: (...) -> None
        super(SubprocessInferBackend, self).__init__(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
        self.impl_script = Path(inspect.getfile(self.__class__)).resolve()

    def load(self):
        # type: () -> None
        super(SubprocessInferBackend, self).load()
        ensure_file(self.impl_script, "infer_impl")

    def _build_cmd(self, request):
        # type: (object) -> List[str]
        py_exec = sys.executable if getattr(sys, "executable", "") else "python3"
        if int(request.gpus) > 1:
            cmd = [
                py_exec,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node={}".format(int(request.gpus)),
                str(self.impl_script),
            ]
        else:
            cmd = [py_exec, str(self.impl_script)]

        cmd += [
            "--gpus",
            str(int(request.gpus)),
            "--batch",
            "--frames_dir",
            str(request.frames_dir),
            "--dataset_fps",
            str(int(request.fps)),
            "--fps",
            str(int(request.fps)),
            "--kf_gap",
            str(int(request.kf_gap)),
            "--segment_seconds",
            "{:.9f}".format(self._segment_seconds(request)),
            "--base_idx",
            str(int(request.base_idx)),
            "--num_segments",
            str(int(request.num_segments)),
            "--seed_base",
            str(int(request.seed_base)),
            "--prompt_file",
            str(request.prompt_file_path),
            "--execution_plan",
            str(request.execution_plan_path),
            "--runs_parent",
            DirectInferBackend._runs_parent(request),
            "--exp_name",
            "runs",
        ]
        if self._model_dir:
            cmd += ["--model_name", str(self._model_dir)]
        elif self._model_id:
            cmd += ["--model_name", str(self._model_id)]
        if self._config_path:
            cmd += ["--config_path", str(self._config_path)]

        cmd += list(request.infer_extra or [])
        return cmd

    def run(self, request):
        # type: (object) -> dict
        if not self._loaded:
            self.load()

        env = os.environ.copy()
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.videox_root) + (os.pathsep + old_pp if old_pp else "")

        cmd = self._build_cmd(request)
        rc = _run_filtered(cmd, cwd=self.videox_root, env=env)
        if rc != 0:
            raise SystemExit(rc)
        return self.meta()
