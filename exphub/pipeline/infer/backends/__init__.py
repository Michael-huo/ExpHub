from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .wan_fun_5b_inp import WanFun5BInpBackend
from .wan_fun_a14b_inp import WanFunA14BInpBackend


def create_backend(backend_name, videox_root, model_ref="", backend_python_phase="infer"):
    # type: (str, str, str, str) -> object
    name = str(backend_name or "wan_fun_5b_inp").strip().lower()
    if name == "wan_fun_a14b_inp":
        return WanFunA14BInpBackend(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
    if name == "wan_fun_5b_inp":
        return WanFun5BInpBackend(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
    raise SystemExit("[ERR] unsupported formal infer backend: {}".format(backend_name))
