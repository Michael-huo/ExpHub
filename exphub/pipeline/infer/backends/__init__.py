from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .wan_fun_5b_inp import WanFun5BInpBackend


def create_backend(backend_name, videox_root, model_ref="", backend_python_phase="infer"):
    # type: (str, str, str, str) -> object
    name = str(backend_name or "wan_fun_5b_inp").strip().lower()
    if name != "wan_fun_5b_inp":
        raise SystemExit("[ERR] Phase 1 infer supports only wan_fun_5b_inp: {}".format(backend_name))
    # Temporary transition constructor. Remove after infer binds directly to decode.image_gen.
    return WanFun5BInpBackend(
        videox_root=videox_root,
        model_ref=model_ref,
        backend_python_phase=backend_python_phase,
    )
