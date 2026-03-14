from __future__ import annotations

import sys
from pathlib import Path

try:
    from .wan_fun_a14b_inp_backend import WanFunA14BInpBackend, run_wan_fun_backend_cli
except Exception:
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2]
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from _infer.backends.wan_fun_a14b_inp_backend import WanFunA14BInpBackend, run_wan_fun_backend_cli


class WanFun5BInpBackend(WanFunA14BInpBackend):
    name = "wan_fun_5b_inp"
    default_phase = "infer_fun_5b"
    model_config_keys = ("wan2_2_fun_5b_inp",)


def main():
    # type: () -> None
    run_wan_fun_backend_cli()


if __name__ == "__main__":
    main()
