#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .wan_fun_runtime import WanFunInferBackend, WanFunRuntimeProfile, run_wan_fun_backend_cli
except Exception:
    _SCRIPTS_DIR = Path(__file__).resolve().parents[2]
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from _infer.backends.wan_fun_runtime import WanFunInferBackend, WanFunRuntimeProfile, run_wan_fun_backend_cli


WAN_FUN_A14B_BACKEND_PROFILE = WanFunRuntimeProfile(
    backend_name="wan_fun_a14b_inp",
    profile_name="wan_fun_a14b_inp",
    default_phase="infer",
    model_config_keys=("wan2_2_fun_a14b_inp", "wan2_2"),
    gpu_memory_mode="model_cpu_offload_and_qfloat8",
    prefer_quantization=True,
    compile_dit=False,
    enable_teacache=True,
    teacache_threshold=0.10,
    cfg_skip_ratio=0.0,
    fsdp_text_encoder=True,
    backend_entry_type="wan_fun_runtime",
)


class WanFunA14BInpBackend(WanFunInferBackend):
    name = "wan_fun_a14b_inp"
    default_phase = "infer"
    model_config_keys = ("wan2_2_fun_a14b_inp", "wan2_2")
    backend_profile = WAN_FUN_A14B_BACKEND_PROFILE


def run_compat_cli(argv=None):
    # type: (object) -> None
    run_wan_fun_backend_cli(argv, backend_profile=WAN_FUN_A14B_BACKEND_PROFILE)


def main():
    # type: () -> None
    run_wan_fun_backend_cli(backend_profile=WAN_FUN_A14B_BACKEND_PROFILE)


if __name__ == "__main__":
    main()
