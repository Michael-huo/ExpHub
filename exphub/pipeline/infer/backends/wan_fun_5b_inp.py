#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from exphub.pipeline.infer.backends.wan_runtime import WanFunInferBackend, WanFunRuntimeProfile, run_wan_fun_backend_cli
else:
    from .wan_runtime import WanFunInferBackend, WanFunRuntimeProfile, run_wan_fun_backend_cli


WAN_FUN_5B_BACKEND_PROFILE = WanFunRuntimeProfile(
    backend_name="wan_fun_5b_inp",
    profile_name="wan_fun_5b_inp",
    default_phase="infer_fun_5b",
    model_config_keys=("wan2_2_fun_5b_inp",),
    gpu_memory_mode="model_cpu_offload",
    prefer_quantization=False,
    compile_dit=False,
    enable_teacache=True,
    teacache_threshold=0.10,
    cfg_skip_ratio=0.0,
    fsdp_text_encoder=True,
    backend_entry_type="wan_fun_runtime",
)


class WanFun5BInpBackend(WanFunInferBackend):
    name = "wan_fun_5b_inp"
    default_phase = "infer_fun_5b"
    model_config_keys = ("wan2_2_fun_5b_inp",)
    backend_profile = WAN_FUN_5B_BACKEND_PROFILE


def run_wan_fun_5b_backend_cli(argv=None):
    # type: (object) -> None
    run_wan_fun_backend_cli(argv, backend_profile=WAN_FUN_5B_BACKEND_PROFILE)


def main():
    # type: () -> None
    run_wan_fun_5b_backend_cli()


if __name__ == "__main__":
    main()
