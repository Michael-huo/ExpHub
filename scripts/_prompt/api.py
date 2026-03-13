from __future__ import annotations

from typing import Optional


def create_backend(
    backend_name,  # type: str
    model_ref="",  # type: str
    dtype="bfloat16",  # type: str
    use_fast=False,  # type: bool
    min_pixels=0,  # type: int
    max_pixels=0,  # type: int
    max_new_tokens=48,  # type: int
):
    # type: (...) -> object
    name = str(backend_name or "qwen").strip().lower()
    if name == "qwen":
        from .backends.qwen_backend import QwenPromptBackend

        return QwenPromptBackend(
            model_ref=model_ref,
            use_fast=bool(use_fast),
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
            max_new_tokens=int(max_new_tokens),
        )
    if name == "smolvlm2":
        from .backends.smolvlm2_backend import SmolVlm2PromptBackend

        return SmolVlm2PromptBackend(
            model_ref=model_ref,
            dtype=dtype,
            max_new_tokens=int(max_new_tokens),
        )
    raise SystemExit("[ERR] unsupported prompt backend: {}".format(backend_name))
