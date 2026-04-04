from __future__ import annotations

from .smolvlm2 import SmolVlm2PromptBackend


def create_backend(model_ref="", max_new_tokens=48):
    # type: (str, int) -> object
    return SmolVlm2PromptBackend(
        model_ref=model_ref,
        max_new_tokens=int(max_new_tokens),
    )
