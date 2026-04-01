from __future__ import annotations

from .smolvlm2 import SmolVlm2PromptBackend


def create_backend(backend_name, model_ref="", dtype="bfloat16", max_new_tokens=48):
    # type: (str, str, str, int) -> object
    name = str(backend_name or "smolvlm2").strip().lower()
    if name == "smolvlm2":
        return SmolVlm2PromptBackend(
            model_ref=model_ref,
            dtype=dtype,
            max_new_tokens=int(max_new_tokens),
        )
    raise SystemExit("[ERR] unsupported formal prompt backend: {}".format(backend_name))
