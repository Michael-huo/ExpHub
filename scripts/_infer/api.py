from __future__ import annotations


def create_backend(
    backend_name,  # type: str
    videox_root,  # type: str
    model_ref="",  # type: str
    backend_python_phase="infer",  # type: str
):
    # type: (...) -> object
    name = str(backend_name or "wan_fun_a14b_inp").strip().lower()
    if name == "wan_fun_a14b_inp":
        from .backends.wan_fun_a14b_inp_backend import WanFunA14BInpBackend

        return WanFunA14BInpBackend(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
    if name == "wan_fun_5b_inp":
        from .backends.wan_fun_5b_inp_backend import WanFun5BInpBackend

        return WanFun5BInpBackend(
            videox_root=videox_root,
            model_ref=model_ref,
            backend_python_phase=backend_python_phase,
        )
    raise SystemExit("[ERR] unsupported infer backend: {}".format(backend_name))
