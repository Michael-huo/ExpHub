__all__ = [
    "PrepareResult",
    "infer_prepare",
    "run_prepare",
    "run",
    "save_prepare_result",
    "train_prepare",
]


def __getattr__(name):
    if name in __all__:
        from . import prepare

        return getattr(prepare, name)
    raise AttributeError(name)
