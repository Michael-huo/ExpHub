def extract_frames(*_args, **_kwargs):
    raise RuntimeError(
        "exphub.input.frames_prepare.extract_frames is deprecated; "
        "use exphub.prepare.prepare infer_prepare/run_prepare instead."
    )

__all__ = ["extract_frames"]
