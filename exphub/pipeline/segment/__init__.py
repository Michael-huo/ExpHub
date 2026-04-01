def run(runtime):
    from .service import run as _run

    return _run(runtime)


__all__ = ["run"]
