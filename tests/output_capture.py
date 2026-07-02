from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stderr, redirect_stdout


@contextmanager
def captured_stdio():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield stdout, stderr


@contextmanager
def silent_stdio():
    with captured_stdio():
        yield
