"""ExpHub orchestrator package.

This package provides a single entrypoint: `python -m exphub ...`.
It orchestrates existing ExpHub steps
(encode/decode/eval/export/doctor)
while keeping outputs clean via keep_level policies.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
