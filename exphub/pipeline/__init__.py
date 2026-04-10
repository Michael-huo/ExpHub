"""ExpHub pipeline package.

Current stage ownership:

- input: raw input preparation
- encode: scene split + text generation
- decode: image generation + sequence merge
- eval: slam + metrics + diagnostics
- export: training-set export
"""

from .orchestrator import OrchestrationResult, run

__all__ = ["OrchestrationResult", "run"]
