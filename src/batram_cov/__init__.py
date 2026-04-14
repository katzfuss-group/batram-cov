from __future__ import annotations

__version__ = "0.1.0"

import warnings
from collections.abc import Callable
from typing import NamedTuple

try:
    import jax
except ImportError:
    raise ImportError(
        "batram-cov requires JAX. Install with an extra:\n"
        "  uv sync --extra cpu    # CPU (uv project)\n"
        "  uv sync --extra cuda   # GPU (uv project)\n"
        "  pip install 'batram-cov[cpu]'   # CPU (pip)\n"
        "  pip install 'batram-cov[cuda]'  # GPU (pip)"
    ) from None
