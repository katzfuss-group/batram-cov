from __future__ import annotations

__version__ = "0.1.0"

from importlib.util import find_spec

import numpy as np

if find_spec("jax") is None:
    raise ImportError(
        "batram-cov requires JAX. Install with an extra:\n"
        "  uv sync --extra cpu    # CPU (uv project)\n"
        "  uv sync --extra cuda   # GPU (uv project)\n"
        "  pip install 'batram-cov[cpu]'   # CPU (pip)\n"
        "  pip install 'batram-cov[cuda]'  # GPU (pip)"
    ) from None


def calc_li(locs, nn):
    li = np.zeros(locs.shape[0])

    for i in range(1, locs.shape[0]):
        loc_i = locs[i]
        loc_nn = locs[nn[i, 0]]
        li[i] = np.linalg.norm(loc_i - loc_nn)

    li[0] = li[1] ** 2 / li[5]
    li /= li[0]

    return li
