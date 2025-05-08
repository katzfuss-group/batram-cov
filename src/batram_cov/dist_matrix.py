"""
dist_matrix: A module for computing the pairwise Euclidean distance between two
sets of vectors.
"""

from jax import Array, jit
from jax.numpy import linalg, prod, sqrt, square, sum

__all__ = ["dist", "dist2"]


def batch_centering(x1: Array, x2: Array) -> tuple[Array, Array]:
    """Center two sets of vectors by the mean of their values.

    Batch dimensions are preserved if they exist.

    ARGS
    ----
    x1 (Array): A batch of vectors with shape (*batch, n1, d)

    x2 (Array): A batch of vectors with shape (*batch, n2, d)

    RETURNS
    -------
    tuple[Array, Array]: The centered versions of x1 and x2
    """
    n1, n2 = prod(x1.shape[-2]), prod(x2.shape[-2])
    m1 = x1.mean(axis=tuple(_ for _ in range(x1.ndim - 1)))
    m2 = x2.mean(axis=tuple(_ for _ in range(x2.ndim - 1)))
    m = (n1 * m1 + n2 * m2) / (n1 + n2)

    return x1 - m, x2 - m


def batched_diffs(x1: Array, x2: Array) -> Array:
    """Compute pairwise difference between two sets of vectors.

    Batch dimensions are preserved if they exist.

    ARGS
    ----
    x1 (Array): A batch of vectors with shape (*batch, n1, d)

    x2 (Array): A batch of vectors with shape (*batch, n2, d)

    RETURNS
    -------
    tuple[Array, Array]: The centered versions of x1 and x2
    """
    return x1[..., None, :] - x2[..., None, :, :]


@jit
def dist(x1: Array, x2: Array) -> Array:
    """Compute the pairwise Euclidean distance between two sets of vectors.

    NOTES
    -----
    Batch dimensions are preserved if they exist.

    Uses naive implementation of distance computation.

    ARGS
    ----
    x1 (Array): A batch of vectors with shape (*batch, n1, d)

    x2 (Array): A batch of vectors with shape (*batch, n2, d)

    RETURNS
    -------
    tuple[Array, Array]: The centered versions of x1 and x2
    """
    x1, x2 = batch_centering(x1, x2)
    d = batched_diffs(x1, x2)
    d = square(d)
    sq_diffs: Array = sum(d, axis=-1)
    dists = sqrt(sq_diffs)

    return dists


@jit
def dist2(x1: Array, x2: Array) -> Array:
    """Compute the pairwise Euclidean distance between two sets of vectors.

    NOTES
    -----
    Batch dimensions are preserved if they exist.

    Uses jax.numpy.linalg.norm to compute the distance.

    ARGS
    ----
    x1 (Array): A batch of vectors with shape (*batch, n1, d)

    x2 (Array): A batch of vectors with shape (*batch, n2, d)

    RETURNS
    -------
    tuple[Array, Array]: The centered versions of x1 and x2
    """
    x1, x2 = batch_centering(x1, x2)
    d = batched_diffs(x1, x2)
    dists: Array = linalg.norm(d, axis=-1)

    return dists
