from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .rchol import rchol

"""Tools for deriving estimands of the Vecchia approximation of a GP.

The Vecchia approximation of a GP is a sparse representation of the precision
matrix of a GP. This module provides tools for deriving the UDB decomposition
of the precision matrix in Proposition 1 of [1].

Classes:
--------
UDB: A named tuple representing the UDB decomposition of the precision matrix.

Functions:
----------
calc_udb: Calculate the UDB decomposition of a precision matrix.

plot_b: Plot the b matrix from a Vecchia UDB decomposition.

[1] Katzfuss & Guinness (2021, _Statistical Science_). A General Framework
    for Vecchia Approximations of Gaussian Processes.
    doi:10.1214/19-STS755.
"""


__all__ = ["UDB", "calc_udb", "plot_b"]


class UDB(NamedTuple):
    """Decomposition of precision matrix U in Proposition 1 of [1].

    [1] Katzfuss & Guinness (2021, _Statistical Science_). A General Framework
        for Vecchia Approximations of Gaussian Processes.
        doi:10.1214/19-STS755.
    """

    u: np.ndarray
    d: np.ndarray
    b: np.ndarray


def calc_udb(A: np.ndarray, prec: bool = False) -> UDB:
    """Calculates the components of U in Proposition 1 of [1].

    [1] Katzfuss & Guinness (2021, _Statistical Science_). A General Framework
        for Vecchia Approximations of Gaussian Processes.
        doi:10.1214/19-STS755.
    """
    if prec:
        A = A
    else:
        L = np.linalg.cholesky(A)
        L_inv = np.linalg.inv(L)
        A = L_inv @ L_inv.T
    u = rchol(A)
    d = np.diag(u)
    b = -(u / d)
    return UDB(u, d, b)


def plot_b(
    idx: int, b: np.ndarray, nbrs: np.ndarray, *args, ax: Axes | None = None, **kwargs
) -> None:
    """Helper to plot the b matrix from a Vecchia UDB decomposition [1].

    Args:
    -----
    idx: int
        The index (spatial location) of the row in the b matrix to plot.

    b: np.ndarray
        The b matrix from a UDB decomposition in a Vecchia approximation.

    nbrs: np.ndarray
        The conditioning sets / neighbors of a Vecchia approximation.

    ax: Axes | None
        The axes to plot on. If None, a new figure is created for plotting.
        This tends to work well for quick plots, but it is best to pass user-
        created Axes object for better control of the plotting contents.

    *args, **kwargs:
        Passed to ax.plot() to control the appearance of the plot with the same
        flexibility as ax.plot().

    [1] Katzfuss & Guinness (2021, _Statistical Science_). A General Framework
        for Vecchia Approximations of Gaussian Processes.
        doi:10.1214/19-STS755.
    """
    if ax is None:
        _, ax = plt.subplots()

    if idx < nbrs.shape[-1]:
        mask = np.where(nbrs[idx] == -1, 0.0, 1.0)

    ax.plot(-b.T[idx, nbrs[idx]] * mask, *args, **kwargs)
