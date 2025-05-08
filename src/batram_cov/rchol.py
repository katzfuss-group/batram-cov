import numpy as np


def cov_exponential(d, sigma, range):
    "exponential covariance function"
    return sigma**2 * np.exp(-d / range)


def rev_mat(x):
    """
    Calculates the reverse matrix.

    See Section 4.1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.
    """
    return x[-1::-1, -1::-1]


def rchol(x):
    """
    reverse cholesky factor

    Defined in Sec 4.1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.

    """
    return rev_mat(np.linalg.cholesky(rev_mat(x)))


class BDDecomposition:
    """
    Calculates u, d, b as defined in Proposition 1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.
    """

    def __init__(self, cov: np.array, igi: list[tuple[int, np.s_]]):
        self._len_i = len(igi)
        self.igi = igi
        self.bs, self.ds = _calc_bs_ds(cov, igi)
        self.ds = np.array(self.ds)

    def calc_u(self):
        u = np.zeros((self._len_i, self._len_i))
        ds_sqrt_inv = 1.0 / np.sqrt(self.ds)
        np.fill_diagonal(u, ds_sqrt_inv)
        for i in range(self._len_i):
            b = self.bs[i]
            if not b.size:
                continue
            gi = self.igi[i][1]

            u[gi, i] = -b * ds_sqrt_inv[i]
        return u


def _calc_b_d(cov: np.array, i: int, gi: np.s_) -> tuple[np.array, np.array]:
    if (gi == -1).all():
        return np.array([]), cov[i, i]

    kii = np.atleast_2d(cov[i, i])
    kig = np.atleast_2d(cov[i, gi])
    kgg = np.atleast_2d(cov[gi, :][:, gi])
    b = kig @ np.linalg.inv(kgg)
    d = kii - b @ kig.T
    return b.squeeze(), d.squeeze()


def _calc_bs_ds(
    cov: np.array, igi: list[tuple[int, np.s_]]
) -> tuple[np.array, np.array]:
    bs, ds = [], []
    for i, gi in igi:
        b, d = _calc_b_d(cov, i, gi)
        bs.append(b)
        ds.append(d)

    return bs, ds


def calc_u_d_b(x):
    """
    @isa ignore me


    Calculates u, d, b as defined in Proposition 1 in [1].

    [1] Katzfuss, and Guinness. A General Framework for Vecchia Approximations
    of Gaussian Processes. Statistical Science 36, no. 1 (February 1, 2021).
    https://doi.org/10.1214/19-STS755.
    """

    u_direct = rchol(x)
    d = u_direct.diagonal() ** (-2)
    u = u_direct * d**0.5
    return u_direct, d, -u
