from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
from scipy import linalg, special

"""A simple Gaussian process library for sample generation.

Data generation for various experiments requires obtaining draws from a Gaussian
process. This module provides a GP class and kernel functions to easily generate
samples from Matern fields. The GP itself minimally wraps a kernel function and
stores the parameters required to generate samples.

Classes:
--------
GP: A simple Gaussian process class for generating samples.

Functions:
----------
make_grid: Make a grid of equally spaced points in a unit hypercube.

cdist: Computes the Euclidean distance between two sets of points.

kernel: Calculates the kernel between two sets of points.
"""

__all__ = ["GP", "VecchiaGP", "make_grid", "cdist", "kernel"]

type MappedBD = Iterable[tuple[np.ndarray, np.ndarray]]
type Samples = np.ndarray
type Means = np.ndarray
type CondVars = np.ndarray
type VecchiaResult = tuple[Samples, Means, CondVars]


class _GP(ABC):
    """ABC for a GP generator class.

    This suite of classes is designed primarily for data generation, not for
    training or inference.
    """

    locs: np.ndarray
    kernel: Callable[..., np.ndarray]
    params: dict

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sample method must be implemented.")

    def update_params(self, **params):
        """Update the parameters for sample generation."""
        self.params.update(params)


class GP(_GP):
    """A wrapper around parameterized kernel functions for sample generation.

    Attributes:
    -----------
    kernel: Callable[[...], np.ndarray]
        The kernel function to use for sample generation. See the `kernel` function
        provided here for more details, or provide a custom kernel which returns
        numpy arrays.

    params: dict
        The parameters to pass to the kernel function when generating samples.
        Parameters may affect the anisotropy, nugget, scale (amplitude), characteristic
        length scale, and smoothness of the kernel. See the `kernel` function for
        more details.

    Methods:
    --------
    update_params: Update the parameters for sample generation. Any method call
        can update the parameters for the kernel function. They will be stored
        in the dict.

    sample: Generate samples from the Gaussian process.
    """

    def __init__(
        self,
        locs: np.ndarray,
        kernel: Callable[..., np.ndarray],
        seed: int | None = None,
        **params,
    ):
        super().__init__(seed)
        self.locs = locs
        self.kernel = kernel
        self.params = params

    def sample(self, n: int = 1, seed: int = 42, **params) -> np.ndarray:
        """Generate samples from the Gaussian process.

        Args:
        -----
        s: np.ndarray
            The set of points in 2d space to generate samples at.

        n: int (default=1)
            The number of samples to generate.

        seed: int (default=42)
            The random seed for sample generation.

        **params:
            Additional parameters to pass to the kernel function when generating
            samples. These will be updated in the stored parameters attribute
            of the class.

        Returns:
        --------
        np.ndarray
            An array of shape (n, len(s)) containing the samples at the points s.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        s = self.locs
        self.update_params(**params)
        kernel = self.kernel(s, s, **self.params)
        return rng.multivariate_normal(
            np.zeros(len(s)), kernel, size=n, check_valid="ignore"
        )


class VecchiaGP(_GP):
    """A Gaussian process data generator based on a Vecchia approximation.

    In some cases we may wish to generate samples from a Vecchia GP instead of
    the normal GP. This class provides tools for that use case.
    """

    def __init__(
        self,
        locs: np.ndarray,
        nbrs: np.ndarray,
        kernel: Callable[..., np.ndarray],
        seed: int | None = None,
        **params,
    ):
        """Initialize a VecchiaGP instance.

        Args:
        -----
        locs: np.ndarray
            The set of points in 2d space to generate samples at.

        nbrs: np.ndarray
            The nearest neighbors for each location in locs.

        kernel: Callable[[...], np.ndarray]
            The kernel function to use for sample generation. See the `kernel`
            function provided for more details, or provide a custom kernel which
            returns numpy arrays.
        """
        super().__init__(seed)
        locs_with_nbrs = locs[nbrs] * (nbrs != -1)[..., None]
        self.locs = locs
        self.nbrs = nbrs
        self.nbrs_locs = locs_with_nbrs
        self.kernel = kernel
        self.params = params

    @staticmethod
    def calc_db(s, s_nbrs, cov_fn) -> MappedBD:
        s = [s[i, None, :] for i in range(len(s))]
        s_nbrs = [s_nbrs[i, :i] for i in range(len(s_nbrs))]

        def apply(k00, k01, k11, i):
            L = linalg.cholesky(k00, lower=True)
            b = linalg.cho_solve((L, True), k01).T if i > 0 else np.zeros_like(k01)
            d = k11 - np.dot(b, k01) if i > 0 else k11
            return (b.squeeze(), d.squeeze())

        def k00_covfn(s, i):
            if s[:i].size == 0:
                return np.array([[1]])
            else:
                return cov_fn(s[:i], s[:i])

        def k01_covfn(s_nbrs, s, i):
            if s_nbrs[:i].size == 0:
                return np.array([[1]])
            else:
                return cov_fn(s_nbrs[:i], s)

        k00 = map(k00_covfn, s_nbrs, range(len(s)))
        k01 = map(k01_covfn, s_nbrs, s, range(len(s)))
        k11 = map(cov_fn, s, s)
        bd = map(apply, k00, k01, k11, range(len(s)))

        return bd

    @staticmethod
    def vecchia_sampler(
        bd: MappedBD,
        nbrs: np.ndarray,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        def get_U(bd, nbrs):
            N = nbrs.shape[0]
            U = np.zeros((N, N))
            for i, (b, d) in enumerate(bd):
                U[nbrs[i, :i], i] = -b / np.sqrt(d)
                U[i, i] = 1 / np.sqrt(d)
            return U

        z = rng.normal(size=(nbrs.shape[0], num_samples))
        U = get_U(bd, nbrs)
        return linalg.solve_triangular(U.T, z, lower=True)

    @staticmethod
    def ar_vecchia_sampler(
        bd: MappedBD,
        nbrs: np.ndarray,
        num_samples: int,
        mean_transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rng: np.random.Generator,
    ) -> VecchiaResult:
        """Sample from a Vecchia GP using an autoregressive sampler.

        Draws samples from a Vecchia GP using the autoregressive formulation of
        the model. This version also supports applying a transformation model
        to the mean (in addition to the linear mean), which allows for drawing
        samples that are marginally Gaussian with non-Gaussian densities.
        """

        def _mean(yh, b):
            return 0

        bs, ds = [], np.empty(nbrs.shape[0])
        y = np.empty((nbrs.shape[0], num_samples))
        means = np.zeros_like(y)
        mean_transform = mean_transform or _mean

        for i, (b, d) in enumerate(bd):
            condset = nbrs[i, :i]
            yh = y[condset]
            mean = np.dot(yh.T, b).squeeze() if i > 0 else 0
            mt = mean_transform(yh.T, b) if i > 0 else 0
            mean = mean + mt
            means[i] = mean
            ds[i] = d
            if d < 0:
                raise RuntimeError(
                    f"Negative variance {d = :.3f} for {i = } in AR Vecchia sampler."
                )
            y[i] = rng.normal(loc=mean, scale=np.sqrt(d), size=num_samples)
            bs.append(b)
        return y.T, means.T, ds

    def sample(
        self,
        n: int,
        seed: int | None = None,
        mean_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        **params,
    ) -> VecchiaResult:
        """Draw sample from the gp

        Arguments:
        ---
        :param:``

        Returns:
        ----
        :param:`samples (np.ndarray)`
        :param:`means (np.ndarray)`
        :param:`vars (np.ndarray)`
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        s = self.locs
        s_nbrs = self.nbrs_locs
        params = {**self.params, **params}
        cov_fn = partial(self.kernel, **params)
        bd = self.calc_db(s, s_nbrs, cov_fn)
        return self.ar_vecchia_sampler(
            bd=bd,
            nbrs=self.nbrs,
            num_samples=n,
            mean_transform=mean_transform,
            rng=rng,
        )


def make_grid(nlocs: int, ndims: int) -> np.ndarray:
    """Make a grid of equally spaced points in a unit hypercube.

    Args:
        nlocs: np.ndarrayhe number of locations in each dimension.
        ndims: np.ndarrayhe number of dimensions.

    Returns:
        A numpy array of shape (nlocs**ndims, ndims) containing the locations
        of the data points.
    """
    _ = np.linspace(0, 1, nlocs)
    return np.stack(np.meshgrid(*[_] * ndims), axis=-1).reshape(-1, ndims)


def _get_anisotropy(theta: float, scales: float | np.ndarray = 1.0) -> np.ndarray:
    """Builds a 2d rotation matrix from an angle theta."""
    if isinstance(scales, float):
        scales = scales * np.ones(2)
    if isinstance(scales, list):
        assert len(scales) == 2
        scales = np.array(scales)

    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([cos, -sin, sin, cos]).reshape(2, 2)

    return rot * scales


def cdist(s1: np.ndarray, s2: np.ndarray, A: np.ndarray | None = None) -> np.ndarray:
    """Computes the Euclidean distance between two sets of points."""
    diffs = s1[:, np.newaxis, :] - s2[np.newaxis, :, :]

    if A is not None:
        diffs = diffs @ A

    dists = np.sum(diffs**2, axis=-1)

    return np.sqrt(dists)


def _kv_matern(d: np.ndarray, nu: float) -> np.ndarray:
    """Calculates the Matern kernel using Bessel functions of the second kind."""
    d_ = np.sqrt(2 * nu) * d
    scale = (1 - nu) * np.log(2) - special.loggamma(nu)
    scale = np.exp(scale)
    kv = special.kv(nu, d_)
    kv = np.where(np.isinf(kv), np.nan, kv)
    kernel = scale * d_**nu * kv
    kernel = np.where(np.isnan(kernel), 1.0, kernel)
    return kernel


def _matern(d: np.ndarray, nu: float) -> np.ndarray:
    """Computes a Matern kernel with smoothness nu.

    The smoothness can be arbitrary. For half integers, the kernel is computed
    using polynomials. For other values, the kernel is computed using Bessel
    functions of the second kind (see `scipy.special.kv`).
    """
    if nu == 0.5:
        return np.exp(-d)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    elif nu == 2.5:
        return (1 + np.sqrt(5) * d + 5 / 3 * d**2) * np.exp(-np.sqrt(5) * d)
    else:
        type_conditions = (
            isinstance(nu, int | float),
            isinstance(nu, np.ndarray) and nu.size == 1,
        )
        if not any(type_conditions) and nu <= 0:
            raise TypeError(
                "The smoothness must be a positive number, got %s of type %s."
                % (nu, type(nu))
            )
        return _kv_matern(d, nu)


def kernel(s1: np.array, s2: np.array, warnings: bool = True, **params) -> np.ndarray:
    """Calculates the kernel between two sets of points.

    Args:
    -----
    s1, s2: np.ndarray
        The sets of points in 2d space.

    nu: float (default=2.5)
        The smoothness of the kernel.

    theta: float (default=0.0)
        The angle of anisotropy.

    ls: float (default=1.0)
        The length scale.

    scale: float (default=1.0)
        The scale of the kernel.

    ascales: np.ndarray (default=[1.0, 1.0])
        The anisotropy scaling factors of the kernel.

    nugget: float (default=0.0)
        The nugget of the kernel when s1 == s2.

    warnings: bool (default=True)
        Whether to print warnings for unused keyword arguments.

    Returns:
    --------
    np.ndarray
        The kernel matrix.
    """
    kwargs = {"nu", "theta", "ls", "scale", "ascales", "nugget"}
    if warnings and not set(params.keys()).issubset(kwargs):
        for key in set(params.keys()) - kwargs:
            print("Invalid keyword argument %s not used by kernel." % key)

    ls = params.get("ls", 1.0)
    s1 = s1.copy() / np.sqrt(ls)
    s2 = s2.copy() / np.sqrt(ls)

    theta = params.get("theta", 0.0)
    scaling = params.get("ascales", 1.0)
    R = _get_anisotropy(theta, scaling)
    d = cdist(s1, s2, R)

    scale = params.get("scale", 1.0)
    nu = params.get("nu", 2.5)
    kernel = scale * _matern(d, nu)

    if np.equal(s1, s2).all():
        nugget = params.get("nugget", 0.0) * np.eye(len(s1))
        kernel += nugget

    return kernel
