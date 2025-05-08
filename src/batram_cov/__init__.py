from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.linalg
import tensorflow_probability.substrates.jax as tfp
from jax import Array
from scipy import linalg
from sklearn.gaussian_process import kernels
from tqdm import tqdm
from veccs import orderings
from veccs.utils import inverse_permutation

__version__ = "0.1.0"


class Data(NamedTuple):
    locs: np.ndarray | Array
    nearest_neighbors: np.ndarray | Array
    li: np.ndarray | Array
    maxmin_permutation: np.ndarray | Array
    inv_maxmin_permuation: np.ndarray | Array
    x: np.ndarray | Array
    nus: np.ndarray | Array
    cond_vars: np.ndarray | Array
    samples: np.ndarray | Array
    noise_var: float | Array
    seed: int | Array
    description: str

    def get_description(self) -> str:
        if isinstance(self.description, str):
            return self.description
        else:
            return self.description.tobytes().decode("utf-8")

    def to_jax(self) -> JaxData:
        d = self._asdict()
        d.pop("cov_fn", None)
        d = jax.tree.map(
            lambda a: jax.device_put(a) if isinstance(a, jnp.ndarray) else a, d
        )

        return JaxData(**d)


class RegressionProblem(NamedTuple):
    location: int
    li: float
    x: np.ndarray
    response: np.ndarray
    y_cs: np.ndarray
    nn_idx: np.ndarray


class Data2(NamedTuple):
    locs: np.ndarray
    nearest_neighbors: np.ndarray
    li: np.ndarray
    maxmin_permutation: np.ndarray
    inv_maxmin_permuation: np.ndarray
    x: np.ndarray
    cond_vars: np.ndarray
    samples: np.ndarray
    noise_var: float
    seed: int
    description: str
    cov_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def to_jax(self) -> JaxData:
        d = self._asdict()
        d.pop("cov_fn", None)
        d = jax.tree.map(
            lambda a: jax.device_put(a) if isinstance(a, jnp.ndarray) else a, d
        )

        return JaxData(**d)

    def extract_regression_problem(self, location: int) -> RegressionProblem:
        nn_idx = self.nearest_neighbors[location, :]
        nn_idx = nn_idx[nn_idx >= 0]
        y_cs = self.samples[:, 0, nn_idx]
        return RegressionProblem(
            location=location,
            li=self.li[location],
            x=self.x,
            response=self.samples[:, 0, location],
            y_cs=y_cs,
            nn_idx=nn_idx,
        )


class JaxData(NamedTuple):
    locs: Array
    nearest_neighbors: Array
    li: Array
    maxmin_permutation: Array
    inv_maxmin_permuation: Array
    x: Array
    cond_vars: Array
    samples: Array
    noise_var: Array
    seed: Array
    description: str


def matern(
    x: np.ndarray,
    noise: float = 0.0,
    ls: float = 0.25,
    nu: float = 1.5,
) -> np.ndarray:
    matern = kernels.Matern(nu=nu, length_scale=ls)
    return matern(x) + noise * np.eye(len(x))


def make_locs(n):
    _ = np.linspace(0, 1, n)
    x, y = np.meshgrid(_, _)
    locs = np.stack([x.flatten(), y.flatten()], axis=1)
    maxmin_ordering = orderings.maxmin_cpp(locs)
    return locs[maxmin_ordering], maxmin_ordering


def scipy_ldl(kernel: np.ndarray) -> np.ndarray:
    _, d, _ = linalg.ldl(kernel, lower=True)
    return d.diagonal()


def calc_li(locs, nn):
    li = np.zeros(locs.shape[0])

    for i in range(1, locs.shape[0]):
        loc_i = locs[i]
        loc_nn = locs[nn[i, 0]]
        li[i] = np.linalg.norm(loc_i - loc_nn)

    li[0] = li[1] ** 2 / li[5]
    li /= li[0]

    return li


def gen_data2(
    xs: np.ndarray,
    cov_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    nsamples: int = 1,
    nlocs_side: int = 32,
    noise_var: float = 0.0,
    max_nn: int = 20,
    seed=0,
    description: str = "",
) -> Data2:
    if nsamples > 1:
        raise ValueError("Only one sample is supported, but nsamples != 1.")

    rng = np.random.default_rng(seed)
    locs, perm = make_locs(nlocs_side)
    ds = np.zeros((len(xs), len(locs)))
    samples = np.zeros((len(xs), nsamples, len(locs)))

    for i, x in enumerate(xs):
        cov = cov_fn(locs, x)
        if noise_var > 0.0:
            cov += noise_var * np.eye(len(locs))

        ds[i] = scipy_ldl(cov)
        samples[i] = rng.multivariate_normal(
            mean=np.zeros(len(locs)), cov=cov, size=nsamples
        )

    nn = orderings.find_nns_l2(locs, max_nn=max_nn)

    return Data2(
        locs=locs,
        nearest_neighbors=nn,
        li=calc_li(locs, nn),
        maxmin_permutation=perm,
        inv_maxmin_permuation=inverse_permutation(perm),
        x=xs[:, np.newaxis],
        cond_vars=ds,
        samples=samples,
        seed=seed,
        noise_var=noise_var,
        description=description,
        cov_fn=cov_fn,
    )


def gen_data(nus, nsamples=1, nlocs_side=32, noise_var=0.0, max_nn=20, seed=0) -> Data:
    rng = np.random.default_rng(seed)
    locs, perm = make_locs(nlocs_side)
    d = np.zeros((len(nus), len(locs)))
    samples = np.zeros((len(nus), nsamples, len(locs)))
    for i, nu in enumerate(nus):
        cov = matern(x=locs, nu=nu, noise=noise_var)
        d[i] = scipy_ldl(cov)
        samples[i] = rng.multivariate_normal(
            mean=np.zeros(len(locs)), cov=cov, size=nsamples
        )

    nn = orderings.find_nns_l2(locs, max_nn=max_nn)

    # TODO: remove the dict to named tuple conversion below
    return Data(
        **{
            "locs": locs,
            "nearest_neighbors": nn,
            "li": calc_li(locs, nn),
            "maxmin_permutation": perm,
            "inv_maxmin_permuation": inverse_permutation(perm),
            "nus": nus,
            "x": np.log(nus)[:, np.newaxis],
            "cond_vars": d,
            "samples": samples,
            "seed": seed,
            "noise_var": noise_var,
            "description": "varies smoothness nu",
        }
    )


class Params(NamedTuple):
    int_gamma_f: Array
    int_q: Array
    int_q_offset: Array
    int_sigma_0: Array
    int_sigma_1: Array
    int_x_0: Array  # p dimensional, p = number of covariates
    int_x_1: Array  # p dimensional

    def sigma2(self, li):
        return jnp.exp(self.int_sigma_0 + jnp.exp(self.int_sigma_1) * li)

    def q_y_scale(self, m):
        return jnp.exp(
            0.5 * self.int_q_offset + -0.5 * jnp.exp(self.int_q) * jnp.arange(1, m + 1)
        )

        # prior variances are the same for 3 neighbors (repeatetly)
        # mm = m // 3
        # js = jnp.arange(1, mm + 1)
        # js = np.repeat(js, 3)
        # zeros = jnp.zeros(m)
        # js = zeros.at[:3*mm].set(js)
        # qs = jnp.exp(-0.5 * jnp.exp(self.int_q) * js)
        # return qs

    def gamma_f(self):
        return jnp.exp(self.int_gamma_f)

    def q_x_scale(self, which_p, li):
        # return jnp.exp(
        #     0.5 * self.int_x_0[which_p] + 0.5 * jnp.exp(self.int_x_1[which_p]) * li
        # )
        return jnp.exp(self.int_x_0[which_p])

    @staticmethod
    def new(data: Data) -> "Params":
        return Params(
            int_gamma_f=jnp.zeros((), jnp.float32),
            int_q=jnp.array(0.1, jnp.float32),
            int_q_offset=jnp.array(0.0, jnp.float32),
            int_sigma_0=jnp.array(0.0, jnp.float32),
            int_sigma_1=jnp.array(0.0, jnp.float32),
            int_x_0=jnp.zeros(data.x.shape[1], jnp.float32),
            int_x_1=jnp.zeros(data.x.shape[1], jnp.float32),
        )


def scale_y(y, params: Params, m):
    return y * params.q_y_scale(m)


def scale_x(x, params: Params, which_p, li):
    return x * params.q_x_scale(which_p, li)


def threshold_num_nn(data, params: Params) -> int:
    max_num_nn = data.nearest_neighbors.shape[1]
    # stop gradient probably not needed
    scales = jax.lax.stop_gradient(params.q_y_scale(max_num_nn))

    below = scales < 0.01
    num_nn = jax.lax.cond(
        jnp.any(below),  # pred
        lambda: jnp.argmax(below),  # true fun
        lambda: max_num_nn,  # false fun
    )
    return num_nn


def build_R_i(i: int, data: Data) -> Array:
    return jnp.diag(data.cond_vars[:, i])


def build_C_i(
    i: int,
    data: Data,
    params: Params,
    disable_x: bool = False,
    linear_only: bool = False,
) -> Array:
    num_nn = threshold_num_nn(data, params)
    nn_i = data.nearest_neighbors[i, :]
    nn = data.samples[:, 0, nn_i]

    # setting columns to 0 has the has same effect as slicing. we do that when y_idx
    # has not enough neighbours idx < max_neightbors
    # or when we want to consider less neighbots because of the thresholding of
    # num of nn.
    # This, shapes are not changing between idx which makes function jit-compilable.
    nn_zeroed0 = jnp.where(nn_i >= 0, nn, 0.0)
    nn_zeroed = jnp.where(jnp.arange(nn.shape[1]) < num_nn, nn_zeroed0, 0.0)

    # scale the values from conditioning set in accordaince with their position in the
    # conditioning set
    nn_scaled = scale_y(nn_zeroed, params, data.nearest_neighbors.shape[1])

    # compute the scaled x values
    x_scaled = jnp.empty_like(data.x)
    for p in range(data.x.shape[1]):
        x_p = data.x[:, p]
        x_scaled = x_scaled.at[:, p].set(scale_x(x_p, params, p, data.li[i]))

    # linear part
    linear_part_y = tfp.math.psd_kernels.Linear().matrix(nn_scaled, nn_scaled)
    if disable_x:
        linear_part = linear_part_y
    else:
        linear_part_x = tfp.math.psd_kernels.MaternThreeHalves().matrix(
            x_scaled, x_scaled
        )
        linear_part = linear_part_y * linear_part_x

    cov = linear_part

    # non-linear part
    # concatenate nn_scaled, x_scaled along axis 1 such that we get a matrix of size
    # n x (p + max_nn)
    if not linear_only:
        if disable_x:
            w_scaled = nn_scaled
            length_scale = 1.0
        else:
            w_scaled = jnp.concatenate([nn_scaled, x_scaled], axis=1)
            length_scale = params.gamma_f()

        sigma2 = params.sigma2(data.li[i])
        nl_part = sigma2 * tfp.math.psd_kernels.MaternThreeHalves(
            length_scale=length_scale
        ).matrix(w_scaled, w_scaled)

        cov = cov + nl_part

    return cov


def loglik_i(
    idx: jax.Array,
    params: Params,
    data: Data,
    eps: None | float,
    disable_x: bool = False,
    linear_only: bool = False,
) -> jax.Array:
    y = data.samples[:, 0, idx]

    mat_c_i = build_C_i(idx, data, params, disable_x, linear_only)
    mat_r_i = build_R_i(idx, data)

    cov = mat_c_i + mat_r_i
    # cov might be not PD due to numerical issues, so there is an option to add a small
    # value to the diagonal
    if eps is not None:
        cov += eps * jnp.eye(cov.shape[0])

    lp = tfp.distributions.MultivariateNormalFullCovariance(
        jnp.zeros_like(y), cov
    ).log_prob(y)
    return lp


def loglik(
    params: Params,
    data: Data,
    eps: float | None = None,
    disable_x: bool = False,
    linear_only: bool = False,
) -> jax.Array:
    return jnp.sum(
        jax.vmap(lambda i: loglik_i(i, params, data, eps, disable_x, linear_only))(
            jnp.arange(data.locs.shape[0])
        )
    )


class Fit(NamedTuple):
    data: Data
    params_track: list[Params]
    loss_track: list[float]
    linear_only: bool
    disable_x: bool
    is_invalid: bool

    @property
    def params(self) -> Params:
        return self.params_track[-1]

    @property
    def loss(self) -> float:
        return self.loss_track[-1]


def train(
    params: Params,
    dataj: Data,
    num_iter: int,
    eps,
    stop_crit=None,
    disable_x=False,
    linear_only=False,
    step_size: float = 1e-2,
    do_not_train_x_scale: bool = False,
    do_not_train_q: bool = False,
) -> Fit:
    optimizer = optax.adam(step_size)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params):
        if do_not_train_x_scale:
            warnings.warn("stop gradient is used on int_x_0. This keeps it constant.")
            params = params._replace(int_x_0=jax.lax.stop_gradient(params.int_x_0))

        if do_not_train_q:
            warnings.warn("stop gradient is used on int_q. This keeps it constant.")
            params = params._replace(int_q=jax.lax.stop_gradient(params.int_q))

        return -loglik(params, dataj, eps, disable_x, linear_only)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params_track = []
    loss_track = []
    # optimzation loop
    for i in (pbar := tqdm(range(num_iter))):
        params, opt_state, loss = step(params, opt_state)
        loss_track.append(loss)
        params_track.append(params)
        pbar.set_description(f"Loss: {loss:.2f}")

        if (
            stop_crit is not None
            and i > 20
            and np.mean(loss_track[-16:-1]) - np.mean(loss_track[-15:]) < stop_crit
        ):
            print("loss does not decrease anymore")
            break

        if np.isnan(loss):
            warnings.warn("loss is nan. results are be invalid.")
            break

    result = Fit(
        data=dataj,
        params_track=params_track,
        loss_track=loss_track,
        linear_only=linear_only,
        disable_x=disable_x,
        is_invalid=np.isnan(loss),
    )

    return result


def _prep_new_data(data: Data, new_nus: np.ndarray) -> Data:
    num_locs = data.locs.shape[0]
    new_x = np.log(new_nus)[:, np.newaxis]
    new_samples = np.zeros((new_x.shape[0], 1, num_locs))
    data = data._replace(x=np.concatenate([data.x, new_x]))
    data = data._replace(samples=np.concatenate([data.samples, new_samples]))
    data = data._replace(nus=np.concatenate([data.nus, new_nus]))

    # calculate the conditional variances for the new nu locations
    new_cond_vars = gen_data(
        new_nus,
        nlocs_side=np.int32(np.sqrt(num_locs)),
        noise_var=data.noise_var,
        max_nn=1,
    ).cond_vars
    data = data._replace(cond_vars=np.concatenate([data.cond_vars, new_cond_vars]))

    return data


def _prep_new_data2(data: Data2, new_xs: np.ndarray) -> Data2:
    num_locs = data.locs.shape[0]
    new_x = new_xs[:, np.newaxis]
    new_samples = np.zeros((new_x.shape[0], 1, num_locs))
    data = data._replace(x=np.concatenate([data.x, new_x]))
    data = data._replace(samples=np.concatenate([data.samples, new_samples]))

    # calculate the conditional variances for the new nu locations
    new_cond_vars = gen_data2(
        new_xs,
        cov_fn=data.cov_fn,
        nlocs_side=np.int32(np.sqrt(num_locs)),
        noise_var=data.noise_var,
        max_nn=1,
    ).cond_vars
    data = data._replace(cond_vars=np.concatenate([data.cond_vars, new_cond_vars]))

    return data


class PredData(NamedTuple):
    orig_data: Data
    nus: np.ndarray | Array
    samples: np.ndarray | Array
    cond_vars: np.ndarray | Array
    pred_cond_vars: np.ndarray | Array
    pred_cond_mean: np.ndarray | Array


class PredData2(NamedTuple):
    orig_data: Data2
    xs: np.ndarray | Array
    samples: np.ndarray | Array
    cond_vars: np.ndarray | Array
    pred_cond_vars: np.ndarray | Array
    pred_cond_mean: np.ndarray | Array


def sample_field2(
    fit: Fit,
    data: Data2,
    new_xs: np.ndarray,
    nsamples=1,
    diag_eps: float | None = None,
    seed=0,
) -> PredData2:
    if nsamples > 1:
        raise ValueError("Currently only one sample is supported, but nsamples!=1.")

    if new_xs.shape[0] > 1:
        raise ValueError(
            "Currently only one new_x is supported, but new_x.shape[0] != 1."
        )
    original_data = data
    data = _prep_new_data2(data, new_xs)
    num_locs = data.locs.shape[0]

    rng = np.random.default_rng(seed)
    params = fit.params

    locs_with_neg_cond_var = []
    pred_cond_vars = []
    pred_cond_means = []

    for i in range(num_locs):
        if i == 0:
            # all x values should have the same value for variance in the first location
            var = data.cond_vars[-1, 0]
            if diag_eps is not None:
                var += diag_eps
            data.samples[-1, :, 0] = rng.normal(0, var, size=nsamples)
            pred_cond_means.append(np.array(0.0))
            pred_cond_vars.append(np.array(var))

        else:
            mat_c_i = build_C_i(
                i, data, params, disable_x=fit.disable_x, linear_only=fit.linear_only
            )
            mat_r_i = build_R_i(i, data)

            cov = mat_c_i + mat_r_i
            if diag_eps is not None:
                cov += diag_eps * jnp.eye(cov.shape[0])

            if not np.all(np.isfinite(cov)):
                print(f"cov contains non-finite values, i = {i}, cov = {cov}")

            cov00 = cov[:-1, :-1]
            cov11 = cov[-1:, -1:]
            cov01 = cov[:-1, -1:]

            # naive implementation
            # cov00_inv = np.linalg.inv(cov00)
            # pred_cond_mean = cov01.T @ cov00_inv @ data.samples[:-1, 0, i]
            # pred_cond_cov = cov11 - cov01.T @ cov00_inv @ cov01

            # cholesky implementation
            cov00_chol = scipy.linalg.cho_factor(cov00, lower=True)
            if not np.all(np.isfinite(cov00_chol[0])):
                print(f"Chol contains non-finate values, i = {i}")
            pred_cond_mean = cov01.T @ scipy.linalg.cho_solve(
                cov00_chol, data.samples[:-1, 0, i]
            )
            try:
                tmp = scipy.linalg.solve_triangular(cov00_chol[0], cov01, lower=True)
            except Exception as e:
                print(f"solve_triangular solve failed at i = {i}")
                raise e

            pred_cond_cov = cov11 - tmp.T @ tmp

            pred_cond_mean = pred_cond_mean.squeeze()
            pred_cond_cov = pred_cond_cov.squeeze()

            if pred_cond_cov < 0:
                locs_with_neg_cond_var.append(i)
                pred_cond_cov = 0.0

            pred_cond_means.append(pred_cond_mean)
            pred_cond_vars.append(pred_cond_cov)
            data.samples[-1, :, i] = rng.normal(
                pred_cond_mean, np.sqrt(pred_cond_cov), size=nsamples
            )

    if len(locs_with_neg_cond_var) > 0:
        warnings.warn(
            f"Negative conditional variance at n locs {len(locs_with_neg_cond_var)}. "
            f"Those have been set to 0. \n locations {locs_with_neg_cond_var}."
        )

    pred_data = PredData2(
        orig_data=original_data,
        xs=data.x[-1:],
        samples=data.samples[-1:, 0:, :],
        cond_vars=data.cond_vars[-1:, :],
        pred_cond_mean=np.array(pred_cond_means),
        pred_cond_vars=np.array(pred_cond_vars),
    )

    return pred_data


def sample_field(
    fit: Fit,
    data: Data,
    new_nus: np.ndarray,
    nsamples=1,
    diag_eps: float | None = None,
    seed=0,
) -> np.ndarray:
    if nsamples > 1:
        raise ValueError("Currently only one sample is supported, but nsamples!=1.")

    if new_nus.shape[0] > 1:
        raise ValueError(
            "Currently only one new_x is supported, but new_x.shape[0] != 1."
        )
    original_data = data
    data = _prep_new_data(data, new_nus)
    num_locs = data.locs.shape[0]

    rng = np.random.default_rng(seed)
    params = fit.params

    locs_with_neg_cond_var = []
    pred_cond_vars = []
    pred_cond_means = []

    for i in range(num_locs):
        if i == 0:
            # all x values should have the same value for variance in the first location
            var = data.cond_vars[-1, 0]
            if diag_eps is not None:
                var += diag_eps
            data.samples[-1, :, 0] = rng.normal(0, var, size=nsamples)
            pred_cond_means.append(np.array(0.0))
            pred_cond_vars.append(np.array(var))

        else:
            mat_c_i = build_C_i(
                i, data, params, disable_x=fit.disable_x, linear_only=fit.linear_only
            )
            mat_r_i = build_R_i(i, data)

            cov = mat_c_i + mat_r_i
            if diag_eps is not None:
                cov += diag_eps * jnp.eye(cov.shape[0])

            if not np.all(np.isfinite(cov)):
                print(f"cov contains non-finite values, i = {i}, cov = {cov}")

            cov00 = cov[:-1, :-1]
            cov11 = cov[-1:, -1:]
            cov01 = cov[:-1, -1:]

            # naive implementation
            # cov00_inv = np.linalg.inv(cov00)
            # pred_cond_mean = cov01.T @ cov00_inv @ data.samples[:-1, 0, i]
            # pred_cond_cov = cov11 - cov01.T @ cov00_inv @ cov01

            # cholesky implementation
            cov00_chol = scipy.linalg.cho_factor(cov00, lower=True)
            if not np.all(np.isfinite(cov00_chol[0])):
                print(f"Chol contains non-finate values, i = {i}")
            pred_cond_mean = cov01.T @ scipy.linalg.cho_solve(
                cov00_chol, data.samples[:-1, 0, i]
            )
            try:
                tmp = scipy.linalg.solve_triangular(cov00_chol[0], cov01, lower=True)
            except Exception as e:
                print(f"Chol solve failed at i = {i}")
                raise e

            pred_cond_cov = cov11 - tmp.T @ tmp

            pred_cond_mean = pred_cond_mean.squeeze()
            pred_cond_cov = pred_cond_cov.squeeze()

            if pred_cond_cov < 0:
                locs_with_neg_cond_var.append(i)
                pred_cond_cov = 0.0

            pred_cond_means.append(pred_cond_mean)
            pred_cond_vars.append(pred_cond_cov)
            data.samples[-1, :, i] = rng.normal(
                pred_cond_mean, np.sqrt(pred_cond_cov), size=nsamples
            )

    if len(locs_with_neg_cond_var) > 0:
        warnings.warn(
            f"Negative conditional variance at n locs {len(locs_with_neg_cond_var)}. "
            f"Those have been set to 0. \n locations {locs_with_neg_cond_var}."
        )

    pred_data = PredData(
        orig_data=original_data,
        nus=data.nus[-1:],
        samples=data.samples[-1:, 0:, :],
        cond_vars=data.cond_vars[-1:, :],
        pred_cond_mean=np.array(pred_cond_means),
        pred_cond_vars=np.array(pred_cond_vars),
    )

    return pred_data


def posterior_alpha_mean_var(fit: Fit, i: int) -> tuple[np.ndarray, np.ndarray]:
    """
    derivation: see paul's notes in notes/derivation_alpha_posterior.pdf
    """

    nu = fit.data.nus[0]
    assert np.all(fit.data.nus == nu), "Currently only one nu is supported."
    assert fit.linear_only, "Currently only linear model is supported."
    assert fit.disable_x, "Currently no covariates are supported."

    data = fit.data
    params = fit.params
    m = threshold_num_nn(data, params)
    y_i = data.samples[:, 0, i]

    nn_idx = data.nearest_neighbors[i, :m]
    nn_idx = nn_idx[nn_idx >= 0]
    m = len(nn_idx)
    y_mat = data.samples[:, 0, nn_idx]
    q_mat = np.diag(params.q_y_scale(m) ** 2)

    p_star = 1 / data.cond_vars[0, i] * y_mat.T @ y_mat + np.linalg.inv(q_mat)
    cov_star = np.linalg.inv(p_star)
    mu_star = 1 / data.cond_vars[0, i] * cov_star @ y_mat.T @ y_i

    return mu_star, cov_star


def posterior_alpha_mean_var_with_x(
    fit: Fit,
    i: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    derivation: see paul's notes good notes.
    """

    warnings.warn("This function may not work yet. Derivation may be flawed.")

    assert fit.linear_only, "Currently only linear model is supported."
    assert not fit.disable_x, "Supports only with covariates are supported."
    assert fit.data.x.shape[1] == 1, "Only one covariate is supported."

    data = fit.data
    params = fit.params
    n = data.x.shape[0]
    m = threshold_num_nn(data, params)
    y_i = data.samples[:, 0, i]

    nn_idx = data.nearest_neighbors[i, :m]
    nn_idx = nn_idx[nn_idx >= 0]
    m = len(nn_idx)
    y_mat = data.samples[:, 0, nn_idx]
    qs = np.diag(params.q_y_scale(m) ** 2)

    x = data.x
    x_scaled = scale_x(x, params, 0, data.li[i])
    k_mat_y = build_C_i(
        i, data, params, disable_x=fit.disable_x, linear_only=fit.linear_only
    ) + build_R_i(i, data)

    res = []
    for j in range(m):
        q_jj = qs[j, j]
        k_mat_alpha = q_jj * tfp.math.psd_kernels.MaternThreeHalves().matrix(
            x_scaled, x_scaled
        )
        y_cij = y_mat[:, j]
        k_mat_y_alpha = np.diag(y_cij) @ k_mat_alpha
        cov = np.block([[k_mat_y, k_mat_y_alpha], [k_mat_y_alpha.T, k_mat_alpha]])
        cov_00 = cov[:n, :n]
        cov_01 = cov[:n, n:]
        cov_11 = cov[n:, n:]

        mu_alpha = cov_01.T @ np.linalg.inv(cov_00) @ y_i
        cov_alpha = cov_11 - cov_01.T @ np.linalg.inv(cov_00) @ cov_01

        res.append((mu_alpha.squeeze(), cov_alpha.squeeze()))

    return res
