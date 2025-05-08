from typing import Protocol, TypeVar

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array
from tensorflow_probability.substrates.jax.distributions import (
    Distribution as TfpDistribution,
)

from .hermgauss import hermgauss
from .hgpip import (
    induced_variational,
    induced_variational_diag,
    induced_variational_diag_whitened,
)
from .natgrad import XiTransformation

T = TypeVar("T")


class JointParam(nnx.Param[T]):
    pass


def _pwb_set_value_hook(pwb: "ParamWithBijector", value):
    return pwb.bijector.inverse(value)


def _pwb_get_value_hook(pwb: "ParamWithBijector", raw_value):
    return pwb.bijector.forward(raw_value)


class ParamWithBijector(nnx.Param[T]):
    def __init__(self, value: T, bijector: tfp.bijectors.Bijector):
        raw_value = bijector.inverse(value)
        super().__init__(
            raw_value,
            set_value_hooks=_pwb_set_value_hook,
            get_value_hooks=_pwb_get_value_hook,
        )
        self.bijector = bijector


T = TypeVar("T")


class JointParamWithBijector(JointParam[T]):
    def __init__(self, value: T, bijector: tfp.bijectors.Bijector):
        raw_value = bijector.inverse(value)
        super().__init__(
            raw_value,
            set_value_hooks=_pwb_set_value_hook,
            get_value_hooks=_pwb_get_value_hook,
        )
        self.bijector = bijector


class VarMVNPar(nnx.Param[T]):
    def __init__(self, m: Array, s: Array, trf: XiTransformation):
        trf, xi = trf.from_ms(m, s)
        super().__init__(xi)
        self.trf = trf


class Data(nnx.Variable[T]):
    pass


class MVNParams:
    def __init__(self, m: Array, *, sl: Array = None, s: Array = None):
        self.m = m
        self._sl = sl
        self._s = s

    @property
    def s(self):
        if self._s is None:
            self._s = self._sl @ self._sl.T
        return self._s

    @property
    def sl(self):
        if self._sl is None:
            self._sl = jnp.linalg.cholesky(self.s)
        return self._sl


class DataModule(Protocol):
    def input_f(self, mb_idx: Array) -> Array: ...

    def input_g(self, mb_idx: Array) -> Array: ...

    def response(self, mb_idx: Array) -> Array: ...

    @property
    def size(self) -> int:
        return self._response.value.shape[-1]


class SimpleDataModule(nnx.Module):
    def __init__(self, response: Array, input: Array):
        self._response = Data(response)
        self._input = Data(input)

    def input_f(self, mb_idx):
        return self._input[mb_idx]

    def input_g(self, mb_idx):
        return self._input[mb_idx]

    def response(self, mb_idx):
        return self._response[mb_idx]

    @property
    def size(self):
        return self._response.value.shape[-1]


# FIXME: shoild be part of the tmcov module
class TMDataModule(nnx.Module):
    """
    Attributes:
    ---
    :param:`position (Array)`
    :param:`response (Array)`
    :param:`conditioning_set (Array)`
    :param:`covariates (Array)`
    :param:`dist_nn (Array)`
    :param:`nn_idx (Array)`
    """

    def __init__(
        self,
        position: Array,
        response: Array,
        conditioning_set: Array,
        covariates: Array,
        dist_nn: Array,
        nn_idx: Array,
    ):
        # ensure that the conditioning sets are zeroed for missing neighbors
        mask = nn_idx != -1
        conditioning_set = conditioning_set * mask

        self._response = Data(response)
        self.conditioning_set = Data(conditioning_set)
        self.covariates = Data(covariates)
        self.dist_nn = Data(dist_nn)
        self.position = Data(position)
        self.nn_idx = Data(nn_idx)

    @property
    def feature_dim(self):
        return self.covariates.value.shape[-1]

    def input_f(self, mb_idx):
        covs = self.covariates[mb_idx]
        cs = self.conditioning_set[mb_idx]
        return jnp.concatenate([covs, cs], axis=-1)

    def input_g(self, mb_idx):
        return self.covariates[mb_idx]

    def response(self, mb_idx):
        return self._response[mb_idx]

    @property
    def size(self):
        return self._response.value.shape[-1]


class Kernel(Protocol):
    def matrix(self, x1: Array, x2: Array) -> Array: ...

    def apply(self, x1: Array, x2: Array) -> Array: ...

    def tfp(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel: ...


class MaternThreeHalvesKernel(nnx.Module):
    def __init__(self, lengthscale: float, variance: float):
        self.lengthscale = ParamWithBijector(
            lengthscale,
            tfp.bijectors.Softplus(),
        )
        self.variance = ParamWithBijector(variance, tfp.bijectors.Softplus())

    def _tfp_kernel(self) -> tfp.math.psd_kernels.MaternThreeHalves:
        return tfp.math.psd_kernels.MaternThreeHalves(
            amplitude=self.variance.value, length_scale=self.lengthscale.value
        )

    def tfp(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        return self._tfp_kernel()

    def matrix(self, x1: Array, x2: Array) -> Array:
        return self._tfp_kernel().matrix(x1, x2)

    def apply(self, x1: Array, x2: Array) -> Array:
        return self._tfp_kernel().apply(x1, x2)


class MeanFunction(nnx.Module):
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError()


class ConstMeanFunction(MeanFunction):
    def __init__(self, value: Array):
        self._value = nnx.Param(jnp.asarray(value, dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        return jnp.repeat(self._value.value, x.shape[-2], axis=0)


class RegressionProblem(Protocol):
    def loss(self, mb: Array) -> Array: ...

    def predict(self, input_f: Array, input_g: Array) -> TfpDistribution: ...


class HGPIPProblem(nnx.Module):
    def __init__(
        self,
        data: TMDataModule,
        kernel_f: Kernel,
        kernel_g: Kernel,
        ip_f: Array,
        ip_g: Array,
        xi_transformation: XiTransformation,
        mean_g: MeanFunction | None = None,
        whiten: bool = False,
        ip_fixed: bool = True,
        diag_krn_noise_f: float = 0.0,
        diag_krn_noise_g: float = 0.0,
        ip_distance_penalty: None | float = None,
        clip_predictive_variances: bool = False,
    ):
        self.data = data
        self.kernel_f = kernel_f
        self.kernel_g = kernel_g
        self.whiten = whiten
        self.diag_krn_noise_f = diag_krn_noise_f
        self.diag_krn_noise_g = diag_krn_noise_g
        self.ip_distance_penalty_factor = ip_distance_penalty
        self._clip_predictive_variances = clip_predictive_variances

        self.ip_fixed = ip_fixed
        self._ip_f = Data(ip_f)
        self._ip_g = Data(ip_g)

        self.mean_g = mean_g

        if not self.ip_fixed:
            self._ip_f = nnx.Param(ip_f)
            self._ip_g = nnx.Param(ip_g)

        var_m = jnp.zeros(self.ip_g.shape[-2])
        if self.whiten:
            var_s = jnp.eye(self.ip_g.shape[-2])
        else:
            var_s = self.kernel_g.tfp().matrix(self.ip_g, self.ip_g)
            if self.mean_g is not None:
                var_m = self.mean_g(self.ip_g)
        self.xi = VarMVNPar(var_m, var_s, xi_transformation)

    @property
    def ip_f(self) -> Array:
        # FIXME: hack for the TM
        ip_f = self._ip_f.value
        if isinstance(self.data, TMDataModule):
            # zero the columns that are not used because the feature dimension of the
            # first positions is smaller
            # we have [covariates, conditioning_set] as input_f
            feature_dim_f = self.data.feature_dim + self.data.position.value
            ip_f = jnp.where(jnp.arange(ip_f.shape[-1]) < feature_dim_f, ip_f, 0.0)
            # does the same as the following line
            # but self.data.position is dynamic and can not be used for slicing
            # ip_f = ip_f.at[:, feature_dim_f:].set(0.0)

            # we do not reduce the number of inducing points even for earler positions
            # because to allow for complexity with respect to the covariates

        return ip_f

    @property
    def ip_g(self) -> Array:
        return self._ip_g.value

    def ip_distance_penalty(self) -> Array:
        def penalty_fn(x):
            dists = x[:, None] - x[None, :]
            dists = jnp.square(dists).sum(-1)

            mask = jnp.triu(jnp.ones_like(dists), k=1)
            penalties = 1 / (dists + 1e-16)

            return jnp.sum(penalties * mask)

        penalty_f = penalty_fn(self.ip_f)
        penalty_g = penalty_fn(self.ip_g)

        # jax.debug.print("{pf}, {pg}", pf=penalty_f, pg=penalty_g)

        return penalty_f + penalty_g

    def var_msl(self) -> MVNParams:
        m, sl = self.xi.trf.xi_to_msl(self.xi.value)
        return MVNParams(m, sl=sl)

    def loss(self, mb: Array) -> Array:
        loss = -self.elbo(mb)
        if self.ip_distance_penalty_factor is not None:
            loss += self.ip_distance_penalty_factor * self.ip_distance_penalty()
        return loss

    def _induced_distribution_g(
        self, var_par_v: MVNParams, mb_idx, inputs=None
    ) -> MVNParams:
        if inputs is None:
            inputs = self.data.input_g(mb_idx)
        if self.whiten:
            raise NotImplementedError("whitening not implemented for non-diagonal")
        m, s = induced_variational(
            self.kernel_g, self.ip_g, inputs, var_par_v.m, var_par_v.s, self.mean_g
        )
        return MVNParams(m, s=s)

    def _induced_distribution_g_diag(
        self, var_par_v: MVNParams, mb_idx, inputs=None
    ) -> tuple[Array, Array]:
        if inputs is None:
            inputs = self.data.input_g(mb_idx)

        if self.whiten:
            m, s_diag = induced_variational_diag_whitened(
                self.kernel_g,
                self.ip_g,
                inputs,
                var_par_v.m,
                var_par_v.s,
                self.mean_g,
                diag_krn_eps=self.diag_krn_noise_g,
            )
        else:
            m, s_diag = induced_variational_diag(
                self.kernel_g,
                self.ip_g,
                inputs,
                var_par_v.m,
                var_par_v.s,
                self.mean_g,
                diag_krn_eps=self.diag_krn_noise_g,
            )
        return m, s_diag

    def var_par_u(self, var_par_v: MVNParams, mb_idx) -> MVNParams:
        m_g, s_g_diag = self._induced_distribution_g_diag(var_par_v, mb_idx)
        # r = jnp.exp(m_g - 0.5*jnp.diag(s_g))
        r_inv = jnp.exp(-m_g + 0.5 * s_g_diag)

        Kuu = self.kernel_f.matrix(
            self.ip_f, self.ip_f
        ) + self.diag_krn_noise_f * jnp.eye(self.ip_f.shape[-2])
        Kuf = self.kernel_f.matrix(self.ip_f, self.data.input_f(mb_idx))
        # Kuf_r_inv_Kfu = Kuf @ jnp.diag(r_inv) @ Kuf.T
        Kuf_r_inv_Kfu = (Kuf * r_inv) @ Kuf.T

        Ainv_C_inv = Kuu @ jnp.linalg.solve(Kuf_r_inv_Kfu, Kuu) + Kuu
        s_u = Kuu - Kuu @ jnp.linalg.solve(Ainv_C_inv, Kuu)

        # mwu = Kuf @ jnp.diag(r_inv) @ self.obs[mb_idx]
        mwu = (Kuf * r_inv) @ self.data.response(mb_idx)
        m_u = mwu - Kuu @ jnp.linalg.solve(Ainv_C_inv, mwu)

        return MVNParams(m_u.squeeze(), s=s_u)

    def var_par_u_whitened(self, var_par_v: MVNParams, mb_idx) -> MVNParams:
        m_g, s_g_diag = self._induced_distribution_g_diag(var_par_v, mb_idx)
        r_inv = jnp.exp(-m_g + 0.5 * s_g_diag)

        Kuu = self.kernel_f.matrix(
            self.ip_f, self.ip_f
        ) + self.diag_krn_noise_f * jnp.eye(self.ip_f.shape[-2])
        Kuf = self.kernel_f.matrix(self.ip_f, self.data.input_f(mb_idx))

        Luu = jnp.linalg.cholesky(Kuu)
        Luu_inv_Kuf = jax.scipy.linalg.solve_triangular(Luu, Kuf, lower=True)

        s_w_inv = ((Luu_inv_Kuf * r_inv) @ Luu_inv_Kuf.T) + jnp.eye(Kuu.shape[-2])
        s_w = jnp.linalg.inv(s_w_inv)
        m_w = jnp.linalg.solve(
            s_w_inv, (Luu_inv_Kuf * r_inv) @ self.data.response(mb_idx)
        )

        return MVNParams(m_w.squeeze(), s=s_w)

    def _induced_distribution_f(
        self, var_par_v: MVNParams, mb_idx, inputs=None
    ) -> MVNParams:
        if self.whiten:
            raise NotImplementedError("whitening not implemented for non-diagonal")
        if inputs is None:
            inputs = self.data.input_f(mb_idx)

        var_par_u = self.var_par_u(var_par_v, mb_idx)
        m, s = induced_variational(
            self.kernel_f,
            self.ip_f,
            inputs,
            var_par_u.m,
            var_par_u.s,
            diag_krn_eps=self.diag_krn_noise_f,
        )

        return MVNParams(m, s=s)

    def _induced_distribution_f_diag(
        self, var_par_v: MVNParams, mb_idx, inputs=None
    ) -> tuple[Array, Array]:
        if inputs is None:
            inputs = self.data.input_f(mb_idx)

        if self.whiten:
            var_par_w = self.var_par_u_whitened(var_par_v, mb_idx)
            m, s_diag = induced_variational_diag_whitened(
                self.kernel_f,
                self.ip_f,
                inputs,
                var_par_w.m,
                var_par_w.s,
                diag_krn_eps=self.diag_krn_noise_f,
            )
        else:
            var_par_u = self.var_par_u(var_par_v, mb_idx)
            m, s_diag = induced_variational_diag(
                self.kernel_f,
                self.ip_f,
                inputs,
                var_par_u.m,
                var_par_u.s,
                diag_krn_eps=self.diag_krn_noise_f,
            )

        return m, s_diag

    def expected_loglikelihood(self, var_par_v: MVNParams, mb_idx) -> Array:
        # TODO: does this miss the correction terms
        # - 1/4 tr(s_g) - 1/2 tr(Sf @ R{-1})
        # probably not because I calculate everything by hand and do
        # not use the N(y| m_f, R) formulation

        n = mb_idx.shape[0]
        t0 = n * jnp.log(2 * jnp.pi)

        # use diag
        g_mean, g_cov_diag = self._induced_distribution_g_diag(var_par_v, mb_idx)
        t1 = jnp.sum(g_mean)

        f_mean, f_cov_diag = self._induced_distribution_f_diag(var_par_v, mb_idx)
        sq_diff = (self.data.response(mb_idx) - f_mean) ** 2
        sq_diff_plus_f = sq_diff + f_cov_diag
        sigma2s = jnp.exp(g_mean - 0.5 * g_cov_diag)

        t2 = jnp.sum(sq_diff_plus_f / sigma2s)
        t_all = t0 + t1 + t2

        return -0.5 * t_all

    def elbo(self, mb_idx=None) -> Array:
        n = self.data.size
        if mb_idx is None:
            mb_idx = jnp.arange(n)
        n_mb = mb_idx.shape[0]

        if self.whiten:
            prior_gv = tfp.distributions.MultivariateNormalDiag(
                scale_diag=jnp.ones(self.ip_g.shape[-2])
            )
            prior_fu = tfp.distributions.MultivariateNormalDiag(
                scale_diag=jnp.ones(self.ip_f.shape[-2])
            )
        else:
            prior_fu = tfp.distributions.GaussianProcess(
                kernel=self.kernel_f.tfp(),
                index_points=self.ip_f,
                observation_noise_variance=0.0,
            )
            prior_gv = tfp.distributions.GaussianProcess(
                kernel=self.kernel_g.tfp(),
                mean_fn=self.mean_g,
                index_points=self.ip_g,
                observation_noise_variance=0.0,
            )

        var_par_v = self.var_msl()
        if self.whiten:
            var_par_u = self.var_par_u_whitened(var_par_v, mb_idx)
        else:
            var_par_u = self.var_par_u(var_par_v, mb_idx)

        # q_gv = tfp.distributions.MultivariateNormalTriL(var_par_v.m, var_par_v.sl)
        # q_fu = tfp.distributions.MultivariateNormalTriL(var_par_u.m, var_par_u.sl)
        q_gv = tfp.distributions.MultivariateNormalFullCovariance(
            var_par_v.m, var_par_v.s
        )
        q_fu = tfp.distributions.MultivariateNormalFullCovariance(
            var_par_u.m, var_par_u.s
        )

        kl_g = tfp.distributions.kl_divergence(q_gv, prior_gv)
        kl_f = tfp.distributions.kl_divergence(q_fu, prior_fu)

        e_loglik = self.expected_loglikelihood(var_par_v, mb_idx)
        mb_scale = n / n_mb

        return mb_scale * e_loglik - kl_g - kl_f

    # TODO: Rename fn diag_mean_sd => diag_mean_var
    # This is returning variances
    def predict_fg_diag_mean_sd(
        self, input_f=None, input_g=None
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        mb_idx = jnp.arange(self.data.size)
        var_par_v = self.var_msl()

        mu_g, sigma_g_diag = self._induced_distribution_g_diag(
            var_par_v, mb_idx, input_g
        )
        mu_f, sigma_f_diag = self._induced_distribution_f_diag(
            var_par_v, mb_idx, input_f
        )

        if self._clip_predictive_variances:
            sigma_f_diag = jnp.maximum(sigma_f_diag, 1e-12)
            sigma_g_diag = jnp.maximum(sigma_g_diag, 1e-12)

        return (mu_f, sigma_f_diag), (mu_g, sigma_g_diag)

    def predict_fg_diag(
        self, input_f=None, input_g=None
    ) -> tuple[TfpDistribution, TfpDistribution]:
        (mu_f, sigma_f_diag), (mu_g, sigma_g_diag) = self.predict_fg_diag_mean_sd(
            input_f, input_g
        )
        d_f = tfp.distributions.MultivariateNormalDiag(
            mu_f, scale_diag=jnp.sqrt(sigma_f_diag)
        )
        d_g = tfp.distributions.MultivariateNormalDiag(
            mu_g, scale_diag=jnp.sqrt(sigma_g_diag)
        )

        return d_f, d_g

    def predict_with_data(self, data: DataModule) -> TfpDistribution:
        input_f = data.input_f(jnp.arange(data.size))
        input_g = data.input_g(jnp.arange(data.size))
        return self.predict(input_f, input_g)

    def fs_and_gs_from_data(
        self, data: DataModule
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Helper to compute fs and gs params from data"""

        input_f = data.input_f(jnp.arange(data.size))
        input_g = data.input_g(jnp.arange(data.size))
        f_params, g_params = self.predict_fg_diag_mean_sd(input_f, input_g)
        return f_params, g_params

    def log_prob(self, data: DataModule) -> Array:
        """Computes logprob using Gaussian quadrature.

        Estimate .. math:
            \\log E_{q(g)} [p(y | g)] = \\sum_{i=1}^{N} h_i * p(y | w_i),

        where :math:`h_i` is the height of the polynomial and :math:`g_i` an
        augmentation :math:`w_i = \\sqrt{2} \\sd{g} x_i + \\mu_g` for the locs
        of the polynomial :math:`x_i`.
        """

        def logp(x):
            mb_idx = jnp.arange(data.size)
            y = data.response(mb_idx)[..., None]

            # TODO: make sure that this is the correct formula for xt
            # dg[1] is the variance of g, dg[0] is the mean of g
            # xt should be the log variance of y
            xt = jnp.sqrt(2 * dg[1][..., None]) * x + dg[0][..., None]
            # xt = dg[0][..., None]
            f_mean = df[0][..., None]
            f_var = df[1][..., None]
            total_sd = jnp.sqrt(f_var + jnp.exp(xt))
            return jax.scipy.stats.norm.logpdf(y, loc=f_mean, scale=total_sd)

        df, dg = self.fs_and_gs_from_data(data)
        x, w = hermgauss(50)
        x = x[None, :]
        w = w[None, :]
        logprobs = logp(x) + jnp.log(w)
        assert logprobs.shape[0] == data.size, f"{logprobs.shape[0]}, {data.size}"
        assert logprobs.shape[1] == w.size, f"{logprobs.shape[1]}, {w.size}"
        logprobs = jax.scipy.special.logsumexp(logprobs, axis=-1)
        logprobs = logprobs - 0.5 * jnp.log(jnp.pi)
        # eventually shape of (spatial_points, data_size) => (sum, mean)
        return logprobs

    def predict(self, input_f=None, input_g=None) -> TfpDistribution:
        mean, sd = self.predict_mean_sd(input_f, input_g)
        return tfp.distributions.Normal(mean, sd)

    def predict_mean_sd(self, input_f=None, input_g=None) -> tuple[Array, Array]:
        df, dg = self.predict_fg_diag(input_f, input_g)
        mean = df.mean()
        var = df.variance() + jnp.exp(dg.mean() + 0.5 * dg.variance())
        return mean, jnp.sqrt(var)

    def mc_logprob(self, data: DataModule) -> Array:
        """mc log prob estimates with f and g both sampled"""
        input_f = data.input_f(jnp.arange(data.size))
        input_g = data.input_g(jnp.arange(data.size))
        df, dg = self.predict_fg_diag(input_f, input_g)

        keys = jax.random.split(jax.random.key(1), 2)
        num_mc_pts = 100_000
        f = df.sample((num_mc_pts,), keys[0])
        g = dg.sample((num_mc_pts,), keys[1])

        mb_idx = jnp.arange(data.size)
        y = data.response(mb_idx)[None, :]
        # TODO: next line is wrong as g is the log variance of y
        logprobs = jax.scipy.stats.norm.logpdf(y, loc=f, scale=jnp.exp(g))
        assert logprobs.shape[0] == num_mc_pts, f"{logprobs.shape[0]}"
        assert logprobs.shape[1] == y.size, f"{logprobs.shape[1]}, {y.size}"
        return jax.scipy.special.logsumexp(logprobs, axis=0) - jnp.log(num_mc_pts)

    def mc_logprob2(self, data: DataModule) -> Array:
        """more stable because it only samples g and integrates out f"""
        input_f = data.input_f(jnp.arange(data.size))
        input_g = data.input_g(jnp.arange(data.size))
        df, dg = self.predict_fg_diag(input_f, input_g)

        keys = jax.random.split(jax.random.key(1), 2)
        num_mc_pts = 100_000
        g = dg.sample((num_mc_pts,), keys[1])
        # TODO: next line is wrong as g is the log variance of y
        total_sd = jnp.sqrt(df.variance()[None, ...] + jnp.exp(2 * g))

        mb_idx = jnp.arange(data.size)
        y = data.response(mb_idx)[None, :]
        logprobs = jax.scipy.stats.norm.logpdf(
            y, loc=df.mean()[None, ...], scale=total_sd
        )
        assert logprobs.shape[0] == num_mc_pts, f"{logprobs.shape[0]}"
        assert logprobs.shape[1] == y.size, f"{logprobs.shape[1]}, {y.size}"
        return jax.scipy.special.logsumexp(logprobs, axis=0) - jnp.log(num_mc_pts)


def merge_states(states: list[nnx.State]) -> nnx.State:
    def merge_values(*vss: nnx.VariableState) -> nnx.VariableState:
        import copy

        tmp = vss[0].value
        vss[0].value = None
        vs = copy.copy(vss[0])
        vss[0].value = tmp

        values = jnp.stack([v.value for v in vss], axis=0)
        vs.value = values
        return vs

    return jax.tree.map(
        merge_values, *states, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    )


def split_state(state: T) -> list[T]:
    def is_variable_state(x):
        return isinstance(x, nnx.VariableState)

    size_leading_dim = jax.tree_util.tree_leaves(state, is_variable_state)[
        0
    ].value.shape[0]

    def index_value(i, vs: nnx.VariableState) -> nnx.VariableState:
        import copy

        value = vs.value
        vs.value = None
        vs_i = copy.copy(vs)
        vs.value = value
        vs_i.value = value[i]
        return vs_i

    return [
        jax.tree.map(lambda x: index_value(i, x), state, is_leaf=is_variable_state)
        for i in range(size_leading_dim)
    ]


def merge_modules(modules: list[T]) -> T:
    gs = []
    jps = []
    rs = []
    for m in modules:
        g, jp, r = nnx.split(m, JointParam, object)
        gs.append(g)
        jps.append(jp)
        rs.append(r)

    r_stacked = merge_states(rs)
    m = nnx.merge(gs[0], jps[0], r_stacked)
    return m


def split_module(module: T) -> list[T]:
    g, jp, rs = nnx.split(module, JointParam, object)
    return [nnx.merge(g, jp, r) for r in split_state(rs)]
