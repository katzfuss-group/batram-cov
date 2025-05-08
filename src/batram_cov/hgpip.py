import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array

PSDKernel = tfp.math.psd_kernels.PositiveSemidefiniteKernel


# TODO provide diagonalized versions of induced variationals
# In the code below, we get away with not using the full covariance matrix of f/g but
# only the diagonal. Thus the code runs in O(n) instead of O(n^2) for n being the
# number of observations/prediction points.


def induced_variational(
    kernel, inducing_points, locations, mu, sigma, mean_function=None, diag_krn_eps=0.0
) -> tuple[Array, Array]:
    """
    mean and covariance of the variational distribution q(f) = int p(f | u) q(u) du
    """

    Kuu = kernel.matrix(inducing_points, inducing_points) + diag_krn_eps * jnp.eye(
        inducing_points.shape[-2]
    )
    Kuf = kernel.matrix(inducing_points, locations)
    Kff = kernel.matrix(locations, locations) + diag_krn_eps * jnp.eye(
        inducing_points.shape[-2]
    )
    mean_u = (
        mean_function(inducing_points)
        if mean_function is not None
        else jnp.zeros(inducing_points.shape[-2])
    )
    mean_f = (
        mean_function(locations)
        if mean_function is not None
        else jnp.zeros(locations.shape[-2])
    )

    # use chol to calc Kuu_inv @ Kuf
    Kuu_chol = jax.scipy.linalg.cho_factor(Kuu, lower=True)

    # Kuu_inv = jnp.linalg.inv(Kuu)
    m = mean_f + (Kuf.T @ jax.scipy.linalg.cho_solve(Kuu_chol, mu - mean_u)).squeeze()
    Kuu_inv_Kuf = jax.scipy.linalg.cho_solve(Kuu_chol, Kuf)
    s = Kff - Kuu_inv_Kuf.T @ (Kuu - sigma) @ Kuu_inv_Kuf
    return m, s


def induced_variational_diag(
    kernel, inducing_points, locations, mu, sigma, mean_function=None, diag_krn_eps=0.0
) -> tuple[Array, Array]:
    """
    mean and variance of the variational distribution q(f_i) = int p(f_i | u) q(u) du

    this is equivaltent to m, diag(s) where m, s = induced_variational(...)
    """
    # TODO: check if this can be speed up
    Kuu = kernel.matrix(inducing_points, inducing_points) + diag_krn_eps * jnp.eye(
        inducing_points.shape[-2]
    )
    Kuf = kernel.matrix(inducing_points, locations)
    Kff_diag = kernel.apply(locations, locations) + diag_krn_eps
    mean_u = (
        mean_function(inducing_points)
        if mean_function is not None
        else jnp.zeros(inducing_points.shape[-2])
    )
    mean_f = (
        mean_function(locations)
        if mean_function is not None
        else jnp.zeros(locations.shape[-2])
    )

    # use chol to calc Kuu_inv @ Kuf
    Kuu_chol = jax.scipy.linalg.cho_factor(Kuu, lower=True)

    # Kuu_inv = jnp.linalg.inv(Kuu)
    m = mean_f + (Kuf.T @ jax.scipy.linalg.cho_solve(Kuu_chol, mu - mean_u)).squeeze()
    Kuu_inv_Kuf = jax.scipy.linalg.cho_solve(Kuu_chol, Kuf)

    # the expression inside of diag can be computed faster sicne we only need the
    # diagonal
    t0 = jnp.diag(Kuu_inv_Kuf.T @ (Kuu - sigma) @ Kuu_inv_Kuf)
    s_diag = Kff_diag - t0
    return m, s_diag


def induced_variational_diag_whitened(
    kernel, inducing_points, locations, mu, sigma, mean_function=None, diag_krn_eps=0.0
) -> tuple[Array, Array]:
    """
    mean and variance of the variational distribution q(f_i) = int p(f_i | w) q(w) dw

    where w is whitend
    """
    # TODO: check if this can be speed up
    Kuu = kernel.matrix(inducing_points, inducing_points) + diag_krn_eps * jnp.eye(
        inducing_points.shape[-2]
    )
    Kuf = kernel.matrix(inducing_points, locations)
    Kff_diag = kernel.apply(locations, locations) + diag_krn_eps
    mean_f = (
        mean_function(locations)
        if mean_function is not None
        else jnp.zeros(locations.shape[-2])
    )

    # use chol to calc Kuu_inv @ Kuf
    Luu = jax.scipy.linalg.cholesky(Kuu, lower=True)
    Luu_inv_Kuf = jax.scipy.linalg.solve_triangular(Luu, Kuf, lower=True)

    m = mean_f + (Luu_inv_Kuf.T @ mu).squeeze()
    # eye_m_sigma = jnp.eye(sigma.shape[-1]) - sigma
    idx = jnp.diag_indices_from(sigma)
    eye_m_sigma = -sigma.at[idx].add(-1.0)
    t0 = jnp.diag(Luu_inv_Kuf.T @ eye_m_sigma @ Luu_inv_Kuf)

    s_diag = Kff_diag - t0
    return m, s_diag


class InducingPointHGP2:
    def __init__(self, kernel_f, kernel_g, ip_g, input_f, input_g, obs):
        self.kernel_f_fn = kernel_f
        self.kernel_g_fn = kernel_g
        # self.ip_f = ip_f
        self.ip_g = ip_g
        self.input_f = input_f
        self.input_g = input_g
        self.obs = obs.squeeze()

    def kernel_g(self, params) -> PSDKernel:
        ls = params.g_ls
        amp = params.g_amp
        base_kernel = self.kernel_g_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def kernel_f(self, params) -> PSDKernel:
        ls = params.f_ls
        amp = params.f_amp
        base_kernel = self.kernel_f_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def induced_distribution_g(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        m, s = induced_variational(
            self.kernel_g(hp), self.ip_g, self.input_g, var_mu, var_sigma
        )
        return m, s

    def best_f(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        m_g, s_g = self.induced_distribution_g(var_mu, var_sigma, hp)
        sigma2s = jnp.exp(m_g - 0.5 * jnp.diag(s_g))
        Lambda = jnp.diag(sigma2s)
        Lambda_inv = jnp.diag(1 / sigma2s)

        Kff = self.kernel_f(hp).matrix(self.input_f, self.input_f)

        s_inv = jnp.linalg.inv(Kff) + Lambda_inv
        # m = s_inv @ Lambda_inv @ self.obs
        m = self.obs - Lambda @ jnp.linalg.solve(Kff + Lambda, self.obs)
        s = jnp.linalg.inv(s_inv)

        return m, s

    def expected_loglikelihood(self, var_mu, var_sigma, hp) -> Array:
        n = self.obs.shape[0]
        g_mean, g_cov = self.induced_distribution_g(var_mu, var_sigma, hp)
        t0 = n * jnp.log(2 * jnp.pi)
        t1 = jnp.sum(g_mean)

        f_mean, f_cov = self.best_f(var_mu, var_sigma, hp)

        sq_diff = (self.obs - f_mean) ** 2
        sq_diff_plus_f = sq_diff + jnp.diag(f_cov)
        sigma2s = jnp.exp(g_mean - 0.5 * jnp.diag(g_cov))

        t2 = jnp.sum(sq_diff_plus_f / sigma2s)
        t_all = t0 + t1 + t2

        return -0.5 * t_all

    def elbo(self, var_mu, var_sigma, hp) -> Array:
        prior_gv = tfp.distributions.GaussianProcess(
            kernel=self.kernel_g(hp),
            index_points=self.ip_g,
            observation_noise_variance=0.0,
        )
        prior_f = tfp.distributions.GaussianProcess(
            kernel=self.kernel_f(hp),
            index_points=self.input_f,
            observation_noise_variance=0.0,
        )

        q_gv = tfp.distributions.MultivariateNormalFullCovariance(var_mu, var_sigma)

        m_f, s_f = self.best_f(var_mu, var_sigma, hp)
        q_f = tfp.distributions.MultivariateNormalFullCovariance(m_f, s_f)

        kl_g = tfp.distributions.kl_divergence(q_gv, prior_gv)
        kl_f = tfp.distributions.kl_divergence(q_f, prior_f)

        e_loglik = self.expected_loglikelihood(var_mu, var_sigma, hp)

        return e_loglik - kl_g - kl_f

    def predict(
        self, var_mu, var_sigma, hp
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        mu_g, sigma_g = self.induced_distribution_g(var_mu, var_sigma, hp)
        # g_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_g, sigma_g)

        mu_f, sigma_f = self.best_f(var_mu, var_sigma, hp)
        # f_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_f, sigma_f)

        # return f_dist, g_dist
        return ((mu_f, sigma_f), (mu_g, sigma_g))

    def predict_y(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        pf, pg = self.predict(var_mu, var_sigma, hp)

        mu_y = pf[0]
        sigma_y = jnp.diag(pf[1]) + jnp.exp(pg[0] + 0.5 * jnp.diag(pg[1]))

        return mu_y, sigma_y


class InducingPointHGP:
    def __init__(self, kernel_f, kernel_g, ip_f, ip_g, input_f, input_g, obs):
        self.ip_f = ip_f
        self.ip_g = ip_g
        self.input_f = input_f
        self.input_g = input_g
        self.obs = obs.squeeze()

        self.kernel_f_fn = kernel_f
        self.kernel_g_fn = kernel_g

    def kernel_g(self, params) -> PSDKernel:
        ls = params.g_ls
        amp = params.g_amp
        base_kernel = self.kernel_g_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def kernel_f(self, params) -> PSDKernel:
        ls = params.f_ls
        amp = params.f_amp
        base_kernel = self.kernel_f_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def induced_distribution_g(
        self, var_mu, var_sigma, hp, mb_idx, inputs=None
    ) -> tuple[Array, Array]:
        if inputs is None:
            inputs = self.input_g[mb_idx]
        m, s = induced_variational(
            self.kernel_g(hp), self.ip_g, inputs, var_mu, var_sigma
        )
        return m, s

    def dist_u(self, var_mu, var_sigma, hp, mb_idx) -> tuple[Array, Array]:
        m_g, s_g = self.induced_distribution_g(var_mu, var_sigma, hp, mb_idx)
        # r = jnp.exp(m_g - 0.5*jnp.diag(s_g))
        r_inv = jnp.exp(-m_g + 0.5 * jnp.diag(s_g))

        kernel_f = self.kernel_f(hp)
        Kuu = kernel_f.matrix(self.ip_f, self.ip_f)
        Kuf = kernel_f.matrix(self.ip_f, self.input_f[mb_idx])
        # Kuf_r_inv_Kfu = Kuf @ jnp.diag(r_inv) @ Kuf.T
        Kuf_r_inv_Kfu = (Kuf * r_inv) @ Kuf.T

        Ainv_C_inv = Kuu @ jnp.linalg.solve(Kuf_r_inv_Kfu, Kuu) + Kuu
        s_u = Kuu - Kuu @ jnp.linalg.solve(Ainv_C_inv, Kuu)

        # mwu = Kuf @ jnp.diag(r_inv) @ self.obs[mb_idx]
        mwu = (Kuf * r_inv) @ self.obs[mb_idx]
        m_u = mwu - Kuu @ jnp.linalg.solve(Ainv_C_inv, mwu)

        return m_u, s_u

    def induced_distribution_f(
        self, var_mu, var_sigma, hp, mb_idx, inputs=None
    ) -> tuple[Array, Array]:
        m_u, s_u = self.dist_u(var_mu, var_sigma, hp, mb_idx)
        if inputs is None:
            inputs = self.input_f[mb_idx]

        return induced_variational(self.kernel_f(hp), self.ip_f, inputs, m_u, s_u)

    def expected_loglikelihood(self, var_mu, var_sigma, hp, mb_idx) -> Array:
        n = mb_idx.shape[0]

        g_mean, g_cov = self.induced_distribution_g(var_mu, var_sigma, hp, mb_idx)
        t0 = n * jnp.log(2 * jnp.pi)
        t1 = jnp.sum(g_mean)

        f_mean, f_cov = self.induced_distribution_f(var_mu, var_sigma, hp, mb_idx)

        sq_diff = (self.obs[mb_idx] - f_mean) ** 2
        sq_diff_plus_f = sq_diff + jnp.diag(f_cov)
        sigma2s = jnp.exp(g_mean - 0.5 * jnp.diag(g_cov))

        t2 = jnp.sum(sq_diff_plus_f / sigma2s)
        t_all = t0 + t1 + t2

        return -0.5 * t_all

    def elbo(self, var_mu, var_sigma_l, hp, mb_idx=None) -> Array:
        if mb_idx is None:
            mb_idx = jnp.arange(self.obs.shape[0])

        n = self.obs.shape[0]
        n_mb = mb_idx.shape[0]

        prior_gv = tfp.distributions.GaussianProcess(
            kernel=self.kernel_g(hp),
            index_points=self.ip_g,
            observation_noise_variance=0.0,
        )
        prior_fu = tfp.distributions.GaussianProcess(
            kernel=self.kernel_f(hp),
            index_points=self.ip_f,
            observation_noise_variance=0.0,
        )

        var_sigma = var_sigma_l @ var_sigma_l.T

        q_gv = tfp.distributions.MultivariateNormalTriL(var_mu, var_sigma_l)

        m_u, s_u = self.dist_u(var_mu, var_sigma, hp, mb_idx)
        q_fu = tfp.distributions.MultivariateNormalFullCovariance(m_u, s_u)

        kl_g = tfp.distributions.kl_divergence(q_gv, prior_gv)
        kl_f = tfp.distributions.kl_divergence(q_fu, prior_fu)

        e_loglik = self.expected_loglikelihood(var_mu, var_sigma, hp, mb_idx)
        mb_scale = n / n_mb

        return mb_scale * e_loglik - kl_g - kl_f

    def predict(
        self, var_mu, var_sigma, hp, inputs=None
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        mb_idx = jnp.arange(self.obs.shape[0])
        mu_g, sigma_g = self.induced_distribution_g(
            var_mu, var_sigma, hp, mb_idx, inputs
        )
        # g_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_g, sigma_g)

        mu_f, sigma_f = self.induced_distribution_f(
            var_mu, var_sigma, hp, mb_idx, inputs
        )
        # f_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_f, sigma_f)

        # return f_dist, g_dist
        return ((mu_f, sigma_f), (mu_g, sigma_g))

    def predict_y(self, var_mu, var_sigma, hp, inputs=None) -> tuple[Array, Array]:
        pf, pg = self.predict(var_mu, var_sigma, hp, inputs)

        mu_y = pf[0]
        sigma_y = jnp.diag(pf[1]) + jnp.exp(pg[0] + 0.5 * jnp.diag(pg[1]))

        return mu_y, sigma_y


class InducingPointHGP3:
    def __init__(self, kernel_f, kernel_g, ip_f, ip_g, input_f, input_g, obs):
        self.ip_f = ip_f
        self.ip_g = ip_g
        self.input_f = input_f
        self.input_g = input_g
        self.obs = obs.squeeze()

        self.kernel_f_fn = kernel_f
        self.kernel_g_fn = kernel_g

    def kernel_g(self, params) -> PSDKernel:
        ls = params.g_ls
        amp = params.g_amp
        base_kernel = self.kernel_g_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def kernel_f(self, params) -> PSDKernel:
        ls = params.f_ls
        amp = params.f_amp
        base_kernel = self.kernel_f_fn(length_scale=ls, amplitude=amp)
        # shifted_kernel = DiagonalShiftKernel(base_kernel, 1e-6)
        shifted_kernel = base_kernel
        return shifted_kernel

    def var_params(self, hp, log_lambda_diag) -> tuple[Array, Array]:
        kernel_g = self.kernel_g(hp)
        Kvv = kernel_g.matrix(self.ip_g, self.ip_g)
        Kvv_inv = jnp.linalg.inv(Kvv)

        lambda_matrix = jnp.diag(jnp.exp(log_lambda_diag))

        var_sigma = jnp.linalg.inv(
            Kvv_inv + lambda_matrix
        )  # TODO: should be Kvv_inv here
        var_mu = (
            Kvv @ (lambda_matrix - 0.5 * jnp.eye(Kvv.shape[0])) @ jnp.ones(Kvv.shape[0])
        )

        return var_mu, var_sigma

    def induced_distribution_g(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        m, s = induced_variational(
            self.kernel_g(hp), self.ip_g, self.input_g, var_mu, var_sigma
        )
        return m, s

    def dist_u(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        m_g, s_g = self.induced_distribution_g(var_mu, var_sigma, hp)
        # r = jnp.exp(m_g - 0.5*jnp.diag(s_g))
        r_inv = jnp.exp(-m_g + 0.5 * jnp.diag(s_g))

        kernel_f = self.kernel_f(hp)
        Kuu = kernel_f.matrix(self.ip_f, self.ip_f)
        Kuf = kernel_f.matrix(self.ip_f, self.input_f)
        Kuf_r_inv_Kfu = Kuf @ jnp.diag(r_inv) @ Kuf.T
        Ainv_C_inv = Kuu @ jnp.linalg.solve(Kuf_r_inv_Kfu, Kuu) + Kuu
        s_u = Kuu - Kuu @ jnp.linalg.solve(Ainv_C_inv, Kuu)

        mwu = Kuf @ jnp.diag(r_inv) @ self.obs
        m_u = mwu - Kuu @ jnp.linalg.solve(Ainv_C_inv, mwu)

        return m_u, s_u

    def induced_distribution_f(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        return self.induced_distribution_f_ip(var_mu, var_sigma, hp)

    def induced_distribution_f_ip(self, var_mu, var_sigma, hp) -> tuple[Array, Array]:
        m_u, s_u = self.dist_u(var_mu, var_sigma, hp)

        return induced_variational(self.kernel_f(hp), self.ip_f, self.input_f, m_u, s_u)

    def expected_loglikelihood(self, var_mu, var_sigma, hp) -> Array:
        n = self.obs.shape[0]
        g_mean, g_cov = self.induced_distribution_g(var_mu, var_sigma, hp)
        t0 = n * jnp.log(2 * jnp.pi)
        t1 = jnp.sum(g_mean)

        f_mean, f_cov = self.induced_distribution_f(var_mu, var_sigma, hp)

        sq_diff = (self.obs - f_mean) ** 2
        sq_diff_plus_f = sq_diff + jnp.diag(f_cov)
        sigma2s = jnp.exp(g_mean - 0.5 * jnp.diag(g_cov))

        t2 = jnp.sum(sq_diff_plus_f / sigma2s)
        t_all = t0 + t1 + t2

        return -0.5 * t_all

    def elbo(self, log_lambda_diag, hp) -> Array:
        var_mu, var_sigma = self.var_params(hp, log_lambda_diag)
        prior_gv = tfp.distributions.GaussianProcess(
            kernel=self.kernel_g(hp),
            index_points=self.ip_g,
            observation_noise_variance=0.0,
        )
        prior_fu = tfp.distributions.GaussianProcess(
            kernel=self.kernel_f(hp),
            index_points=self.ip_f,
            observation_noise_variance=0.0,
        )

        q_gv = tfp.distributions.MultivariateNormalFullCovariance(var_mu, var_sigma)

        m_u, s_u = self.dist_u(var_mu, var_sigma, hp)
        q_fu = tfp.distributions.MultivariateNormalFullCovariance(m_u, s_u)

        kl_g = tfp.distributions.kl_divergence(q_gv, prior_gv)
        kl_f = tfp.distributions.kl_divergence(q_fu, prior_fu)

        e_loglik = self.expected_loglikelihood(var_mu, var_sigma, hp)

        return e_loglik - kl_g - kl_f

    def predict(
        self, log_lambda_diag, hp
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        var_mu, var_sigma = self.var_params(hp, log_lambda_diag)
        mu_g, sigma_g = self.induced_distribution_g(var_mu, var_sigma, hp)
        # g_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_g, sigma_g)

        mu_f, sigma_f = self.induced_distribution_f(var_mu, var_sigma, hp)
        # f_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_f, sigma_f)

        # return f_dist, g_dist
        return ((mu_f, sigma_f), (mu_g, sigma_g))

    def predict_y(self, log_lambda_diag, hp) -> tuple[Array, Array]:
        pf, pg = self.predict(log_lambda_diag, hp)

        mu_y = pf[0]
        sigma_y = jnp.diag(pf[1]) + jnp.exp(pg[0] + 0.5 * jnp.diag(pg[1]))

        return mu_y, sigma_y
