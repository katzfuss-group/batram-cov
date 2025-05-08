import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from flax import nnx
from jax import Array
from tensorflow_probability.substrates.jax.bijectors import Softplus
from veccs.orderings import maxmin_cpp

from .natgrad import MSLTransformation
from .regression_problem import (
    Data,
    HGPIPProblem,
    JointParam,
    JointParamWithBijector,
    MeanFunction,
    TMDataModule,
)


class MeanFunctionG(MeanFunction):
    def __init__(self, var_at_first_loc, data: TMDataModule):
        self.data = data
        self.intercept = JointParam(jnp.asarray(var_at_first_loc, dtype=jnp.float32))
        self.slope = JointParamWithBijector(
            jnp.asarray(0.2, dtype=jnp.float32), Softplus()
        )

    def __call__(self, x: Array) -> Array:
        value = self.intercept.value + self.slope.value * jnp.log(
            self.data.dist_nn.value
        )
        return jnp.repeat(value, x.shape[-2], axis=0)


class TransportMapKernelF(tfp.math.psd_kernels.AutoCompositeTensorPsdKernel):
    def __init__(
        self,
        position,
        length_scale_x,
        tm_q: JointParam,
        tm_q_offset: JointParamWithBijector,
        linear_only: bool,
        x_mean: Data,
        x_std: Data,
        nonlinear_variance: None | Array = None,
        nonlinear_lengthscale: None | Array = None,
        feature_ndims: int = 1,
        normalize_x: bool = True,
        validate_args: bool = False,
        name: str = "TransportMapKernelF",
    ):
        parameters = dict(locals())
        self._length_scale_x = length_scale_x
        self._tm_q = tm_q
        self._tm_q_offset = tm_q_offset
        self._linear_only = linear_only
        self._normalize_x = normalize_x
        self._nonlinear_variance = nonlinear_variance
        self._nonlinear_lengthscale = nonlinear_lengthscale
        self._position = position

        self.x_mean = x_mean
        self.x_std = x_std

        if not self._linear_only:
            if nonlinear_variance is None:
                raise ValueError(
                    "non_linear_variance must be provided if linear_only is False"
                )
            if nonlinear_lengthscale is None:
                raise ValueError(
                    "nonlinear_lengthscale must be provided if linear_only is False"
                )

        # TODO get common dtype and convert all to that
        dtype = length_scale_x.dtype

        super().__init__(
            feature_ndims=feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters,
        )

    @property
    def _dim_covar(self):
        return self._length_scale_x.shape[-1]

    def _prep_data(self, x) -> tuple[Array, Array]:
        covariate = x[..., : self._dim_covar]
        cond_set = x[..., self._dim_covar :]

        dim_cs = cond_set.shape[-1]
        cs_scales = self._calc_cs_scales(dim_cs)
        cond_set = cond_set * cs_scales
        if self._normalize_x:
            covariate = (covariate - self.x_mean.value) / self.x_std.value
        return covariate, cond_set

    # TODO: Do we actually need this? Seems like in `_prep_data` we already have
    # the data in the right format to input into the kernel.
    def _join(self, ys, xs):
        return jnp.concatenate([ys, xs], axis=-1)

    def _calc_cs_scales(self, num_cs: int) -> Array:
        return jnp.exp(
            0.5 * self._tm_q_offset - 0.5 * self._tm_q * jnp.arange(1, num_cs + 1)
        )

    def cs_cutoff(self) -> Array:
        cs_scales = self._calc_cs_scales(100)
        return jnp.argmax(cs_scales < 1e-3)

    def _apply(self, x1, x2, example_ndims=1):
        covar1, cs1 = self._prep_data(x1)
        covar2, cs2 = self._prep_data(x2)

        tfp_krn_x = tfp.math.psd_kernels.FeatureScaled(
            tfp.math.psd_kernels.MaternThreeHalves(), scale_diag=self._length_scale_x
        )
        krn_x = tfp_krn_x.apply(covar1, covar2, example_ndims=example_ndims)

        krn_y = tfp.math.psd_kernels.Linear().apply(
            cs1, cs2, example_ndims=example_ndims
        )
        assert not isinstance(self._position, nnx.Variable)
        linear_part = jax.lax.cond(
            self._position == 0,  # TODO: do we need to test for cs1 == cs2?
            lambda: krn_x,
            lambda: krn_y * krn_x,
        )

        if not self._linear_only:
            ls = self._nonlinear_lengthscale
            nl_krn = tfp.math.psd_kernels.MaternThreeHalves()
            nl_krn = nl_krn.apply(
                self._join(cs1, covar1) / ls,
                self._join(cs2, covar2) / ls,
                example_ndims=example_ndims,
            )
            non_linear_part = nl_krn * self._nonlinear_variance
        else:
            non_linear_part = 0.0

        return linear_part + non_linear_part

    def _matrix(self, x1, x2):
        covar1, cs1 = self._prep_data(x1)
        covar2, cs2 = self._prep_data(x2)

        # TODO: deal with postion == 0 when x1 == x2

        # linear part
        krn_y = tfp.math.psd_kernels.Linear().matrix(cs1, cs2)
        tfp_krn_x = tfp.math.psd_kernels.FeatureScaled(
            tfp.math.psd_kernels.MaternThreeHalves(), scale_diag=self._length_scale_x
        )
        krn_x = tfp_krn_x.matrix(covar1, covar2)
        assert not isinstance(self._position, nnx.Variable)
        linear_part = jax.lax.cond(
            self._position == 0,  # TODO: do we need to test for cs1 == cs2?
            lambda: krn_x,
            lambda: krn_y * krn_x,
        )

        if not self._linear_only:
            ls = self._nonlinear_lengthscale
            nl_krn = tfp.math.psd_kernels.MaternThreeHalves()
            nl_krn = nl_krn.matrix(
                self._join(cs1, covar1) / ls,
                self._join(cs2, covar2) / ls,
            )
            non_linear_part = nl_krn * self._nonlinear_variance
        else:
            non_linear_part = 0.0

        return linear_part + non_linear_part


class KernelF(nnx.Module):
    def __init__(
        self,
        data: TMDataModule,
        linear_only: bool,
        normalize_x: bool = True,
    ):
        self.data = data
        self.tm_q_offset = JointParam(0.0)
        self.tm_q = JointParamWithBijector(1.0, Softplus())
        self.nl_amp_offset = JointParam(0.0)
        self.nl_amp = JointParam(0.0)
        self.nl_lengthscale = JointParamWithBijector(0.3, Softplus())
        self.x_lengthscale = JointParamWithBijector(
            0.1
            * jnp.ones(
                data.feature_dim,
            ),
            Softplus(),
        )
        self.linear_only = linear_only
        self.normalize_x = normalize_x

        m = data.covariates.value.mean(axis=-2, keepdims=True)
        s = data.covariates.value.std(axis=-2, keepdims=True)
        self.x_mean = Data(m)
        self.x_std = Data(s)

    def tfp(self) -> TransportMapKernelF:
        if self.linear_only:
            nonlinear_variance = None
        else:
            res = self.nl_amp_offset.value + self.nl_amp.value * jnp.log(
                self.data.dist_nn.value
            )
            nonlinear_variance = jnp.exp(res)
        return TransportMapKernelF(
            position=self.data.position.value,
            length_scale_x=self.x_lengthscale.value,
            tm_q=self.tm_q.value,
            tm_q_offset=self.tm_q_offset.value,
            linear_only=self.linear_only,
            x_mean=self.x_mean,
            x_std=self.x_std,
            nonlinear_variance=nonlinear_variance,
            nonlinear_lengthscale=self.nl_lengthscale.value,
            normalize_x=self.normalize_x,
            feature_ndims=1,
        )

    def matrix(self, x1: Array, x2: Array) -> Array:
        return self.tfp().matrix(x1, x2)

    def apply(self, x1: Array, x2: Array) -> Array:
        return self.tfp().apply(x1, x2)

    def cs_cutoff(self) -> Array:
        return self.tfp().cs_cutoff()


class MaternThreeHalvesKernel(nnx.Module):
    def __init__(
        self,
        lengthscale: Array,
        variance: Array,
        data: TMDataModule,
        normalize_x: bool = True,
    ):
        self.lengthscale = JointParamWithBijector(
            lengthscale,
            tfp.bijectors.Softplus(),
        )
        self.variance = JointParamWithBijector(variance, tfp.bijectors.Softplus())
        m = data.covariates.value.mean(axis=-2, keepdims=True)
        s = data.covariates.value.std(axis=-2, keepdims=True)
        self.data_mean = Data(m)
        self.data_std = Data(s)
        self.normalize_x = normalize_x

    def _tfp_kernel(self) -> tfp.math.psd_kernels.MaternThreeHalves:
        bk = tfp.math.psd_kernels.MaternThreeHalves(
            amplitude=self.variance.value,
            # length_scale=self.lengthscale.value
        )
        k = tfp.math.psd_kernels.FeatureScaled(bk, scale_diag=self.lengthscale.value)
        return k

    def tfp(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        return self._tfp_kernel()

    def _get_x(self, x: Array) -> Array:
        if self.normalize_x:
            return (x - self.data_mean.value) / self.data_std.value
        return x

    def matrix(self, x1: Array, x2: Array) -> Array:
        x1 = self._get_x(x1)
        x2 = self._get_x(x2)
        return self._tfp_kernel().matrix(x1, x2)

    def apply(self, x1: Array, x2: Array) -> Array:
        x1 = self._get_x(x1)
        x2 = self._get_x(x2)
        return self._tfp_kernel().apply(x1, x2)


def setup_tm_rp(
    response: Array,
    conditioning_set: Array,
    covariates: Array,
    dist_nn: Array,
    position: Array,
    nn_idx: Array,
    num_ip_f: int,
    num_ip_g: int,
    log_var_at_first_loc: float,
    whiten: bool = False,
    variational_noise_f: float = 0.0,
    variational_noise_g: float = 0.0,
    ip_fixed: bool = True,
    ip_distance_penalty: None | float = None,
    linear_only: bool = True,
    normalize_x: bool = True,
) -> HGPIPProblem:
    data = TMDataModule(
        position,
        jnp.asarray(response, dtype=jnp.float32),
        jnp.asarray(conditioning_set, dtype=jnp.float32),
        jnp.asarray(covariates, dtype=jnp.float32),
        jnp.asarray(dist_nn, dtype=jnp.float32),
        jnp.asarray(nn_idx, dtype=jnp.int32),
    )

    kernel_f = KernelF(data, linear_only=linear_only, normalize_x=normalize_x)
    kernel_g = MaternThreeHalvesKernel(
        lengthscale=0.1 * jnp.ones((data.feature_dim,)),
        variance=jnp.ones(()),
        data=data,
        normalize_x=normalize_x,
    )

    # what does make_ips do
    # maxmin order the locations that is (x, cs)
    # select the first num_ips rows
    # if that is not enough then add ips with random noise to get to num_ips rows
    # if num_ips is larger than unqiue x (which should only have for ips for f) then
    #   add noise to the x dimension of the ips that are later than num_unique_x.
    #   that is an issue, since the first num_unique_x rows may contain
    #   repeated x because of maxmin ordering jointly with cs
    #
    # what do we want?
    # initialize the inducing points such that they are sufficently differently
    # even if
    #  - we consider only x
    #  - we consider only cs
    #
    # what should the penalty achieve?
    #  - keep the ips only concerning x sufficiently separated
    #  - keep the ips only concerning x sufficiently separated
    #  - avoid linear dependence
    def make_ips(locs: Array, num_ips: Array, locsg: Array) -> Array:
        if not True:
            return locs[maxmin_cpp(locs)[:num_ips]]
        else:
            nunique_x = np.unique(locsg, axis=0).shape[0]
            feat_dim_x = locsg.shape[1]
            xx = np.unique(locs, axis=0)
            ord = maxmin_cpp(xx)
            ips = xx[ord][:num_ips]
            rng = np.random.default_rng(0)
            add_points_shape = (num_ips - ips.shape[0], ips.shape[1])
            add_points = 1e3 * rng.standard_normal(add_points_shape)
            ips = np.concatenate([ips, add_points], axis=0)
            if num_ips > nunique_x:
                add_x_noise = rng.standard_normal((num_ips - nunique_x, feat_dim_x))
                ips[nunique_x:, :feat_dim_x] += add_x_noise
            # ips += 0.25 * rng.standard_normal(ips.shape)
            return ips

    mb_idx = jnp.arange(data.size)
    locsf = np.asarray(data.input_f(mb_idx), dtype=np.float32)
    locsg = np.asarray(data.input_g(mb_idx), dtype=np.float32)

    ip_f = make_ips(locsf, num_ip_f, locsg)
    ip_f[:, data.feature_dim + position :] = 0.0
    ip_g = make_ips(locsg, num_ip_g, locsg)

    m = jnp.zeros(num_ip_g)
    s = jnp.eye(num_ip_g)
    trf, _ = MSLTransformation.from_ms(m, s)

    return HGPIPProblem(
        data,
        kernel_f,
        kernel_g,
        ip_f,
        ip_g,
        trf,
        # mean_g=ConstMeanFunction(0.0),
        mean_g=MeanFunctionG(log_var_at_first_loc, data),
        whiten=whiten,
        diag_krn_noise_f=variational_noise_f,
        diag_krn_noise_g=variational_noise_g,
        ip_fixed=ip_fixed,
        ip_distance_penalty=ip_distance_penalty,
        clip_predictive_variances=True,
    )
