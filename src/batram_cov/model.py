from __future__ import annotations

from functools import partial
from time import time_ns
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax import nnx
from tqdm import tqdm
from veccs.orderings import maxmin_cpp

from . import natgrad as ngm
from . import regression_problem as rp
from .stopping import early_stopper
from .tmcov import setup_tm_rp
from .typing import ArraysProtocol
from .utils import to_strong_jax_type

type RngKey = jax.Array
type MVNParams = tuple[jax.Array, jax.Array]
type FsAndGs = tuple[MVNParams, MVNParams]


class FitStatus(NamedTuple):
    model: rp.HGPIPProblem
    train_loss: np.ndarray | None
    validation_loss: np.ndarray | None
    debugging: np.ndarray | None
    fit_passed: bool


# The main things we should instantiate this object with are datasets, not
# models. Models can be built from data and configuration parameters.
class CovariateTransportMap:
    def __init__(
        self,
        problems: rp.HGPIPProblem,
        validation_data: rp.TMDataModule | None = None,
        seed: int | None = None,
    ):
        self.model = problems
        self.validation_data = validation_data
        self.key = jax.random.key(seed) if seed else jax.random.key(time_ns())

    def fit(
        self,
        num_steps: int = 1000,
        warmup_steps: int = 200,
        init_lr: float = 0.0001,
        peak_lr: float = 0.03,
        min_lr: float = 0.00001,
        num_epochs_only_var_par: float = 0,
        stopper_patience: int = 30,
        stopper_tol: float = 0.0,
    ) -> FitStatus:
        """fit covariate transport map using initialized data

        Arguments:
        ---
        :param:`num_steps (int = 1000)`
            The number of post-warmup steps to train the model for
        :param:`warmup_steps (int = 200)`
            The number of warmup steps to take before training
        :param:`init_lr (float = 0.0001)`
            The initial learning rate to use during warmup training
        :param:`peak_lr (float = 0.03)`
            The leak learning rate to apply during training
        :param:`min_lr (float = 0.00001)`
            The minimum learning rate to use with schedling
        :param:`num_epochs_only_var_par (float = 0)`
            The number of epochs to train only variational parameters for
        :param:`stopper_patience (int = 30)`
            The patience for an early stopper

        Returns:
        ---
        :param:`model (rp.HGPIPProblem)`
            The fitted model
        :param:`loss (np.ndarray)`
            The training (elbo) loss
        :param:`validation_loss (np.ndarray)`
            Prediction loss on the validation set
        :param:`fit_failed (bool)`
            Whether or not the fit failed
        """

        fit_status = fit_model(
            self.model,
            self.validation_data,
            num_steps,
            warmup_steps,
            init_lr,
            peak_lr,
            min_lr,
            num_epochs_only_var_par,
            stopper_patience,
            stopper_tol,
        )

        if fit_status.fit_passed:
            self.model = fit_status.model

        return fit_status

    def logprob(self, data: rp.TMDataModule) -> jax.Array:
        return log_prob_with_data(self.model, data)

    def predict(self, data: rp.TMDataModule) -> FsAndGs:
        return predict_fs_and_gs(self.model, data)

    def sample(
        self,
        x: jax.Array,
        num_samples: int = 1,
        seed: int | None = None,
        sample_fixed_noise: bool = False,
    ) -> jax.Array:
        if seed:
            rng_key = jax.random.key(seed)
        else:
            new_key, rng_key = jax.random.split(self.key)
            self.key = new_key

        model = self.model
        samples = []

        for i in range(num_samples):
            if sample_fixed_noise:
                prediction = sample(rng_key, model, sample_fixed_noise, covar=x)
            else:
                rng_key = jax.random.fold_in(rng_key, i)
                prediction = sample(rng_key, model, sample_fixed_noise, covar=x)
            samples.append(prediction._response)

        return jnp.stack(samples)


def build_data_module(
    data: ArraysProtocol, sample_idx: int = 0, max_nns: int = 50
) -> rp.TMDataModule:
    dataj = jax.tree.map(
        lambda a: jax.device_put(a) if isinstance(a, np.ndarray) else a,
        data,
    )

    modules = []
    for i in range(dataj.samples.shape[2]):
        x = dataj.x
        nn_idx = dataj.nearest_neighbors[i, :max_nns]
        cond_set = dataj.samples[sample_idx][:, nn_idx]
        dist_to_nn = dataj.li[i]
        resp = dataj.samples[sample_idx, :, i]

        modules.append(
            rp.TMDataModule(
                position=jnp.array(i, dtype=jnp.int32),
                response=resp,
                conditioning_set=cond_set,
                covariates=x,
                dist_nn=dist_to_nn,
                nn_idx=nn_idx,
            )
        )

    return rp.merge_modules(modules)


def create_and_merge_problems(
    data: ArraysProtocol,
    num_ip_f: int,
    num_ip_g: int,
    sample_idx: int = 0,
    max_nns: int = 30,
    whiten: bool = False,
    variational_noise_f: float = 0.0,
    variational_noise_g: float = 0.0,
    ip_fixed: bool = True,
    ip_distance_penalty: None | float = None,
    linear_only: bool = True,
    normalize_x: bool = True,
) -> rp.HGPIPProblem:
    location_subset = None
    if location_subset is None:
        location_subset = np.arange(data.samples.shape[2])

    problems = []
    for i in location_subset:
        x = data.x
        nn_idx = data.nearest_neighbors[i, :max_nns]
        cond_set = data.samples[sample_idx][:, nn_idx]
        dist_to_nn = data.li[i]
        resp = data.samples[sample_idx, :, i]
        var_at_first_loc = np.var(data.samples[sample_idx, :, 0])

        problems.append(
            setup_tm_rp(
                resp,
                cond_set,
                x,
                dist_to_nn,
                i,
                nn_idx,
                num_ip_f,
                num_ip_g,
                log_var_at_first_loc=np.log(var_at_first_loc),
                whiten=whiten,
                variational_noise_f=variational_noise_f,
                variational_noise_g=variational_noise_g,
                ip_fixed=ip_fixed,
                ip_distance_penalty=ip_distance_penalty,
                linear_only=linear_only,
                normalize_x=normalize_x,
            )
        )

    # merge problems
    return rp.merge_modules(problems)


def sample_location(
    rng: jax.Array,
    problem: rp.HGPIPProblem,
    responses: jax.Array,
    sample_fixed_noise: bool,
    x: None | jax.Array = None,
) -> jax.Array:
    """sample an hgp regression at one location.

    Arguments
    ---
    :param:`rng (jax.Array)` key for generating samples

    :param:`problem (rp.HGPIPProblem)` fitted model

    :param:`responses (jax.Array)` empty buffer to store results in

    :param:`sample_fixed_noise (bool)` whether to fix the noise across samples
        or draw different samples for each x

    :param:`x (jax.Array | None = None)` new data locations to draw samples at.
        If `None` then use the training data
    """
    x = problem.data.covariates.value if x is None else jnp.asarray(x, jnp.float32)
    nn_idx = problem.data.nn_idx.value
    cond_set = responses[0][:, nn_idx]
    mask = nn_idx != -1
    cond_set = cond_set * mask

    data = rp.TMDataModule(
        position=problem.data.position.value,
        response=jnp.zeros(x.shape[0]),
        conditioning_set=cond_set,
        covariates=x,
        dist_nn=problem.data.dist_nn.value,
        nn_idx=problem.data.nn_idx.value,
    )

    dist = problem.predict_with_data(data)

    if sample_fixed_noise:
        z = tfp.distributions.Uniform().sample(seed=rng)
        samples = dist.quantile(z)
    else:
        samples = dist.sample(seed=rng)

    assert isinstance(samples, jax.Array), (
        f"got type(samples) {type(samples).__name__} instead of a jax.Array"
    )

    data._response.value = samples
    responses = responses.at[0, :, problem.data.position.value].set(samples)
    return responses


def sample(
    rng: jax.Array,
    model: rp.HGPIPProblem,
    sample_fixed_noise: bool = False,
    covar: None | jax.Array = None,
) -> rp.TMDataModule:
    problems = rp.split_module(model)

    if covar is not None:
        covar_size = covar.shape[0]
    else:
        covar_size = problems[0].data.covariates.value.shape[0]

    responses = jnp.empty((1, covar_size, len(problems)))
    g, jp, rs = nnx.split(model, rp.JointParam, object)

    def loop_body(i, carry):
        rng, responses, g, jp = carry
        rng, subkey = jax.random.split(rng)
        r = jax.tree_util.tree_map(lambda x: x[i], rs)
        problem = nnx.merge(g, jp, r)
        responses = sample_location(
            subkey, problem, responses, sample_fixed_noise, covar
        )
        return rng, responses, g, jp

    _, responses, _, _ = jax.lax.fori_loop(
        0, len(problems), loop_body, (rng, responses, g, jp)
    )

    samples = []
    for i in range(len(problems)):
        nn_idx = problems[i].data.nn_idx.value
        data = rp.TMDataModule(
            position=problems[i].data.position.value,
            response=responses[0, :, i],
            conditioning_set=responses[0, :, nn_idx],
            covariates=problems[i].data.covariates.value if covar is None else covar,
            dist_nn=problems[i].data.dist_nn.value,
            nn_idx=nn_idx,
        )
        samples.append(data)

    return rp.merge_modules(samples)


@jax.jit
def _log_prob_one_location_vmap_jit(g, joint_params, rest, data_g, data_val):
    def _apply(g, joint_params, rest, data_g, data_val):
        m: rp.HGPIPProblem = nnx.merge(g, joint_params, rest)
        data = nnx.merge(data_g, data_val)
        log_prob = m.log_prob(data)
        return log_prob

    apply = jax.vmap(_apply, (None, None, 0, None, 0))
    return apply(g, joint_params, rest, data_g, data_val)


def log_prob_with_data(
    model: rp.HGPIPProblem, score_data: rp.TMDataModule
) -> jax.Array:
    g, joint_params, rest = nnx.split(model, rp.JointParam, object)
    data_g, data_val = nnx.split(score_data, object)
    scores = _log_prob_one_location_vmap_jit(g, joint_params, rest, data_g, data_val)
    return scores


@jax.jit
def _fs_gs_one_loc_vmap_jit(g, joint_params, rest, data_g, data_val):
    def _apply(g, joint_params, rest, data_g, data_val):
        m: rp.HGPIPProblem = nnx.merge(g, joint_params, rest)
        data = nnx.merge(data_g, data_val)
        fs, gs = m.fs_and_gs_from_data(data)
        return fs, gs

    apply = jax.vmap(_apply, (None, None, 0, None, 0))
    return apply(g, joint_params, rest, data_g, data_val)


def predict_fs_and_gs(
    model: rp.HGPIPProblem, score_data: rp.TMDataModule
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    g, joint_params, rest = nnx.split(model, rp.JointParam, object)
    data_g, data_val = nnx.split(score_data, object)
    fs, gs = _fs_gs_one_loc_vmap_jit(g, joint_params, rest, data_g, data_val)
    return fs, gs


# use natgrads by default,
def fit_model(
    problem: rp.HGPIPProblem,
    score_data: rp.TMDataModule | None = None,
    num_steps: int = 1000,
    warmup_steps: int = 200,
    init_lr: float = 0.0001,
    peak_lr: float = 0.03,
    min_lr: float = 0.00001,
    num_epochs_only_var_par: float = 0,
    stopper_patience: int = 30,
    stopper_tol: float = 0.0,
) -> FitStatus:
    """fit covariate transport map using initialized data

    Arguments:
    ---
    :param:`problem (rp.HGPIPProblem)`
        The model to fit
    :param:`score_data (rp.TMDataModule | None = None)`
        Validation data to use in the function
    :param:`num_steps (int = 1000)`
        The number of post-warmup steps to train the model for
    :param:`warmup_steps (int = 200)`
        The number of warmup steps to take before training
    :param:`init_lr (float = 0.0001)`
        The initial learning rate to use during warmup training
    :param:`peak_lr (float = 0.03)`
        The leak learning rate to apply during training
    :param:`min_lr (float = 0.00001)`
        The minimum learning rate to use with schedling
    :param:`num_epochs_only_var_par (float = 0)`
        The number of epochs to train only variational parameters for
    :param:`stopper_patience (int = 30)`
        The patience for an early stopper

    Returns:
    ---
    :param:`model (rp.HGPIPProblem)`
        The fitted model
    :param:`loss (np.ndarray)`
        The training (elbo) loss
    :param:`validation_loss (np.ndarray)`
        Prediction loss on the validation set
    :param:`fit_failed (bool)`
        Whether or not the fit failed
    """
    # split module and convert to strong types to avoid jax recompilation
    g, var_mvn_params, params, gstatic = to_strong_jax_type(
        nnx.split(problem, rp.VarMVNPar, nnx.Param, object)
    )
    problem = nnx.merge(g, params, var_mvn_params, gstatic)

    if num_steps - num_epochs_only_var_par < 0:
        raise ValueError(
            "num_epochs_only_var_par must be less than or equal to num_epochs."
        )

    # define step function
    def update(epoch, g, params, var_mvn_params, gstatic, opt_states):
        opt_ng_state, opt_state = opt_states

        # updates inducing points
        def _elbo_natgrad(g, jp, var_mvn_params, r):
            return ngm.elbo_and_nat_grad(
                lambda vmp: nnx.merge(g, jp, vmp, r).elbo(None),
                var_mvn_params["xi"].trf,
                var_mvn_params,
            )

        # vmap inducing point updates
        def _velbo_natgrad(g, jp, var_mvn_params, r):
            elbos, grad = jax.vmap(_elbo_natgrad, (None, None, 0, 0))(
                g, jp, var_mvn_params, r
            )
            return elbos.sum(), grad

        # updates mean / covariance kernel params
        def _param_update():
            _, grad = jax.value_and_grad(_vloss)(ps)
            updates, opt_state_new = opt.update(grad, opt_state)
            ps_updated = optax.apply_updates(ps, updates)
            return ps_updated, opt_state_new

        # vmap updates
        def _vloss(params):
            m = nnx.merge(g, params, var_mvn_params, gstatic)
            ig, jp, rs = nnx.split(m, rp.JointParam, object)
            vloss = jax.vmap(
                lambda ig, jp, r: nnx.merge(ig, jp, r).loss(None), (None, None, 0)
            )
            # use mean instead of sum to make the learning rate more
            # independent of the number of locations
            return vloss(ig, jp, rs).mean()

        ps = params
        opt_state = opt_states[1]

        # update parameters only when epoch > num_epochs_only_var_par
        params, opt_state = jax.lax.cond(
            epoch < num_epochs_only_var_par,
            lambda: (params, opt_state),
            _param_update,
        )

        # merge and split to have different parameter separation
        m = nnx.merge(g, params, var_mvn_params, gstatic)
        g, jp, var_mvn_params, gstatic = nnx.split(
            m, rp.JointParam, rp.VarMVNPar, object
        )
        elbo_val, grad = _velbo_natgrad(g, jp, var_mvn_params, gstatic)

        updates, opt_ng_state = opt_ng.update(grad, opt_ng_state)
        var_mvn_params = optax.apply_updates(var_mvn_params, updates)  # type: ignore

        # build return values
        loss = -elbo_val
        new_params = (params, var_mvn_params)
        new_opt_states = (opt_ng_state, opt_state)

        return new_params, new_opt_states, loss

    stopper = partial(early_stopper, warmup_phase=warmup_steps, patience=stopper_patience, tol=stopper_tol)
    scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_lr, peak_lr, warmup_steps, num_steps, min_lr
    )
    opt = optax.chain(optax.clip(10), optax.adam(scheduler))

    opt_ng = optax.chain(
        optax.clip(10),
        optax.scale(1e-1),
    )

    _, stop_state = stopper(val=float("inf"), params=None, stop_state=None)
    opt_state = opt.init(params)
    opt_ng_state = opt_ng.init(var_mvn_params)
    opt_states = (opt_ng_state, opt_state)

    def calc_standard_score(score_data):
        fs, gs = predict_fs_and_gs(problem, score_data)
        mu = fs[0]
        var = fs[1] + jnp.exp(gs[0])
        y_test = score_data.response(np.arange(score_data._response.shape[0]))
        lp = jax.scipy.stats.norm.logpdf(y_test, loc=mu, scale=jnp.sqrt(var)).sum(0).mean()
        return lp

    # initialize tracking
    tracker = {
        "loss": np.nan * np.zeros(num_steps, dtype=np.float32),
    }


    if score_data is not None:
        tracker["mean_pred_log_prob"] = np.nan * np.zeros(num_steps, dtype=np.float32)
        # tracker['debugging'] = np.nan * np.zeros(num_steps, dtype=np.float32)


    fit_pass = True

    update_jitted = jax.jit(
        update, donate_argnames=("opt_states", "params", "var_mvn_params")
    )

    # run training loop
    for epoch in (bar := tqdm(range(num_steps))):
        g, var_mvn_params, params, gstate = nnx.split(
            problem, rp.VarMVNPar, nnx.Param, object
        )
        (params, var_mvn_params), opt_states, loss = update_jitted(
            epoch, g, params, var_mvn_params, gstate, opt_states
        )

        nnx.update(problem, params, var_mvn_params)

        # update tracker and progress bar
        tracker["loss"][epoch] = loss.item()

        if score_data is not None:
            log_scores = log_prob_with_data(problem, score_data)
            log_prob_mean = log_scores.sum(0).mean()
            tracker["mean_pred_log_prob"][epoch] = log_prob_mean
            # tracker['debugging'][epoch] = calc_standard_score(score_data)

            stop, stop_state = stopper(
                val=-log_prob_mean.item(),
                params=(params, var_mvn_params),
                stop_state=stop_state,
            )
            if stop:
                params, var_mvn_params = stop_state[-1]
                nnx.update(problem, params, var_mvn_params)
                break

        desc = ", ".join([f"{k}: {v[epoch]:.3f}" for k, v in tracker.items()]) 
        # desc += f"{stop_state[:-1]}"
        bar.set_description(desc)

        # check for nan loss
        if np.isnan(loss):
            fit_pass = False
            break

    # print(stop_state[:-1])

    return FitStatus(
        problem,
        tracker.get("loss", None),
        -tracker.get("mean_pred_log_prob", None),
        tracker.get('debugging', None),
        fit_pass,
    )


def _make_ips(locs: np.ndarray, num_ips: int, locsg: np.ndarray) -> np.ndarray:
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
