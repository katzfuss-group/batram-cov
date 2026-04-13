from __future__ import annotations

import os
import pickle
from functools import partial
from time import time_ns
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow_probability.substrates.jax as tfp
from deprecated import deprecated
from flax import nnx
from tqdm import tqdm
from veccs.orderings import maxmin_cpp

from . import natgrad as ngm
from . import regression_problem as rp
from .stopping import early_stopper
from .tmcov import KernelF, setup_tm_rp
from .typing import ArraysProtocol
from .utils import to_strong_jax_type

RngKey = jax.Array


class MVNParams(NamedTuple):
    mean: np.ndarray | jax.Array
    var: np.ndarray | jax.Array


class FitStatus(NamedTuple):
    model: rp.HGPIPProblem
    train_loss: np.ndarray | None
    validation_loss: np.ndarray | None
    debugging: np.ndarray | None
    fit_passed: bool
    num_neighbors: np.ndarray | None = None  # Compat with old versions


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

    @property
    def num_neighbors(self) -> int:
        assert isinstance(self.model, rp.HGPIPProblem)
        assert isinstance(self.model.kernel_f, KernelF)
        return self.model.kernel_f.cs_cutoff().item()

    def save_model(self, abs_path, checkpointer=None):
        """Save models using `orbax.checkpoint` module

        NOTES:
        - This is not well tested, and it may be easier to save models yourself.

        - This function tries to save everything in the path (directory) provided.

        - See the `flax` docs on checkpointing [1,2] for details on how to do this, and
          note you will want to save both the internal `model` and `validation_data` if
          it is present. These will need to be stored in separate paths within a larger
          file path (consider: os.path.join to create subpaths).

        Args:
        - abs_path (str): an absolute path (e.g. /tmp/models) to a directory for saving
          checkpoint data
        - checkpointer (ocp.checkpoint class): a checkpointing class to save weights
          with. We provide ocp.StandardCheckpointer if no checkpointer is provided.

        Refs:
        [1] Flax checkpointing: docs https://flax.readthedocs.io/en/latest/guides/checkpointing.html
        [2] Orbax checkpointing: https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html
        """
        if not checkpointer:
            checkpointer = ocp.StandardCheckpointer()

        model_path = os.path.join(abs_path, "model")
        model_state = nnx.state(self.model)
        checkpointer.save(model_path, model_state)

        if self.validation_data:
            validation_path = os.path.join(abs_path, "validation_state")
            validation_state = nnx.split(self.validation_data)
            checkpointer.save(validation_path, validation_state)

        key_path = os.path.join(abs_path, "key")
        with open(key_path, "wb") as f:
            pickle.dump(self.key, f)

        return

    def load_model(self, abs_path, checkpointer=None):
        """Load models using `orbax.checkpoint` module

        NOTES:
        - This is not well tested, and you may be better off implementing it yourself.
          This was not used in our original experiments because training was cheap.

        - This function tries to load previous model weights, validation_data, and a
          random key assuming all are present. It does not perform any error handling.

        - See the `flax` docs on checkpointing [1,2] for details on how to do this, and
          note you will want to load both the internal `model` and `validation_data` if
          it is present. These will need to be stored in separate paths within a larger
          file path (consider: os.path.join for subpaths).

        Args:
        - abs_path (str): an absolute path (e.g. /tmp/models) to a directory for saving
          checkpoint data
        - checkpointer (ocp.checkpoint class): a checkpointing class to save weights
          with. We provide ocp.StandardCheckpointer if no checkpointer is provided.

        Refs:
        [1] Flax checkpointing: docs https://flax.readthedocs.io/en/latest/guides/checkpointing.html
        [2] Orbax checkpointing: https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html
        """
        if not checkpointer:
            checkpointer = ocp.StandardCheckpointer()

        model_path = os.path.join(abs_path, "model")
        state = nnx.state(self.model)
        checkpointer.restore(model_path, state)

        if self.validation_data:
            validation_path = os.path.join(abs_path, "validation_state")
            validation_state = nnx.split(self.validation_data)
            checkpointer.restore(validation_path, validation_state)

        key_path = os.path.join(abs_path, "key")
        with open(key_path, "bb") as f:
            pickle.load(f)

        return

    def update_train_data(self, new_x: np.ndarray, new_samples: np.ndarray):
        """Add new_x and new_samples to training data module

        NOTE: The function assumes all necessary preprocessing has been conducted by
        the user _before_ calling this function. This function only concatenates data
        into the training module state.

        Args:
        new_x: an x value with shape (num_samples, x_dim)
        new_samples: fields with shape (num_samples, num_locs)

        Returns:
        None
        """
        graph, state = nnx.split(self.model)
        _response, condsets, covariates, _, nn_idx, _ = state["data"].values()

        new_x = np.broadcast_to(new_x, (covariates.value.shape[0], *new_x.shape))
        new_condsets = new_samples[:, nn_idx.value].transpose(1, 0, -1)
        x = jnp.concat([covariates.value, new_x], axis=-2)
        y = jnp.concat([_response.value, new_samples.T], axis=-1)
        condsets = jnp.concat([condsets.value, new_condsets], axis=-2)

        state["data"]["_response"].value = y
        state["data"]["covariates"].value = x
        state["data"]["conditioning_set"].value = condsets

        self.model = nnx.merge(graph, state)

        return

    def update_model_params(self, previous_state: nnx.State, use_data: bool = False):
        """Update params based on previously saved model state.

        By default, restores a previous parameter state without copying training data
        (the model state binds training data and parameters in the same object). When
        `use_data=True`, we replace the entire previous model state with training data
        and parameters.

        NOTE: This method should only be used in problems that are identically shaped
        to the task of the current model. This means the same number of spatial
        locations _and_ the same dimension of covariates. The sample size can change.

        Args:
        previous_state (nnx.State): some previously stored model state that we wish
            to reload

        use_data (bool=False): whether to copy training data from the previous state
        """
        if use_data:
            # TODO: Confirm split/merge gives a copy of the data rather than bumping a
            # ref counter. The goal here was to replace a deep copy by creating a new
            # object/ref.
            graph, state = nnx.split(previous_state)
            self.model = nnx.merge(graph, state)
            return

        _ = previous_state.pop("data")
        graph, state = nnx.split(self.model)
        for k, v in previous_state.items():
            state[k] = v
        new_state = nnx.merge(graph, state)
        self.model = new_state

        return

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

    @deprecated("Use `CovariateTransportMap.predict_fs_and_gs(data=data)`.")
    def predict(self, data: rp.TMDataModule) -> tuple[MVNParams, MVNParams]:
        f, g = self.predict_fs_and_gs(data=data)
        assert isinstance(f, MVNParams) and isinstance(g, MVNParams)
        return f, g

    def predict_fs_and_gs(
        self,
        *,
        data: rp.TMDataModule | None = None,
        x: np.ndarray | jax.Array | None = None,
        y: np.ndarray | jax.Array | None = None,
    ) -> tuple[MVNParams | None, MVNParams | None]:
        """Predict fs and gs from `x` or `y`.

        This method wraps `predict_from_new_fields` and `predict_from_score_data`
        to simplify scoring new fields. The unified interface means that arguments
        must be passed by name. The two behaviors are defined below.

        **Case 1** `data is not None`: This was the default from the initial
        release. It is predicated on constructing `data` as a `TMDataModule`,
        which makes sense in fully automated workflows but can be painful in
        notebooks or exploratory work. The `Option<MVNParams>` return types
        never hold in this case, but linters may require proving this. Do this
        as follows:

        ```python
        >>> f, g = model.predict(data=prediction_data)
        >>> assert f is not None and g is not None, (f"{type(f).__name__}, {type(g).__name__}")
        ```

        **Case 2** `data is None`: This function simplifies getting `f` and `g`
        from data without building a full `TMDataModule`. There are three
        possible cases:
        - If `x` and `y` are both `None` then `f` and `g` are returned over the
          training data only.
        - If there is only interest in drawing the `g` functions to visualize the
          variance part of the model as a function of `x` then `y` can be omitted.
        - If `x` and `y` are conformably shaped arrays then this is a specialized
          version of `predict_fs_and_gs_from_data_module` which builds the module
          internally to make predictions.
        - See the **Raises** section for error cases.

        **Note**: If `data is None` then the second case is *always* taken.


        **Args**:
        - `rng`: random key for generating samples.
        - `model`: Fitted model.
        - `x`: New covariate values `x` to evaluate the model at. Values may be part
          of training data or disjoint to it.
        - `y` (Optional): Response fields corresponding to the values of `x`.

        **Raises**:
        - `ValueError` if
          - `x is None` but `y is not None` or
          - `x` and `y` are not conformable in dimension or
          - `x` or `y` is not an array castable to a `jax.Array`.
        """

        if data is not None:
            if not isinstance(data, rp.TMDataModule):
                raise ValueError(
                    f"'data' not a TMDataModule. Received {type(data).__name__}."
                )
            return predict_fs_and_gs_from_data_module(self.model, data)

        return predict_from_new_fields(self.model, x, y)

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
            samples.append(prediction)

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

    return responses[0]


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


def predict_fs_and_gs_from_data_module(
    model: rp.HGPIPProblem, score_data: rp.TMDataModule
) -> tuple[MVNParams, MVNParams]:
    g, joint_params, rest = nnx.split(model, rp.JointParam, object)
    data_g, data_val = nnx.split(score_data, object)
    fs, gs = _fs_gs_one_loc_vmap_jit(g, joint_params, rest, data_g, data_val)
    fs = MVNParams(*fs)
    gs = MVNParams(2 * gs[0], 2 * gs[1])
    return fs, gs


# TODO: Implement with running notebook.
def predict_from_new_fields(
    model: rp.HGPIPProblem,
    x: np.ndarray | jax.Array | None = None,
    y: np.ndarray | jax.Array | None = None,
) -> tuple[MVNParams | None, MVNParams | None]:
    """Predict fs and gs from `x` or `y`.

    This function simplifies getting `f` and `g` from data without building a
    full `TMDataModule`. There are three possible cases:
    - If `x` and `y` are both `None` then `f` and `g` are returned over the
      training data only.
    - If there is only interest in drawing the `g` functions to visualize the
      variance part of the model as a function of `x` then `y` can be omitted.
    - If `x` and `y` are conformably shaped arrays then this is a specialized
      version of `predict_fs_and_gs_from_data_module` which builds the module
      internally to make predictions.
    - See the **Raises** section for error cases.

    **Args**:
    - `model`: Fitted model.
    - `x`: New covariate values `x` to evaluate the model at. Values may be part
      of training data or disjoint to it.
    - `y` (Optional): Response fields corresponding to the values of `x`.

    **Raises**:
    - `ValueError` if
      - `x is None` but `y is not None` or
      - `x` and `y` are not conformable in dimension or
      - `x` or `y` is not an array castable to a `jax.Array`.
    """

    _arraylike_t = jax.Array | np.ndarray

    if x is None and y is None:
        f, g = predict_fs_and_gs_from_data_module(model, model.data)
        return MVNParams(*f), MVNParams(*g)

    elif isinstance(x, _arraylike_t) and isinstance(y, _arraylike_t):
        x = jax.device_put(x)
        response = jax.device_put(y)

    elif x is not None and y is None:
        if not isinstance(x, _arraylike_t):
            raise ValueError(
                f"'x' is not a numpy or jax Array. Got {type(x).__name__} and 'y' None."
            )
        x = jax.device_put(x)
        N = model.data._response.shape[0]
        response = jnp.zeros((x.shape[0], N))

    elif x is None and y is not None:
        raise ValueError(
            "got 'x' as None and data 'y'. Provide 'x' conformable with 'y' "
            "to generate data in this case."
        )

    else:
        raise ValueError(
            f"invalid inputs 'x' and 'y'. Received "
            f"'x' of type {type(x).__name__} and 'y' of type "
            f"{type(y).__name__}. Must provide x and y of "
            "None | np.ndarray | jax.Array."
        )

    # DEBUG:
    nn_idx = model.data.nn_idx.value
    dist_nn = model.data.dist_nn.value
    mask = nn_idx != -1

    modules = []
    for i in range(response.shape[-1]):
        nns_i = nn_idx[i]
        cond_set = response[:, nns_i] * mask[i]
        modules.append(
            rp.TMDataModule(
                position=jnp.array(i, dtype=jnp.int32),
                response=response[:, i],
                conditioning_set=cond_set,
                covariates=x,
                dist_nn=dist_nn[i],
                nn_idx=nn_idx[i],
            )
        )

    data = rp.merge_modules(modules)

    f, g = predict_fs_and_gs_from_data_module(model, data)

    # NOTE: These are the only two possible paths.
    if y is None:
        f = None
        assert f is None
        return f, g
    else:
        assert f is not None
        return f, g


# NOTE: This forces using natgrads for training. We found this was substantially
# faster in pre-release development.
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
        var_mvn_params = optax.apply_updates(var_mvn_params, updates)

        # build return values
        loss = -elbo_val
        new_params = (params, var_mvn_params)
        new_opt_states = (opt_ng_state, opt_state)

        return new_params, new_opt_states, loss

    stopper = partial(
        early_stopper,
        warmup_phase=warmup_steps,
        patience=stopper_patience,
        tol=stopper_tol,
    )
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
        fs, gs = predict_fs_and_gs_from_data_module(problem, score_data)
        mu = fs[0]
        var = fs[1] + jnp.exp(gs[0])
        y_test = score_data.response(np.arange(score_data._response.shape[0]))
        lp = (
            jax.scipy.stats.norm.logpdf(y_test, loc=mu, scale=jnp.sqrt(var))
            .sum(0)
            .mean()
        )
        return lp

    # initialize tracking
    tracker = {
        "loss": np.nan * np.zeros(num_steps, dtype=np.float32),
        "num_nbrs": np.zeros(num_steps, dtype=np.uint8),
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
        tracker["num_nbrs"][epoch] = problem.kernel_f.cs_cutoff()

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

    mean_pred_log_prob = tracker.get("mean_pred_log_prob", None)
    if mean_pred_log_prob is not None:
        mean_pred_log_prob = -mean_pred_log_prob
    num_nbrs = tracker.get("num_nbrs", None)
    if isinstance(num_nbrs, np.ndarray):
        num_nbrs = num_nbrs[:epoch]
    assert isinstance(num_nbrs, np.ndarray)
    return FitStatus(
        problem,
        tracker.get("loss", None),
        mean_pred_log_prob,
        tracker.get("debugging", None),
        fit_pass,
        num_nbrs,
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
