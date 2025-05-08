from time import time_ns

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from flax import nnx

from . import regression_problem as rp
from .data_gen import SimulationData
from .typing import AnomalizedData


def sample_location(
    rng: jax.Array,
    problem: rp.HGPIPProblem,
    responses: jax.Array,
    sample_fixed_noise: bool,
    covar: None | jax.Array = None,
) -> list[rp.TMDataModule]:
    covar = (
        problem.data.covariates.value
        if covar is None
        else jnp.asarray(covar, jnp.float32)
    )
    nn_idx = problem.data.nn_idx.value
    cond_set = responses[0][:, nn_idx]
    mask = nn_idx != -1
    cond_set = cond_set * mask

    data = rp.TMDataModule(
        position=problem.data.position.value,
        response=jnp.zeros(covar.shape[0]),
        conditioning_set=cond_set,
        covariates=covar,
        dist_nn=problem.data.dist_nn.value,
        nn_idx=problem.data.nn_idx.value,
    )

    dist = problem.predict_with_data(data)

    if sample_fixed_noise:
        z = tfp.distributions.Uniform().sample(seed=rng)
        samples = dist.quantile(z)
    else:
        samples = dist.sample(seed=rng)

    data._response.value = samples
    responses = responses.at[0, :, problem.data.position.value].set(samples)
    return responses


def sample(
    rng: jax.Array,
    model: nnx.Module,
    nsamples: int = 1,
    sample_fixed_noise: bool = False,
    covar: None | jax.Array = None,
) -> rp.TMDataModule:
    problems = rp.split_module(model)

    if nsamples > 1:
        raise NotImplementedError("nsamples > 1 is not implemented yet.")
    if covar is not None:
        covar_size = covar.shape[0]
    else:
        covar_size = problems[0].data.covariates.value.shape[0]

    responses = jnp.empty((nsamples, covar_size, len(problems)))
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

    datas = []
    for i in range(len(problems)):
        nn_idx = problems[i].data.nn_idx.value
        data = rp.TMDataModule(
            position=problems[i].data.position.value,
            response=responses[0, :, i],
            conditioning_set=responses[0][:, nn_idx],
            covariates=problems[i].data.covariates.value if covar is None else covar,
            dist_nn=problems[i].data.dist_nn.value,
            nn_idx=nn_idx,
        )
        datas.append(data)

    return rp.merge_modules(datas)


def exp1_1d_covariates(
    rng_key: jax.Array,
    model: nnx.Module,
    data: SimulationData,
    sample_fixed_noise: bool,
    num_samples: int = 1,
) -> tuple[jax.Array, list[jax.Array]]:
    subkeys = jax.random.split(rng_key, num_samples)

    num_fields = data.x.shape[0]
    if num_fields > 8:
        how_many = 8
    else:
        how_many = num_fields

    # step_size = num_fields // how_many
    # x = data.x[::step_size, ...]
    x = data.x
    x = jax.device_put(x)
    assert x.ndim == 2, f"{x.shape=}"
    assert x.shape == (data.samples.shape[1], 1), f"{x.shape=}"

    assert data.samples.ndim == 3, f"{data.samples.shape=}"
    assert data.samples.shape == (1, x.shape[0], data.li.size), f"{data.samples.shape=}"

    rev_order = data.inv_maxmin_permutation
    # samples = [data.samples[0][::step_size, rev_order].T]
    samples = [data.samples[0][:, rev_order].T]
    for i in range(num_samples):
        if sample_fixed_noise:
            prediction = sample(rng_key, model, 1, sample_fixed_noise, covar=x)
        else:
            prediction = sample(subkeys[i], model, 1, sample_fixed_noise, covar=x)
        # samples.append(prediction._response[rev_order, ::step_size])
        samples.append(prediction._response[rev_order, :])

    return x, samples


def exp1_2d_covariates(
    rng_key: jax.Array,
    model: nnx.Module,
    data: SimulationData,
    sample_fixed_noise: bool,
    num_samples: int = 1,
) -> tuple[jax.Array, list[jax.Array]]:
    subkeys = jax.random.split(rng_key, num_samples)

    x = data.x
    x = jax.device_put(x)
    assert x.ndim == 2, f"{x.shape=}"
    assert x.shape == (data.samples.shape[1], 1), f"{x.shape=}"

    rev_order = data.inv_maxmin_permutation
    samples = [data.samples[0][:, rev_order].T]
    for i in range(num_samples):
        if sample_fixed_noise:
            prediction = sample(rng_key, model, 1, sample_fixed_noise, covar=x)
        else:
            prediction = sample(subkeys[i], model, 1, sample_fixed_noise, covar=x)
        samples.append(prediction._response[rev_order, :])

    return x, samples


def cmip_app(
    rng_key: jax.Array,
    model: nnx.Module,
    data: AnomalizedData,
    sample_fixed_noise: bool,
    num_samples: int = 1,
) -> tuple[jax.Array, list[jax.Array]]:
    x = data.x
    x = jax.device_put(x)
    assert x.ndim == 2, f"{x.shape=}"
    assert x.shape == (data.samples.shape[1], 1), f"{x.shape=}, {data.samples.shape=}"

    assert data.samples.ndim == 3, f"{data.samples.shape=}"
    assert data.samples.shape == (1, x.shape[0], data.li.size), f"{data.samples.shape=}"

    rev_order = data.maxmin_permutation.argsort()
    samples = [data.samples[0][:, rev_order].T]
    subkeys = jax.random.split(rng_key, num_samples)
    for i in range(num_samples):
        rng_key = jax.random.fold_in(rng_key, i)
        if sample_fixed_noise:
            prediction = sample(rng_key, model, 1, sample_fixed_noise, covar=x)
        else:
            prediction = sample(subkeys[i], model, 1, sample_fixed_noise, covar=x)
        samples.append(prediction._response[rev_order, :])

    return x, samples


def just_predict(
    model: rp.HGPIPProblem,
    dataset: AnomalizedData | SimulationData,
):
    rng_key = jax.random.key(time_ns() % 2**64)
    if isinstance(dataset, AnomalizedData):
        x, fixed_noise_samples = cmip_app(rng_key, model, dataset, True, num_samples=20)
        x, random_noise_samples = cmip_app(
            rng_key, model, dataset, False, num_samples=20
        )
    else:
        x_shape = dataset.x.shape[-1]
        sample_fn = exp1_1d_covariates if x_shape == 1 else exp1_2d_covariates
        x, fixed_noise_samples = sample_fn(
            rng_key, model, dataset, True, num_samples=20
        )
        x, random_noise_samples = sample_fn(
            rng_key, model, dataset, False, num_samples=20
        )

    return x, fixed_noise_samples, random_noise_samples
