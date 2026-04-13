from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

import batram_cov.regression_problem as rp


@pytest.fixture
def one_point() -> rp.TMDataModule:
    """Used in `test_one_point`."""

    key = jax.random.key(0)
    position = jnp.array(0, dtype=int)
    response = jax.random.normal(key, shape=(1,))
    conditioning_set = jax.random.normal(key, shape=(1, 30))
    covariates = jax.random.normal(key, shape=(1, 1))
    dist_nn = 1 / (position + 1)
    nn_idx = jnp.zeros((1, 30), dtype=int).at[:, 1:].set(-1)
    data = rp.TMDataModule(
        position, response, conditioning_set, covariates, dist_nn, nn_idx
    )
    return data


@pytest.fixture
def n_points() -> rp.TMDataModule:
    """Used in `test_n_points`."""

    key = jax.random.key(0)
    position = jnp.arange(100, dtype=int).reshape(-1, 1)
    response = jax.random.normal(key, shape=(100, 5))
    conditioning_set = jax.random.normal(key, shape=(100, 30))
    covariates = jax.random.normal(key, shape=(5, 1))
    dist_nn = 1 / (position + 1)
    nn_idx = jnp.zeros((100, 30), dtype=int).at[:, 1:].set(-1)
    data = rp.TMDataModule(
        position, response, conditioning_set, covariates, dist_nn, nn_idx
    )
    return data


def test_one_point(one_point):
    def _check_shape(
        field: Callable, mb_idx: int, expect: tuple[int | None, ...]
    ) -> None:
        assert field(mb_idx).shape == expect, field(mb_idx).shape

    _check_shape(one_point.response, 0, ())
    _check_shape(one_point.input_f, 0, (31,))
    _check_shape(one_point.input_g, 0, (1,))


def test_n_points(n_points):
    # NOTE: mb_idx must be provided as an array type based on current jax
    # version (0.5.0) for lax interfaces.
    def _check_shape(
        field: Callable, mb_idx: jax.Array, expect: tuple[int | None, ...]
    ) -> None:
        assert field(mb_idx).shape == expect, field(mb_idx).shape

    _check_shape(n_points.response, jnp.array([0]), (1, 5))
    _check_shape(n_points.input_f, jnp.array([0]), (1, 31))
    _check_shape(n_points.input_g, jnp.array([0]), (1, 1))
