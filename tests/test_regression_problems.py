import jax
import jax.numpy as jnp
import pytest

import batram_cov.regression_problem as rp


@pytest.fixture
def data() -> rp.TMDataModule:
    key = jax.random.key(0)
    position = jnp.array(100, dtype=int)
    response = jax.random.normal(key, shape=(100,))
    conditioning_set = jax.random.normal(key, shape=(100, 30))
    covariates = jax.random.normal(key, shape=(100,))
    dist_nn = 1 / (jnp.arange(100) + 1)
    nn_idx = jnp.arange(100, dtype=int)
    data = rp.TMDataModule(
        position, response, conditioning_set, covariates, dist_nn, nn_idx
    )

    return data


def test_data(data):
    assert data.response([0]).shape == (), data.response([0]).shape
