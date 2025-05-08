import jax
import jax.numpy as jnp
import pytest

from batram_cov.hermgauss import hermgauss


@pytest.mark.parametrize(
    "fn, expect", [(lambda x: x, 0.0), (lambda x: x**2, 1.0), (lambda x: x**3, 0.0)]
)
def test_hermgauss50(fn, expect):
    w, h = hermgauss(50)
    w = jnp.sqrt(2) * w
    h = h / jnp.sqrt(jnp.pi)

    val = jnp.dot(fn(w), h)
    assert jnp.isclose(val, expect)


@pytest.mark.parametrize(
    "fn, expect", [(lambda x: x, 0.0), (lambda x: x**2, 1.0), (lambda x: x**3, 0.0)]
)
def test_hermgauss50_with_jit(fn, expect):
    w, h = hermgauss(50)
    w = jnp.sqrt(2) * w
    h = h / jnp.sqrt(jnp.pi)

    val = jnp.dot(jax.jit(fn)(w), h)
    assert jnp.isclose(val, expect)
