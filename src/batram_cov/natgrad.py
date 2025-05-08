from collections.abc import Callable
from typing import Protocol, Self

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_static


def _flip(x: Array) -> Array:
    """
    Flips the rows and columns of a matrix.

    Equivalent to np.fliplr(np.flipud(x)) or P x P where P is the antidiagonal
    matrix.
    """
    return jnp.flipud(jnp.fliplr(x))


def _msl_to_nat(m: Array, sl: Array) -> tuple[Array, Array]:
    nat1 = jax.scipy.linalg.cho_solve((sl, True), m)
    sl_inv = jax.scipy.linalg.solve_triangular(sl, jnp.eye(sl.shape[0]), lower=True)
    nat2 = -0.5 * sl_inv.T @ sl_inv
    return nat1, nat2


def _nat_to_msl(nat1: Array, nat2: Array) -> tuple[Array, Array]:
    # s = -0.5 * jnp.linalg.inv(nat2)
    nat2_chol = jnp.linalg.cholesky(-2 * _flip(nat2))
    nat2_chol_inv = jax.scipy.linalg.solve_triangular(
        nat2_chol, jnp.eye(nat2_chol.shape[0]), lower=True
    )
    sl = _flip(nat2_chol_inv).T

    # m = s @ nat1
    m = sl @ sl.T @ nat1

    # s = jnp.linalg.inv(-2 * nat2)
    # m = s @ nat1
    # sl = jnp.linalg.cholesky(s)

    return m, sl


def _msl_to_exp(m: Array, sl: Array) -> tuple[Array, Array]:
    exp1 = m
    mvec = m[:, jnp.newaxis]
    s = sl @ sl.T
    exp2 = s + mvec @ mvec.T
    return exp1, exp2


def _exp_to_msl(exp1: Array, exp2: Array) -> tuple[Array, Array]:
    m = exp1
    mvec = m[:, jnp.newaxis]
    s = exp2 - mvec @ mvec.T
    sl = jnp.linalg.cholesky(s)
    return m, sl


def _nat_to_exp(nat1: Array, nat2: Array) -> tuple[Array, Array]:
    return _msl_to_exp(*_nat_to_msl(nat1, nat2))


def _exp_to_nat(exp1: Array, exp2: Array) -> tuple[Array, Array]:
    return _msl_to_nat(*_exp_to_msl(exp1, exp2))


class XiTransformation(Protocol):
    @staticmethod
    def from_ms(m: Array, s: Array) -> tuple[Self, Array]: ...

    def xi_to_msl(self, xi: Array) -> tuple[Array, Array]: ...

    def xi_to_exp(self, xi: Array) -> Array: ...

    def exp_to_xi(self, exp: Array) -> Array: ...

    def nat_to_xi(self, nat: Array) -> Array: ...

    def xi_to_nat(self, xi: Array) -> Array: ...

    def jvp_needed(self) -> bool:
        return True


@register_static
class MSLTransformation(XiTransformation):
    def __init__(self, xi_to_msl, unflatten_nat, unflatten_exp, bijector):
        self._xi_to_msl = xi_to_msl
        self._unflatten_nat = unflatten_nat
        self._unflatten_exp = unflatten_exp
        self._bijector = bijector

    @staticmethod
    def from_ms(m: Array, s: Array, diag_shift=None) -> tuple[Self, Array]:
        m = m
        sl = jnp.linalg.cholesky(s)

        bijector = tfp.bijectors.FillScaleTriL(diag_shift=diag_shift)

        xi_pair = (m, bijector.inverse(sl))
        nat_pair = _msl_to_nat(m, sl)
        exp_pair = _msl_to_exp(m, sl)

        xi_flat, unravel_xi = ravel_pytree(xi_pair)

        def xi_to_msl(xi: Array) -> tuple[Array, Array]:
            m, sl_uc = unravel_xi(xi)
            sl = bijector.forward(sl_uc)
            return m, sl

        _, unravel_nat = ravel_pytree(nat_pair)
        _, unravel_exp = ravel_pytree(exp_pair)

        return MSLTransformation(xi_to_msl, unravel_nat, unravel_exp, bijector), xi_flat

    def xi_to_msl(self, xi: Array) -> tuple[Array, Array]:
        m, sl = self._xi_to_msl(xi)
        return m, sl

    def msl_to_xi(self, m: Array, sl: Array) -> Array:
        xi_pair = (m, self._bijector.inverse(sl))
        xi, _ = ravel_pytree(xi_pair)
        return xi

    def xi_to_exp(self, xi: Array) -> Array:
        m, sl = self._xi_to_msl(xi)
        exp_pair = _msl_to_exp(m, sl)
        exp, _ = ravel_pytree(exp_pair)
        return exp

    def exp_to_xi(self, exp: Array) -> Array:
        exp1, exp2 = self._unflatten_exp(exp)
        m, sl = _exp_to_msl(exp1, exp2)
        xi_pair = self.msl_to_xi(m, sl)
        xi, _ = ravel_pytree(xi_pair)
        return xi

    def nat_to_xi(self, nat: Array) -> Array:
        nat1, nat2 = self._unflatten_nat(nat)
        m, sl = _nat_to_msl(nat1, nat2)
        xi_pair = self.msl_to_xi(m, sl)
        xi, _ = ravel_pytree(xi_pair)
        return xi

    def xi_to_nat(self, xi: Array) -> Array:
        m, sl = self._xi_to_msl(xi)
        nat_pair = _msl_to_nat(m, sl)
        nat, _ = ravel_pytree(nat_pair)
        return nat


@register_static
class NatTrilTransformation(XiTransformation):
    def __init__(self, xi_to_nat, nat_to_xi, unflatten_exp, unflatten_nat, bijector):
        self._xi_to_nat = xi_to_nat
        self._nat_to_xi = nat_to_xi
        self._unflatten_exp = unflatten_exp
        self._unflatten_nat = unflatten_nat
        self._bijector = bijector

    @staticmethod
    def from_ms(m: Array, s: Array, diag_shift) -> tuple[Self, Array]:
        bijector = tfp.bijectors.FillScaleTriL(diag_shift=diag_shift)

        sl = jnp.linalg.cholesky(s)
        nat_pair = _msl_to_nat(m, sl)

        def nat_to_xi(nat1: Array, nat2: Array) -> Array:
            neg_nat2_l = jnp.linalg.cholesky(-nat2)
            l_uc = bijector.inverse(neg_nat2_l)
            xi_pair = (nat1, l_uc)
            xi_flat, unravel_xi = ravel_pytree(xi_pair)
            return xi_flat, unravel_xi

        xi_flat, unravel_xi = nat_to_xi(*nat_pair)

        def xi_to_nat(xi: Array) -> tuple[Array, Array]:
            nat1, l_uc = unravel_xi(xi)
            neg_nat2_l = bijector.forward(l_uc)
            nat2 = -neg_nat2_l @ neg_nat2_l.T
            return nat1, nat2

        exp_pair = _nat_to_exp(*nat_pair)

        _, unravel_nat = ravel_pytree(nat_pair)
        _, unravel_exp = ravel_pytree(exp_pair)

        return (
            NatTrilTransformation(
                xi_to_nat,
                lambda n1, n2: nat_to_xi(n1, n2)[0],
                unravel_exp,
                unravel_nat,
                bijector,
            ),
            xi_flat,
        )

    def xi_to_msl(self, xi: Array) -> tuple[Array, Array]:
        nat1, nat2 = self._xi_to_nat(xi)
        m, sl = _nat_to_msl(nat1, nat2)
        return m, sl

    def xi_to_exp(self, xi: Array) -> Array:
        nat1, nat2 = self._xi_to_nat(xi)
        exp_pair = _nat_to_exp(nat1, nat2)
        exp, _ = ravel_pytree(exp_pair)
        return exp

    def exp_to_xi(self, exp: Array) -> Array:
        exp1, exp2 = self._unflatten_exp(exp)
        nat1, nat2 = _exp_to_nat(exp1, exp2)
        xi = self._nat_to_xi(nat1, nat2)
        return xi

    def nat_to_xi(self, nat: Array) -> Array:
        nat1, nat2 = self._unflatten_nat(nat)
        xi = self._nat_to_xi(nat1, nat2)
        return xi

    def xi_to_nat(self, xi: Array) -> Array:
        nat_pair = self._xi_to_nat(xi)
        nat, _ = ravel_pytree(nat_pair)
        return nat


@register_static
class NatTransformation(XiTransformation):
    def __init__(self, xi_to_nat, nat_to_xi, unflatten_exp, unflatten_nat):
        self._xi_to_nat = xi_to_nat
        self._nat_to_xi = nat_to_xi
        self._unflatten_exp = unflatten_exp
        self._unflatten_nat = unflatten_nat

    @staticmethod
    def from_ms(m: Array, s: Array) -> tuple[Self, Array]:
        sl = jnp.linalg.cholesky(s)
        nat_pair = _msl_to_nat(m, sl)

        def nat_to_xi(nat1: Array, nat2: Array) -> Array:
            xi_pair = (nat1, nat2)
            xi_flat, unravel_xi = ravel_pytree(xi_pair)
            return xi_flat, unravel_xi

        xi_flat, unravel_xi = nat_to_xi(*nat_pair)

        def xi_to_nat(xi: Array) -> tuple[Array, Array]:
            nat1, nat2 = unravel_xi(xi)
            return nat1, nat2

        exp_pair = _nat_to_exp(*nat_pair)

        _, unravel_nat = ravel_pytree(nat_pair)
        _, unravel_exp = ravel_pytree(exp_pair)

        return (
            NatTransformation(
                xi_to_nat, lambda n1, n2: nat_to_xi(n1, n2)[0], unravel_exp, unravel_nat
            ),
            xi_flat,
        )

    def xi_to_msl(self, xi: Array) -> tuple[Array, Array]:
        nat1, nat2 = self._xi_to_nat(xi)
        m, sl = _nat_to_msl(nat1, nat2)
        return m, sl

    def xi_to_exp(self, xi: Array) -> Array:
        nat1, nat2 = self._xi_to_nat(xi)
        exp_pair = _nat_to_exp(nat1, nat2)
        exp, _ = ravel_pytree(exp_pair)
        return exp

    def exp_to_xi(self, exp: Array) -> Array:
        exp1, exp2 = self._unflatten_exp(exp)
        nat1, nat2 = _exp_to_nat(exp1, exp2)
        xi = self._nat_to_xi(nat1, nat2)
        return xi

    def nat_to_xi(self, nat: Array) -> Array:
        nat1, nat2 = self._unflatten_nat(nat)
        xi = self._nat_to_xi(nat1, nat2)
        return xi

    def xi_to_nat(self, xi: Array) -> Array:
        nat_pair = self._xi_to_nat(xi)
        nat, _ = ravel_pytree(nat_pair)
        return nat

    def jvp_needed(self) -> bool:
        return False


def elbo_and_nat_grad(
    elbo: Callable[[Array], Array], trf: XiTransformation, xi: Array
) -> tuple[Array, Array]:
    xi_flat, xi_unravel = ravel_pytree(xi)
    exp = trf.xi_to_exp(xi_flat)

    def elbo_exp(exp):
        xi_flat = trf.exp_to_xi(exp)
        xi = xi_unravel(xi_flat)
        return elbo(xi)

    elbo_val, grad = jax.value_and_grad(elbo_exp)(exp)

    if trf.jvp_needed():
        nat = trf.xi_to_nat(xi_flat)
        _, nat_grad = jax.jvp(trf.nat_to_xi, (nat,), (grad,))
    else:
        nat_grad = grad

    nat_grad_structured = xi_unravel(nat_grad)

    return elbo_val, nat_grad_structured
