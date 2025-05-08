"""hermgauss

A lookup table for Gauss-Hermite quadrature polynomial weights. For a function
:math:`f(x)` supported on the real line, Gauss-Hermite quadrature is a
numerical integration technique for estimating integrals of the form

.. math:

    I(f) = \\int_{-\\infty}^{+\\infty} e^{-x^2} f(x) dx

Gauss-Hermite quadrature approximates this integral by a finite polynomial

.. math:

   I(f) \\approx \\sum_{i=1}^{N} w_i f(x_i)

There is no support for these in JAX, so we provide a limited lookup table
of values to use when integrating a variable under a Gaussian measure.

Notes:
- Scaling of data:  The locations `x` and `w` are given on an
  unnoramlized scale, and may require a change of variables to be useful.
  For a random variable :math:`y \\sim N(\\mu, \\sigma^2)`, the appropriate
  change of variables is :math:`y = \\sqrt{2} \\sigma x + \\mu`. More
  details are available at [1].

- Normalizing results:  The polynomial is unnormalized. This is easily fixed
  by recalling that :math:`\\int e^{-x^2} dx = \\sqrt{\\pi}`, so normalizing
  the weights will transform calculations to expectations under a Gaussian
  measure.

References:
[1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
"""

import jax
import numpy as np

GaussHermiteLocsAndValues = tuple[jax.Array, jax.Array]


def hermgauss(num_pts: int) -> GaussHermiteLocsAndValues:
    locs, vals = np.polynomial.hermite.hermgauss(num_pts)
    locs = locs.astype(np.float32)
    vals = vals.astype(np.float32)
    return locs, vals
