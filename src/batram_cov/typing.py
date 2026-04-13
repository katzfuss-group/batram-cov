from typing import NamedTuple, Protocol

from jaxtyping import Array, Float, Int
from numpy import ndarray


class ArraysProtocol(Protocol):
    """Protocol for arrays to follow.

    **Attributes**:
    - `locs (N, d)`: Ordered locations of samples on a grid.
    - `nearest_neighbors (N, max_m)`: Nearest neighbors for building
      conditioning sets.
    - `li (N, )`: Distance to nearest neighbor (scaling used by spatial priors).
    - `maxmin_permutation (N, )`: The permutation from unordered locs to ordered
      locs.
    - `x (n, p)`: The covariate values corresponding to each sample.
    - `samples (nr, n, N)`: Climate variables from an unknown distribution, set
      to match the shape of simulation data.
    """

    # NOTE: ruff does not understand these annotations yet.
    # See https://github.com/astral-sh/ruff/issues/13824 for details, and note
    # it is an ongoing issue. Have to inline the ignores for now.
    locs: Float[Array, "N d"]  # noqa: F722
    nearest_neighbors: Int[Array | ndarray, "N max_m"]  # noqa: F722, 821
    li: Float[Array | ndarray, "N"]  # noqa: F821
    maxmin_permutation: Float[Array | ndarray, "N"]  # noqa: F821
    x: Float[Array | ndarray, "n p"]  # noqa: F722
    samples: Float[Array | ndarray, "nr n N"]  # noqa: F722


class AppData(NamedTuple):
    """Pre-processed CMIP6 data demonstrating the model.

    **Attributes**:
    - `locs (N, d)`: Ordered locations of samples on a grid.
    - `nearest_neighbors (N, max_m)`: Nearest neighbors for building
      conditioning sets.
    - `li (N, )`: Distance to nearest neighbor (scaling used by spatial priors).
    - `maxmin_permutation (N, )`: The permutation from unordered locs to ordered
      locs.
    - `x (n, p)`: The covariate values corresponding to each sample.
    - `samples (nr, n, N)`: Climate variables from an unknown distribution, set
      to match the shape of simulation data.
    - `description`: A description of the dataset in use
    """

    # NOTE: ruff does not understand these annotations yet.
    # See https://github.com/astral-sh/ruff/issues/13824 for details, and note
    # it is an ongoing issue. Have to inline the ignores for now.
    locs: Float[Array, "N d"]  # noqa: F722
    nearest_neighbors: Int[Array | ndarray, "N max_m"]  # noqa: F722, 821
    li: Float[Array | ndarray, "N"]  # noqa: F821
    maxmin_permutation: Float[Array | ndarray, "N"]  # noqa: F821
    x: Float[Array | ndarray, "n p"]  # noqa: F722
    samples: Float[Array | ndarray, "nr n N"]  # noqa: F722
    description: str


class AnomalizedData(NamedTuple):
    """Anomalized data for a given AppData instance.

    Acts as a superset of the AppData class with `samples` as anomalies instead
    of realized values from the observed distribution. The mean and standard
    deviation fields are provided to reconstruct the original samples as
    needed.

    **Attributes**:
    - `locs (N, d)`: Ordered locations of samples on a grid.
    - `nearest_neighbors (N, max_m)`: Nearest neighbors for building
      conditioning sets.
    - `li (N, )`: Distance to nearest neighbor (scaling used by spatial priors).
    - `maxmin_permutation (N, )`: The permutation from unordered locs to ordered
      locs.
    - `x (n, p)`: The covariate values corresponding to each sample.
    - `samples (nr, n, N)`: Climate variables from an unknown distribution, set
      to match the shape of simulation data.
    - `description`: A description of the dataset in use
    """

    # NOTE: ruff does not understand these annotations yet.
    # See https://github.com/astral-sh/ruff/issues/13824 for details, and note
    # it is an ongoing issue. Have to inline the ignores for now.
    locs: Float[Array, "N d"]  # noqa: F722
    nearest_neighbors: Int[Array | ndarray, "N max_m"]  # noqa: F722, 821
    li: Float[Array | ndarray, "N"]  # noqa: F821
    maxmin_permutation: Float[Array | ndarray, "N"]  # noqa: F821
    x: Float[Array | ndarray, "n p"]  # noqa: F722
    samples: Float[Array | ndarray, "nr n N"]  # noqa: F722
    description: str


class SimulationData(NamedTuple):
    """Data for simulations (generated in Numpy, consumed in JAX).

    **Attributes**:
    - `locs (N, d)`: Ordered locations of samples on a grid.
    - `nearest_neighbors (N, max_m)`: Nearest neighbors for building
      conditioning sets.
    - `li (N, )`: Distance to nearest neighbor (scaling used by spatial priors).
    - `maxmin_permutation (N, )`: The permutation from unordered locs to ordered
      locs.
    - `x (n, p)`: The covariate values corresponding to each sample.
    - `samples (nr, n, N)`: Climate variables from an unknown distribution, set
      to match the shape of simulation data.
    - `means (n N)`: Oracle estimands for the target field.
    - `conditional_variances (n N)`: Oracle estimands for the target field.
    - `noise_var`: Nugget used in generation.
    - `description`: Description of the dataset.
    - `seed`: Random seed used to generate the dataset.
    - `config`: Configuration used to generate the dataset.
    - `timestamp`: Timestamp of the dataset generation (yyyy-mm-dd hh:mm:ss).
    """

    locs: Float[Array, "N d"]  # noqa: F722
    nearest_neighbors: Int[Array | ndarray, "N max_m"]  # noqa: F722, 821
    li: Float[Array | ndarray, "N"]  # noqa: F821
    maxmin_permutation: Float[Array | ndarray, "N"]  # noqa: F821
    x: Float[Array | ndarray, "n p"]  # noqa: F722
    samples: Float[Array | ndarray, "nr n N"]  # noqa: F722
    means: Float[Array | ndarray, "n N"]  # noqa: F722
    conditional_variances: Float[Array | ndarray, "n N"]  # noqa: F722
    noise_var: float
    seed: int
    config: dict
    timestamp: str
    description: str


class SimulatedDatasets(NamedTuple):
    """A collection of train, test, and validation datasets.

    Datasets parsed from `batram_cov.data_config_parser.parse_experiment_config`
    and a corresponding `config.yml` file.

    Attributes:
    -----------
    train: SimulationData
        Training dataset generated with [ntrain, nreps] samples

    valid: SimulationData
        Validation dataset generated with [nvalid, nreps] samples

    test: SimulationData
        Testing dataset generated with [ntest, nreps] samples
    """

    train: SimulationData
    valid: SimulationData
    test: SimulationData
