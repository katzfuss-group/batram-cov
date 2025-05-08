import json
import os
from hashlib import sha256
from typing import Any, NamedTuple

Seeds = tuple[int, int, int]


class ConfigDataParsingError(Exception):
    """A meaningful error to raise."""

    pass


class Bounds(NamedTuple):
    """Bounds for a kernel parameter."""

    left: float
    right: float


class Sizes(NamedTuple):
    """A tuple of sizes for number of unique sites and repeated samples at each site.

    Attributes:
    -----------
    nx: int
        The number of unique sites to generate.

    nr: int
        The number of repeated samples to generate at each site.
    """

    nx: tuple[int, int, int]
    nr: tuple[int, int, int]


class KernelParameterSpec(NamedTuple):
    """A specification for generating kernel parameter lists.

    Attributes:
    -----------
    bounds: Bounds
        The bounds for the parameter.

    sizes: Sizes
        The sizes for the number of unique sites and repeated samples to use
        when generating the parameter.

    is_scalar: bool
        Whether the parameter is a scalar or not. When not a scalar, the parameter
        is used as a covariate in generated datasets.

    covariate_scale: None | tuple[str, str]
        The scale to use when generating the parameter. Valid strings are
        "linear" and "log". The first string is the scale to generate the bounds
        on. The second scale is the scale to use when returning a covariate.

    Example:
    --------
    # Example 1
    >>> bounds = Bounds(0.25, 2.5)
    >>> sizes = Sizes(10, 5)
    >>> is_scalar = True
    >>> covariate_scale = ("log", "log")
    >>> smoothness = KernelParameterSpec(bounds, sizes, is_scalar, covariate_scale)

    # Example 2
    >>> amplitude = KernelParameterSpec.from_scalar(1.0)
    """

    bounds: Bounds
    sizes: Sizes
    is_scalar: bool
    covariate_scale: None | tuple[str, str]

    @classmethod
    def from_scalar(cls, value: float):
        size = (1, 1, 1)
        return cls(Bounds(value, value), Sizes(size, size), True, None)


class KernelSpec(NamedTuple):
    """Kernel parameters for an anisotropic Matern field.

    Any of the parameters can be a scalar or configurable as a list of parameters
    to generate depending on the experiment. Typically a large number of the
    parameters will be scalars and only one or two will be covariates.

    Attributes:
    -----------
    amplitude: KernelParameterSpec
        The amplitude of the kernel.

    length_scale: KernelParameterSpec
        The length scale of the kernel.

    nugget: KernelParameterSpec
        The nugget parameter of the kernel.

    smoothness: KernelParameterSpec
        The smoothness parameter of the kernel.

    rotation_angle: KernelParameterSpec
        The rotation angle of the kernel.

    rotation_axis1: KernelParameterSpec
        The first axis of rotation for the kernel.

    rotation_axis2: KernelParameterSpec
        The second axis of rotation for the kernel.
    """

    amplitude: KernelParameterSpec
    length_scale: KernelParameterSpec
    nugget: KernelParameterSpec
    smoothness: KernelParameterSpec
    rotation_angle: KernelParameterSpec
    rotation_axis1: KernelParameterSpec
    rotation_axis2: KernelParameterSpec


class ParsedSimulationDataConfig(NamedTuple):
    """Validated configs for an experiment.

    Attributes:
    -----------
    experiment_type: str
        The type of experiment to generate. Currently smoothness or anisotropy only.

    nlocs: int
        The number of spatial locations to generate per side of a unit square.

    max_neighbors: int
        The maximum number of neighbors to consider for each location.

    ntrain: int
        The number of independent copies of the training data to generate.

    nvalid: int
        The number of independent copies of the validation data to generate.

    ntest: int
        The number of independent copies of the test data to generate.

    seeds: tuple[int, int, int]
        The seeds for random number generation in ntrain, nvalid, and ntest.

    kernel: KernelSpec
        The kernel parameters to use when generating a dataset.
    """

    experiment_type: str
    nlocs: int
    max_neighbors: int
    ntrain: int
    nvalid: int
    ntest: int
    seeds: tuple[int, int, int]
    kernel: KernelSpec


class ParsedExperimentalDataConfig(NamedTuple):
    """Parsed experiment configs to use in data generation.

    Attributes:
    -----------
    configs: ParsedExperimentConfig
        Config args for any experiment to be generated
    save_target: str
        Path to use when saving the experiment
    """

    configs: ParsedSimulationDataConfig
    save_target: str


def validate_type_and_bounds(
    *args: tuple[Any, ...],
    dtype: type,
    lower: int | float | None = None,
    upper: int | float | None = None,
    equal: list[str] | set[str] | None = None,
    length: int | None = None,
) -> None:
    """Validate the types and bounds of arguments.

    Bounds are inclusive. If equal is not None, the values checked must be in
    the set of valid arguments provided for the check. Length is the number of
    arguments passed in args.
    """

    def is_equal(value: Any) -> bool:
        if equal is None:
            return True
        return value in equal

    def is_in_bounds(value: Any) -> bool:
        lb = lower is None or value >= lower
        ub = upper is None or value <= upper
        return lb and ub

    def is_type(value: Any) -> bool:
        return isinstance(value, dtype)

    if length:
        if len(args) != length:
            raise ConfigDataParsingError(
                f"Expected {length} arguments. Got {len(args)} instead."
            )

    for arg in args:
        conditions = (is_type(arg), is_in_bounds(arg), is_equal(arg))
        if not all(conditions):
            raise ConfigDataParsingError(
                f"Argument must be a {dtype} between {lower} and {upper}. "
                f"Got {arg} of type {type(arg)} instead."
            )


def get_key(key: str, config: dict):
    try:
        return config.pop(key)
    except KeyError:
        raise ConfigDataParsingError(f"Missing key: {key}")


def get_experiment_name(path: str, exp_type: str, config_hash: str):
    return os.path.join(path, f"{exp_type}_{config_hash}.pkl")


def hash_encodings(config: str | dict[str, Any]):
    config_str = json.dumps(config, sort_keys=True)
    config_encoding = config_str.encode("utf-8")
    return sha256(config_encoding).hexdigest()


def parse_grid_and_nbrs(config: dict) -> tuple[int, int]:
    nlocs = get_key("nlocs", config)
    max_neighbors = get_key("max_neighbors", config)
    validate_type_and_bounds(nlocs, max_neighbors, dtype=int, lower=1)
    return nlocs, max_neighbors


def parse_kernel_param(param: int | float | dict) -> KernelParameterSpec:
    if isinstance(param, int | float):
        return KernelParameterSpec.from_scalar(param)
    bounds = get_key("bounds", param)
    validate_type_and_bounds(*bounds, dtype=float, length=2)
    bounds = Bounds(*bounds)

    nx = get_key("nx", param)
    validate_type_and_bounds(*nx, dtype=int, lower=2, length=3)

    nr = get_key("nr", param)
    if isinstance(nr, int):
        nr = (nr, 1, 1)
    validate_type_and_bounds(*nr, dtype=int, lower=1, length=3)

    sizes = Sizes(nx, nr)
    is_scalar = param.get("is_scalar", None)
    if not is_scalar:
        is_scalar = sizes.nx == 1 and sizes.nr == 1

    covariate_scale = param.get("covariate_scale", None)
    if covariate_scale:
        validate_type_and_bounds(*covariate_scale, dtype=str, equal={"log", "linear"})

    return KernelParameterSpec(
        bounds=bounds,
        sizes=Sizes(nx, nr),
        is_scalar=is_scalar,
        covariate_scale=covariate_scale,
    )


def parse_kernel_tree(kernel_tree: dict) -> KernelSpec:
    """Parses a kernel tree into a KernelSpec object."""
    expected_keys = {
        "amplitude",
        "length_scale",
        "nugget",
        "smoothness",
        "rotation_angle",
        "rotation_axis1",
        "rotation_axis2",
    }
    kernel_spec = {}
    while kernel_tree.keys():
        key = next(iter(kernel_tree.keys()))
        if key not in expected_keys:
            raise ConfigDataParsingError(f"Unrecognized kernel key: {key}")
        kernel_spec[key] = parse_kernel_param(kernel_tree.pop(key))

    return KernelSpec(**kernel_spec)


def parse_sample_sizes_and_seeds(
    config: dict,
) -> tuple[int, int, int, Seeds]:
    ntrain = get_key("ntrain", config)
    nvalid = get_key("nvalid", config)
    ntest = get_key("ntest", config)
    seeds = get_key("seeds", config)

    validate_type_and_bounds(ntrain, nvalid, ntest, dtype=int, lower=1)
    validate_type_and_bounds(*seeds, dtype=int, length=3)
    return ntrain, nvalid, ntest, seeds


def parse_simulation_data_config(config: dict) -> ParsedExperimentalDataConfig:
    """Parser for experiments with different data generation requirements."""

    config_hash = hash_encodings(config)
    experiment_path = get_key("experiment_path", config)
    exp_type = get_key("exp_type", config)

    _ = config.pop("name", None)
    _ = config.pop("tags", None)
    _ = config.pop("desc", None)

    nlocs, max_neighbors = parse_grid_and_nbrs(config)
    ntrain, nvalid, ntest, seeds = parse_sample_sizes_and_seeds(config)
    kernel_spec = parse_kernel_tree(get_key("kernel", config))

    if config.keys():
        raise ConfigDataParsingError(f"Unrecognized keys: {config.keys()}")

    save_target = get_experiment_name(experiment_path, exp_type, config_hash)

    parsed_configs = ParsedSimulationDataConfig(
        experiment_type=exp_type,
        nlocs=nlocs,
        max_neighbors=max_neighbors,
        ntrain=ntrain,
        nvalid=nvalid,
        ntest=ntest,
        seeds=seeds,
        kernel=kernel_spec,
    )

    return ParsedExperimentalDataConfig(parsed_configs, save_target)
