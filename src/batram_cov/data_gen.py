import itertools
import logging
import pickle
import time
from collections.abc import Callable, Generator, Iterable
from typing import NamedTuple, TypeVar

import numpy as np
from veccs import orderings

from . import calc_li, gp
from .data_config_parser import (
    Bounds,
    KernelParameterSpec,
    KernelSpec,
    ParsedSimulationDataConfig,
    parse_simulation_data_config,
)
from .typing import SimulatedDatasets, SimulationData

# Type aliases
T = TypeVar("T")
BD = tuple[np.ndarray, np.ndarray]
MappedBD = Iterable[BD]
XandParam = tuple[None | np.ndarray, np.ndarray]
KernelParamSequence = list[dict[str, float]]
XandKernelParamSequence = tuple[np.ndarray, KernelParamSequence]


class Bijector(NamedTuple):
    """Bijector for transforming kernel parameters."""

    forward: Callable[[T], T]
    inverse: Callable[[T], T]

    @classmethod
    def from_str(cls, transform: str):
        if transform == "log":
            return cls(np.log, np.exp)
        elif transform == "linear":
            return cls(lambda x: x, lambda x: x)
        else:
            raise ValueError(f"Transform {transform} is not supported.")


def generate_ordered_locs_and_neighbors(
    nlocs: int, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates ordered locations and nearest neighbors on a unit square."""
    locs = gp.make_grid(nlocs=nlocs, ndims=2)
    maxmin_ordering = orderings.maxmin_cpp(locs)
    ordered_locs = locs[maxmin_ordering]
    nearest_neighbors = orderings.find_nns_l2(ordered_locs, max_nn=max_neighbors)
    return ordered_locs, maxmin_ordering, nearest_neighbors


def repeated_linspace(bounds: Bounds[float], nx: int, nr: int) -> np.ndarray:
    """Generates a repeated linspace of values for kernel parameters.

    Args:
    -----
    bounds: float | Bounds[float]
        The bounds (start, stop) for the linspace

    nx: int
        The number of points in the linspace

    nreps: int
        The number of times to repeat the generated linspace

    Returns:
    --------
    np.ndarray
        The repeated linspace of values
    """
    xs = np.linspace(bounds.left, bounds.right, nx)
    xs = np.repeat(xs, nr)
    return xs


def x_and_p_from_param_spec(
    param_spec: KernelParameterSpec,
) -> Generator[XandParam, None, None]:
    """Converts a KernelParamSpec into a tuple of arrays."""

    def apply(func: Callable[[float], float], bounds: Bounds) -> Bounds:
        return Bounds(func(bounds.left), func(bounds.right))

    def get_transforms(scale: None | tuple[str]):
        if scale is None:
            bijectors = Bijector.from_str("linear")
            return (bijectors, bijectors)
        else:
            return (Bijector.from_str(scale[0]), Bijector.from_str(scale[1]))

    def x_and_param(bounds: Bounds, nx: int, nr: int, funcs: tuple[Bijector]):
        input_space = apply(funcs[0].forward, bounds)
        x = repeated_linspace(input_space, nx, nr)
        param = funcs[0].inverse(x)
        x = funcs[1].forward(funcs[0].inverse(x))
        return x, param

    bounds = param_spec.bounds
    nx, nr = param_spec.sizes.nx, param_spec.sizes.nr
    transforms = get_transforms(param_spec.covariate_scale)

    xtrain, ptrain = x_and_param(bounds, nx[0], nr[0], transforms)
    xvalid, pvalid = x_and_param(bounds, nx[1], nr[1], transforms)
    xtest, ptest = x_and_param(bounds, nx[2], nr[2], transforms)

    if param_spec.is_scalar:
        xtrain = xvalid = xtest = None
    yield from (
        (xtrain, ptrain),
        (xvalid, pvalid),
        (xtest, ptest),
    )


def _const_and_var_tree(
    kernel_spec: KernelSpec,
) -> dict[str, dict[str, KernelParameterSpec]]:
    """Parse the kernel trees into a dictionary for generation."""
    consts_and_vars = {"const": {}, "var": {}}
    for k, v in kernel_spec._asdict().items():
        if v.is_scalar:
            consts_and_vars["const"][k] = v
        else:
            consts_and_vars["var"][k] = v

    return consts_and_vars


def _x_and_param_tree(
    const_and_var_tree: dict[dict[str, KernelParameterSpec]],
) -> list[dict[str, dict[str, np.ndarray]]]:
    def x_and_param_tree():
        return {"x": {}, "param": {}}

    trees = [x_and_param_tree() for _ in range(3)]
    for k, v in const_and_var_tree["const"].items():
        x_and_p = x_and_p_from_param_spec(v)
        for i, (_, p) in enumerate(x_and_p):
            trees[i]["param"][k] = p

    for k, v in const_and_var_tree["var"].items():
        x_and_p = x_and_p_from_param_spec(v)
        for i, (x, p) in enumerate(x_and_p):
            trees[i]["x"][k] = x
            trees[i]["param"][k] = p

    return trees


def _unpack_kernel_spec(
    x_and_param_tree: list[dict[str, dict[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    def remap_keys(param_dict):
        parsing_keys = [
            "amplitude",
            "length_scale",
            "nugget",
            "smoothness",
            "rotation_angle",
            "rotation_axis",
        ]
        kernel_keys = ["scale", "ls", "nugget", "nu", "theta", "ascales"]
        return {k: param_dict[v] for k, v in zip(kernel_keys, parsing_keys)}

    def get_param_sequence(tree) -> list[dict[str, float]]:
        param_keys = tree.keys()
        param_values = itertools.product(*tree.values())
        param_seq = []
        for p in param_values:
            one_seq = dict(zip(param_keys, p))
            # We use different names for things in data generation than we do
            # in the GP kernel, so we need to remap the dict keys to the GP
            # kernel's arguments. This is a bit of a hack, but provides the
            # simplest transition from one set of keys to the other.
            one_seq["rotation_axis"] = np.array(
                [
                    one_seq.pop("rotation_axis1"),
                    one_seq.pop("rotation_axis2"),
                ]
            )
            one_seq = remap_keys(one_seq.copy())
            param_seq.append(one_seq)

        return param_seq

    def get_x_array(tree) -> np.ndarray:
        arrays = tree.values()
        grids = np.meshgrid(*arrays)
        return np.vstack([g.ravel() for g in grids]).T

    for tree in x_and_param_tree:
        x = get_x_array(tree["x"])
        param_seq = get_param_sequence(tree["param"])
        yield x, param_seq


def generate_kernel_parameter_sequence(
    kernel_spec: KernelSpec,
) -> tuple[XandKernelParamSequence, XandKernelParamSequence, XandKernelParamSequence]:
    """Generates a sequence of kernel parameters for the GP."""
    const_and_var_tree = _const_and_var_tree(kernel_spec)
    x_and_param_tree = _x_and_param_tree(const_and_var_tree)
    return _unpack_kernel_spec(x_and_param_tree)


def create_gp_samples(
    nreps: int,
    seed: int,
    ordered_locs: np.ndarray,
    nearest_neighbors: np.ndarray,
    kernel_param_sequence: KernelParamSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = len(kernel_param_sequence)
    samples = np.empty((nreps, nx, ordered_locs.shape[0]))
    means = np.empty((nreps, nx, ordered_locs.shape[0]))
    ds = np.full((nx, ordered_locs.shape[0]), np.nan)
    vecc_gp = gp.VecchiaGP(
        locs=ordered_locs, nbrs=nearest_neighbors, kernel=gp.kernel, seed=seed
    )
    for i, params in enumerate(kernel_param_sequence):
        vecc_gp.update_params(**params)
        res = vecc_gp.sample(n=nreps, **params)
        samples[:, i, :] = res[0]
        means[:, i, :] = res[1]
        ds[i] = res[2]
    return samples, means, ds


def create_simulation_datasets(
    configs: ParsedSimulationDataConfig,
) -> SimulatedDatasets:
    def get_description(samples, experiment_type):
        nfields, nreps = samples.shape[:2]
        description = (
            f"{experiment_type} dataset with {nfields} samples and {nreps} "
            f"reps per sample generated with seed {seed}. Has samples with "
            f"shape {samples.shape}"
        )
        return description

    (
        ordered_locs,
        maxmin_ordering,
        nearest_neighbors,
    ) = generate_ordered_locs_and_neighbors(
        nlocs=configs.nlocs, max_neighbors=configs.max_neighbors
    )

    nreps = [configs.ntrain, configs.nvalid, configs.ntest]
    seeds = configs.seeds
    xs_and_kernel_params_seq = generate_kernel_parameter_sequence(configs.kernel)

    simulation_datasets = []

    for (
        n,
        seed,
        xs_and_kernel_params_seq,
    ) in zip(nreps, seeds, xs_and_kernel_params_seq):
        xs, params = xs_and_kernel_params_seq
        nugget = params[0]["nugget"]
        samples, bs, ds = create_gp_samples(
            nreps=n,
            seed=seed,
            ordered_locs=ordered_locs,
            nearest_neighbors=nearest_neighbors,
            kernel_param_sequence=params,
        )

        simulation_dataset = SimulationData(
            locs=ordered_locs,
            nearest_neighbors=nearest_neighbors,
            li=calc_li(ordered_locs, nearest_neighbors),
            maxmin_permutation=maxmin_ordering,
            inv_maxmin_permutation=maxmin_ordering.argsort(),
            x=xs,
            samples=samples,
            means=bs,
            conditional_variances=ds,
            noise_var=nugget,
            seed=seed,
            description=get_description(samples, configs.experiment_type),
            config=configs,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        simulation_datasets.append(simulation_dataset)

    return SimulatedDatasets(
        train=simulation_datasets[0],
        valid=simulation_datasets[1],
        test=simulation_datasets[2],
    )


def load_simulation_dataset(filename: str) -> SimulatedDatasets:
    """Load a dataset from disk"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_simulation_dataset(data: SimulatedDatasets, filename: str) -> None:
    """Write a dataset to disk"""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def retrieve_or_generate_simulation_datasets(
    config: dict,
) -> tuple[str, SimulatedDatasets]:
    parsed_experiment = parse_simulation_data_config(config)
    filename = parsed_experiment.save_target
    try:
        data = load_simulation_dataset(filename)
    except FileNotFoundError:
        logging.info(f"Dataset not found, generating {filename}")
        data = create_simulation_datasets(parsed_experiment.configs)
        write_simulation_dataset(data, filename)
    logging.info(f"Loaded dataset with filename: {filename}")

    return filename, data
