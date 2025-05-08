import itertools
from copy import deepcopy
from io import TextIOWrapper

import jax
import jax.numpy as jnp
import numpy as np
import yaml


def to_strong_jax_type(x):
    def _to_strong_leave(x):
        if isinstance(x, jax.Array) or isinstance(x, np.ndarray):
            return jnp.asarray(x, dtype=x.dtype)
        elif isinstance(x, float):
            return jnp.array(x, dtype=jnp.float32)
        elif isinstance(x, int):
            return jnp.array(x, dtype=jnp.int32)
        elif isinstance(x, bool):
            return jnp.array(x, dtype=jnp.bool_)
        else:
            return x

    return jax.tree.map(_to_strong_leave, x)


def _merge_dot_keys(config: dict) -> dict:
    """merge keys with dot notation into the dictionary"""

    # if there are no keys with dot notation, return the original dictionary
    if not any(key for key in config.keys() if "." in key):
        return config

    # create a deep copy of the dictionary to avoid modifying nested dictionaries
    # which
    config = deepcopy(config)

    keys_to_delete = []
    for key, value in config.items():
        if "." in key:
            # print(key, value)
            keys = key.split(".")
            last_key = keys[-1]
            current = config
            # traverse through the nested dictionaries
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}  # create a new dictionary if it does not exist
                current = current[k]  # move to the next level

            # merge the value into the current dictionary
            match current[last_key]:
                case dict():
                    current[last_key].update(value)
                case _:
                    current[last_key] = value

            keys_to_delete.append(key)

    for key in keys_to_delete:
        del config[key]

    return config


def _build_config_one_type(one_config_raw, defaults):
    grid_params = one_config_raw.get("grid", None)
    defaults2 = one_config_raw.get("defaults", None)
    if defaults2 is None:
        defaults2 = {}
    if grid_params is None:
        grid_params = []

    defaults = defaults.copy()
    defaults.update(defaults2)

    # handle the case when there is no grid search
    if not grid_params:
        return [defaults]

    # build all possible combinations of grid search parameters
    grid_dicts = []
    for grid_entry in grid_params:
        grid_combinations = list(itertools.product(*grid_entry.values()))

        grid_dicts += [
            dict(zip(grid_entry.keys(), combination))
            for combination in grid_combinations
        ]

    # combine with fixed parameters and default parameters
    configurations = []
    for comb in grid_dicts:
        config = defaults.copy()
        config.update(comb)
        configurations.append(config)

    return configurations


def _build_config(one_config_raw, defaults):
    data = one_config_raw.get("data", None)
    model = one_config_raw.get("model", None)

    if data is None:
        data = {}
    if model is None:
        model = {}

    data_configs = _build_config_one_type(data, defaults["data"])
    model_configs = _build_config_one_type(model, defaults["model"])

    # merge keys with dot notation
    data_configs = [_merge_dot_keys(config) for config in data_configs]
    model_configs = [_merge_dot_keys(config) for config in model_configs]

    # combine data and model configurations
    all_configs = []
    for data_config in data_configs:
        for model_config in model_configs:
            config = {
                "data": data_config,
                "model": model_config,
                "skip": one_config_raw.get("skip", False),
            }
            all_configs.append(config)

    return all_configs


def read_config(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return read_config_from_stream(f)


def read_config_from_stream(f: TextIOWrapper) -> list[dict]:
    config_raw = yaml.safe_load(f)

    defaults = config_raw["defaults"]
    all_configs_raw = config_raw["configs"]
    all_configs = []
    for one_config_raw in all_configs_raw:
        configs = _build_config(one_config_raw, defaults)
        for config in configs:
            if not config["skip"]:
                all_configs.append(config)

    return all_configs
