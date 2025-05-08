from functools import partial

import pytest

from batram_cov import data_config_parser


@pytest.fixture
def base_config() -> dict:
    config = {
        "experiment_path": "root/experiments/testpath",
        "nlocs": 32,
        "max_neighbors": 60,
        "ntrain": 1,
        "nvalid": 1,
        "ntest": 1,
        "seeds": (0, 1, 2),
        "kernel": {
            "amplitude": 1.0,
            "length_scale": 1.0,
            "nugget": 1e-6,
            "smoothness": 1.0,
            "rotation_angle": 0.0,
            "rotation_axis1": 1.0,
            "rotation_axis2": 1.0,
        },
    }
    return config


def test_base_config_raises(base_config):
    with pytest.raises(data_config_parser.ConfigDataParsingError):
        data_config_parser.parse_simulation_data_config(base_config)


def test_argument_validation():
    validate = partial(data_config_parser.validate_type_and_bounds, dtype=int)

    n1, n2, n3 = (1, 2, 3)
    n = (1, 2, 3)
    validate(n1, n2, n3, lower=1, upper=3, length=3)
    validate(*n, lower=1, upper=3, length=3)
    validate(*list(n), lower=1, upper=3, length=3)
    validate(*n, dtype=int, length=3)


def test_parse_smoothness(base_config: dict):
    config = base_config.copy()
    config["exp_type"] = "smoothness"
    config["kernel"]["smoothness"] = {
        "bounds": [0.25, 3.5],
        "nx": [3, 2, 2],
        "nr": [1, 1, 1],
        "covariate_scale": ["log", "log"],
    }
    pe = data_config_parser.parse_simulation_data_config(config)
    assert isinstance(pe, data_config_parser.ParsedExperimentalDataConfig)
    assert pe.configs.experiment_type == "smoothness"
    assert isinstance(pe.configs.kernel, data_config_parser.KernelSpec)
    for k, v in pe.configs.kernel._asdict().items():
        assert isinstance(v.is_scalar, bool), f"{k} has no bool attribute"


def test_parse_anisotropy(base_config: dict):
    config = base_config.copy()
    config["exp_type"] = "anisotropy"
    config["kernel"]["rotation_angle"] = {
        "bounds": [0.0, 3.14],
        "nx": [3, 2, 2],
        "nr": [1, 1, 1],
        "covariate_scale": ["log", "log"],
    }
    pe = data_config_parser.parse_simulation_data_config(config)
    assert isinstance(pe, data_config_parser.ParsedExperimentalDataConfig)
    assert pe.configs.experiment_type == "anisotropy"
    assert isinstance(pe.configs.kernel, data_config_parser.KernelSpec)


def test_parse_smoothness_and_anisotropy(base_config: dict):
    config = base_config.copy()
    config["exp_type"] = "smoothness_and_anisotropy"
    config["kernel"]["smoothness"] = {
        "bounds": [0.25, 3.5],
        "nx": [5, 5, 5],
        "nr": [5, 5, 5],
        "covariate_scale": ["log", "log"],
    }
    config["kernel"]["rotation_angle"] = {
        "bounds": [0.0, 3.14],
        "nx": [3, 2, 2],
        "nr": [1, 1, 1],
    }
    pe = data_config_parser.parse_simulation_data_config(config)
    assert isinstance(pe, data_config_parser.ParsedExperimentalDataConfig)
    assert pe.configs.experiment_type == "smoothness_and_anisotropy"
    assert isinstance(pe.configs.kernel, data_config_parser.KernelSpec)


# This originally raised an error because `exp_type` was a directive argument,
# and led to an error if it did not satisfy a small number of expected values.
# Now it is simply a descriptive argument, which is shown in this test.
def test_parse_other(base_config: dict):
    config = base_config.copy()
    config["exp_type"] = "other"
    pe = data_config_parser.parse_simulation_data_config(config)
    assert isinstance(pe, data_config_parser.ParsedExperimentalDataConfig)
    assert pe.configs.experiment_type == "other"
