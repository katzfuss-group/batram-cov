import pytest

from batram_cov.data_config_parser import parse_simulation_data_config
from batram_cov.data_gen import SimulationData, create_simulation_datasets


@pytest.fixture
def config() -> dict:
    config = {
        "experiment_path": "root/experiments/testpath",
        "nlocs": 16,
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


def test_calc_smoothness(config):
    config = config.copy()
    config["exp_type"] = "smoothness"
    config["kernel"]["smoothness"] = {
        "bounds": [0.25, 3.5],
        "nx": [20, 2, 2],
        "nr": [1, 1, 1],
        "covariate_scale": ["log", "log"],
    }
    parsed_experiment = parse_simulation_data_config(config)
    parsed_configs = parsed_experiment.configs
    simulation_datasets = create_simulation_datasets(parsed_configs)
    assert parsed_configs.experiment_type == "smoothness"
    assert len(simulation_datasets) == 3
    assert simulation_datasets.train.x.shape == (20, 1)


def test_calc_anisotropy(config):
    config = config.copy()
    config["exp_type"] = "anisotropy"
    config["kernel"]["rotation_angle"] = {
        "bounds": [0.0, 3.14],
        "nx": [20, 2, 2],
        "nr": [1, 1, 1],
    }
    parsed_experiment = parse_simulation_data_config(config)
    parsed_configs = parsed_experiment.configs
    simulation_datasets = create_simulation_datasets(parsed_configs)
    assert parsed_configs.experiment_type == "anisotropy"
    assert len(simulation_datasets) == 3
    assert simulation_datasets.train.x.shape == (20, 1)


def test_array_shapes(config):
    config = config.copy()
    config["exp_type"] = "anisotropy"
    config["ntrain"] = 9
    config["kernel"]["rotation_angle"] = {
        "bounds": [0.0, 3.14],
        "nx": [20, 2, 2],
        "nr": [4, 1, 1],
    }
    parsed_experiment = parse_simulation_data_config(config)
    parsed_configs = parsed_experiment.configs
    simulation_datasets = create_simulation_datasets(parsed_configs)
    train: SimulationData = simulation_datasets[0]

    expected_shape = (9, 80, 16 * 16)
    assert train.samples.shape == expected_shape
    assert train.means.shape == expected_shape
    assert train.conditional_variances.shape == (80, 16 * 16)


def test_calc_smoothness_anisotropy(config):
    config = config.copy()
    config["exp_type"] = "smoothness_and_anisotropy"
    config["ntrain"] = 1
    config["kernel"]["smoothness"] = {
        "bounds": [0.25, 3.5],
        "nx": [12, 2, 2],
        "nr": [1, 1, 1],
        "covariate_scale": ["log", "log"],
    }
    config["kernel"]["rotation_angle"] = {
        "bounds": [0.0, 3.14],
        "nx": [12, 2, 2],
        "nr": [1, 1, 1],
    }
    parsed_experiment = parse_simulation_data_config(config)
    parsed_configs = parsed_experiment.configs
    simulation_datasets = create_simulation_datasets(parsed_configs)
    assert parsed_configs.experiment_type == "smoothness_and_anisotropy"
    assert len(simulation_datasets) == 3
    assert simulation_datasets.train.samples.shape == (1, 144, 16 * 16)


@pytest.mark.skip(reason="refactor")
def test_tuple_train_configs(config):
    config = config.copy()
    config["exp_type"] = "smoothness_and_anisotropy"
    train_configs = [
        ((12, 1), (1, 1), 1),
        (1, (2, 1), 1),
        ((5, 5), (2, 2), 1),
        (5, 1, 1),
    ]
    expects = [
        (1, 12, 256),
        (1, 2, 256),
        (1, 100, 256),
        (1, 25, 256),
    ]
    for ntrain, expect in zip(train_configs, expects):
        config["ntrain"] = ntrain
        parsed_experiment = parse_simulation_data_config(config.copy())
        parsed_configs = parsed_experiment.configs
        simulation_datasets = create_simulation_datasets(parsed_configs)
        assert parsed_configs.experiment_type == "smoothness_and_anisotropy"
        assert simulation_datasets.train.samples.shape == expect
        assert simulation_datasets.train.x.shape == (expect[1], 2), (
            ntrain,
            expect,
            simulation_datasets.train.x.shape,
            simulation_datasets.train.x,
        )
        assert simulation_datasets.test.x.shape == (25, 2)
