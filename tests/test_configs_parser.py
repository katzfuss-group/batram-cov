import io

import batram_cov.utils as tmutils

YAML_CONFIG = """

defaults:
    data:
        data_path: "./data"
        type: "missing"
    model:
        num_epochs: 250
        num_samples: 160

configs:
    -   data:
            defaults:
                type: "smoothness"
            grid:
                -   length_scale: [0.25, 0.5]
        model:
            defaults:
                data_path: data_rotated_anisotropic_matern.pkl
            grid:
                -   num_ip_f: [16, 32]
                    num_ip_g: [4, 12]

                -   num_samples: [5, 10, 20]
                    learning_rate: [0.1]
                    num_ip_f: [32]
                    num_ip_g: [32]

    -   skip: true
        data:
            defaults:
                type: "smoothness"
            grid:
                -   length_scale: [0.25, 0.5]
        model:
            defaults:
                data_path: data_rotated_anisotropic_matern.pkl
            grid:
                -   num_ip_f: [16, 32]
                    num_ip_g: [4, 12]
                    learning_rate: [0.1]



"""


def test_parsing():
    yaml_io = io.StringIO(YAML_CONFIG)
    configs = tmutils.read_config_from_stream(yaml_io)

    assert len(configs) == 2 * (4 + 3)

    _ = {
        "skip": False,
        "data": {"data_path": "./data", "type": "smoothness", "length_scale": 0.25},
        "model": {
            "num_epochs": 250,
            "num_samples": 160,
            "data_path": "data_rotated_anisotropic_matern.pkl",
            "num_ip_f": 32,
            "num_ip_g": 4,
            "learning_rate": 0.1,
        },
    }

    assert configs[0]["data"]["type"] == "smoothness"
    assert configs[0]["model"]["num_samples"] == 160
