# batram-cov

This repository extends Bayesian transport maps ([Katzfuss and Schaefer (2023)](https://doi.org/10.1080/01621459.2023.2197158)) that incorporates covariate-dependent transformations. This facilities modeling conditional distributions with ocvariates, thereby allowing for a rich spatial structure with nonlinear and nonstationary depdendence that changes as a function of covariates.

There is no accompanying publication for this extension, yet; this work is under active development.

## Related projects

There are numerous related research projects. Please refer to [katzfuss-group/batram](https://github.com/katzfuss-group/batram) for an overview.

## Installation

Make sure [uv](https://docs.astral.sh/uv/) is installed, then clone the repository and run:

```bash
# CPU
uv sync --extra cpu

# GPU
uv sync --extra cuda
```

To add as a dependency in another uv project without cloning:

- CPU:
  ```bash
  # CPU
  uv add "batram-cov @ git+https://github.com/katzfuss-group/batram-cov.git" --extra cpu
  ```

- GPU:
  ```bash
  # GPU
  uv add "batram-cov @ git+https://github.com/katzfuss-group/batram-cov.git" --extra cuda
  ```

- CPU (via `uv pip`):
  ```bash
  # CPU
  uv pip install "batram-cov[cpu] @ git+https://github.com/katzfuss-group/batram-cov.git"
  ```

- GPU (via `uv pip`):
  ```bash
  # GPU
  uv pip install "batram-cov[cuda] @ git+https://github.com/katzfuss-group/batram-cov.git"
  ```

Or using **pip**:

- CPU:
  ```bash
  # CPU
  pip install "batram-cov[cpu] @ git+https://github.com/katzfuss-group/batram-cov.git"
  ```

- GPU:
  ```bash
  # GPU
  pip install "batram-cov[cuda] @ git+https://github.com/katzfuss-group/batram-cov.git"
  ```

## Getting Started

A [tutorial notebook](notebooks/getting-started.ipynb) demonstrating the covariate extension is available in the notebooks folder.

## System Requirements

Models in this package require a minimum of ~8 GB of RAM for small problems; real development workloads typically need 20 GB or more. HPC environments are strongly recommended for anything beyond prototyping.

For local exploration, the built-in data generators (`gen_data`, `gen_data2`) are designed to produce small-scale datasets suitable for prototyping on a laptop.
