# batram-cov

This repository extends Bayesian transport maps ([Katzfuss and Schaefer (2023)](https://doi.org/10.1080/01621459.2023.2197158)) that incorporates covariate-dependent transformations. This facilities modeling conditional distributions with ocvariates, thereby allowing for a rich spatial structure with nonlinear and nonstationary depdendence that changes as a function of covariates.

There is no accompanying publication for this extension, yet; this work is under active development.

## Related projects

There are numerous related research projects. Please refer to [katzfuss-group/batram](https://github.com/katzfuss-group/batram) for an overview.

## Installing as a dependency

If you'd like to use this repo in your packages, select one of the following installation settings:

```bash
# CPU-only to try code.
uv add "batram-cov @ git+https://github.com/katzfuss-group/batram-cov.git"

# Install cuda and let package manager infer it
uv add "batram-cov[cuda] @ git+https://github.com/katzfuss-group/batram-cov.git"

# Select a CUDA version that matches your hardware
uv add "batram-cov[cuda12] @ git+https://github.com/katzfuss-group/batram-cov.git"
uv add "batram-cov[cuda13] @ git+https://github.com/katzfuss-group/batram-cov.git"

# LOCAL CUDA versions (substantially reduces package size)
uv add "batram-cov[cuda12-local] @ git+https://github.com/katzfuss-group/batram-cov.git"
uv add "batram-cov[cuda13-local] @ git+https://github.com/katzfuss-group/batram-cov.git"
```

> [!NOTE]
> The `cuda12-local` and `cuda13-local` options can drastically shrink the dependency list if you already have CUDA installed on your machine/HPC. It is the recommended path to install over if you are using this code on an HPC or in an environment where there are file/memory limits.
>
> Using a `cuda-local` library requires having a full CUDA install in your system path.

## Installation in repo

Make sure [uv](https://docs.astral.sh/uv/) is installed, then clone the repository and run:

```bash
uv sync
```

The `experiments` extra adds notebook and visualization dependencies (cartopy, jupyter, matplotlib, etc.):

```bash
uv sync --extra experiments               # CPU
uv sync --extra experiments,cuda          # Infers CUDA version
uv sync --extra experiments,cuda12        # Uses CUDA12
uv sync --extra experiments,cuda12-local  # Uses local CUDA12
uv sync --extra experiments,cuda13        # Uses CUDA13
uv sync --extra experiments,cuda13-local  # Uses local CUDA13
```

### Development

The dev environment adds dependencies for running the test suite and rendering notebooks. Dev dependencies can be installed using

```bash
uv sync --dev
```

Development should not require a GPU in most cases.

## Getting Started

A [tutorial notebook](notebooks/getting-started.ipynb) demonstrating the covariate extension is available in the notebooks folder.

## System Requirements

Models in this package require a minimum of ~8 GB of RAM for small problems; real development workloads typically need 20 GB or more. HPC environments are strongly recommended for anything beyond prototyping.

For local exploration, the built-in data generators (`gen_data`, `gen_data2`) are designed to produce small-scale datasets suitable for prototyping on a laptop.
