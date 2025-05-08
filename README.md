# batram-cov

This repository provides an extension of the Bayesian transport map methodology ([Katzfuss and Schaefer (2023)](https://doi.org/10.1080/01621459.2023.2197158)) that incorporates covariate-dependent transformations. Thereby enabling conditional modeling with covariates, allowing for a rich spatial structure with nonlinear and nonstationarity dependence that changes with covariates.

There is no accompanying publication for this extension, yet; this work is under active development.

### Related projects

There are numerous related research projects. Please refer to [katzfuss-group/batram](https://github.com/katzfuss-group/batram) for an overview.

### Installation

Clone the repository make sure the project manager [uv](https://docs.astral.sh/uv/) is installed and run the following command:

```bash
uv sync
```

>[!NOTE]
> The dependencies assume a cuda device is present on the machine running this
> code. If no such device is present, replace `jax[cuda]==0.5` in the
> `pyproject.toml` file with `jax==0.5`.

### Getting Started

A [tutorial notebook](notebooks/getting-started.ipynb) demonstrating the
covariate extension is available in the notebooks folder.
