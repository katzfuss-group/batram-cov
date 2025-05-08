import argparse
import logging
import os
import pickle
import pprint
from copy import deepcopy
from functools import partial
from io import BytesIO
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax import nnx
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from . import natgrad as ngm
from . import plotting, predictions
from . import regression_problem as rp
from .data_config_parser import ConfigDataParsingError
from .data_gen import (
    SimulatedDatasets,
    SimulationData,
    retrieve_or_generate_simulation_datasets,
)
from .stopping import early_stopper
from .tmcov import setup_tm_rp
from .utils import read_config, to_strong_jax_type


class VariationalNugget:
    def __init__(self, config: dict):
        self.config = config

        self.type = config["type"]
        self.nugget = config["nugget"]
        self.p0_nugget = config.get("pos_0_nugget", self.nugget)
        match self.type:
            case "const":
                pass
            case "exp_decay":
                self.decay = config["decay"]
            case _:
                raise ValueError(f"Unknown variational nugget type: {self.type}")

    def __call__(self, i: int) -> float:
        match self.type:
            case "const":
                return (i == 0) * self.p0_nugget + (i > 0) * self.nugget
            case "exp_decay":
                return (i == 0) * self.p0_nugget + (i > 0) * self.nugget * self.decay**i
            case _:
                raise ValueError(f"Unknown variational nugget type: {self.type}")


def parse_variational_nugget(
    config: dict,
) -> tuple[VariationalNugget, VariationalNugget]:
    dict_f = config.get("ip_diagonal_add_nugget_f", None)
    if dict_f is None:
        dict_f = config["ip_diagonal_add_nugget"]

    dict_g = config.get("ip_diagonal_add_nugget_g", None)
    if dict_g is None:
        dict_g = config["ip_diagonal_add_nugget"]

    return VariationalNugget(dict_f), VariationalNugget(dict_g)


def create_and_merge_problems(
    data: SimulationData,
    config: dict,
) -> rp.HGPIPProblem:
    # this is for later, when we want to use just a random
    # subset to estimate the model parameters
    # before fitting the entire model
    location_subset = None
    if location_subset is None:
        location_subset = np.arange(data.samples.shape[2])

    num_ip_f = config["num_ip_f"]
    num_ip_g = config["num_ip_g"]
    ip_fixed = config["ip_fixed"]
    ip_distance_penalty = config.get("ip_distance_penalty", None)
    if ip_distance_penalty == 0.0:
        ip_distance_penalty = None

    # add value to diagonal of ip inverses for numerical stability
    var_nugg_f, var_nugg_g = parse_variational_nugget(config)

    max_num_nn = config["num_neighbors"]
    num_nn_in_data = data.nearest_neighbors.shape[1]
    if num_nn_in_data < max_num_nn:
        raise ValueError(
            f"Not enough neighbors in data. Number of neighbors in data is "
            f"{num_nn_in_data}, but max_num_nn is set to {max_num_nn}."
        )

    sample_idx = config["sample_idx"]
    linear_only = config.get("linear_only", False)
    normalize_x = config.get("normalize_x", False)

    problems = []
    for i in location_subset:
        x = data.x
        nn_idx = data.nearest_neighbors[i, :max_num_nn]
        cond_set = data.samples[sample_idx][:, nn_idx]
        dist_to_nn = data.li[i]
        resp = data.samples[sample_idx, :, i]
        var_at_first_loc = np.var(data.samples[sample_idx, :, 0])

        problems.append(
            setup_tm_rp(
                resp,
                cond_set,
                x,
                dist_to_nn,
                i,
                nn_idx,
                num_ip_f,
                num_ip_g,
                log_var_at_first_loc=np.log(var_at_first_loc),
                whiten=True,
                variational_noise_f=var_nugg_f(i),
                variational_noise_g=var_nugg_g(i),
                ip_fixed=ip_fixed,
                ip_distance_penalty=ip_distance_penalty,
                linear_only=linear_only,
                normalize_x=normalize_x,
            )
        )

    # merge problems
    return rp.merge_modules(problems)


def build_data_module(data: SimulationData, config: dict) -> rp.TMDataModule:
    sample_idx = config["sample_idx"]
    max_num_nn = config["num_neighbors"]

    modules = []
    for i in range(data.samples.shape[2]):
        x = data.x
        nn_idx = data.nearest_neighbors[i, :max_num_nn]
        cond_set = data.samples[sample_idx][:, nn_idx]
        dist_to_nn = data.li[i]
        resp = data.samples[sample_idx, :, i]

        modules.append(
            rp.TMDataModule(
                position=i,
                response=resp,
                conditioning_set=cond_set,
                covariates=x,
                dist_nn=dist_to_nn,
                nn_idx=nn_idx,
            )
        )

    return rp.merge_modules(modules)


def log_media_with_run(fig: Figure, run: Run) -> None:
    """Log a matplotlib figure to wandb as a media file."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    wandb_image = wandb.Image(image)
    image.close()
    run.log({"prediction": wandb_image})
    buf.close()


def lr_parser(config: list[str, float, ...], num_epochs: int):
    """Parser to configure a learning rate scheduler.

    Consumes a list of arguments for a learning rate schedule, returning either
    the initial learning rate or a configured optax.warmup_cosine_decay_schedule.

    Args:
    -----
    config: list[str, float, ...]
        case: "const" -> init_lr: float
        case: "cosine" -> optax.warmup_cosine_decay_schedule:
            init_value: float
            peak_value: float
            warmup_steps: int
            decay_steps: int
            end_value: float (optional, default=0.0)
            exponent: float (optional, default=1.0)
    """

    def validate_arg(arg, expected_type):
        if not isinstance(arg, expected_type):
            raise ValueError(f"Expected {expected_type}, got {type(arg)}.")

    validate_arg(config, dict)
    if config["type"] == "constant":
        init_lr = config["init_value"]
        validate_arg(init_lr, float)
        return init_lr
    elif config["type"] == "cosine":
        init_value = config["init_value"]
        peak_value = config["peak_value"]
        warmup_steps = config["warmup_steps"]
        decay_steps = num_epochs
        end_value = config.get("end_value", 0.0)

        validate_arg(init_value, float)
        validate_arg(peak_value, float)
        validate_arg(warmup_steps, int)
        validate_arg(decay_steps, int)
        validate_arg(end_value, float)

        return optax.warmup_cosine_decay_schedule(
            init_value, peak_value, warmup_steps, decay_steps, end_value
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config[0]}")


def stopping_parser(config: dict, num_epochs: int):
    patience = config.get("patience", num_epochs)
    tol = config.get("tol", 0.0)
    stopper = partial(early_stopper, patience=patience, tol=tol)
    return stopper


def _get_run_name(config: dict, idx: int) -> str:
    name = f"{config['data'].get('name', '')}-{config['model'].get('name', '')}"
    if name == "-":
        name = None
    else:
        name += f"-{idx}"

    return name


def _get_run_tags(config: dict) -> list[str] | None:
    data_tags = config["data"].get("tags", [])
    model_tags = config["model"].get("tags", [])
    tags = data_tags + model_tags
    if not tags:
        tags = None

    return tags


def _get_run_description(config: dict):
    desc_data = config["data"].get("desc", "").format(**config["data"])
    desc_model = config["model"].get("desc", "").format(**config["model"])
    descriptions = []
    if desc_data != "":
        descriptions.append(desc_data)
    if desc_model != "":
        descriptions.append(desc_model)

    description = " - ".join(descriptions)

    return description


def _log_prob_one_location(g, joint_params, rest, data_g, data_val):
    m: rp.HGPIPProblem = nnx.merge(g, joint_params, rest)
    data = nnx.merge(data_g, data_val)
    log_prob = m.log_prob(data)
    # log_prob = m.mc_logprob(data)
    # log_prob = m.mc_logprob2(data)
    return log_prob


@jax.jit
def _log_prob_one_location_vmap_jit(g, joint_params, rest, data_g, data_val):
    vlpol = jax.vmap(_log_prob_one_location, (None, None, 0, None, 0))
    return vlpol(g, joint_params, rest, data_g, data_val)


def log_prob_with_data(model, score_data: rp.TMDataModule) -> jax.Array:
    g, joint_params, rest = nnx.split(model, rp.JointParam, object)
    data_g, data_val = nnx.split(score_data, object)
    scores = _log_prob_one_location_vmap_jit(g, joint_params, rest, data_g, data_val)
    return scores


def _fs_gs_one_loc(g, joint_params, rest, data_g, data_val):
    m: rp.HGPIPProblem = nnx.merge(g, joint_params, rest)
    data = nnx.merge(data_g, data_val)
    fs, gs = m.fs_and_gs_from_data(data)
    return fs, gs


@jax.jit
def _fs_gs_one_loc_vmap_jit(g, joint_params, rest, data_g, data_val):
    fn = jax.vmap(_fs_gs_one_loc, (None, None, 0, None, 0))
    return fn(g, joint_params, rest, data_g, data_val)


def get_fs_and_gs(
    model: rp.HGPIPProblem, score_data: rp.TMDataModule
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    g, joint_params, rest = nnx.split(model, rp.JointParam, object)
    data_g, data_val = nnx.split(score_data, object)
    fs, gs = _fs_gs_one_loc_vmap_jit(g, joint_params, rest, data_g, data_val)
    return fs, gs


def nd_array_dict(key, value) -> list[(str, float)]:
    if isinstance(value, float):
        return [(key, value)]
    elif value.ndim == 0:
        return [(key, value)]
    elif value.ndim == 1:
        return [(f"{key}_{i}", v) for i, v in enumerate(value)]
    else:
        raise ValueError(f"Not implemented for ndim > 1. ndim is {value.ndim}")


def build_dict_from_state(state: nnx.State) -> dict:
    d = {}
    for k, v in state.items():
        if isinstance(v, nnx.VariableState):
            bijector = getattr(v, "bijector", lambda x: x)
            val = bijector(v.value)
            for k_, v_ in nd_array_dict(k, val):
                d[k_] = v_
        elif isinstance(v, nnx.State):
            d2 = build_dict_from_state(v)
            for k_, v_ in d2.items():
                d[f"hparam/{k}.{k_}"] = v_
        else:
            raise ValueError(f"Unknown state type: {type(v)}")

    return d


def extract_params_for_logging(problem: nnx.Module) -> dict:
    _, jp, __ = nnx.split(problem, rp.JointParam, object)
    d = build_dict_from_state(jp)

    # conditioning set cutoff
    cs_cutoff = problem.kernel_f.cs_cutoff()
    d["cs_cutoff"] = cs_cutoff

    return d


def fit_model(
    problem: rp.HGPIPProblem,
    config: dict,
    run: Run,
    score_data: SimulationData | None = None,
) -> tuple[Any, list[float] | None, list[float] | None, bool]:
    # split module and convert to strong types to avoid jax recompilation
    g, var_mvn_params, params, gstatic = to_strong_jax_type(
        nnx.split(problem, rp.VarMVNPar, nnx.Param, object)
    )
    problem = nnx.merge(g, params, var_mvn_params, gstatic)

    nepochs = config["num_epochs"]
    num_epochs_only_var_par = config["num_epochs_only_var_par"]
    # if num_epochs_only_var_par > 0:
    #     raise NotImplementedError(
    #         "num_epochs_only_var_par > 0 is not implemented yet."
    #     )

    if nepochs - num_epochs_only_var_par < 0:
        raise ValueError(
            "num_epochs_only_var_par must be less than or equal to num_epochs."
        )

    # set up optimizers
    scheduler = lr_parser(config["lr_schedule"], nepochs)
    stopper = stopping_parser(config.get("stopper", {}), nepochs)
    opt = optax.chain(optax.clip(100), optax.adam(scheduler))

    opt_ng = optax.chain(
        optax.clip(10),
        optax.scale(1e-1),
    )

    _, stop_state = stopper(val=float("inf"), params=None, stop_state=None)
    opt_state = opt.init(params)
    opt_ng_state = opt_ng.init(var_mvn_params)

    if score_data is not None:
        score_data_module = build_data_module(score_data, config)

    # define step function
    def update(epoch, g, params, var_mvn_params, gstatic, opt_states):
        logging.info("compiling update function...")

        opt_ng_state, opt_state = opt_states

        # update parameters without
        def _vloss(params):
            m = nnx.merge(g, params, var_mvn_params, gstatic)
            ig, jp, rs = nnx.split(m, rp.JointParam, object)
            vloss = jax.vmap(
                lambda ig, jp, r: nnx.merge(ig, jp, r).loss(None), (None, None, 0)
            )
            return vloss(
                ig, jp, rs
            ).mean()  # use mean instead of sum to make the learning rate more
            # independent of the number of locations

        ps = params
        opt_state = opt_states[1]

        def _param_update():
            _, grad = jax.value_and_grad(_vloss)(ps)
            updates, opt_state_new = opt.update(grad, opt_state)
            ps_updated = optax.apply_updates(ps, updates)
            return ps_updated, opt_state_new

        # update parameters only when epoch > num_epochs_only_var_par
        params, opt_state = jax.lax.cond(
            epoch < num_epochs_only_var_par,
            lambda: (params, opt_state),
            _param_update,
        )

        # update xi
        def _elbo_natgrad(g, jp, var_mvn_params, r):
            return ngm.elbo_and_nat_grad(
                lambda vmp: nnx.merge(g, jp, vmp, r).elbo(None),
                var_mvn_params["xi"].trf,
                var_mvn_params,
            )

        def _velbo_natgrad(g, jp, var_mvn_params, r):
            elbos, grad = jax.vmap(_elbo_natgrad, (None, None, 0, 0))(
                g, jp, var_mvn_params, r
            )
            return elbos.sum(), grad

        # merge and split to have different parameter separation
        m = nnx.merge(g, params, var_mvn_params, gstatic)
        g, jp, var_mvn_params, gstatic = nnx.split(
            m, rp.JointParam, rp.VarMVNPar, object
        )
        elbo_val, grad = _velbo_natgrad(g, jp, var_mvn_params, gstatic)

        updates, opt_ng_state = opt_ng.update(grad, opt_ng_state)
        var_mvn_params = optax.apply_updates(var_mvn_params, updates)

        # build return values
        loss = -elbo_val

        new_params = (params, var_mvn_params)
        new_opt_states = (opt_ng_state, opt_state)

        return new_params, new_opt_states, loss

    # run training loop
    logging.info(f"Running {nepochs} epochs with {1} steps per epoch")

    opt_states = (opt_ng_state, opt_state)
    tracker = {
        "loss": np.zeros(nepochs, dtype=np.float32),
    }
    if score_data is not None:
        tracker["mean_pred_log_prob"] = np.zeros(nepochs, dtype=np.float32)

    invalid_state_encountered = False

    update_jitted = jax.jit(
        update, donate_argnames=("opt_states", "params", "var_mvn_params")
    )

    for epoch in (bar := tqdm(range(nepochs))):
        g, var_mvn_params, params, gstate = nnx.split(
            problem, rp.VarMVNPar, nnx.Param, object
        )
        (params, var_mvn_params), opt_states, loss = update_jitted(
            epoch, g, params, var_mvn_params, gstate, opt_states
        )

        nnx.update(problem, params, var_mvn_params)

        # update tracker and progress bar
        tracker["loss"][epoch] = loss
        wandb_log_dict = {
            "loss": loss,
        }

        if score_data is not None:
            n_test = score_data_module.size
            log_scores = log_prob_with_data(problem, score_data_module)
            log_prob_mean = log_scores.sum(0).mean()
            tracker["mean_pred_log_prob"][epoch] = log_prob_mean

            wandb_log_dict["mean_pred_log_prob"] = log_prob_mean
            stop, stop_state = stopper(
                val=-log_prob_mean.item(),
                params=(params, var_mvn_params),
                stop_state=stop_state,
            )
            if stop:
                params, var_mvn_params = stop_state[-1]
                nnx.update(problem, params, var_mvn_params)
                logging.info("Early stopping triggered. Exiting training loop early.")
                break

        desc = ", ".join([f"{k}: {v[epoch]:.3f}" for k, v in tracker.items()])
        bar.set_description(desc)

        param_dict = extract_params_for_logging(problem)
        wandb_log_dict.update(param_dict)
        run.log(wandb_log_dict)

        # check for nan loss
        if np.isnan(loss):
            logging.info("NaN loss. Stop training.")
            invalid_state_encountered = True
            break

    # restore best parameters
    if not invalid_state_encountered:
        logging.info("Training finished successfully.")

    return (
        problem,
        tracker.get("loss", None),
        tracker.get("mean_pred_log_prob", None),
        (not invalid_state_encountered),
    )


def predict_and_plot(
    model: nnx.Module,
    dataset: SimulationData,
    sample_fixed_noise: bool,
    desc: str = "",
) -> Figure:
    logging.info("Predicting and plotting.")

    # x always has shape (n, p), so need to identify size of p before picking
    # the plotting function
    rng_key = jax.random.key(dataset.seed)
    x_shape = dataset.x.shape[-1]
    if x_shape == 1:
        x, samples = predictions.exp1_1d_covariates(
            rng_key, model, dataset, sample_fixed_noise, num_samples=2
        )
        fig = plotting.exp1_1d_covariates(x, samples, desc=desc)
    if x_shape == 2:
        x, samples = predictions.exp1_2d_covariates(
            rng_key, model, dataset, sample_fixed_noise, num_samples=2
        )
        fig = plotting.exp1_2d_covariates(x, samples, desc=desc)

    return fig


def do_one_run(
    parsed_args: argparse.Namespace, config: dict, idx: int, sample_fixed_noise: bool
) -> None:
    config = deepcopy(config)
    name = _get_run_name(config, idx)
    tags = _get_run_tags(config)
    description = _get_run_description(config)

    logging.info(f"Running experiment with name: '{name}'")
    if description:
        logging.info(f"Description: {description}")

    with wandb.init(
        project="batram_cov_experiment1",
        entity="danjdrennan-Texas A&M University",
        config=config,
        name=name,
        tags=tags,
        notes=description,
    ) as run:
        data_config = config["data"]
        model_config = config["model"]

        save_exp_name, data = retrieve_or_generate_simulation_datasets(data_config)

        def get_exp_name(filename: str) -> str:
            path, fname = os.path.split(filename)
            fname = fname.split("_")[-1].split(".")[0]
            return os.path.join(path, fname[:8])

        save_exp_name = get_exp_name(save_exp_name)

        assert isinstance(data, SimulatedDatasets), f"{type(data)}"

        logging.info(
            f"Create and merge {data.train.samples.shape[-1]} regression problems"
        )
        model = create_and_merge_problems(data.train, model_config)

        if config["model"]["validation_data"]:
            validation_data = data.valid
        else:
            validation_data = None

        logging.info("Fitting model.")
        model, elbo_loss, mean_log_prob, success = fit_model(
            model, config["model"], run, validation_data
        )
        if not success:
            logging.error("Model fitting failed. Skipping remaining steps.")
            run.finish(1)
            return
        logging.info("Fitting finished.")
        fig = predict_and_plot(model, data.test, sample_fixed_noise, desc=description)
        log_media_with_run(fig, run)
        plt.close(fig)

        save_experiment(
            model,
            data,
            config,
            elbo_loss,
            mean_log_prob,
            save_exp_name,
        )

        return


def save_experiment(
    model: rp.HGPIPProblem,
    data: SimulatedDatasets,
    config: dict[str, Any],
    training_loss: list[float] | None,
    validation_loss: list[float] | None,
    hashed_name: str,
):
    preds = predictions.just_predict(model, data.test)
    x, fixed_noise_samples, random_noise_samples = jax.tree.map(jax.device_get, preds)

    score_data_module = build_data_module(data.test, config["model"])
    log_probs = log_prob_with_data(model, score_data_module)
    fs, gs = get_fs_and_gs(model, score_data_module)

    with open(hashed_name + "inference.pkl", "wb") as f:
        inference_dict = dict(
            x=x,
            training_loss=training_loss,
            validation_loss=validation_loss,
            test_log_probs=log_probs,
            test_f_mean=fs[0],
            test_f_variance=fs[1],
            test_g_mean=gs[0],
            test_g_variance=gs[1],
            fixed_noise_samples=fixed_noise_samples,
            random_noise_samples=random_noise_samples,
        )
        pickle.dump(inference_dict, f)

    with open(hashed_name + "data.pkl", "wb") as f:
        data_dict = dict(
            train=data.train._asdict(),
            valid=data.valid._asdict(),
            test=data.test._asdict(),
        )
        pickle.dump(data_dict, f)

    with open(hashed_name + "config.pkl", "wb") as f:
        pickle.dump(config, f)

    return


def main(args: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="experiment1",
        description="Runs experiment 1 using a config.yaml file",
    )

    parser.add_argument("--config", type=str, help="path to config file to load")

    parser.add_argument(
        "--wandb-disable", action="store_true", help="disable wandb logging"
    )

    parser.add_argument(
        "--run",
        type=int,
        default=-1,
        help="run only one experiment with config corresponding to the given index.",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the loaded configurations and exit",
    )

    parser.add_argument(
        "--sample-fixed-noise",
        action="store_true",
        help=(
            "Sampling is done via transorming noise instead of sampling with "
            "independent random variables."
        ),
    )

    if args is None:
        print(parser.print_help())
        exit(0)

    parsed_args = parser.parse_args(args)

    config_path = parsed_args.config

    if parsed_args.wandb_disable:
        logging.info("Disabling wandb logging.")
        os.environ["WANDB_MODE"] = "disabled"

    configs = read_config(config_path)
    logging.info(f"Loaded {len(configs)} configurations from {config_path}")

    # TODO: Decrement run by 1 to match 0-based indexing in Python with the
    # 1-based counting of wandb's cli reports
    if parsed_args.run >= 0:
        configs = [configs[parsed_args.run]]
        logging.info(f"Running only one experiment with index {parsed_args.run}")

    if parsed_args.print_config:
        pprint.pprint(configs)
        return None

    logging.info(f"Running {len(configs)} experiments")
    for i, config in enumerate(configs):
        logging.info(f"Running experiment {i + 1} out of {len(configs)}.")
        try:
            do_one_run(parsed_args, config, i, parsed_args.sample_fixed_noise)
        except ConfigDataParsingError as e:
            logging.error(f"Error in experiment {i}. Config file:\n{e}")
        logging.info(f"Finished run {i + 1} out of {len(configs)}.")

    return


if __name__ == "__main__":
    main()
