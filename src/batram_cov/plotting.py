import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure


def exp1_1d_covariates(
    x: jax.Array, plot_data: list[jax.Array], desc: str = ""
) -> Figure:
    side_dim = int(np.sqrt(plot_data[0].shape[0]))
    vminmax = (np.min(plot_data), np.max(plot_data))
    how_many = len(x)
    fig, axs = plt.subplots(3, how_many, figsize=(how_many * 2, 6.2))

    for j in range(3):
        for i in range(how_many):
            pred_field = plot_data[j][:, i].reshape(side_dim, side_dim)
            axs[j, i].imshow(pred_field, vmin=vminmax[0], vmax=vminmax[1])

            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

            if j == 0:
                axs[j, i].set_title(f"x={x[i, 0]:.3f}")

    axs[0, 0].set_ylabel("DGP")
    axs[1, 0].set_ylabel("model")
    axs[2, 0].set_ylabel("model")
    fig.suptitle(desc, fontsize=16)
    fig.tight_layout(pad=1.3)

    return fig


def exp1_2d_covariates(
    x: jax.Array, plot_data: list[jax.Array], desc: str = ""
) -> Figure:
    def plot_subgrid(x, y, vminmax, fig, outer_grid, start_col, title, ylabels=False):
        subgrid = gridspec.GridSpecFromSubplotSpec(
            5,
            5,
            subplot_spec=outer_grid[:, start_col : start_col + 5],
            wspace=0.1,
            hspace=0.1,
        )

        side_dim = int(np.sqrt(y.shape[0]))
        assert y.shape == (side_dim**2, 25)
        y = y.reshape(side_dim, side_dim, 5, 5)
        for i in range(5):
            for j in range(5):
                ax = plt.Subplot(fig, subgrid[i, j])
                ax.imshow(
                    y[..., j, i],
                    vmin=vminmax[0],
                    vmax=vminmax[1],
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"$x_1$ = {x[:5][j, 0]:.2f}", fontsize=10)
                if ylabels and j == 0:
                    ax.set_ylabel(f"$x_2$ = {x[::5][i, 1]:.2f}", fontsize=10)
                fig.add_subplot(ax)

        title_pos = [3.65 / 15, 7.75 / 15, 11.75 / 15]
        fig.text(
            title_pos[start_col // 5],
            0.95,
            title,
            ha="center",
            va="center",
            fontsize=16,
        )

    # plots currently (1024, 5) shape implies that we cannot fill the array

    fig = plt.figure(figsize=(15, 5))
    outer_grid = gridspec.GridSpec(5, 15, wspace=0.4, hspace=0.4)

    vminmax = (np.min(plot_data), np.max(plot_data))
    plot_subgrid(x, plot_data[0], vminmax, fig, outer_grid, 0, "Data", ylabels=True)
    plot_subgrid(x, plot_data[1], vminmax, fig, outer_grid, 5, "Sample 1")
    plot_subgrid(x, plot_data[2], vminmax, fig, outer_grid, 10, "Sample 2")

    fig.suptitle(desc, fontsize=20, y=1.05)
    plt.subplots_adjust(top=0.85)
    return fig


def plot_app_run(x: jax.Array, plot_data: list[jax.Array], desc: str = "") -> Figure:
    match plot_data[0].shape[0]:
        case 2220:
            data_shape = (37, 60)
        case 1426:  # .67391, ASIA
            data_shape = (31, 46)
        case 1500:  # 0.6, NORTH_AMERICA
            data_shape = (30, 50)
        case 18432:  # 0.5, GLOBAL
            data_shape = (96, 192)
        case 8192:
            data_shape = (64, 128)
        case _:
            raise ValueError(f"Unknown shape: {plot_data[0].shape}")

    v = np.max([np.abs(_).max() for _ in plot_data])
    how_many = len(x)
    fig, axs = plt.subplots(5, how_many, figsize=(how_many * 1.8, 7.2))

    for j in range(5):
        for i in range(how_many):
            pred_field = plot_data[j][:, i].reshape(*data_shape)
            _ = axs[j, i].imshow(pred_field, vmin=-v, vmax=v, cmap="RdBu_r")

            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

            if j == 0:
                axs[j, i].set_title(f"x={x[i, 0]:.3f}")

    cax = fig.add_axes((0.0225, 0.04, 0.945, 0.02))
    fig.colorbar(_, cax=cax, orientation="horizontal")

    axs[0, 0].set_ylabel("Test data")
    axs[1, 0].set_ylabel("Model")
    axs[2, 0].set_ylabel("Model")
    fig.suptitle(desc, fontsize=16)
    fig.tight_layout(pad=1.3)

    return fig
