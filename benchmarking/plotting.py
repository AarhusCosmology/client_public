import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from getdist import plots
from matplotlib.lines import Line2D


TEXTWIDTH_PTS = 440
WIDTH_INCHES = TEXTWIDTH_PTS / 72.27
FONT_SIZE = 11 / 1.2


def configure_matplotlib():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": "cmr10",
        "font.size": FONT_SIZE,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
    })


def save_triangle_plot(
    figure_dir,
    timestamp,
    iteration,
    samples,
    reference_samples,
    plot_params,
    param_indices,
    getdist_ranges,
    training_samples=None,
    surrogate_sampler="ensemble",
    reference_sampler=None,
):
    figure_dir.mkdir(exist_ok=True)
    configure_matplotlib()

    plotter = plots.get_subplot_plotter(width_inch=WIDTH_INCHES)
    plotter.settings.axes_fontsize = FONT_SIZE
    plotter.settings.axes_labelsize = FONT_SIZE
    plotter.settings.legend_fontsize = FONT_SIZE * 0.9
    plotter.settings.figure_legend_frame = False

    plot_data = [reference_samples, samples] if reference_samples else samples
    plot_args = {
        "filled": False,
        "param_limits": {name: getdist_ranges[name] for name in plot_params},
    }
    if reference_samples:
        plot_args.update({
            "line_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}],
            "contour_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}],
        })
    else:
        plot_args.update({
            "line_args": [{"lw": 2, "color": "C0"}],
            "contour_args": [{"lw": 2, "color": "C0"}],
        })

    plotter.triangle_plot(plot_data, plot_params, **plot_args)
    if training_samples is not None:
        _scatter_training_samples(plotter, training_samples, plot_params, param_indices)

    for legend in list(plotter.fig.legends):
        legend.remove()

    plotter.fig.legend(
        handles=_legend_elements(reference_samples, training_samples, surrogate_sampler, reference_sampler),
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=FONT_SIZE * 0.9,
        framealpha=0.9,
    )

    output_path = figure_dir / f"{timestamp}_triangle_plot_it_{iteration}.pdf"
    plotter.export(str(output_path))
    return output_path


def save_training_history_plot(figure_dir, timestamp, iteration, history_path):
    if not history_path.exists():
        return None

    figure_dir.mkdir(exist_ok=True)
    configure_matplotlib()
    history = pd.read_csv(history_path)

    fig, ax = plt.subplots(figsize=(WIDTH_INCHES, WIDTH_INCHES * 0.6))
    epochs = range(len(history["loss"]))
    ax.plot(epochs, history["loss"].to_numpy(), label="Training", color="blue", alpha=0.8)
    ax.plot(epochs, history["val_loss"].to_numpy(), label="Validation", color="orange", alpha=0.8)
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"Training History - Iteration {iteration}", yscale="log")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend()
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.92)

    output_path = figure_dir / f"{timestamp}_training_history_it_{iteration}.pdf"
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def _scatter_training_samples(plotter, training_samples, plot_params, param_indices):
    # Keep dot size a fixed fraction of panel width regardless of n_params.
    n_plot = len(plot_params)
    panel_pts = WIDTH_INCHES * 72 / n_plot
    size = (0.01 * panel_pts) ** 2

    for j in range(1, len(plot_params)):
        for i in range(j):
            ax = plotter.get_axes_for_params(plot_params[i], plot_params[j])
            if ax:
                ax.scatter(
                    training_samples[:, param_indices[i]],
                    training_samples[:, param_indices[j]],
                    s=size,
                    alpha=0.15,
                    color="black",
                    zorder=1,
                    edgecolors="none",
                    rasterized=True,
                )


def _legend_elements(reference_samples, training_samples, surrogate_sampler, reference_sampler):
    ref_label = f"Reference ({reference_sampler})" if reference_sampler else "True Posterior"
    surr_label = f"Surrogate ({surrogate_sampler})"

    elements = [Line2D([0], [0], color="C0", lw=2, label=surr_label)]
    if reference_samples:
        elements.append(Line2D([0], [0], color="C1", lw=2, label=ref_label))
    if training_samples is not None:
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                lw=0,
                ms=4,
                alpha=0.7,
                label="Training Data",
            )
        )
    return elements
