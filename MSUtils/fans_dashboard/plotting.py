import numpy as np
import shutil
import matplotlib.pyplot as plt


def setup_mpl_style(fontsize=12):

    use_tex = shutil.which("latex") is not None

    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_subplots(
    data1,
    data2,
    labels_x=None,
    labels_y=None,
    subplot_titles=None,
    title="",
    nrows=None,
    ncols=None,
    linewidth=1.4,
    markersize=4.5,
    linecolor=None,
    markercolor=None,
    fontsize=12,
    fig=None,
    axes=None,
    figsize=None,
    sharex=False,
    sharey=False,
    marker="o",
    linestyle="--",
    grid=True,
    grid_alpha=0.3,
    grid_linestyle="--",
    grid_linewidth=0.5,
    minor_ticks=True,
    markerfacecolor="white",
    markeredgewidth=1.0,
    fillstyle="full",
    hide_unused_axes=True,
    tight_layout=True,
):
    """
    Plot a grid of subplots using Matplotlib.

    Parameters
    ----------
    data1, data2 : np.ndarray
        Arrays of shape (n_points,) or (n_points, n_components).
    labels_x, labels_y : list[str], optional
        Axis labels per component.
    subplot_titles : list[str], optional
        Titles per subplot.
    title : str, optional
        Figure title.
    nrows, ncols : int, optional
        Grid dimensions. Auto-chosen if omitted.
    linewidth, markersize : float, optional
        Line and marker sizes.
    linecolor, markercolor : str or list[str], optional
        Per-component colors.
    fontsize : int, optional
        Base font size.
    fig, axes : optional
        Existing figure/axes for overlaying.
    figsize : tuple, optional
        Figure size in inches.
    sharex, sharey : bool, optional
        Shared axes.
    marker, linestyle : str or list[str], optional
        Marker/linestyle per component.
    grid : bool, optional
        Whether to draw grid.
    grid_alpha, grid_linestyle, grid_linewidth : optional
        Grid style.
    minor_ticks : bool, optional
        Enable minor ticks.
    markerfacecolor : str or list[str], optional
        Marker face color(s). Default is white for hollow markers.
    markeredgewidth : float, optional
        Marker edge width.
    fillstyle : str, optional
        Marker fillstyle.
    hide_unused_axes : bool, optional
        Hide empty subplot slots.
    tight_layout : bool, optional
        Apply tight_layout at the end.

    Returns
    -------
    fig, axes
    """

    def _as_2d(arr, name):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D.")
        return arr

    def _normalize(arg, n, name, default):
        if arg is None:
            return [default] * n
        if isinstance(arg, str):
            return [arg] * n
        if len(arg) != n:
            raise ValueError(
                f"The length of {name} must match the number of components ({n})."
            )
        return list(arg)

    data1 = _as_2d(data1, "data1")
    data2 = _as_2d(data2, "data2")

    if data1.shape != data2.shape:
        raise ValueError("data1 and data2 must have identical shape.")

    n_components = data1.shape[1]

    if nrows is None or ncols is None:
        nrows = int(np.ceil(np.sqrt(n_components)))
        ncols = int(np.ceil(n_components / nrows))

    if labels_x is None:
        labels_x = [""] * n_components
    elif len(labels_x) != n_components:
        raise ValueError(
            f"The length of labels_x must match the number of components ({n_components})."
        )

    if labels_y is None:
        labels_y = [""] * n_components
    elif len(labels_y) != n_components:
        raise ValueError(
            f"The length of labels_y must match the number of components ({n_components})."
        )

    if subplot_titles is None:
        subplot_titles = [f"Component {i + 1}" for i in range(n_components)]
    elif len(subplot_titles) != n_components:
        raise ValueError(
            f"The length of subplot_titles must match the number of components ({n_components})."
        )

    linecolor = _normalize(linecolor, n_components, "linecolor", "C0")
    markercolor = _normalize(markercolor, n_components, "markercolor", "C0")
    markerfacecolor = _normalize(
        markerfacecolor, n_components, "markerfacecolor", "white"
    )
    marker_list = _normalize(marker, n_components, "marker", "o")
    linestyle_list = _normalize(linestyle, n_components, "linestyle", "--")

    setup_mpl_style(fontsize=fontsize)

    if fig is None or axes is None:
        if figsize is None:
            figsize = (3.2 * ncols, 2.8 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
    else:
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(nrows, ncols)
        elif axes.ndim != 2:
            raise ValueError("axes must be 1D or 2D array-like.")

    if subplot_titles is None:
        subplot_titles = [f"Component {i + 1}" for i in range(n_components)]
    elif len(subplot_titles) != n_components:
        raise ValueError(
            f"The length of subplot_titles must match the number of components ({n_components})."
        )

    linecolor = _normalize(linecolor, n_components, "linecolor", "C0")
    markercolor = _normalize(markercolor, n_components, "markercolor", "C0")
    markerfacecolor = _normalize(
        markerfacecolor, n_components, "markerfacecolor", "white"
    )
    marker_list = _normalize(marker, n_components, "marker", "o")
    linestyle_list = _normalize(linestyle, n_components, "linestyle", "--")

    setup_mpl_style(fontsize=fontsize)

    if fig is None or axes is None:
        if figsize is None:
            figsize = (3.2 * ncols, 2.8 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
    else:
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(nrows, ncols)
        elif axes.ndim != 2:
            raise ValueError("axes must be 1D or 2D array-like.")

    flat_axes = axes.ravel()

    for i in range(n_components):
        ax = flat_axes[i]

        ax.plot(
            data1[:, i],
            data2[:, i],
            linestyle=linestyle_list[i],
            linewidth=linewidth,
            marker=marker_list[i],
            markersize=markersize,
            markerfacecolor=markerfacecolor[i],
            markeredgecolor=markercolor[i],
            markeredgewidth=markeredgewidth,
            fillstyle=fillstyle,
            color=linecolor[i],
        )

        ax.set_xlabel(labels_x[i], fontsize=fontsize)
        ax.set_ylabel(labels_y[i], fontsize=fontsize)
        ax.set_title(subplot_titles[i], fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        if grid:
            ax.grid(
                True,
                alpha=grid_alpha,
                linestyle=grid_linestyle,
                linewidth=grid_linewidth,
            )

        if minor_ticks:
            ax.minorticks_on()

    if hide_unused_axes:
        for j in range(n_components, nrows * ncols):
            flat_axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=fontsize)

    if tight_layout:
        plt.tight_layout()

    return fig, axes
