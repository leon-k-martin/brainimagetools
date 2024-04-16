import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def formatted_colorbar(
    data,
    ax=None,
    unit=r"$R_{KL}$",
    orientation="vertical",
    title_pos="right",
    cmap=None,
    **kwargs,
):
    """
    Plot a formatted colorbar into a Matplotlib figure or stand-alone.

    Parameters
    ----------
    data : numpy.ndarray
        TVBase data array.
    ax : matplotlib.pyplot.axis, optional
        Axis of a Matplotlib subplot. If None, a new figure is created. Default is None.
    unit : str, optional
        Unit of the data. Default is '$R_{KL}$'.
    orientation : str, optional
        Orientation of the colorbar ('vertical' or 'horizontal'). Default is 'vertical'.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use. If None, 'hot_r' is used. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to the ColorbarBase constructor.

    Returns
    -------
    matplotlib.pyplot.figure
        Plotted colorbar.

    """

    fontsize = 12

    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = fontsize

    # data = np.round(data, 3)
    if "vmin" not in kwargs.keys():
        vmin = np.nanmin(data)
    else:
        vmin = kwargs["vmin"]

    if "vmax" not in kwargs.keys():
        vmax = np.nanmax(data)
    else:
        vmax = kwargs["vmax"]

    abs_extrema = np.max([abs(vmin), abs(vmax)])

    # Set cbar tick labels.
    if vmin < 0:
        if vmax > 0:
            ticks = [-abs_extrema, 0, abs_extrema]
            norm = colors.TwoSlopeNorm(vmin=-abs_extrema, vcenter=0.0, vmax=abs_extrema)
        if vmax < 0:
            ticks = [vmin, 0]
            norm = mpl.colors.Normalize(vmin=vmin, vmax=0)
    else:
        ticks = [0, vmax]
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    # Check if inserted in figure or create new one.
    if not ax:
        fig_exist = False
        fig, ax = plt.subplots(figsize=(0.25, 2))
    else:
        fig_exist = True

    # Get colorbar from matplotlib if cbar input is string.
    if isinstance(cmap, type(None)):
        # TODO: Change back when tvbase_cmap is adapted to new nilearn style.
        # cmap = tvbase_cmap(data)
        cmap = "hot_r"

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # cb = mpl.colorbar.ColorbarBase(
    #     ax, orientation=orientation, cmap=cmap, norm=norm, ticks=ticks  # vmax and vmin
    # )
    cb = mpl.colorbar.ColorbarBase(
        ax,
        orientation=orientation,
        ticks=ticks,  # [0, 0.5, 1],
        cmap=cmap,
        norm=norm,
    )

    cb.set_ticklabels(np.round(ticks, 3))
    cb.outline.set_visible(False)
    if title_pos in ["up", "top"]:
        cb.ax.set_title(unit, size=fontsize)

    elif title_pos == "right":
        cb.ax.set_ylabel(
            unit, fontsize=fontsize, labelpad=-0.5, **{"rotation": "horizontal"}
        )
    else:
        cb.ax.set_title("")
        cb.ax.set_ylabel("")

    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    ax.patch.set_alpha(0)
    if not fig_exist:
        plt.close(fig)
        return fig


def add_colorbar(data, fig, ax, cmap="viridis", unit="", **kwargs):
    """
    Add a colorbar to a Matplotlib figure.

    Parameters
    ----------
    data : numpy.ndarray
        TVBase data array.
    fig : matplotlib.pyplot.figure
        Figure to which the colorbar will be added.
    ax : matplotlib.pyplot.axis
        Axis to which the colorbar will be added.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use. Default is 'viridis'.
    unit : str, optional
        Unit of the data. Default is an empty string.
    **kwargs : dict
        Additional keyword arguments to pass to the ScalarMappable constructor.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        Added colorbar.

    """

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap),
        ax=ax,
        **kwargs,
    )
    cbar.outline.set_visible(False)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([np.round(vmin, 2), np.round(vmax, 2)])

    cbar.set_label(unit, loc="center")

    return cbar
