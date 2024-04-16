from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from brainvistools import style
from brainvistools import cbar as bvcbar
from brainvistools import colormap as bvcm
from nilearn import plotting

style.init_style()


def formatted_glass_brain(
    img,
    title=None,
    unit=r"$R_{KL}$",
    colorbar=True,
    display_mode="ortho",
    contours=None,
    cmap="viridis",
    fig=None,
    ax=None,
    figsize=(12, 4),
    cbar_kwargs={"anchor": (5.5, 0.5), "shrink": 0.7},
    **kwargs,
):
    """
    Plots formatted version of the `nilearn.plotting.plot_glass_brain` function.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        Brain image in NIfTI format.
    title : str, optional
        Plot title, by default None
    unit : str, optional
        Unit for the colorbar, defaults to r"$R_{KL}$"
    colorbar : bool, optional
        Adds colorbar to plot with `tvbase.plot.add_colorbar`, by default True
    display_mode : str, optional
        display_mode from `nilearn.plotting.plot_glass_brain`, by default "ortho"
    contours : None, optional
        Adds contours of the parcellation, by default None
    alpha : int, optional
        Alpha value (transparency) of the color map, by default 0
    cmap : str, optional
        Color map: accepts string input for matplotlib cmap names, by default "viridis"
    fig : matplotlib.figure.Figure, optional
        Figure instance, by default None
    ax : matplotlib.axes.Axes, optional
        Axes instance, by default None
    figsize : tuple, optional
        Figure size, by default (12, 4)
    cbar_kwargs : dict, optional
        Keyword arguments for `tvbase.plot.add_colorbar`, by default {"anchor": (5.5, 0.5), "shrink": 0.7}
    **kwargs : dict, optional
        Additional keyword arguments to pass to `nilearn.plotting.plot_glass_brain`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plotted figure.

    Examples
    --------
    >>> import nibabel as nib
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Load brain image in NIfTI format
    >>> img = nib.load('brain_image.nii.gz')
    >>>
    >>> # Plot formatted version of the brain image
    >>> fig = formatted_glass_brain(img, title='Brain Image', cmap='viridis', alpha=0.5)
    >>>
    >>> # Show the plot
    >>> plt.show()
    """

    style.init_style()

    data = img.get_fdata()
    # TODO: Check new cmap handling in nilearn!
    # if isinstance(cmap, type(None)):
    #    cmap = tvbase_cmap(data)

    if isinstance(cmap, str) and cmap not in ["bwr", "seismic"]:
        cmap_cbar = cmap
        cmap = bvcm.double_cmap(cmap)
    else:
        cmap_cbar = cmap

    if isinstance(ax, type(None)) or isinstance(fig, type(None)):
        fig, ax = plt.subplots(figsize=figsize)
        is_subplot = False
    else:
        is_subplot = True

    display = plotting.plot_glass_brain(
        img,
        plot_abs=False,
        axes=ax,
        display_mode=display_mode,
        cmap=cmap,
        **kwargs,
    )

    if colorbar is True:
        bvcbar.add_colorbar(
            np.unique(data), fig, ax, unit=unit, cmap=cmap_cbar, **cbar_kwargs
        )

    if not isinstance(title, type(None)) and not is_subplot:
        fig.suptitle(title, ha="center", y=1)
    else:
        ax.set_title(title, loc="left")

    if not isinstance(contours, type(None)):
        display.add_contours(contours, colors="#4dbbd5")
    plt.close(fig)
    return fig


def formatted_stat_map(
    img, title, unit="R", cmap=None, title_pad=1.1, colorbar=True, **kwargs
):
    """Plots formatted version of the :func:`nilearn.plotting.plot_stat_map`

    :param img: _description_
    :type img: _type_
    :param title: _description_
    :type title: _type_
    :param unit: _description_, defaults to 'R'
    :type unit: str, optional
    :param cmap: _description_, defaults to None
    :type cmap: _type_, optional
    :param title_pad: _description_, defaults to 1.1
    :type title_pad: float, optional
    :return: _description_
    :rtype: _type_
    """
    data = img.get_fdata()

    if isinstance(cmap, type(None)):
        cmap = "hot_r"

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    plt.axis("off")

    display = plotting.plot_stat_map(
        img,
        # bg_img=constants.mask_mni152,
        black_bg=False,
        figure=fig,
        cmap=cmap,
        colorbar=False,
        draw_cross=False,
        **kwargs,
    )

    if colorbar is True:
        bvcbar.add_colorbar(
            data, fig, unit=unit, cmap=cmap, orientation="vertical", borderpad=-2
        )

    if title:
        fig.suptitle(title, ha="center", va="top", y=title_pad)

    fig.patch.set_alpha(0)  # Transparent background.

    return display
