from os.path import join

import hcp_utils as hcp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from tvbase.parcellations import mmp

from brainvistools import colormap, colors, constants, legend, style, templates

style.init_style()


def normalize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def plot_hcp_connectome(
    fc, edge_thresh=0.7, edge_cmap="inferno", labels=None, fout=None, ax=None, **kwargs
):
    """Plot a connectome based on the functional connectivity matrix `fc`.

    Parameters
    ----------
    fc : numpy.ndarray, pandas.DataFrame
        A 2D numpy array or pandas data frame of shape (N, N) representing the functional (or structural) connectivity matrix.
    edge_thresh : float, optional
        The threshold value for the edges to display. Default is 0.7.
    edge_cmap : str or matplotlib colormap, optional
        The colormap for the edges. If a string is passed, it should be the name of a valid colormap. Default is 'inferno'.
    labels : pandas.DataFrame or None, optional
        A pandas DataFrame object containing labels for each node in the connectome. If None is passed, the node labels are inferred from the index of `fc`. Default is None.
    fout : str or None, optional
        The filename to save the plot. If None is passed, the plot is not saved. Default is None.
    ax : matplotlib AxesSubplot or None, optional
        The subplot to plot on. If None is passed, a new figure is created. Default is None.
    **kwargs : optional
        Additional keyword arguments to pass to the `plotting.plot_connectome()` function.

    Returns
    -------
    matplotlib Figure or None
        The plotted figure. If `fout` is passed, the figure is saved to file and None is returned.

    Raises
    ------
    TypeError
        If `fc` is not a 2D numpy array.

    Notes
    -----
    The connectome is plotted using the `hcp.mmp` atlas.

    Examples
    --------
    Plot a connectome using the default settings:

    >>> plot_hcp_connectome(fc)

    Plot a connectome with custom edge threshold and colormap:

    >>> plot_hcp_connectome(fc, edge_thresh=0.5, edge_cmap='coolwarm')

    Plot a connectome and save the plot to file:

    >>> plot_hcp_connectome(fc, fout='connectome.png')
    """

    if isinstance(edge_cmap, str):
        edge_cmap = colormap.double_cmap(edge_cmap)

    centers = np.loadtxt(
        join(constants.DATA_DIR, "tvbase_atlas_mni_center_coordinates.txt")
    )
    if isinstance(labels, type(None)) and isinstance(labels, pd.DataFrame):
        labels = fc.index.to_list()

    colors = list()
    for k, v in hcp.mmp["rgba"].items():
        colors.append(v)

    edge_threshold = edge_thresh

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(10, 5))

    display = plotting.plot_connectome(
        fc,
        centers,
        node_color=colors[1:],
        edge_threshold=edge_threshold,
        display_mode="x",
        node_size=40,
        edge_vmin=edge_threshold,
        edge_vmax=1,
        node_kwargs={"alpha": 0.6},
        edge_kwargs={"alpha": 0.8, "linewidth": 0.5},
        edge_cmap=edge_cmap,
        axes=ax,
        **kwargs
    )

    if fout:
        display.savefig(fout + ".png", dpi=500)

    try:
        fig
    except NameError:
        fig = None

    if not isinstance(fig, type(None)):
        plt.close()
        return fig


# %%


def plot_brain_connectome(
    cm,
    parcellation=None,
    centers=None,
    colors=colors.mmp1_colors,
    ax=None,
    plot_both_hemis=False,
    area_info=None,
    edge_threshold=0,
    edge_size_factor=1,
):
    """
    Plot a brain connectome.

    Parameters
    ----------
    cm : ndarray of shape (n_regions, n_regions)
        The connectivity matrix.
    parcellation : str or None, optional
        The parcellation file or None. Default is None.
    centers : ndarray or None, optional
        The coordinates of the regions or None. Default is None.
    colors : str or ndarray or None, optional
        The colors for each region or None. Default is colors.mmp1_colors.
    ax : matplotlib Axes3DSubplot or None, optional
        The Axes3DSubplot object to plot on or None. Default is None.
    plot_both_hemis : bool, optional
        Whether to plot both hemispheres or not. Default is False.
    area_infos : list of str or None, optional
        The list of area names or None. Default is None.

    Raises
    ------
    ValueError
        If neither parcellation nor centers are specified.

    Returns
    -------
    None

    Notes
    -----
    The function plots a brain connectome with subcortical and cortical regions, and their interconnections.

    Examples
    --------
    >>> plot_brain_connectome(cm, parcellation="parcellation.nii.gz", centers=centers, colors=colors)
    """

    hemi = "right"
    view = "lateral"

    cm = (cm - np.min(cm)) / (np.max(cm) - np.min(cm))

    if isinstance(parcellation, str):
        print("parcellation given")
        nib.load(parcellation)

    if parcellation and isinstance(centers, type(None)):
        print("centers given")
        centers = plotting.find_parcellation_cut_coords(parcellation)

    if isinstance(centers, type(None)) and isinstance(parcellation, type(None)):
        print("no centers and parcellation specified. loading default HCP-MMP1 centers")
        centers = np.loadtxt(
            join(constants.DATA_DIR, "tvbase_atlas_mni_center_coordinates.txt")
        )
        area_info = mmp.area_info

    if isinstance(colors, str):
        colors = np.repeat(colors, cm.shape[0])
    centers = np.array(centers)

    # Geometry
    coords, tri = templates.geometry_from_gifti(templates.get_fsLR(hemi='L'))

    sc_dict = list()

    for ind_i, i in enumerate(cm):
        if isinstance(area_info, pd.DataFrame):
            hemi_i = area_info.loc[ind_i, "hemisphere"]

            if hemi_i != "L":
                continue

        for ind_j, j in enumerate(i):
            if isinstance(area_info, pd.DataFrame):
                hemi_j = area_info.loc[ind_j, "hemisphere"]

                if hemi_j != "L":
                    continue

            center_j = centers[ind_j]
            center_i = centers[ind_i]

            x = np.array((center_j[0], center_i[0]))
            y = np.array((center_j[1], center_i[1]))
            z = np.array((center_j[2], center_i[2]))

            if j > 0:
                sc_dict.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "weight": j,
                        "type": "{}-{}".format(ind_i, ind_j),
                    }
                )

    cmap = plt.cm.magma

    if isinstance(ax, type(None)):
        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_axes((0, 0, 1, 1), projection="3d", aspect="auto")

    limits = [coords.min(), coords.max()]
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    # set view
    ax.view_init(
        elev=0,
        azim=180,  # roll=10
    )
    ax.set_axis_off()

    # ax.dist = 8

    # cortical nodes
    for i, (x, y, z) in enumerate(centers[:180]):
        ax.scatter(x, y, z, color=colors[i], s=100, alpha=0.8, marker="s")

    # subcortex nodes
    for i, (x, y, z) in enumerate(centers[360:]):
        ax.scatter(
            x,
            y,
            z,
            color=colors[179 + i],
            s=200,
        )

    # cortical connectivity
    for entry in sc_dict:
        if entry["weight"] <= edge_threshold:
            continue
        ax.plot(
            entry["x"],
            entry["y"],
            entry["z"],
            "-",
            c=cmap(entry["weight"]),
            alpha=entry["weight"],  # np.log(1+entry['weight'])*10**-2,
            linewidth=entry["weight"]*edge_size_factor,
        )

    egrey=.4
    ealpha=.2
    fgrey=.5
    falpha=.2

    # plot lh
    ax.plot_trisurf(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        triangles=tri,
        antialiased=False,
        edgecolor=(egrey,egrey,egrey,ealpha),
        facecolor=(fgrey,fgrey,fgrey,falpha),
        # facealpha=0.5,
        linewidth=0.001,
        color="grey",
    )

    # if plot_both_hemis:
    #     coords_rh, tri_rh = templates.geometry_from_gifti(templates.get_fsLR(hemi="R"))
    #     # plot rh
    #     ax.plot_trisurf(
    #         coords_rh[:, 0],
    #         coords_rh[:, 1],
    #         coords_rh[:, 2],
    #         triangles=tri_rh,
    #         antialiased=False,
    #         edgecolor="k",
    #         facecolor="grey",
    #         alpha=0.1,
    #         linewidth=10**-8,
    #         color="grey",
    #     )

    legend.legend_circles(["Subcortical areas"], colors[180:], loc=0)

    img = plt.imshow(cm, cmap=cmap, aspect="auto")
    img.set_visible(False)

    cax = plt.colorbar(cmap=cmap, **{"shrink": 0.2}, pad=-0.1)
    cax.ax.set_ylabel("connectivity")
    cax.set_ticks([0, 1])
    cax.set_ticklabels([0, np.max(cm)])

    plt.close()

    return figure

# %%
def load_parcellation(parcellation):
    """
    Load the parcellation file.

    Parameters
    ----------
    parcellation : str
        Path to the parcellation file.

    Returns
    -------
    nibabel.Nifti1Image
        The loaded parcellation image.
    """
    if isinstance(parcellation, str):
        return nib.load(parcellation)
    return None

def get_centers(parcellation, centers):
    """
    Get the centers of the regions.

    Parameters
    ----------
    parcellation : str or None
        Path to the parcellation file or None.
    centers : ndarray or None
        Pre-defined centers or None.

    Returns
    -------
    ndarray
        The centers of the regions.
    """
    if parcellation and centers is None:
        return plotting.find_parcellation_cut_coords(parcellation)
    if centers is None and parcellation is None:
        # Load default centers (update with actual path and module)
        return np.loadtxt(join(constants.DATA_DIR, "tvbase_atlas_mni_center_coordinates.txt"))
    return np.array(centers)

def plot_nodes(ax, centers, colors):
    """
    Plot the nodes (both cortical and subcortical) on the 3D plot.

    Parameters
    ----------
    ax : matplotlib Axes3DSubplot
        The 3D subplot to plot on.
    centers : ndarray
        The centers of the regions.
    colors : ndarray
        The colors for each region.

    Returns
    -------
    None
    """
    # Cortical nodes
    for i, (x, y, z) in enumerate(centers[:180]):
        ax.scatter(x, y, z, color=colors[i], s=100, alpha=0.8, marker="s")

    # Subcortex nodes
    for i, (x, y, z) in enumerate(centers[360:]):
        ax.scatter(x, y, z, color=colors[179 + i], s=200)


def plot_connections(ax, cm, centers, area_info, edge_threshold, edge_size_factor):
    """
    Plot the connections between nodes on the 3D plot.

    Parameters
    ----------
    ax : matplotlib Axes3DSubplot
        The 3D subplot to plot on.
    cm : ndarray
        The connectivity matrix.
    centers : ndarray
        The centers of the regions.
    area_info : list of str or None
        The list of area names or None.
    edge_threshold : float
        The threshold for edge weights.
    edge_size_factor : float
        The factor to scale the size of the edges.

    Returns
    -------
    None
    """
    cmap = plt.cm.magma
    sc_dict = list()

    for ind_i, i in enumerate(cm):
        if isinstance(area_info, pd.DataFrame):
            hemi_i = area_info.loc[ind_i, "hemisphere"]
            if hemi_i != "L":
                continue

        for ind_j, j in enumerate(i):
            if isinstance(area_info, pd.DataFrame):
                hemi_j = area_info.loc[ind_j, "hemisphere"]
                if hemi_j != "L":
                    continue

            center_j = centers[ind_j]
            center_i = centers[ind_i]

            x = np.array((center_j[0], center_i[0]))
            y = np.array((center_j[1], center_i[1]))
            z = np.array((center_j[2], center_i[2]))

            if j > edge_threshold:
                sc_dict.append(
                    {"x": x, "y": y, "z": z, "weight": j, "type": f"{ind_i}-{ind_j}"}
                )

    # Plot connections
    for entry in sc_dict:
        ax.plot(
            entry["x"], entry["y"], entry["z"], "-", c=cmap(entry["weight"]), alpha=entry["weight"], linewidth=entry["weight"] * edge_size_factor,
        )

