from os.path import join

import hcp_utils as hcp
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import plotting
from nltools.data import Adjacency
from nltools.mask import expand_mask, roi_to_brain

from brainvistools import colormap, constants, style

style.init_style()


def upper(df):
    """Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    """
    try:
        assert type(df) == np.ndarray
    except:
        if type(df) == pd.DataFrame:
            df = df.values
        else:
            raise TypeError("Must be np.ndarray or pd.DataFrame")
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]


def make_fc_graph(fc, quant=0.35):
    a = Adjacency(fc, matrix_type="similarity")
    a_thresholded = a.threshold(upper=np.quantile(fc, quant), binarize=False)
    G = a_thresholded.to_graph()
    return G


def get_degree(fc, quant=0.25, roi_labels=None, return_nii=False):
    G = make_fc_graph(fc, quant=quant)
    degree = pd.Series(dict(G.degree()))

    if not isinstance(roi_labels, type(None)):
        degree.index = roi_labels

    if return_nii:
        mask = nib.load(
            join(
                constants.ATLAS_DIR,
                "MNI_ICBM2009c",
                "label",
                "aparc+aseg-mni_09c.nii.gz",
            )
        )

        mask_data = mask.get_fdata()
        mask_data_reind = np.zeros(mask_data.shape)

        for i, c in enumerate(roi_labels):
            ind = reparc.fs_mapper(index_as_key=False)[c]
            mask_data_reind = np.where(mask_data == ind, int(i + 1), mask_data_reind)

        mask = nib.Nifti1Image(mask_data_reind, mask.affine)
        mask_x = expand_mask(mask)

        brain_degree = roi_to_brain(degree, mask_x)

        brain_degree.nii = brain_degree.to_nifti()

        return brain_degree.nii

    return degree


def plot_fc_graph(
    fc, ax=None, node_color=None, linewidths=0.5, node_size=1, labelsize=9, thresh=0.3
):
    G = make_fc_graph(fc, quant=thresh)
    pos = nx.spring_layout(G)
    node_and_degree = G.degree()

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(10, 5))

    nx.draw_networkx_edges(G, pos, width=linewidths, alpha=0.2, ax=ax)

    if labelsize > 0:
        nx.draw_networkx_labels(
            G, pos, font_size=labelsize, font_color="darkslategray", ax=ax
        )

    if isinstance(node_color, type(None)):
        node_color = list(dict(node_and_degree).values())

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(dict(node_and_degree).keys()),
        node_size=[x[1] * node_size for x in node_and_degree],
        node_color=node_color,
        cmap=plt.cm.Reds_r,
        linewidths=linewidths,
        edgecolors="darkslategray",
        alpha=1,
        ax=ax,
    )


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


def plot_fc_overview(datapath):
    """
    Plots timeseries of tvb-input folder for each subject.
    """

    n_subjects = 27

    ncols = 7
    nrows = n_subjects // ncols + (n_subjects % ncols > 0)

    row = 0
    col = 0

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharex=True, sharey=True)

    mean_fcs = []

    for n, subid in enumerate(range(1, 28)):
        if col == ncols:
            col = 0
            row += 1

        if subid < 10:
            subid = "0" + str(subid)

        fin_ts = join(datapath, "tvb_input/fc/sub-{}".format(subid))
        fin = join(
            fin_ts,
            "sub-{}_task-rest_run-average_space-fsLR_den-91k_atlas-aparc+aseg_desc-residual_bold.pconn.csv".format(
                subid
            ),
        )
        df = pd.read_csv(fin, index_col=0)

        mean_fcs.append(df)

        # ax = plt.subplot(nrows, ncols, n + 1#, sharex=ax, sharey=ax)
        ax = axes[row, col]
        ax.imshow(df.values, vmin=-1, vmax=1)
        ax.set_title("sub-{}".format(subid))

        col += 1

    # If there is space, add average timeseries.
    if n_subjects % ncols != 0:
        mfc = mean_fcs[0]
        for fc in mean_fcs[1:]:
            mfc += fc
        mfc = mfc / len(mean_fcs)

        axes[row, col].imshow(mfc, vmin=-1, vmax=1)
        axes[row, col].set_title("average FC")

    plt.subplots_adjust(hspace=-0.5)
    plt.tight_layout()
    plt.suptitle("BOLD FCs (averaged across all 6 runs)", fontsize=18, y=0.95)
    plt.savefig(join(datapath, "tvb_input/fc/", "BOLD_FC_overview.png"), dpi=500)
