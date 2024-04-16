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
import hcp_utils as hcp

from brainvistools import colormap, constants
from tvbase.parcellations import mmp


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


# def plot_fc_graph(
#     fc, ax=None, node_color=None, linewidths=0.5, node_size=1, labelsize=9, thresh=0.3
# ):
#     G = make_fc_graph(fc, quant=thresh)
#     pos = nx.spring_layout(G)
#     node_and_degree = G.degree()

#     if isinstance(ax, type(None)):
#         fig, ax = plt.subplots(figsize=(10, 5))

#     nx.draw_networkx_edges(G, pos, width=linewidths, alpha=0.2, ax=ax)

#     if labelsize > 0:
#         nx.draw_networkx_labels(
#             G, pos, font_size=labelsize, font_color="darkslategray", ax=ax
#         )

#     if isinstance(node_color, type(None)):
#         node_color = list(dict(node_and_degree).values())

#     nx.draw_networkx_nodes(
#         G,
#         pos,
#         nodelist=list(dict(node_and_degree).keys()),
#         node_size=[x[1] * node_size for x in node_and_degree],
#         node_color=node_color,
#         cmap=plt.cm.Reds_r,
#         linewidths=linewidths,
#         edgecolors="darkslategray",
#         alpha=1,
#         ax=ax,
#     )


# %%


def plot_matrix_to_graph(
    rc,
    labels=None,
    tperc=55,
    min_connected_comp=3,
    ax=None,
    draw_labels=False,
    size_factor=1,
    layout="spring",
):
    if isinstance(ax, type(None)):
        new_fig = True
        fig, ax = plt.subplots(figsize=(15, 15))
    else:
        new_fig = False

    if isinstance(labels, type(None)):
        labels = hcp.mmp["labels"]

    G = nx.from_numpy_array(
        np.where(rc < np.percentile(rc, tperc), 0, rc),
    )

    # shows the edges with their corresponding weights
    G.edges(data=True)

    for n in G.nodes:
        G.nodes[n]["color"] = mmp.get_hexcolor(n)

    for component in list(nx.connected_components(G)):
        if len(component) < min_connected_comp:
            for node in component:
                G.remove_node(node)

    if layout.lower() == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=123)
    elif layout.lower() == "kamada":
        pos = nx.kamada_kawai_layout(G, weight="weight")

    # Draw Nodes.
    hex_colors = nx.get_node_attributes(G, "color").values()
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes),  # [:378],
        node_size=[x[1] * size_factor for x in G.degree()],  # [:378],
        node_color=hex_colors,
        label=labels,
        ax=ax,
    )
    # Draw Labels.
    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="darkslategray", ax=ax)

    # Draw edges.
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())

    nx.draw_networkx_edges(
        G,
        pos,
        width=0.3,
        alpha=0.1,
        edge_color="k",
        # edge_cmap=plt.get_cmap("Greys_r"),
        ax=ax,
    )

    plt.close()

    if new_fig:
        return fig
