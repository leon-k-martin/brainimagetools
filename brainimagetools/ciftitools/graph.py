import gc
import gzip
import os
import pickle
from os.path import dirname, exists, join, realpath

import hcp_utils as hcp
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import plotly.io as pio
from brainvistools.surf import plot
from plotly.subplots import make_subplots
from tqdm import tqdm


def create_connected_vertex_graph(
    vertices,
    triangles,
    vertex_data,
    threshold=0,
    min_size=None,
    connect_only_identical_vertices=False,
):
    """
    Create a graph based on vertices and triangles where each edge is formed between vertices with values above a specified threshold.

    Parameters
    ----------
    vertices : array_like
        Array of vertices in the graph.
    triangles : array_like
        Array of triangles where each triangle is defined by indices of its vertices.
    vertex_data : array_like
        Data associated with each vertex. Used to determine if an edge should be created based on the threshold.
    threshold : float, optional
        The threshold value for vertex data to be considered in creating an edge. Default is 0.
    min_size : int, optional
        Minimum size of connected components to be considered. Default is None.

    Returns
    -------
    networkx.Graph
        A graph object representing the connected components formed above the given threshold.
    """
    if vertex_data.shape[0] != vertices.shape[0]:
        raise ValueError("Data and vertices must have the same length")

    # Create a graph representation
    G = nx.Graph()
    for tri in triangles:
        for i in range(3):
            for j in range(i + 1, 3):
                # Check if both vertices of the edge have values above the threshold
                if connect_only_identical_vertices:
                    if (
                        abs(vertex_data[tri[i]]) > threshold
                        and abs(vertex_data[tri[j]]) > threshold
                        and vertex_data[tri[i]] == vertex_data[tri[j]]
                    ):
                        G.add_node(tri[i], data=vertex_data[tri[i]])
                        G.add_node(tri[j], data=vertex_data[tri[j]])
                        G.add_edge(tri[i], tri[j])
                else:
                    if (
                        vertex_data[tri[i]] > threshold
                        and vertex_data[tri[j]] > threshold
                    ):
                        G.add_node(tri[i], data=vertex_data[tri[i]])
                        G.add_node(tri[j], data=vertex_data[tri[j]])
                        G.add_edge(tri[i], tri[j])
    return G


def sort_connected_components_by_size(G, min_size=0):
    """
    Sort the connected components of a graph by size.

    Parameters
    ----------
    G : networkx.Graph
        The graph whose connected components are to be sorted.
    min_size : int, optional
        Minimum size of the connected components to be considered. Components smaller than this size will be ignored. Default is 0.

    Returns
    -------
    list
        A list of connected components, sorted by size.
    """
    # List connected components sorted by size
    return [
        c
        for c in sorted(nx.connected_components(G), key=len, reverse=False)
        if len(c) > min_size
    ]


def cluster_data_by_graph(data, G, min_size=100):
    """
    Cluster data based on the connected components of a graph and assign cluster labels.

    Parameters
    ----------
    data : array_like
        The data to be clustered.
    G : networkx.Graph
        The graph representing connections between data points.
    min_size : int, optional
        Minimum size of connected components to be considered for clustering. Default is 100.

    Returns
    -------
    ndarray
        An array where each element corresponds to the cluster label of the input data.
    """
    compmap = np.zeros(data.shape)
    for i, comp in enumerate(sort_connected_components_by_size(G, min_size=min_size)):
        compmap[list(comp)] = i
    return compmap


def cluster_vertex_data(
    vertices, triangles, vertex_data, threshold=0, min_size=100, return_counts=False
):
    """
    Cluster vertex data based on a graph created from vertices and triangles, where edges are formed between vertices exceeding a threshold.

    This function first creates a graph where each edge is formed between vertices with values above the specified threshold. It then clusters the vertex data based on the connected components of this graph.

    Parameters
    ----------
    vertices : array_like
        Array of vertices in the graph.
    triangles : array_like
        Array of triangles, where each triangle is defined by indices of its vertices.
    vertex_data : array_like
        Data associated with each vertex. Used for determining edge creation and clustering.
    threshold : float, optional
        The threshold value for vertex data to be considered in creating an edge. Default is 0.
    min_size : int, optional
        Minimum size of connected components to be considered in clustering. Default is 100.
    return_counts : bool, optional
        If True, the function also returns a dictionary with the counts of vertices in each cluster. Default is False.

    Returns
    -------
    compmap : ndarray
        An array where each element corresponds to the cluster label of the input vertex data.
    counts : dict, optional
        A dictionary with cluster labels as keys and the count of vertices in each cluster as values. Returned only if return_counts is True.
    """
    G = create_connected_vertex_graph(
        vertices, triangles, vertex_data, threshold=threshold
    )
    compmap = cluster_data_by_graph(vertex_data, G, min_size=min_size)

    if return_counts:
        unique, counts = np.unique(compmap[compmap != 0], return_counts=True)
        return (compmap, dict(zip(unique, counts)))
    return compmap


def fill_zero_nodes(G, zero_nodes):
    for zero_node in zero_nodes:
        neighbors = G.neighbors(zero_node)
        neighbors_values = [G.nodes(data=True)[n]["data"] for n in neighbors]
        if len(neighbors_values) > 0:
            new_value = max(neighbors_values, key=neighbors_values.count)
            G.nodes[zero_node]["data"] = new_value


def interpolate_data(vertices, triangles, data):
    new_parcellation = data
    G = create_connected_vertex_graph(
        vertices, triangles, new_parcellation, threshold=-1
    )
    # G.remove_nodes_from(np.where(hcp.yeo7.map_all == 0)[0])

    zero_nodes = [n for n, attr in G.nodes(data=True) if attr.get("data") == 0]
    while len(zero_nodes) > 0:
        before = len(zero_nodes)
        fill_zero_nodes(G, zero_nodes)
        zero_nodes = [n for n, attr in G.nodes(data=True) if attr.get("data") == 0]
        after = len(zero_nodes)
        print("Number of zero nodes:", len(zero_nodes), end="\r")
        if before == after:
            break

    parc_filled = np.zeros(len(new_parcellation))
    for n, attr in G.nodes(data=True):
        parc_filled[n] = attr["data"]

    parc_filled = np.where(hcp.cortex_data(hcp.yeo7.map_all) == 0, 0, parc_filled)
    return parc_filled
