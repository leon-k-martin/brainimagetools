import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import hcp_utils
from matplotlib.colors import LinearSegmentedColormap
from brainvistools import colors


def cmap_from_list(color_list, name="listed_cmap"):
    """
    Create a ListedColormap from a list of colors.

    Parameters
    ----------
    color_list : list
        List of colors in RGBA format.
    name : str, optional
        Name of the colormap, defaults to "listed_cmap".

    Returns
    -------
    ListedColormap
        A ListedColormap object.

    Examples
    --------
    >>> from brainvistools.colormap import cmap_from_list
    >>> color_list = [(0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
    >>> cmap = cmap_from_list(color_list, name="my_cmap")
    """
    return LinearSegmentedColormap.from_list(
        name=name, colors=color_list, N=len(color_list)
    )


cmap_hcp = cmap_from_list(list(hcp_utils.mmp["rgba"].values()), name="hcp_cmap")


def get_continuous_cmap(hex_list, float_list=None):
    """
    Create a LinearSegmentedColormap object that can be used in heat map figures.

    If float_list is not provided, the colormap graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list : list
        List of hex code strings representing colors.
    float_list : list, optional
        List of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        If not provided, it is generated linearly from 0 to 1., defaults to None.

    Returns
    -------
    LinearSegmentedColormap
        A LinearSegmentedColormap object.

    Examples
    --------
    >>> hex_list = ['#0000FF', '#00FF00', '#FF0000']
    >>> cmap = get_continuous_cmap(hex_list)
    >>> cmap
    """

    rgb_list = [colors.rgb_to_dec(colors.hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def double_cmap(cmap, reverse=True):
    """
    Create a ListedColormap by doubling the given colormap.

    The colors are extended with a reversed version (mirroring).

    Parameters
    ----------
    cmap : matplotlib.colors.ListedColormap or str
        Matplotlib colormap or string indicating one.
    reverse : bool, optional
        If True, the colors in the colormap are mirrored and extended. Defaults to True.

    Returns
    -------
    ListedColormap
        A ListedColormap object.

    Examples
    --------
    >>> cmap = plt.cm.get_cmap('jet')
    >>> cmap_doubled = double_cmap(cmap)
    """

    if isinstance(cmap, str):
        clrs = plt.get_cmap(cmap)(np.linspace(0.0, 1, 128))
        clrs2 = plt.get_cmap(cmap)(np.linspace(0.0, 1, 128))
    else:
        clrs = cmap(np.linspace(0.0, 1, 128))
        clrs2 = cmap(np.linspace(0.0, 1, 128))

    if reverse:
        clrs = np.vstack(((np.flip(clrs2, axis=0), clrs)))
    else:
        clrs = np.vstack((clrs, np.flip(clrs2, axis=0)))

    mymap = LinearSegmentedColormap.from_list("my_colormap", clrs)
    return mymap
