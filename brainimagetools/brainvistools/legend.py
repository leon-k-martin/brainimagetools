from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def legend_circles(labels, palette, loc=1, markersize=10, marker="o", padding=0):
    """
    Make a legend where the color is indicated by a circle.

    Parameters
    ----------
    labels : list of str
        The labels to use for the legend.
    palette : list of str
        The colors to use for each label.
    loc : int or str, optional
        The location of the legend. Default is 1.
    markersize : int, optional
        The size of the marker used in the legend. Default is 10.
    marker : str, optional
        The shape of the marker used in the legend. Default is "o".
    padding : float, optional
        The padding to add around the legend. Default is 0.

    Returns
    -------
    matplotlib.legend.Legend
        The legend object.

    Examples
    --------
    >>> labels = ["Label 1", "Label 2"]
    >>> palette = ["red", "blue"]
    >>> legend_circles(labels, palette, loc=2)

    """

    legend_markers = [
        Line2D(
            range(1),
            range(1),
            linewidth=0,  # Invisible line
            marker=marker,
            markersize=markersize,
            markerfacecolor=palette[i],
            markeredgecolor=palette[i],
        )
        for i in range(len(labels))
    ]

    legend_markers.append(
        Line2D(
            range(1),
            range(1),
            linewidth=0,  # Invisible line
            marker="s",
            markersize=markersize,
            markerfacecolor="white",
            markeredgecolor="grey",
        )
    )
    labels.append("Glasser areas")

    return plt.legend(legend_markers, labels, numpoints=1, bbox_to_anchor=(0.9, 0.3))
