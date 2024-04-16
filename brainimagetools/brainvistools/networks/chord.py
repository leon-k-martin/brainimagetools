# chord.py
# Author: Leon Martin
import numpy as np
import matplotlib.pyplot as plt
from d3blocks import D3Blocks
from mpl_chord_diagram import chord_diagram


def mpl_chord(m, **kwargs):
    fig = chord_diagram(m, alpha=0.4, figsize=(10, 10), **kwargs)
    plt.close()
    return fig


def matrix2triplets(m):
    d3 = D3Blocks()
    vector = d3.adjmat2vec(m, min_weight=np.percentile(m, 70))

    vector = vector.rename(
        {"source": "from", "target": "to", "weight": "value"}, axis=1
    )

    for i, r in vector.iterrows():
        vector.at[i, "from"] = "subgroup_{}".format(int(r["from"] + 1))
        vector.at[i, "to"] = "subgroup_{}".format(int(r.to + 1))
