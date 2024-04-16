import hcp_utils as hcp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from brainvistools import surface

# matplotlib.use("Agg")

metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
ax = fig.add_subplot()


def animate_surf_data(data, fout, cmap="hot_r", **kwargs):
    if len(data) > 32492:
        data = hcp.left_cortex_data(data)
    with writer.saving(fig, fout, 600):
        for i in range(len(data)):
            surface.fsLR(
                data[i],
                cmap=cmap,
                **kwargs,
            )
            writer.grab_frame()
