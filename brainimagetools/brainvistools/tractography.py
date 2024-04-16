import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dipy.io.streamline import load_tck


def plot_streamlines(tck_file, reference_file):
    streamlines = load_tck(tck_file, reference_file)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # streamline = streamlines.streamlines[0]
    for streamline in tqdm(streamlines.streamlines):
        if len(streamline) < 10:
            continue
        directions = np.diff(streamline, axis=0)
        average_angle = np.arctan2(directions[:, 1], directions[:, 0]).mean()
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        normalized_average_angle = (average_angle - angles.min()) / (
            angles.max() - angles.min()
        )
        color = plt.cm.hsv(normalized_average_angle)

        ax.plot(
            streamline[:, 0],
            streamline[:, 1],
            streamline[:, 2],
            color=color,
            alpha=0.9,
            linewidth=0.1,
        )
