import os

import matplotlib.pyplot as plt
import numpy as np
from brainvistools import surface
from matplotlib import gridspec as mgs
from mayavi import mlab
from surfer import Brain

from brainvistools import constants

mlab.options.offscreen = True


def normalize_ts(ts):
    ts_norm = (ts - ts.mean()) / ts.std()
    return ts_norm


annot_path_lh = os.path.join(constants.DATA_DIR, "lh.HCP-MMP1.annot")
annot_path_rh = os.path.join(constants.DATA_DIR, "rh.HCP-MMP1.annot")


def anmiate_fsaverage_ts(df, annot_path=None, hemi="lh", tstart=0, tend=20, tdil=0.5):

    times = list()
    vtx_ts = np.ndarray((163842, len(df)))
    ind = 0

    for t, r in df.iterrows():
        times.append(float(t * 16))

        vtx_data = surface.parc2fsaverage(r, annot_path=annot_path, hemi=hemi)

        vtx_ts[:, ind] = vtx_data
        ind += 1

    absmax = np.nanmax(
        [abs(np.nanmin(vtx_ts[:, :tend])), abs(np.nanmax(vtx_ts[:, :tend]))]
    )

    # %%

    brain = Brain("fsaverage", "lh", "pial", background="white")

    # %%
    brain.add_data(
        vtx_ts[:, tstart:tend],
        colormap="seismic",
        hemi="lh",
        min=-absmax,
        max=absmax,
        smoothing_steps=10,
        time=times[:tend],
        time_label=lambda t: "%s ms" % int(round(t * 1e3)),
    )

    if not fout.endswith(".mov"):
        fout = fout + ".mov"

    # brain.hide_colorbar(row=1)
    brain.save_movie(fout, time_dilation=tdil)
    brain.close()

    mlab.close(all=True)


def cifti_plot_carpet(
    data,
    atlas=None,
    atlas_labels=None,
    t_r=2.5,
    detrend=True,
    output_file=None,
    figure=None,
    axes=None,
    vmin=None,
    vmax=None,
    title=None,
    cmap="gray",
    cmap_labels=plt.cm.gist_ncar,
):

    # Define TR and number of frames
    n_tsteps = data.shape[1]

    if not isinstance(atlas, type(None)):
        background_label = 0

        atlas_values = np.squeeze(atlas)

        if atlas_labels:
            label_dtype = type(list(atlas_labels.values())[0])
            if label_dtype != atlas_values.dtype:
                print("Coercing atlas_values to {}".format(label_dtype))
                atlas_values = atlas_values.astype(label_dtype)

        # Sort data and atlas by atlas values
        order = np.argsort(atlas_values)
        order = np.squeeze(order)
        atlas_values = atlas_values[order]
        data = data[:, order]

    # Detrend and standardize data
    # if detrend:
    #     data = signal.clean(data, t_r=t_r, detrend=True, standardize='zscore')

    if figure is None:
        if not axes:
            figsize = (10, 5)
            figure = plt.figure(figsize=figsize)
        else:
            figure = axes.figure

    if axes is None:
        axes = figure.add_subplot(1, 1, 1)
    else:
        assert axes.figure is figure, "The axes passed are not in the figure"

    # Determine vmin and vmax based on the full data
    std = np.mean(data.std(axis=0))
    default_vmin = data.mean() - (2 * std)
    default_vmax = data.mean() + (2 * std)

    # Avoid segmentation faults for long acquisitions by decimating the data
    LONG_CUTOFF = 800
    # Get smallest power of 2 greater than the number of volumes divided by the
    # cutoff, to determine how much to decimate (downsample) the data.
    # n_decimations = int(np.ceil(np.log2(np.ceil(n_tsteps / LONG_CUTOFF))))
    # data = data[:: 2**n_decimations, :]

    if not isinstance(atlas, type(None)):
        # Define nested GridSpec
        legend = False
        wratios = [2, 100, 20]
        gs = mgs.GridSpecFromSubplotSpec(
            1,
            2 + int(legend),
            subplot_spec=axes,
            width_ratios=wratios[: 2 + int(legend)],
            wspace=0.0,
        )

        ax0 = plt.subplot(gs[0])
        ax0.set_xticks([])
        ax0.imshow(
            atlas_values[:, np.newaxis],
            interpolation="none",
            aspect="auto",
            cmap=cmap_labels,
        )
        if atlas_labels:
            # Add labels to middle of each associated band
            mask_labels_inv = {v: k for k, v in atlas_labels.items()}
            ytick_locs = [
                np.mean(np.where(atlas_values == i)[0]) for i in np.unique(atlas_values)
            ]
            ax0.set_yticks(ytick_locs)
            ax0.set_yticklabels([mask_labels_inv[i] for i in np.unique(atlas_values)])
        else:
            ax0.set_yticks([])

        # Carpet plot
        axes = plt.subplot(gs[1])  # overwrite axes
        axes.imshow(
            data.T,
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
            vmin=vmin or default_vmin,
            vmax=vmax or default_vmax,
        )
        ax0.tick_params(axis="both", which="both", length=0)
    else:
        axes.imshow(
            data.T,
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
            vmin=vmin or default_vmin,
            vmax=vmax or default_vmax,
        )

    axes.grid(False)
    axes.set_yticks([])
    axes.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = max((int(data.shape[0] + 1) // 10, int(data.shape[0] + 1) // 5, 1))
    xticks = list(range(0, data.shape[0])[::interval])
    axes.set_xticks(xticks)
    axes.set_xlabel("time (s)")

    if title:
        axes.set_title(title)

    labels = t_r * (np.array(xticks))
    # labels *= 2**n_decimations
    axes.set_xticklabels(["%.02f" % t for t in labels.tolist()])

    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        axes.spines[side].set_color("none")
        axes.spines[side].set_visible(False)

    axes.xaxis.set_ticks_position("bottom")
    axes.spines["bottom"].set_position(("outward", 10))

    if not atlas_labels:
        axes.yaxis.set_ticks_position("left")
        buffer = 20 if not isinstance(atlas, type(None)) else 10
        axes.spines["left"].set_position(("outward", buffer))
        axes.set_ylabel("voxels")

    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
        figure = None
    plt.close()

    return figure
