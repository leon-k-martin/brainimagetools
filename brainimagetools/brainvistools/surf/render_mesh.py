import os
from os.path import join

import hcp_utils as hcp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mayavi import mlab
from nibabel.freesurfer.io import read_annot
from surfer import Brain
from tvtk.api import tvtk
from tvtk.common import configure_input_data

from brainvistools import utils

from brainvistools import constants


def fsLR(
    dscalars,
    hemi="lh",
    surface="inflated",
    cmap=None,
    view="lateral",
    figsize=(1024, 1024),
    alpha=0.9,
    alpha_sulc=1,
    bgcolor=(1, 1, 1),
):

    mlab.clf()

    azimuth = 0

    if hemi in ["lh", "left"]:
        surf = hcp.mesh[surface + "_left"]
        dscalars_sulc = hcp.mesh["sulc_left"]
        dscalars_ctx = hcp.left_cortex_data(dscalars)

        if view == "lateral":
            azimuth += 180

    elif hemi in ["rh", "right"]:
        surf = hcp.mesh[surface + "_right"]
        dscalars_sulc = hcp.mesh["sulc_right"]
        dscalars_ctx = hcp.right_cortex_data(dscalars)

        if view == "medial":
            azimuth += 180

    else:
        surf = hcp.mesh[surface]
        dscalars_sulc = hcp.mesh["sulc"]
        dscalars_ctx = hcp.cortex_data(dscalars)

    fig = mlab.figure(size=figsize, bgcolor=bgcolor)

    # Get coordinates and vertices.
    x = surf[0][:, 0]
    y = surf[0][:, 1]
    z = surf[0][:, 2]
    tris = surf[1]

    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, tris, figure=fig)

    return fig, mesh


def overlay_sulc():
    fig, mesh = fsLR
    # generate an rgba matrix, of shape n_vertices x 4
    if not cmap:
        cmap = cm.get_cmap("hot_r")
    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    hue = norm(dscalars_ctx)
    colors = cmap(hue)[:, :3]
    alpha = np.full(z.shape, alpha)
    alpha = np.where(dscalars_ctx == 0, 0, alpha)
    rgba_vals = np.concatenate((colors, alpha[:, None]), axis=1)

    cmap_sulc = plt.get_cmap("Greys")
    hue_sulc = utils.norm(dscalars_sulc)
    colors_sulc = cmap_sulc(hue_sulc)[:, :3]
    alpha_sulc = np.full(z.shape, alpha_sulc)

    # Mix with sulcal depth image by means of alpha compositing
    srcRGB = colors
    dstRGB = colors_sulc

    srcA = alpha
    dstA = alpha_sulc

    # Work out resultant alpha channel
    outA = srcA + dstA * (1 - srcA)

    # Work out resultant RGB
    outRGB = (
        srcRGB * srcA[..., np.newaxis]
        + dstRGB * dstA[..., np.newaxis] * (1 - srcA[..., np.newaxis])
    ) / outA[..., np.newaxis]
    rgba_vals = np.concatenate((outRGB, outA[:, None]), axis=1)

    mesh.data.point_data.scalars.number_of_components = 4  # r, g, b, a
    mesh.data.point_data.scalars = (rgba_vals * 255).astype("ubyte")

    # tvtk for vis
    mapper = tvtk.PolyDataMapper()
    configure_input_data(mapper, mesh.data)
    actor = tvtk.Actor()
    actor.mapper = mapper
    fig.scene.add_actor(actor)

    cam, foc = mlab.move()
    # Define View
    if view.lower() in ["lateral"]:
        elevation = 90

    elif view.lower() in ["medial"]:
        elevation = 90

    elif view.lower() in ["dorsal", "superior"]:
        elevation = 0

    elif view.lower() in ["ventral", "iferior"]:
        elevation = 180

    mlab.view(
        azimuth=azimuth,
        elevation=elevation,
        distance=None,
        focalpoint="auto",
        roll=None,
        reset_roll=True,
    )

    return fig
