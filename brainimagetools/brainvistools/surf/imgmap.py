import hcp_utils as hcp
import matplotlib.pyplot as plt
import numpy as np

# from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data


def norm(x, vmin=None, vmax=None):
    """Normalize array between 0-1 based on given min and max values.

    Parameters:
    - x (np.array): The array to normalize.
    - vmin (float, optional): Minimum value for normalization. Defaults to min of x.
    - vmax (float, optional): Maximum value for normalization. Defaults to max of x.

    Returns:
    - np.array: Normalized array.
    """
    if vmax is None:
        vmax = np.nanmax(x)

    if vmin is None:
        vmin = np.nanmin(x)

    return (x - vmin) / (vmax - vmin)


def get_hemisphere_data(hemi, dscalars, view, surface):
    """Get data specific to a hemisphere.

    Parameters:
    - hemi (str): Hemisphere ('lh', 'rh', 'left', 'right', or other for both).
    - dscalars (np.array): Data values to map onto the surface.
    - view (str): Viewpoint ('lateral', 'medial', etc.).
    - surface (str): Surface type, e.g., 'inflated'.

    Returns:
    - tuple: surf, dscalars_sulc, dscalars_ctx, azimuth.
    """
    if hemi in ["lh", "left"]:
        hemi_key = "_left"
        azimuth = 180 if view == "lateral" else 0
    elif hemi in ["rh", "right"]:
        hemi_key = "_right"
        azimuth = 180 if view == "medial" else 0
    else:
        surf = hcp.mesh[surface]
        dscalars_sulc = hcp.mesh["sulc"]
        dscalars_ctx = hcp.cortex_data(dscalars)
        azimuth = 0
        return surf, dscalars_sulc, dscalars_ctx, azimuth

    surf = hcp.mesh[surface + hemi_key]
    dscalars_sulc = hcp.mesh[f"sulc{hemi_key}"]
    dscalars_ctx = (
        hcp.left_cortex_data(dscalars)
        if "left" in hemi_key
        else hcp.right_cortex_data(dscalars)
    )

    return surf, dscalars_sulc, dscalars_ctx, azimuth


def compute_rgba_vals(dscalars_ctx, dscalars_sulc, cmap, alpha, alpha_sulc):
    """Compute RGBA values for the mesh based on data scalars and colormap.

    Parameters:
    - dscalars_ctx (np.array): Data values for the cortex.
    - dscalars_sulc (np.array): Data values for sulcal depth.
    - cmap (str or colormap, optional): Color map to use. Defaults to 'hot_r'.
    - alpha (float): Alpha transparency for the data.
    - alpha_sulc (float): Alpha transparency for the sulcal depth.

    Returns:
    - np.array: RGBA values.
    """
    cmap = plt.cm.get_cmap(cmap or "hot_r")

    hue = norm(dscalars_ctx)
    colors = cmap(hue)[:, :3]
    alpha_array = np.where(dscalars_ctx == 0, 0, np.full(colors.shape[0], alpha))

    cmap_sulc = plt.get_cmap("Greys")
    hue_sulc = norm(dscalars_sulc)
    colors_sulc = cmap_sulc(hue_sulc)[:, :3]
    alpha_sulc_array = np.full(colors.shape[0], alpha_sulc)

    outA = alpha_array + alpha_sulc_array * (1 - alpha_array)
    outRGB = (
        colors * alpha_array[:, None]
        + colors_sulc * alpha_sulc_array[:, None] * (1 - alpha_array[:, None])
    ) / outA[:, None]

    return np.concatenate((outRGB, outA[:, None]), axis=1)


def render_mesh(rgba_vals, surf, figsize, bgcolor, azimuth, view):
    """Render a mesh and return a screenshot.

    Parameters:
    - rgba_vals (np.array): RGBA values to map on the mesh.
    - surf (tuple): Surface data.
    - figsize (tuple): Figure size for rendering.
    - bgcolor (tuple): Background color for rendering.
    - azimuth (int): Azimuthal viewing angle.
    - view (str): Viewpoint ('lateral', 'medial', etc.).

    Returns:
    - np.array: A screenshot of the rendered surface image in RGBA format.
    """
    mlab.clf()
    mlab.close(all=True)

    fig = mlab.figure(size=figsize, bgcolor=bgcolor)
    x, y, z = surf[0].T
    tris = surf[1]
    mesh = mlab.pipeline.triangular_mesh_source(x, y, z, tris, figure=fig)
    mesh.data.point_data.scalars.number_of_components = 4
    mesh.data.point_data.scalars = (rgba_vals * 255).astype("ubyte")

    mapper = tvtk.PolyDataMapper()
    configure_input_data(mapper, mesh.data)
    actor = tvtk.Actor()
    actor.mapper = mapper
    fig.scene.add_actor(actor)

    elevation_map = {
        "lateral": 90,
        "medial": 90,
        "dorsal": 0,
        "superior": 0,
        "ventral": 180,
        "inferior": 180,
    }
    elevation = elevation_map.get(view.lower(), 90)

    mlab.view(
        azimuth=azimuth,
        elevation=elevation,
        distance=None,
        focalpoint="auto",
        roll=None,
        reset_roll=True,
    )

    imgmap = mlab.screenshot(figure=fig, mode="rgba", antialiased=True)

    mlab.clf(fig)
    mlab.close(fig)
    return imgmap


def mlab_imgmap(
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
    """
    Renders HCP template surface mesh and returns a screenshot of the rendered image.

    Parameters:
    - dscalars: Data values to map onto the surface.
    - hemi (optional): Which hemisphere to use ('lh', 'rh', or both). Defaults to 'lh'.
    - surface (optional): Surface type, e.g., 'inflated'. Defaults to 'inflated'.
    - cmap (optional): Color map to use. Defaults to 'hot_r'.
    - view (optional): Viewpoint ('lateral', 'medial', etc.). Defaults to 'lateral'.
    - figsize (optional): Figure size. Defaults to (1024, 1024).
    - alpha (optional): Alpha transparency for the data. Defaults to 0.9.
    - alpha_sulc (optional): Alpha transparency for the sulcal depth. Defaults to 1.
    - bgcolor (optional): Background color. Defaults to white.
    - vmax (optional): Maximum value for the colormap.
    - vmin (optional): Minimum value for the colormap.

    Returns:
    - imgmap: A screenshot of the rendered surface image in RGBA format.
    """
    surf, dscalars_sulc, dscalars_ctx, azimuth = get_hemisphere_data(
        hemi, dscalars, view, surface
    )
    rgba_vals = compute_rgba_vals(dscalars_ctx, dscalars_sulc, cmap, alpha, alpha_sulc)
    imgmap = render_mesh(rgba_vals, surf, figsize, bgcolor, azimuth, view)
    return imgmap
