import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from math import cos, sin, pi
from matplotlib.pyplot import subplot


def get_rotation_matrix(rotation_axis, deg):

    """Return rotation matrix in the x,y,or z plane"""

    # (note make deg minus to change from anticlockwise to clockwise rotation)
    th = -deg * (pi / 180)  # convert degrees to radians

    if rotation_axis == 0:
        return np.array([[1, 0, 0], [0, cos(th), -sin(th)], [0, sin(th), cos(th)]])
    elif rotation_axis == 1:
        return np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])
    elif rotation_axis == 2:
        return np.array([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])


def get_combined_rotation_matrix(rotations):
    """Return a combined rotation matrix from a dictionary of rotations around
    the x,y,or z axes"""
    rotmat = np.eye(3)

    if type(rotations) is tuple:
        rotations = [rotations]
    for r in rotations:
        newrot = get_rotation_matrix(r[0], r[1])
        rotmat = np.dot(rotmat, newrot)
    return rotmat


def plot_surface_mpl(
    vtx,
    tri,
    data=None,
    rm=None,
    reorient="tvb",
    view="superior",
    shaded=False,
    ax=None,
    figsize=(6, 4),
    title=None,
    lthr=None,
    uthr=None,
    nz_thr=1e-20,
    shade_kwargs={
        "edgecolors": "k",
        "linewidth": 0.1,
        "alpha": None,
        "cmap": "coolwarm",
        "vmin": None,
        "vmax": None,
    },
):

    r"""Plot surfaces, surface patterns, and region patterns with matplotlib

    This is a general-use function for neuroimaging surface-based data, and
    does not necessarily require construction of or interaction with tvb
    datatypes.

    See also:  plot_surface_mpl_mv



    Parameters
    ----------

    vtx           : N vertices x 3 array of surface vertex xyz coordinates

    tri           : N faces x 3 array of surface faces

    data          : array of numbers to colour surface with. Can be either
                    a pattern across surface vertices (N vertices x 1 array),
                    or a pattern across the surface's region mapping
                    (N regions x 1 array), in which case the region mapping
                    bust also be given as an argument.

    rm            : region mapping - N vertices x 1 array with (up to) N
                    regions unique values; each element specifies which
                    region the corresponding surface vertex is mapped to

    reorient      : modify the vertex coordinate frame and/or orientation
                    so that the same default rotations can subsequently be
                    used for image views. The standard coordinate frame is
                    xyz; i.e. first,second,third axis = left-right,
                    front-back, and up-down, respectively. The standard
                    starting orientation is axial view; i.e. looking down on
                    the brain in the x-y plane.

                    Options:

                      tvb (default)   : swaps the first 2 axes and applies a rotation

                      fs              : for the standard freesurfer (RAS) orientation;
                                        e.g. fsaverage lh.orig.
                                        No transformations needed for this; so is
                                        gives same result as reorient=None

    view          : specify viewing angle.

                    This can be done in one of two ways: by specifying a string
                    corresponding to a standard viewing angle, or by providing
                    a tuple or list of tuples detailing exact rotations to apply
                    around each axis.

                    Standard view options are:

                    lh_lat / lh_med / rh_lat / rh_med /
                    superior / inferior / posterior / anterior

                    (Note: if the surface contains both hemispheres, then medial
                     surfaces will not be visible, so e.g. 'rh_med' will look the
                     same as 'lh_lat')

                    Arbitrary rotations can be specied by a tuple or a list of
                    tuples, each with two elements, the first defining the axis
                    to rotate around [0,1,2], the second specifying the angle in
                    degrees. When a list is given the rotations are applied
                    sequentially in the order given.

                    Example: rotations = [(0,45),(1,-45)] applies 45 degrees
                    rotation around the first axis, followed by 45 degrees rotate
                    around the second axis.

    lthr/uthr     : lower/upper thresholds - set to zero any datapoints below /
                    above these values

    nz_thr        : near-zero threshold - set to zero all datapoints with absolute
                    values smaller than this number. Default is a very small
                    number (1E-20), which unless your data has very small numbers,
                    will only mask out actual zeros.

    shade_kwargs  : dictionary specifiying shading options

                    Most relevant options (see matplotlib 'tripcolor' for full details):

                      - 'shading'        (either 'gourand' or omit;
                                          default is 'flat')
                      - 'edgecolors'     'k' = black is probably best
                      - 'linewidth'      0.1 works well; note that the visual
                                         effect of this will depend on both the
                                         surface density and the figure size
                      - 'cmap'           colormap
                      - 'vmin'/'vmax'    scale colormap to these values
                      - 'alpha'          surface opacity

    ax            : figure axis

    figsize       : figure size (ignore if ax provided)

    title         : text string to place above figure




    Usage
    -----


    Basic freesurfer example:

    import nibabel as nib
    vtx,tri = nib.freesurfer.read_geometry('subjects/fsaverage/surf/lh.orig')
    plot_surface_mpl(vtx,tri,view='lh_lat',reorient='fs')



    Basic tvb example:

    ctx = cortex.Cortex.from_file(source_file = ctx_file,
                                  region_mapping_file =rm_file)
    vtx,tri,rm = ctx.vertices,ctx.triangles,ctx.region_mapping
    conn = connectivity.Connectivity.from_file(conn_file); conn.configure()
    isrh_reg = conn.is_right_hemisphere(range(conn.number_of_regions))
    isrh_vtx = np.array([isrh_reg[r] for r in rm])
    dat = conn.tract_lengths[:,5]

    plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat,view='inferior',title='inferior')

    fig, ax = plt.subplots()
    plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat, view=[(0,-90),(1,55)],ax=ax,
                     title='lh angle',shade_kwargs={'shading': 'gouraud', 'cmap': 'rainbow'})


    """

    # Copy things to make sure we don't modify things
    # in the namespace inadvertently.
    # plt.clf()
    # plt.cla()
    vtx, tri = vtx.copy(), tri.copy()
    if data is not None:
        data = data.copy()

    # 1. Set the viewing angle

    if reorient == "tvb":
        # The tvb default brain has coordinates in the order
        # yxz for some reason. So first change that:
        vtx = np.array([vtx[:, 1], vtx[:, 0], vtx[:, 2]]).T.copy()

        # Also need to reflect in the x axis
        vtx[:, 0] *= -1

    # (reorient == 'fs' is same as reorient=None; so not strictly needed
    #  but is included for clarity)

    # ...get rotations for standard view options

    if view == "lh_lat":
        rots = [(0, -90), (1, 90)]
    elif view == "lh_med":
        rots = [(0, -90), (1, -90)]
    elif view == "rh_lat":
        rots = [(0, -90), (1, -90)]
    elif view == "rh_med":
        rots = [(0, -90), (1, 90)]
    elif view == "superior":
        rots = None
    elif view == "inferior":
        rots = (1, 180)
    elif view == "anterior":
        rots = (0, -90)
    elif view == "posterior":
        rots = [(0, -90), (1, 180)]
    elif (type(view) == tuple) or (type(view) == list):
        rots = view

    # (rh_lat is the default 'view' argument because no rotations are
    #  for that one; so if no view is specified when the function is called,
    #  the 'rh_lat' option is chose here and the surface is shown 'as is'

    # ...apply rotations

    if rots is None:
        rotmat = np.eye(3)
    else:
        rotmat = get_combined_rotation_matrix(rots)
    vtx = np.dot(vtx, rotmat)

    # 2. Sort out the data

    # ...if no data is given, plot a vector of 1s.
    #    if using region data, create corresponding surface vector
    if data is None:
        data = np.ones(vtx.shape[0])
    elif data.shape[0] != vtx.shape[0]:
        data = np.array([data[r] for r in rm])

    # ...apply thresholds
    if uthr:
        data *= data < uthr
    if lthr:
        data *= data > lthr
    data *= np.abs(data) > nz_thr

    # 3. Create the surface triangulation object

    x, y, z = vtx.T
    tx, ty, tz = vtx[tri].mean(axis=1).T
    tr = Triangulation(x, y, tri[np.argsort(tz)])

    # 4. Make the figure

    # if ax is None:
    #     fig, ax = plt.subplots(figsize=figsize)

    # if shade = 'gouraud': shade_opts['shade'] =
    tc = ax.tripcolor(tr, np.squeeze(data),
                      **shade_kwargs
                      )

    # ax.set_aspect("equal")
    # ax.axis("off")

    # if title is not None:
    #     ax.set_title(title)

    # plt.close()
    # return fig


def plot_surface_mpl_mv(
    vtx=None,
    tri=None,
    data=None,
    rm=None,
    hemi=None,  # Option 1
    vtx_lh=None,
    tri_lh=None,
    data_lh=None,
    rm_lh=None,  # Option 2
    vtx_rh=None,
    tri_rh=None,
    data_rh=None,
    rm_rh=None,
    title=None,
    **kwargs
):

    r"""Convenience wrapper on plot_surface_mpl for multiple views

    This function calls plot_surface_mpl five times to give a complete
    picture of a surface- or region-based spatial pattern.

    As with plot_surface_mpl, this function is written so as to be
    generally usable with neuroimaging surface-based data, and does not
    require construction of of interaction with tvb datatype objects.

    In order for the medial surfaces to be displayed properly, it is
    necessary to separate the left and right hemispheres. This can be
    done in one of two ways:

    1. Provide single arrays for vertices, faces, data, and
       region mappings, and addition provide arrays of indices for
       each of these (vtx_inds,tr_inds,rm_inds) with 0/False
       indicating left hemisphere vertices/faces/regions, and 1/True
       indicating right hemisphere.

       Note: this requires that

    2. Provide separate vertices,faces,data,and region mappings for
       each hemisphere (vtx_lh,tri_lh; vtx_rh,tri_rh,etc...)



    Parameters
    ----------

    (see also plot_surface_mpl parameters info for more details)

    (Option 1)

    vtx               :  surface vertices

    tri               : surface faces

    data              : spatial pattern to plot

    rm                : surface vertex to region mapping

    hemi              : hemisphere labels for each vertex
                        (1/True = right, 0/False = left) -


    OR

    (Option 2)

    vtx_lh            : left hemisphere surface_vertices
    vtx_rh            : right ``      ``    ``     ``

    tri_lh            : left hemisphere surface faces
    tri_rh            : right ``      ``    ``     ``

    data_lh          : left hemisphere surface_vertices
    data_rh          : right ``      ``    ``     ``

    rm_lh            : left hemisphere region_mapping
    rm_rh            : right ``      ``    ``     ``


    title            : title to show above middle plot

    kwargs           : additional tripcolor kwargs; see plot_surface_mpl



    Examples
    ----------

    # TVB default data

    # Plot one column of the region-based tract lengths
    # connectivity matrix. The corresponding region is
    # right auditory cortex ('rA1')

    ctx = cortex.Cortex.from_file(source_file = ctx_file,
                                  region_mapping_file =rm_file)
    vtx,tri,rm = ctx.vertices,ctx.triangles,ctx.region_mapping
    conn = connectivity.Connectivity.from_file(conn_file); conn.configure()
    isrh_reg = conn.is_right_hemisphere(range(conn.number_of_regions))
    isrh_vtx = np.array([isrh_reg[r] for r in rm])
    dat = conn.tract_lengths[:,5]

    plot_surface_mpl_mv(vtx=vtx,tri=tri,rm=rm,data=dat,
                        hemi=isrh_vtx,title=u'rA1 \ntract length')

    plot_surface_mpl_mv(vtx=vtx,tri=tri,rm=rm,data=dat,
                      hemi=isrh_vtx,title=u'rA1 \ntract length',
                      shade_kwargs = {'shading': 'gouraud',
                                      'cmap': 'rainbow'})


    """

    if vtx is not None:  # Option 1
        tri_hemi = hemi[tri].any(axis=1)
        tri_lh, tri_rh = tri[tri_hemi == 0], tri[tri_hemi == 1]
    elif vtx_lh is not None:  # Option 2
        vtx = np.vstack([vtx_lh, vtx_rh])
        tri = np.vstack([tri_lh, tri_rh + tri_lh.max() + 1])

    if data_lh is not None:  # Option 2
        data = np.hstack([data_lh, data_rh])

    if rm_lh is not None:  # Option 2
        rm = np.hstack([rm_lh, rm_rh + rm_lh.max() + 1])

    # 2. Now do the plots for each view

    # (Note: for the single hemispheres we only need lh/rh arrays for the
    #  faces (tri); the full vertices, region mapping, and data arrays
    #  can be given as arguments, they just won't be shown if they aren't
    #  connected by the faces in tri )

    # LH lateral
    plot_surface_mpl(
        vtx, tri_lh, data=data, rm=rm, view="lh_lat", ax=subplot(2, 3, 1), **kwargs
    )

    # LH medial
    plot_surface_mpl(
        vtx, tri_lh, data=data, rm=rm, view="lh_med", ax=subplot(2, 3, 4), **kwargs
    )

    # RH lateral
    plot_surface_mpl(
        vtx, tri_rh, data=data, rm=rm, view="rh_lat", ax=subplot(2, 3, 3), **kwargs
    )

    # RH medial
    plot_surface_mpl(
        vtx, tri_rh, data=data, rm=rm, view="rh_med", ax=subplot(2, 3, 6), **kwargs
    )

    # Both superior
    plot_surface_mpl(
        vtx,
        tri,
        data=data,
        rm=rm,
        view="superior",
        ax=subplot(1, 3, 2),
        title=title,
        **kwargs
    )

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)
