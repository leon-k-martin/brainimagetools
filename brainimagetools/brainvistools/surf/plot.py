import gc

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from brainvistools.surf import mesh, template
from vtk.util import numpy_support
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)

pv.OFF_SCREEN = True


def surf_pyvista(
    surf=None,
    values=None,
    title="Surface Plot",
    view="lateral",
    off_screen=False,
    space="fsLR",
    hemi="rh",
    surface="inflated",
):
    """
    Plots a surface using PyVista.

    Parameters:
    - surf: Surface data. If not provided, default surface data is fetched.
    - values: Values for the surface data points. If not provided, z-values of vertices are used.
    - title: Title of the plot.
    - view: View direction for the plot ("medial" or "lateral").

    Returns:
    - PyVista Plotter object after plotting.
    """

    if isinstance(surf, type(None)):
        vertices, triangles = template.get_surface_data(
            template=space, surface=surface, hemi=hemi
        )
        surf = mesh.convert_to_vtk_format(vertices, triangles)

    if isinstance(values, type(None)):
        # if not vertices:
        #     raise ValueError("Values not provided and cannot infer from mesh")
        values = vertices[:, 2]

    # Convert the vtkPolyData to pyvista.PolyData
    pv_mesh = pv.wrap(surf)

    # Add scalar data to point data
    pv_mesh.point_data["values"] = values

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.camera.zoom(1.5)
    plotter.set_background([255, 255, 255], top=[255, 255, 255])  # top for color fade
    s = plotter.add_mesh(pv_mesh, scalars="values", show_edges=False, cmap="coolwarm")

    view_config = {"medial": "zy", "lateral": "yz"}

    cpos = view_config.get(view, "yz")

    # Plot the scalar data on the surface
    plotter.render()
    plotter.show(
        cpos=cpos,
        title=title,
    )
    return plotter


view_config = {"medial": "zy", "lateral": "yz"}


def rotate(arr, positions):
    """
    Rotate a numpy array by a specified number of positions.

    Parameters:
    ----------
    arr : np.ndarray
        The array to rotate.
    positions : int
        Number of positions to rotate the array by.

    Returns:
    -------
    np.ndarray
        The rotated array.
    """
    return np.roll(arr, positions)


################
# TS Anmiation #
################
import concurrent.futures
from threading import Lock

lock = Lock()


def generate_frame(i, time_series, pv_mesh, plotter):
    """Function to generate each frame."""
    pv_mesh.point_data["values"] = time_series[i, :]
    plotter.add_text(f"{i} ms", name="time-label", color="black")
    with lock:
        plotter.write_frame()  # Write this frame
    return i  # Return the frame number (or any other result you want to collect)


def ts_surf_pyvista(
    surf=None,
    values=None,
    title="Surface Plot",
    view="lateral",
    filename="ts_movie.mp4",
    fps=60,
    quality=5,
    parallel=False,
    tp=100,
):
    """
    Plots a surface using PyVista.

    Parameters:
    - surf: Surface data. If not provided, default surface data is fetched.
    - values: Values for the surface data points. If not provided, z-values of vertices are used.
    - title: Title of the plot.
    - view: View direction for the plot ("medial" or "lateral").
    - parallel: Bool, if True uses parallel processing to generate frames.

    Returns:
    - PyVista Plotter object after plotting.
    """

    if isinstance(surf, type(None)):
        vertices, triangles = template.get_surface_data(
            template="fsaverage", surface="pial", hemi="both"
        )
        surf = mesh.convert_to_vtk_format(vertices, triangles)

    if isinstance(values, type(None)):
        # if not vertices:
        #     raise ValueError("Values not provided and cannot infer from mesh")
        values = vertices[:, 2]
        time_series = np.array([rotate(values, i) for i in range(tp)])

    # Convert the vtkPolyData to pyvista.PolyData
    pv_mesh = pv.wrap(surf)

    # Add scalar data to point data
    pv_mesh.point_data["values"] = values

    cpos = view_config.get(view, "yz")

    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.set_background([255, 255, 255], top=[255, 255, 255])  # top for color fade
    plotter.open_movie(filename, framerate=fps, quality=quality)

    s = plotter.add_mesh(pv_mesh, scalars="values", show_edges=False, cmap="coolwarm")

    # Plot the scalar data on the surface
    plotter.show(
        auto_close=False,
        cpos=cpos,
        title=title,
    )

    # Run through each frame
    plotter.write_frame()  # write initial data
    # Parallel frame generation
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    generate_frame,
                    range(time_series.shape[0]),
                    [time_series] * time_series.shape[0],
                    [pv_mesh] * time_series.shape[0],
                    [plotter] * time_series.shape[0],
                )
            )
    else:
        for i in range(time_series.shape[0]):
            generate_frame(i, time_series, pv_mesh, plotter)

    plotter.close()


def surf_pyvista2mpl():
    plotter = surf_pyvista(off_screen=True)
    plotter.camera.zoom(1.5)
    plotter.image_scale = 20000
    plotter.render()
    imgmap = plotter.screenshot(transparent_background=True)
    plotter.close()
    return imgmap


#######
# VTK #
#######


def apply_colormap(
    vtk_mesh, values, cmap="viridis", thresh=None, vmin=None, vmax=None, grey=0.8
):
    """
    Apply a colormap to a VTK mesh.

    Parameters:
    ----------
    vtk_mesh : vtk.vtkPolyData
        The VTK mesh to which the colormap will be applied.
    values : array-like
        The scalar values associated with the vertices of the mesh.
    cmap : str or Colormap, optional
        The Matplotlib colormap to use. Default is "viridis".
    vmin : float, optional
        The minimum data value that corresponds to the lower limit of the colormap.
    vmax : float, optional
        The maximum data value that corresponds to the upper limit of the colormap.

    Returns:
    -------
    vtk.vtkPolyDataMapper
        The VTK mapper with the colormap applied.
    """
    import numpy as np
    import vtk
    import matplotlib.pyplot as plt

    # Get the Matplotlib colormap
    if isinstance(cmap, str):
        mpl_cmap = plt.get_cmap(cmap)
    else:
        mpl_cmap = cmap

    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    if thresh is None:
        thresh = np.nanmin(values)

    # Map the scalar values to colors with alpha
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)  # Adding an alpha channel
    colors.SetName("Colors")
    for val in values:
        if np.abs(val) <= thresh:
            # Set color as transparent if value is below vmin
            colors.InsertNextTuple4(grey * 255, grey * 255, grey * 255, 255)
        else:
            # Normalize the value to the range [0, 1]
            norm_val = (val - vmin) / (vmax - vmin)
            # Get the RGBA values from the Matplotlib colormap
            rgba = mpl_cmap(norm_val)
            colors.InsertNextTuple4(
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
                int(rgba[3] * 255),
            )

    # Add the colors to the vtkPolyData object as a point data array
    vtk_mesh.GetPointData().SetScalars(colors)

    # Create a mapper and set the vtkPolyData object
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_mesh)

    return mapper


def setup_renderer(mapper):
    """
    Set up the VTK renderer for visualization.

    Parameters:
    ----------
    mapper : vtkMapper
        The mapper for rendering the data.

    Returns:
    -------
    vtkRenderWindow
        Configured render window.
    """
    renderer = vtkRenderer()

    renderer.UseDepthPeelingOn()
    renderer.SetOcclusionRatio(0.1)
    renderer.SetMaximumNumberOfPeels(100)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

    actor = vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    renderer.ResetCamera()
    return render_window


def surf_vtk(
    vertices=None,
    triangles=None,
    values=None,
    window_width=1024,
    window_height=1024,
    azimuth=0,
    elevation=-90,
    roll=0,
    cmap="viridis",
    **kwargs,
):
    """
    Render surface using VTK.

    Parameters:
    ----------
    window_width : int, optional
        Width of the render window. Default is 1024.
    window_height : int, optional
        Height of the render window. Default is 1024.

    Returns:
    -------
    vtkRenderWindow
        Configured render window for the surface.

    Note:
    ----
    The caller of this function is responsible for cleaning up the
    render_window object by calling its Finalize method and then deleting
    it and forcing garbage collection to release resources,
    like this:

        render_window.Finalize()
        del render_window
        gc.collect()
    """

    if isinstance(vertices, type(None)) or isinstance(triangles, type(None)):
        vertices, triangles = template.get_surface_data(
            template="fsaverage", surface="pial", hemi="both"
        )
    vtk_mesh = mesh.convert_to_vtk_format(vertices, triangles)

    if isinstance(values, type(None)):
        values = vertices[:, 2]
        print(len(values))

    # mapper = mesh.apply_colormap(vtk_mesh, values)
    mapper = apply_colormap(vtk_mesh, values, cmap=cmap, **kwargs)

    render_window = setup_renderer(mapper)
    render_window.SetOffScreenRendering(1)

    # Adjusting the camera view similar to _render_surf
    renderer = render_window.GetRenderers().GetFirstRenderer()
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(azimuth)
    camera.Elevation(elevation)
    camera.Roll(roll)
    camera.Dolly(1)

    render_window.SetSize(int(window_width), int(window_height))

    # Close the renderer and render window to free up resources
    render_window.Finalize()
    return render_window


def surf_vtk2mpl(
    vertices=None,
    triangles=None,
    values=None,
    ax=None,
    view="lateral",  # New parameter
    window_size_factor=10,
    window_width=1024,
    window_height=1024,
    return_array=False,
    hemi="right",
    cmap="viridis",
    **kwargs,
):
    """
    Convert VTK surface render to a Matplotlib figure.

    Parameters:
    ----------
    ax : matplotlib Axes, optional
        Axes on which to plot the image. If not provided, a new figure and axes are created.
    window_width : int, optional
        Width of the VTK render window. Default is 1024.
    window_height : int, optional
        Height of the VTK render window. Default is 1024.

    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the rendered image.
    """
    # Mapping of view orientations to camera positions
    camera_positions = {
        "lateral": (90, 0, -90),
        "medial": (-90, 0, 90),
        "superior": (0, 0, 0),
        "inferior": (180, 0, 0),
        "anterior": (0, 90, 180),
        "posterior": (0, -90, 0),
    }

    if hemi.lower() in ["left", "l", "lh"]:
        camera_positions["lateral"] = (-90, 0, 90)
        camera_positions["medial"] = (90, 0, -90)

    # Set the camera position based on the view orientation
    if view in camera_positions:
        azimuth, elevation, roll = camera_positions[view]
    else:
        raise ValueError(
            f"Invalid view orientation: {view}. Valid options are: {camera_positions.keys()}"
        )

    render_window = surf_vtk(
        vertices=vertices,
        triangles=triangles,
        values=values,
        window_height=window_height * window_size_factor,
        window_width=window_width * window_size_factor,
        azimuth=azimuth,
        elevation=elevation,
        roll=roll,
        cmap=cmap,
        **kwargs,
    )
    render_window.Render()

    # Capture the screen
    window_to_image_filter = vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetInputBufferTypeToRGBA()
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    # Close the renderer and render window to free up resources
    # Delete the VTK objects
    render_window.Finalize()
    del render_window

    # Convert VTK image to numpy array
    vtk_image = window_to_image_filter.GetOutput()
    rows, cols, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array).reshape(rows, cols, components)

    del vtk_image
    del vtk_array
    del window_to_image_filter

    # Force garbage collection
    import gc

    gc.collect()

    if return_array:
        return numpy_array

    # Display numpy array using Matplotlib
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        return_fig = False

    # Padding
    if view == "medial":
        limplus = 200
    else:
        limplus = 0

    ax.set_xlim(
        [100 * window_size_factor + limplus, 900 * window_size_factor + limplus]
    )  # these values are just for demonstration
    ax.set_ylim(
        [100 * window_size_factor + limplus, 900 * window_size_factor + limplus]
    )
    ax.imshow(numpy_array)
    ax.axis("off")

    if return_fig:
        plt.close()
        return fig


## %%
import plotly.graph_objects as go
import numpy as np


def determine_eye_position(view):
    if view == "posterior":
        return dict(x=0, y=-1.5, z=0)
    elif view == "lateral_right" or view == "lateral":
        return dict(x=1.5, y=0, z=0)
    elif view == "lateral_left":
        return dict(x=-1.5, y=0, z=0)
    elif view == "medial_left":
        return dict(x=-1.5, y=0, z=0)
    elif view == "medial_right" or view == "medial":
        return dict(x=1.5, y=0, z=0)
    elif view == "superior":
        return dict(x=0, y=0, z=1.5)
    elif view == "inferior":
        return dict(x=0, y=0, z=-1.5)


import numpy as np
import plotly.graph_objects as go


def create_mesh3d(
    x,
    y,
    z,
    faces,
    data,
    face_colors=None,
    colorscale="viridis",
    colorbar_title="a.u.",
):
    if not isinstance(face_colors, type(None)):
        face_colors = np.mean(data[faces], axis=1)
        face_colors = np.array([plt.cm.viridis(c)[:3] for c in face_colors])
        intensity = None
        colorscale = None
    else:
        intensity = data

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=intensity,
        facecolor=None,
        colorscale=colorscale,
        cmin=np.min(data),
        cmax=np.max(data),
        opacity=1,
        colorbar_title=colorbar_title,
    )


def update_layout_no_axes(fig, x, y, z):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=0,
                showgrid=False,
                showbackground=False,
                showticklabels=False,
                title="",
                range=[min(x), max(x)],
            ),
            yaxis=dict(
                nticks=0,
                showgrid=False,
                showbackground=False,
                showticklabels=False,
                title="",
                range=[min(y), max(y)],
            ),
            zaxis=dict(
                nticks=0,
                showgrid=False,
                showbackground=False,
                showticklabels=False,
                title="",
                range=[min(z), max(z)],
            ),
        )
    )


def adjust_aspect_ratio(fig, x, y, z):
    aspect_ratio = dict(
        x=(max(x) - min(x)) / (max(y) - min(y)),
        y=1,
        z=(max(z) - min(z)) / (max(y) - min(y)),
    )
    fig.update_layout(scene_aspectmode="manual", scene_aspectratio=aspect_ratio)


def plotly_data(
    vertices,
    faces,
    data,
    view="lateral_left",
    show_colorbar=False,
    row=None,
    col=None,
    cmap="viridis",
    face_colors=None,
):
    x, y, z = vertices.T
    colorscale = cmap
    colorbar_title = "Data Value"

    mesh = create_mesh3d(
        x,
        y,
        z,
        faces,
        data,
        colorscale=colorscale,
        colorbar_title=colorbar_title,
        face_colors=face_colors,
    )
    fig = go.Figure(mesh)

    if not show_colorbar:
        mesh.colorbar = None

    if row and col:
        fig.add_trace(mesh, row=row, col=col)

    update_layout_no_axes(fig, x, y, z)
    adjust_aspect_ratio(fig, x, y, z)

    fig.update_layout(width=400, height=300, margin=dict(r=0, l=0, b=0, t=0))
    eye = determine_eye_position(view)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=eye)
    fig.update_layout(scene_camera=camera)

    colorbar_length = 0.4
    fig.data[0].update(colorbar=dict(len=colorbar_length, outlinewidth=1))

    return fig


# def plotly_data(
#     vertices,
#     faces,
#     data,
#     view="lateral_left",
#     animation=False,
#     show_colorbar=False,
#     row=None,
#     col=None,
#     cmap="viridis",
#     face_colors=None,
# ):
#     x, y, z = vertices.T
#     colorscale = cmap
#     colorbar_title = "Data Value"

#     if animation and len(data.shape) == 2:
#         frames = [
#             go.Frame(
#                 data=[
#                     create_mesh3d(
#                         x,
#                         y,
#                         z,
#                         faces,
#                         data[:, t],
#                         colorscale=colorscale,
#                         colorbar_title=colorbar_title,
#                         face_colors=face_colors,
#                     )
#                 ],
#                 name=str(t),
#             )
#             for t in range(data.shape[1])
#         ]

#         mesh = create_mesh3d(x, y, z, faces, data[:, 0], colorscale, colorbar_title)

#         fig = go.Figure(
#             data=[mesh],
#             layout=go.Layout(
#                 updatemenu=[
#                     dict(
#                         type="buttons",
#                         showactive=False,
#                         buttons=[
#                             dict(
#                                 label="Play",
#                                 method="animate",
#                                 args=[
#                                     None,
#                                     dict(
#                                         frame=dict(duration=100, redraw=True),
#                                         fromcurrent=True,
#                                     ),
#                                 ],
#                             )
#                         ],
#                     )
#                 ],
#                 frames=frames,
#             ),
#         )
#     else:
#         mesh = create_mesh3d(x, y, z, faces, data, colorscale, colorbar_title)
#         fig = go.Figure(mesh)

#     if not show_colorbar:
#         mesh.colorbar = None

#     if row and col:
#         fig.add_trace(mesh, row=row, col=col)

#     update_layout_no_axes(fig, x, y, z)
#     adjust_aspect_ratio(fig, x, y, z)

#     fig.update_layout(width=400, height=300, margin=dict(r=0, l=0, b=0, t=0))

#     # NOTE: You'll need to provide the 'determine_eye_position' function for the following to work
#     eye = determine_eye_position(view)

#     camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=eye)
#     fig.update_layout(scene_camera=camera)

#     colorbar_length = 0.4
#     fig.data[0].update(colorbar=dict(len=colorbar_length, outlinewidth=1))

#     return fig


# def plotly_data(
#     vertices,
#     faces,
#     data,
#     view="lateral_left",
#     animation=False,
#     show_colorbar=False,
#     row=None,
#     col=False,
#     cmap="viridis",
# ):
#     x, y, z = vertices.T
#     colorscale = cmap
#     colorbar_title = "Data Value"

#     if animation and len(data.shape) == 2:
#         frames = [
#             go.Frame(
#                 data=[
#                     go.Mesh3d(
#                         x=x,
#                         y=y,
#                         z=z,
#                         i=faces[:, 0],
#                         j=faces[:, 1],
#                         k=faces[:, 2],
#                         intensity=data[:, t],
#                         colorscale=colorscale,
#                         cmin=np.min(data),
#                         cmax=np.max(data),
#                         opacity=1,
#                         colorbar_title=colorbar_title,
#                     )
#                 ],
#                 name=str(t),
#             )
#             for t in range(data.shape[1])
#         ]

#         fig = go.Figure(
#             data=[
#                 go.Mesh3d(
#                     x=x,
#                     y=y,
#                     z=z,
#                     i=faces[:, 0],
#                     j=faces[:, 1],
#                     k=faces[:, 2],
#                     intensity=data[:, 0],  # start with the first time step
#                     colorscale=colorscale,
#                     cmin=np.min(data),
#                     cmax=np.max(data),
#                     opacity=1,
#                     colorbar_title=colorbar_title,
#                 )
#             ],
#             layout=go.Layout(
#                 updatemenu=[
#                     dict(
#                         type="buttons",
#                         showactive=False,
#                         buttons=[
#                             dict(
#                                 label="Play",
#                                 method="animate",
#                                 args=[
#                                     None,
#                                     dict(
#                                         frame=dict(duration=100, redraw=True),
#                                         fromcurrent=True,
#                                     ),
#                                 ],
#                             )
#                         ],
#                     )
#                 ],
#                 frames=frames,
#             ),
#         )

#     else:
#         mesh = go.Mesh3d(
#             x=x,
#             y=y,
#             z=z,
#             i=faces[:, 0],
#             j=faces[:, 1],
#             k=faces[:, 2],
#             intensity=data,
#             colorscale=colorscale,
#             cmin=np.min(data),
#             cmax=np.max(data),
#             opacity=1,
#             colorbar_title=colorbar_title,
#         )

#         fig = go.Figure(mesh)

#     if not show_colorbar:
#         mesh.colorbar = None

#     if fig and row and col:
#         fig.add_trace(mesh, row=row, col=col)
#     else:
#         fig = go.Figure(mesh)

#     # Remove axes ticks and grid
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(
#                 nticks=0,
#                 showgrid=False,
#                 showbackground=False,
#                 showticklabels=False,
#                 title="",
#                 range=[min(x), max(x)],
#             ),
#             yaxis=dict(
#                 nticks=0,
#                 showgrid=False,
#                 showbackground=False,
#                 showticklabels=False,
#                 title="",
#                 range=[min(y), max(y)],
#             ),
#             zaxis=dict(
#                 nticks=0,
#                 showgrid=False,
#                 showbackground=False,
#                 showticklabels=False,
#                 title="",
#                 range=[min(z), max(z)],
#             ),
#         )
#     )

#     # Compute the aspect ratio based on actual data range
#     aspect_ratio = dict(
#         x=(max(x) - min(x)) / (max(y) - min(y)),
#         y=1,
#         z=(max(z) - min(z)) / (max(y) - min(y)),
#     )

#     # Apply the computed aspect ratio to the scene
#     fig.update_layout(scene_aspectmode="manual", scene_aspectratio=aspect_ratio)

#     # Adjust the figure dimensions to be square
#     fig.update_layout(width=400, height=300, margin=dict(r=0, l=0, b=0, t=0))

#     eye = determine_eye_position(view)

#     # Adjust camera view (center the brain)
#     camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=eye)
#     fig.update_layout(scene_camera=camera)
#     colorbar_length = (
#         0.4  # This sets the colorbar to 50% of the figure height. Adjust as needed.
#     )

#     fig.data[0].update(colorbar=dict(len=colorbar_length, outlinewidth=1))

#     return fig
