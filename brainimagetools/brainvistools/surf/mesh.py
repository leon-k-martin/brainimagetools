import matplotlib.pyplot as plt
import numpy as np
from vtk import (
    VTK_FLOAT,
    vtkCellArray,
    vtkClipVolume,
    vtkColorTransferFunction,
    vtkGPUVolumeRayCastMapper,
    vtkImageData,
    vtkImageReslice,
    vtkLookupTable,
    vtkPiecewiseFunction,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkSmartVolumeMapper,
    vtkTransform,
    vtkVolumeProperty,
)
from vtk.util import numpy_support
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkImageActor,
    vtkImageMapper3D,
    vtkImageSliceMapper,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkWindowToImageFilter,
)


def convert_to_vtk_format(vertices, triangles):
    """Converts the data to VTK format."""
    points = vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices))

    polys = vtkCellArray()
    for tri in triangles:
        polys.InsertNextCell(len(tri), tri)

    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(polys)
    return mesh


def _apply_colormap(mesh, values):
    """Applies a colormap to the mesh."""
    vtk_scalars = numpy_support.numpy_to_vtk(values)
    mesh.GetPointData().SetScalars(vtk_scalars)

    lut = vtkLookupTable()
    lut.SetNumberOfColors(256)
    lut.SetHueRange(0.667, 0.0)  # Blue to red
    lut.Build()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarRange(np.min(values), np.max(values))
    mapper.SetLookupTable(lut)
    return mapper


def apply_colormap(mesh, values, colormap="viridis", zero_transparent=True):
    """Applies a colormap to the mesh."""
    vtk_scalars = numpy_to_vtk(values)
    mesh.GetPointData().SetScalars(vtk_scalars)

    # Normalize the values to range [0, 1]
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Get the matplotlib colormap
    cmap = plt.cm.get_cmap(colormap)

    # Create a VTK lookup table and set its properties
    lut = vtkLookupTable()
    lut.SetNumberOfColors(256)

    # Populate the VTK lookup table with RGB values from the matplotlib colormap
    for i in range(256):
        r, g, b, _ = cmap(i / 255.0)  # Extract RGBA from the colormap, but ignore alpha
        alpha = 1.0
        if i == 0 and zero_transparent:
            # alpha = 0.0
            lut.SetTableValue(i, 100, 100, 100, alpha)
        else:
            lut.SetTableValue(
                i, r, g, b, alpha
            )  # Set the RGB values into the lookup table

    lut.Build()

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarRange(np.min(values), np.max(values))
    mapper.SetLookupTable(lut)
    return mapper
