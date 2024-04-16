from tvb.datatypes import surfaces
from brainvistools.surf import mesh, template
from brainvistools import surf
from templateflow.api import get
import nibabel as nib
import numpy as np

from neuromaps import transforms, datasets, nulls

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def timeseries(data, template='fsLR', surface='inflated', hemi='L'):
    gii_img = nib.load(get(template, density='32k', hemi=hemi, suffix=surface, desc=None))



    vertices, triangles = gii_img.darrays[0].data, gii_img.darrays[1].data

    arr = surf.plot.surf_vtk2mpl(
        vertices=vertices, triangles=triangles, values=data[0], window_size_factor=12, view='lateral', hemi='left', return_array=True
    )

    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    ylim = ax.get_ylim()
    ax.set_ylim((2000, 10000))
    ax.set_xlim((1500, 11000))


    # Update function for animation
    def update(frame):
        # Generate new data for each frame (replace with your own data updating logic)
        data=data[frame]

        arr = surf.plot.surf_vtk2mpl(
            vertices=vertices, triangles=triangles, values=data,
            window_size_factor=12, view='lateral', hemi='left', return_array=True)
        im.set_array(arr)
        return [im]

    # Create animation
    ani = FuncAnimation(fig, update, frames=10, interval=10, blit=True)

    return ani
