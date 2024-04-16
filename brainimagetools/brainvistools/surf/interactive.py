import nibabel as nb
import pandas as pd
import plotly.express as px
import plotly.io as pio
import templateflow
from nilearn import plotting as nlp

pio.renderers.default = "vscode"
pio.templates.default = "simple_white"


fsLR_lh = templateflow.api.get(
    template="fsLR", density="32k", suffix="inflated", hemi="L"
)

fsLR_rh = templateflow.api.get(
    template="fsLR", density="32k", suffix="inflated", hemi="R"
)


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for (
        name,
        data_indices,
        model,
    ) in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:  # Just looking for a surface
            data = data.T[
                data_indices
            ]  # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex  # Generally 1-N, except medial wall vertices
            surf_data = np.zeros(
                (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype
            )
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


import plotly.express as px
import plotly.io as pio

pio.renderers.default = "vscode"


def surf(data, template="fsLR", density="32k", surface="inflated", hemi="R", **kwargs):
    surface = templateflow.api.get(
        template=template, density=density, suffix=surface, hemi=hemi
    )

    fig = nlp.plot_surf(surface, data, engine="plotly", **kwargs)

    return fig
