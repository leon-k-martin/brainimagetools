from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv
import matplotlib.pyplot as plt
import matplotlib
from os.path import join
import numpy as np
from matplotlib.colors import ListedColormap
from brainvistools.constants import DATA_DIR
from matplotlib.cm import ScalarMappable


def plot_greyordinates(data=None, dscalar_file=None, cmap=None, ax=None, plot_sctx="all", alpha=1, volume_rendering="spheres_peeled", colorbar=True, view='L', d2c_kwargs={"vlims": (1e-10, 1e10)}):
    if isinstance(cmap, type(None)):
        cmap = matplotlib.colormaps["viridis"]
    elif isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]


    my_brain_viewer = cbv.Cerebro_brain_viewer(
        offscreen=True,
        background_color=(1, 1, 1, 0),
        null_color=(0.9, 0.9, 0.9, 0.5),
        no_color=(0.6, 0.6, 0.6, 0.1),
    )
    surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models("pial")
    cifti_space = my_brain_viewer.visualize_cifti_space(
        volumetric_structures=plot_sctx,
        volume_rendering=volume_rendering,
        cifti_left_right_seperation=0
    )
    my_brain_viewer.add_cifti_dscalar_layer(
        dscalar_file=dscalar_file,
        dscalar_data=data,
        colormap=cmap,
        opacity=alpha,
        **d2c_kwargs
    )

    my_brain_viewer.change_view(view)

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(10,10))
        return_fig = True
    else:
        return_fig = False

    ax.axis("off")
    my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)

    # Clear this viewer
    my_brain_viewer.viewer.window.destroy()


    if colorbar:
        norm = plt.Normalize(min(data), max(data))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=fig.axes[0], shrink=.5)
        cbar.set_label('a.u.', rotation=270, labelpad=0)
        cbar.outline.set_visible(False)
        cbar.set_ticks([min(data), max(data)])
        cbar.set_ticklabels([f'{min(data):.2f}', f'{max(data):.2f}'])
        # cbar.ax.tick_params(size=0)


    if return_fig:
        plt.close()
        return fig


def plot_mmp():
    mmp_file = join(DATA_DIR, "parcellation", "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors_with_Atlas_ROIs2.32k_fs_LR.dlabel.nii")

    roi_info = np.loadtxt(join(DATA_DIR, "parcellation", "hcpmmp1_ordered.txt"),dtype=str)
    mmp_cmap = ListedColormap(roi_info[:, 2:].astype(float) / 255)

    fig = plot_greyordinates(dscalar_file=mmp_file, cmap=mmp_cmap, colorbar=False)
    return fig