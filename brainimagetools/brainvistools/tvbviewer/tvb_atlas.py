#!/usr/bin/env python
# tvb_atlas.py
#
# Created on Thu Aug 10 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
# %%
import functools
import os
import glob
from os.path import abspath, basename, dirname, join
from functools import partial

os.environ["ETS_TOOLKIT"] = "qt5"

import hcp_utils as hcp
import numpy as np
import tvbase
from templateflow.api import get
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import vuetify2 as vuetify
from trame.widgets.vtk import VtkLocalView
from trame.widgets.vuetify import VCol, VRow

from brainvistools.surf.mesh import apply_colormap, convert_to_vtk_format
from brainvistools.surf.plot import setup_renderer
from brainvistools.surf.template import get_surface_data

ROOT = abspath(dirname(__file__))


def normalize_values(values):
    """
    Normalize values to the range [0, 1].
    """
    min_val = values.min()
    max_val = values.max()
    return (values - min_val) / (max_val - min_val)


def randomize_mesh_values(text_box, ctrl, mesh, vertices):
    """
    Generate random values and apply them to the mesh.
    """
    server = get_server()
    state = server.state
    print(state.tvbaseQuery)

    random_values = hcp.left_cortex_data(hcp.unparcellate(np.random.rand(379), hcp.mmp))
    normalized_values = normalize_values(random_values)
    apply_colormap(mesh, random_values)
    ctrl.view_update()  # Update the VTK view


def map_tvbase(text_box, ctrl, mesh, vertices):
    server = get_server()
    state = server.state
    query = state.tvbaseQuery
    print(query)
    data = tvbase.map_scaiview(query, request_doclist=False).parc_data
    values = hcp.left_cortex_data(hcp.unparcellate(data, hcp.mmp))
    values = normalize_values(values)
    apply_colormap(mesh, values)
    ctrl.view_update()  #
    print(query, "mapped")


# Examples.
ROOT_PATH = join(ROOT, "maps", "mesh")


def on_example_data_clicked(f, ctrl, mesh):
    data = tvbase.io.load_map(f).parc_data
    values = hcp.left_cortex_data(hcp.unparcellate(data, hcp.mmp))
    values = normalize_values(values)
    apply_colormap(mesh, values)
    ctrl.view_update()  #


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    tree = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1

    for path, dirs, files in os.walk(rootdir):
        # Filter out dotfiles and consider only files ending with "tvbase.json"
        files = [
            f for f in files if not f.startswith(".") and f.endswith("tvbase.json")
        ]

        folders = path[start:].split(os.sep)
        if files:  # If there are relevant files, put them in the directory structure
            subdir_dict = dict.fromkeys(dirs, {})
            subdir_dict.update({"_files": files})
        else:
            subdir_dict = dict.fromkeys(dirs, {})

        parent = functools.reduce(dict.get, folders[:-1], tree)

        # Check if parent is a dictionary
        if isinstance(parent, dict):
            parent[folders[-1]] = subdir_dict
        else:
            # If this is ever hit, it means our tree structure is not being constructed correctly.
            print(f"Unexpected structure at {path}.")
            continue

    return tree


def get_files_from_directory(rootdir):
    """
    Get all files with '.tvbase.json' extension from rootdir
    """
    return glob.glob(os.path.join(rootdir, "*_tvbase.json"))


def create_file_items_from_list(file_list, ctrl, mesh):
    components = []
    with vuetify.VExpansionPanel():
        vuetify.VExpansionPanelHeader(
            ("examples",),
        )
        vuetify.VExpansionPanelContent(
            [
                vuetify.VListItem(
                    os.path.basename(file_path)
                    .replace("_tvbase.json", "")
                    .replace("MESH-", ""),
                    click=partial(on_example_data_clicked, file_path, ctrl, mesh),
                )
                for file_path in file_list
            ]
        )


def main():
    """Main function to display the rendered data."""

    def update_surface(values, ctrl):
        server = get_server()
        state = server.state
        surface = state.selectSurf
        print(surface)

        vertices, triangles = get_surface_data(
            template="fsLR", surface=surface, hemi="lh"
        )
        mesh = convert_to_vtk_format(vertices, triangles)
        mapper = apply_colormap(mesh, values)
        ctrl.view_update()

    # Default values
    vertices, triangles = get_surface_data(
        template="fsLR", surface="midthickness", hemi="lh"
    )
    mesh = convert_to_vtk_format(vertices, triangles)
    values = normalize_values(vertices[:, 2])
    mapper = apply_colormap(mesh, values)
    render_window = setup_renderer(mapper)

    server = get_server()
    server.client_type = "vue2"
    server.state.selectedDatasetIndex = -1  # Default: No dataset selected

    ctrl = server.controller
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("TVBase viewer")

        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = VtkLocalView(render_window)
                ctrl.on_server_ready.add(view.update)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

        with layout.toolbar:
            # Add search textbox
            with VCol(cols=4):  # Adjust cols for the desired width
                search_box = vuetify.VTextField(
                    label="Search",
                    prepend_icon="mdi-magnify",
                    v_model=("tvbaseQuery", None),
                )
                # Bind the randomize function to the input event of the search box
                search_box.change = lambda: map_tvbase(search_box, ctrl, mesh, vertices)

            # Dropdown for surface selection
            with VCol(cols=4):
                surface_select = vuetify.VSelect(
                    items=("array_list", ["midthickness", "inflated", "very_inflated"]),
                    label="Select Surface",
                    v_model=("selectSurf", "midthickness"),  # Default value
                )
                surface_select.change = lambda: update_surface(values, ctrl)

            vuetify.VSpacer()
            vuetify.VSwitch(
                v_model="$vuetify.theme.dark",
                hide_details=True,
                dense=True,
            )
            vuetify.VDivider(vertical=True, classes="mx-2")
            with vuetify.VBtn(
                icon=True,
                click=ctrl.view_reset_camera,
            ):
                vuetify.VIcon("mdi-crop-free")

        file_list = get_files_from_directory(ROOT_PATH)
        with layout.drawer:
            create_file_items_from_list(file_list, ctrl, mesh)

    server.start()


if __name__ == "__main__":
    main()
