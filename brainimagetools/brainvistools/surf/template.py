#  template.py
#
# Created on Fri Aug 11 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#

from os.path import abspath, dirname, join
import nibabel as nib
import numpy as np
from templateflow.api import get

ROOT = abspath(dirname(__file__))


def get_hcp_surface_data(hemi="both"):
    """Fetches and returns the HCP surface data for the specified hemisphere."""

    if hemi in ["lh", "left"]:
        hemi_key = "_left"
    elif hemi in ["rh", "right"]:
        hemi_key = "_right"
    else:
        surf = hcp.mesh["inflated"]
        return surf

    surf = hcp.mesh["inflated" + hemi_key]
    return surf


def _get_surface_path(template="fsaverage", hemi="lh"):
    if hemi.lower() in ["l", "lh", "right"]:
        if template == "fsaverage":
            surf_paths = get(
                "fsaverage", suffix="pial", desc=None, hemi="L", density="164k"
            )
        elif template == "fsLR":
            surf_paths = get(
                "fsLR", density="32k", hemi="L", suffix="midthickness", desc=None
            )
        elif template == "mni":
            surf_paths = join(ROOT, "data", "BrainMesh_ICBM152.lh.gii")
    elif hemi.lower() in ["r", "rh", "right"]:
        if template == "fsaverage":
            surf_paths = get(
                "fsaverage", suffix="pial", hemi="R", desc=None, density="164k"
            )
        elif template == "fsLR":
            surf_paths = get(
                "fsLR", density="32k", hemi="R", suffix="midthickness", desc=None
            )
        elif template == "mni":
            surf_paths = join(ROOT, "data", "BrainMesh_ICBM152.rh.gii")

    if not isinstance(surf_paths, list):
        surf_paths = [surf_paths]
    return surf_paths


def get_surface_path(template="fsLR", hemi="lh", surface="inflated"):
    template_map = {
        "fsaverage": {
            "suffix": surface,
            "density": "164k",
            "hemi_map": {"l": "L", "lh": "L", "r": "R", "rh": "R", "right": "R"},
        },
        "fsLR": {
            "suffix": surface,
            "density": "32k",
            "hemi_map": {"l": "L", "lh": "L", "r": "R", "rh": "R", "right": "R"},
        },
        "mni": {
            "hemi_map": {
                "l": join(ROOT, "data", "BrainMesh_ICBM152.lh.gii"),
                "lh": join(ROOT, "data", "BrainMesh_ICBM152.lh.gii"),
                "r": join(ROOT, "data", "BrainMesh_ICBM152.rh.gii"),
                "rh": join(ROOT, "data", "BrainMesh_ICBM152.rh.gii"),
                "right": join(ROOT, "data", "BrainMesh_ICBM152.rh.gii"),
            }
        },
    }

    hemi = hemi.lower()

    if template != "mni":
        surf_paths = get(
            template,
            suffix=template_map[template]["suffix"],
            desc=None,
            hemi=template_map[template]["hemi_map"][hemi],
            density=template_map[template]["density"],
        )
    else:
        surf_paths = template_map[template]["hemi_map"][hemi]

    return [surf_paths] if not isinstance(surf_paths, list) else surf_paths


def get_surface_data(template="fsaverage", surface="pial", hemi="both"):
    """Fetches and returns the fsaverage surface data for the specified hemisphere."""

    if hemi == "both":
        surf_paths = get_surface_path(
            template, hemi="l", surface=surface
        ) + get_surface_path(template, hemi="r", surface=surface)
    elif hemi in ["l", "lh", "r", "rh", "right"]:
        surf_paths = get_surface_path(template, hemi=hemi, surface=surface)
    else:
        raise ValueError(
            "Invalid hemisphere choice. Choose 'L', 'lh', 'R', 'rh', or 'both'."
        )

    all_vertices = []
    all_triangles = []

    for surf_path in surf_paths:
        surf = nib.load(surf_path)
        if template == "mni":
            all_vertices.append(surf.darrays[1].data)
            all_triangles.append(surf.darrays[0].data)
        else:
            all_vertices.append(surf.darrays[0].data)
            all_triangles.append(surf.darrays[1].data)

    if len(all_vertices) > 1:
        # Update triangle indices for the right hemisphere
        all_triangles[1] += len(all_vertices[0])

    # Concatenate all vertices and triangles
    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_triangles)

    return vertices, triangles
