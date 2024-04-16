import nibabel as nib
from os.path import join
from templateflow import api as tflow

from brainvistools import constants


def get_fsaverage(density="164k", hemi="L", suffix="pial"):
    """
    Get FreeSurfer fsaverage surface template.

    Parameters
    ----------
    density : str, optional
        Density of surface mesh. Default is "164k".
    hemi : str, optional
        Hemisphere of surface mesh. Default is "L".
    suffix : str, optional
        Suffix of surface mesh. Default is "pial".

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Loaded NIfTI-1 format image from the surface mesh file.
    """

    fpath = tflow.get("fsaverage", density="164k", hemi="L", suffix="pial")
    return nib.load(fpath)


def get_fsLR(hemi="L", suffix="pial"):
    """
    Get FreeSurfer fsLR surface template.

    Parameters
    ----------
    hemi : str, optional
        Hemisphere of surface mesh. Default is "L".
    suffix : str, optional
        Suffix of surface mesh. Default is "pial".

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Loaded NIfTI-1 format image from the surface mesh file.
    """
    if suffix == "pial":
        fsLR = join(
            constants.DATA_DIR,
            "Q1-Q6_RelatedValidation210.{}.pial_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii".format(
                hemi
            ),
        )

    fsLR = tflow.get("fsLR", density="32k", hemi="L", suffix="midthickness", desc=None)
    return nib.load(fsLR)


def geometry_from_gifti(gii):
    """
    Get vertex and triangle coordinates from a GIFTI surface mesh file.

    Parameters
    ----------
    gii : str or nibabel.gifti.GiftiImage
        Path to a GIFTI surface mesh file or a loaded GiftiImage object.

    Returns
    -------
    tuple
        A tuple of numpy arrays representing vertex coordinates and triangle indices.
    """

    if isinstance(gii, str):
        gii = nib.load(gii)

    vtx, tri = gii.agg_data()

    return (vtx, tri)
