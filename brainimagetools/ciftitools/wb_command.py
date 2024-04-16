import os
import tempfile
from os.path import basename, join

import nibabel as nib

from ciftitools import constants

hcp_mmp1_91k = template_cifti = join(constants.DATA_DIR, "hcp-mmp1_91k_fsLR.dlabel.nii")


def create_dscalar(path_lh, path_rh, path_vol=None, cifti_out=None):
    """Create a CIFTI dense scalar file.
    https://www.humanconnectome.org/software/workbench-command/-cifti-create-dense-scalar

    :param path_lh: _description_
    :type path_lh: _type_
    :param path_rh: _description_
    :type path_rh: _type_
    :param path_vol: _description_, defaults to None
    :type path_vol: _type_, optional
    :param cifti_out: _description_, defaults to None
    :type cifti_out: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    cmd = "wb_command -cifti-create-dense-scalar {} -left-metric {} -right-metric {}"

    if not isinstance(path_vol, type(None)):
        cmd = cmd + "-volume {}".format(path_vol)

    if isinstance(cifti_out, type(None)):
        with tempfile.TemporaryDirectory() as tempdir:
            cifti_out = join(
                tempdir,
                basename(path_lh).replace("lh", "LR").replace("L", "LR")
                + ".dscalar.nii",
            )
            os.system(cmd.format(cifti_out, path_lh, path_rh))
            return nib.load(cifti_out)

    else:
        os.system(cmd.format(cifti_out, path_lh, path_rh))
        return cifti_out


def dscalar_from_template(path_dscalar, cifti_out=None, path_template=None):
    """_summary_

    :param path_dscalar: _description_
    :type path_dscalar: _type_
    :param cifti_out: _description_, defaults to None
    :type cifti_out: _type_, optional
    :param path_template: _description_, defaults to None
    :type path_template: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if isinstance(path_template, type(None)):
        path_template = hcp_mmp1_91k

    cmd = "wb_command -cifti-create-dense-from-template {} {} -cifti {}"

    if isinstance(path_dscalar, nib.cifti2.cifti2.Cifti2Image):
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(path_dscalar, join(tempdir, "input.dscalar.nii"))
        path_dscalar = join(tempdir, "input.dscalar.nii")

    if isinstance(cifti_out, type(None)):
        with tempfile.TemporaryDirectory() as tempdir:
            cifti_out = join(tempdir, "91k_" + basename(path_dscalar))
            os.system(cmd.format(path_template, path_dscalar, cifti_out))
            return nib.load(cifti_out)

    else:
        os.system(cmd.format(path_template, cifti_out, path_dscalar))
        return cifti_out


def cifti_parcellate(input, path_dlabel, cifti_out=None, direction="column"):
    cmd = "wb_command -cifti-parcellate {} {} {} {}"  # Input, dlabel file, direction, output

    # if isinstance(path_dscalar, nib.cifti2.cifti2.Cifti2Image):
    #     with tempfile.TemporaryDirectory() as tempdir:
    #         nib.save(path_dscalar, join(tempdir, "input.dscalar.nii"))
    #     path_dscalar = join(tempdir, "input.dscalar.nii")

    if isinstance(cifti_out, type(None)):
        with tempfile.TemporaryDirectory() as tempdir:
            cifti_out = join(tempdir, basename(input.replace(".d", ".p")))
            os.system(cmd.format(input, path_dlabel, direction.upper(), cifti_out))
            return nib.load(cifti_out)

    else:
        os.system(cmd.format(input, path_dlabel, direction.upper(), cifti_out))
        return cifti_out


def smoothing(
    cifti_in,
    cifti_out=None,
    sigma_ctx=2.55,
    sigma_vol=2.55,
    left_surf="default",
    right_surf="default",
    direction="COLUMN",
):
    """
    Applies smoothing to a CIFTI file.

    Parameters:
    cifti_in (str): Path to the input CIFTI file.
    cifti_out (str, optional): Path to the output CIFTI file. If not provided, a new file with "_smoothed" appended to the original filename is created.
    sigma_ctx (float, optional): Sigma value for cortical smoothing. Default is 2.55.
    sigma_vol (float, optional): Sigma value for volume smoothing. Default is 2.55.
    left_surf (str, optional): Path to the left surface file. If "default", uses the default left surface file. Default is "default".
    right_surf (str, optional): Path to the right surface file. If "default", uses the default right surface file. Default is "default".
    direction (str, optional): Direction of smoothing. Default is "COLUMN".

    Returns:
    str: Path to the output CIFTI file.
    """

    if left_surf == "default":
        left_surf = constants.LEFT_SURFACE_INFL
    if right_surf == "default":
        right_surf = constants.RIGHT_SURFACE_INFL

    if isinstance(cifti_out, type(None)):
        cifti_out = ".".join(
            [cifti_in.split(".")[0] + "_smoothed"] + cifti_in.split(".")[1:]
        )
        # print(cifti_out)

    cmd = f"""wb_command -cifti-smoothing "{cifti_in}" {sigma_ctx} {sigma_vol} {direction.upper()} "{cifti_out}" -left-surface "{left_surf}" -right-surface  "{right_surf}" -merged-volume"""

    os.system(cmd)
    # print(cmd)
