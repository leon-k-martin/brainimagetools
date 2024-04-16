# !pip install pandas nibabel nilearn matplotlib
import pandas as pd
import nibabel as nib
from nilearn import plotting
import numpy as np

dk = nib.load("aparc+aseg-mni_09c.nii.gz")

tvbase_atlas = nib.load("tvbase-atlas_mni_icbm152_nlin_asym_09c.nii.gz")

dk_info = pd.read_csv(
    "FreeSurferColorLUT.txt",
    sep="\s+",
    comment="#",
    header=None,
    names=["ID", "Label", "R", "G", "B", "A"],
)


def parc_data2volume(data, parcellation):
    volume = np.zeros(parcellation.shape)
    parcellation_data = parcellation.get_fdata()
    for l, d in data.items():
        roi_idx = dk_info[dk_info["Label"].str.lower() == l.lower()]["ID"].item()
        if not roi_idx in np.unique(parcellation_data):
            print(l, roi_idx)
        volume = np.where(parcellation_data == roi_idx, d, volume)

    return nib.Nifti1Image(volume, parcellation.affine)


