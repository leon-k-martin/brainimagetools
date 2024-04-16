import os
import json

from os.path import join

import nibabel as nib
from nilearn import plotting

from tvbase.constants import DATA_DIR as cts_dir


def lists2dict(key_list, values_list):
    if not isinstance(key_list, list):
        key_list = key_list.tolist()
    if not isinstance(values_list, list):
        values_list = values_list.tolist()
    res = {}
    for key in key_list:
        for value in values_list:
            res[key] = value
            values_list.remove(value)
            break  
    return res


def get_centers(parc_img):
    ccoords = plotting.find_parcellation_cut_coords(parc_img, background_label=0, return_label_names=True, label_hemisphere='left')
    d = lists2dict(ccoords[1], ccoords[0])
    
    return d
