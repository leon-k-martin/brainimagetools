import numpy as np
import pandas as pd
from os.path import join
from brainvistools import constants


def fs_mapper(output="label"):
    """_summary_

    :param index_as_key: _description_, defaults to True
    :type index_as_key: bool, optional
    :return: _description_
    :rtype: _type_
    """
    filename = join(constants.DATA_DIR, "FreeSurferColorLUT.txt")
    lut = pd.read_csv(
        filename, comment="#", sep="\s+", names=["id", "name", "r", "g", "b", "a"]
    )

    lut = pd.DataFrame(lut)
    lut.name = lut.name.str.lower()

    # FS index-name mapper
    mapper = dict()

    # Create index-name pairs.
    for i, r in lut.iterrows():
        if output.lower() in ["label"]:
            mapper[r.id] = r["name"]
        else:
            mapper[r["name"]] = r.id

    return mapper


def idx2label(idx):
    if isinstance(idx, list):
        return [fs_mapper(output="label")[i] for i in idx]
    else:
        return fs_mapper(output="label")[idx]


def label2idx(label):
    if isinstance(label, list):
        return [fs_mapper(output="index")[l] for l in label]
    else:
        return fs_mapper(output="index")[label]


def hcp2fs_labels(hcp_labels):
    fs_labels = list()
    for l in hcp_labels:
        l = l.lower()
        fs_labels.append(l.replace("l_", "ctx-lh-").replace("r_", "ctx-rh-"))

    return fs_labels
