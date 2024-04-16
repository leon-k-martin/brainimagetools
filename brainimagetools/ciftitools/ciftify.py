from niprocessingtools import freesurfer
from ciftitools import io


def relabel_aparcaseg(faparcaseg):
    aparc = io.load_dlabel(faparcaseg)

    for i, r in aparc.area_info.iterrows():
        aparc.area_info.at[i, "label"] = freesurfer.idx2label(r.area_index)
        aparc.area_info = aparc.area_info[["area_index", "label", "rgba"]]
    return aparc
