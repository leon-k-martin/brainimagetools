# parc
import hcp_utils as hcp
import numpy as np
from ciftitools import io


def cifti_unparcellate(data_dict, path_dlabel):
    """_summary_

    :param parc_data: key-value pair of area-names and parcel values
    :type parc_data: pandas.Series, dict
    :param path_dlabel: Path to the parcellation.dlabel file
    :type path_dlabel: _type_
    :return: _description_
    :rtype: _type_
    """
    dlabel = io.load_dlabel(path_dlabel)
    area_info = dlabel.area_info
    data = dlabel.data
    for lab, v in data_dict.items():
        area_info.at[area_info.fs_label == lab, "value"] = v

    parc_data = np.array(area_info.sort_values("area_index")["value"].to_list())
    parc_data = np.nan_to_num(parc_data)
    vtx_data = parc_data[data.astype(int)].flatten()
    return vtx_data
