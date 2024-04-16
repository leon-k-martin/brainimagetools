import nibabel as nib
import pandas as pd
from matplotlib import colors
from sklearn.utils import Bunch

# TODO: Add docstrings.


def load_dscalar(fpath, return_axes=False):
    d = nib.load(fpath)
    hdr = d.header
    label_info = pd.DataFrame()

    scalar_ax = hdr.get_axis(0)

    bm_ax = hdr.get_axis(1)

    data = d.get_fdata()

    return Bunch(data=d, scalar_ax=scalar_ax, bm_ax=bm_ax)


def load_dlabel(fpath):
    d = nib.load(fpath)
    hdr = d.header
    label_info = pd.DataFrame()

    label_ax = hdr.get_axis(0)
    labels = label_ax.label

    bm_ax = hdr.get_axis(1)

    data = d.get_fdata()

    for i, v in label_ax.label[0].items():
        lab = v[0]
        rgba = v[1]
        label_info.at[i, "label"] = lab
        label_info.at[i, "fs_label"] = lab.replace("L_", "ctx-lh-").replace(
            "R_", "ctx-rh-"
        )

        label_info.at[i, "rgba"] = colors.rgb2hex(rgba, keep_alpha=True)

    label_info = label_info.reset_index().rename({"index": "area_index"}, axis=1)

    out = Bunch(
        data=data,
        labels=[v[0] for v in labels[0].values()],
        area_info=label_info,
        brain_model_axis=bm_ax,
        file=d,
    )

    return out


def load_dtseries(fpath):
    tseries = nib.load(fpath)
    series_axis = tseries.header.get_axis(0)
    brain_model_axis = tseries.header.get_axis(1)
    out = Bunch(
        data=tseries.get_fdata(),
        size=series_axis.size,
        start=series_axis.start,
        tr=series_axis.step,
        time=series_axis.time,
        unit=series_axis.unit,
        brain_model_axis=brain_model_axis,
    )
    return out


def load_ptseries(fpath, return_df=False):
    ts = nib.load(fpath)
    hdr = ts.header

    parcelaxis = hdr.get_axis(1)
    tsaxis = hdr.get_axis(0)
    time = tsaxis.time

    labels = parcelaxis.name
    labels = [l.replace("_ROI", "").lower() for l in labels]

    data = ts.get_fdata()

    if return_df:
        df = pd.DataFrame(data)
        df.columns = labels
        df.index = time
        return df

    return Bunch(data=data, time=time, labels=labels)


def ptseries2csv(fpath, fout=None):
    ts = nib.load(fpath)

    hdr = ts.header

    parcelaxis = ts.header.get_axis(1)
    tsaxis = ts.header.get_axis(0)
    time = tsaxis.time

    labels = parcelaxis.name
    labels = [l.replace("_ROI", "").lower() for l in labels]

    data = ts.get_fdata()
    df = pd.DataFrame(data)
    df.columns = labels
    df.index = time
    # df = df[fpp.tvbase_labels_clean]

    if fout:
        df.to_csv(fout)
    else:
        return df


def pconn2csv(fpath, fout=None):
    cm = nib.load(fpath)

    labels = cm.header.get_axis(0).name
    labels = [l.replace("_ROI", "").lower() for l in labels]

    data = cm.get_fdata()
    df = pd.DataFrame(data)
    df.columns = labels
    df.index = labels
    # df = df.loc[fpp.tvbase_labels_clean, fpp.tvbase_labels_clean]

    if fout:
        df.to_csv(fout)
    else:
        return df
