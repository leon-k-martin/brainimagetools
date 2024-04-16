import os
from os.path import join

import nibabel as nib
import pandas as pd

from tvbids import mmp1


def convert_ptseries(fpath, fout=None):
    ts = nib.load(fpath)
    
    hdr = ts.header
    
    parcelaxis = ts.header.get_axis(1)
    tsaxis = ts.header.get_axis(0)
    time = tsaxis.time

    labels = parcelaxis.name
    labels = [l.replace('_ROI', '').lower() for l in labels]

    data = ts.get_fdata()
    df = pd.DataFrame(data)
    df.columns=labels
    df.index=time
    df = df[mmp1.tvbase_labels_clean]
    
    if fout:
        df.to_csv(fout)
    else: return df

    
def convert_pconn(fpath, fout=None):
    cm = nib.load(fpath)
    
    labels = cm.header.get_axis(0).name
    labels = [l.replace('_ROI', '').lower() for l in labels]

    data = cm.get_fdata()
    df = pd.DataFrame(data)
    df.columns=labels
    df.index=labels
    df = df.loc[fpp.tvbase_labels_clean, fpp.tvbase_labels_clean]
    
    if fout:
        df.to_csv(fout)
    else: return df