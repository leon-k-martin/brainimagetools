# Functional Processing utilities
# author: Leon Martin

import os
import json

import numpy as np
from os.path import join
import pandas as pd
import nibabel as nib

from tvbase import reparc
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import plotting

from tvbase import constants, reparc

# Freesurfer 

fs_labels = ['ctx-lh-bankssts', 'ctx-lh-caudalanteriorcingulate',
       'ctx-lh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-lh-entorhinal',
       'ctx-lh-frontalpole', 'ctx-lh-fusiform', 'ctx-lh-inferiorparietal',
       'ctx-lh-inferiortemporal', 'ctx-lh-insula', 'ctx-lh-isthmuscingulate',
       'ctx-lh-lateraloccipital', 'ctx-lh-lateralorbitofrontal',
       'ctx-lh-lingual', 'ctx-lh-medialorbitofrontal', 'ctx-lh-middletemporal',
       'ctx-lh-paracentral', 'ctx-lh-parahippocampal',
       'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis',
       'ctx-lh-parstriangularis', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral',
       'ctx-lh-posteriorcingulate', 'ctx-lh-precentral', 'ctx-lh-precuneus',
       'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal',
       'ctx-lh-superiorfrontal', 'ctx-lh-superiorparietal',
       'ctx-lh-superiortemporal', 'ctx-lh-supramarginal',
       'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 
        'left-accumbens-area', 'left-amygdala', 'left-caudate',
       'left-cerebellum-cortex', 'left-hippocampus', 'left-pallidum',
       'left-putamen', 'left-thalamus', 'left-ventraldc',
          'ctx-rh-bankssts',
       'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal',
       'ctx-rh-cuneus', 'ctx-rh-entorhinal', 'ctx-rh-frontalpole',
       'ctx-rh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal',
       'ctx-rh-insula', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital',
       'ctx-rh-lateralorbitofrontal', 'ctx-rh-lingual',
       'ctx-rh-medialorbitofrontal', 'ctx-rh-middletemporal',
       'ctx-rh-paracentral', 'ctx-rh-parahippocampal',
       'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis',
       'ctx-rh-parstriangularis', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral',
       'ctx-rh-posteriorcingulate', 'ctx-rh-precentral', 'ctx-rh-precuneus',
       'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal',
       'ctx-rh-superiorfrontal', 'ctx-rh-superiorparietal',
       'ctx-rh-superiortemporal', 'ctx-rh-supramarginal',
       'ctx-rh-temporalpole', 'ctx-rh-transversetemporal',
       'right-accumbens-area', 'right-amygdala', 'right-caudate',
       'right-cerebellum-cortex', 'right-hippocampus', 'right-pallidum',
       'right-putamen', 'right-thalamus', 'right-ventraldc']
       

LUT = pd.read_csv(join(constants.DATA_DIR, 'desc-aparcaseg_dseg.tsv'), sep='\t', index_col=1).transpose().drop('index')
LUT.columns = [x.lower() for x in LUT.columns.to_list()]

fs_mapper = reparc.fs_mapper()


datapath = '/data/gpfs-1/users/martinl_c/work/Virtual_PET/ds002898/derivatives/'



for subid in range(1,28):
    if subid < 10:
        subid = '0'+str(subid)
    aparcaseg = nib.load(join(datapath, 'ciftify/sub-{}/aparc+aseg.dlabel.nii'.format(subid)))
    
    labs = list()
    for area_ind in sorted(np.unique(aparcaseg.get_fdata()).astype(int)):
        labs.append(reparc.fs_mapper()[area_ind])
    labs = labs[1:]
    
    avg_con = np.zeros((86, 86))
    fout_con = join(datapath, 'tvb_input/FC/sub-{}'.format(subid))
    os.makedirs(fout_con, exist_ok=True)
    
    concat_ts = pd.DataFrame(index=fs_labels).transpose()
    fout_ts = join(datapath, 'tvb_input/timeseries/sub-{}'.format(subid))
    os.makedirs(fout_ts, exist_ok=True)
    
    
    
    for run in range(1,7):
        # FC
        cm = nib.load(join(datapath, 'xcp_d/sub-{}/func/sub-{}_task-rest_run-{}_space-fsLR_den-91k_atlas-aparc+aseg_desc-residual_bold.pconn.nii'.format(subid, subid, run)))
    
        labels = cm.header.get_axis(0).name
        labels = [fs_mapper[int(c.lower().replace('label_', ''))] for c in labels]

        data = cm.get_fdata()
        df = pd.DataFrame(data)
        df.columns=labels
        df.index=labels
        df = df.loc[fs_labels, fs_labels]
        
        avg_con += df#.values
        # df.to_csv(join(fout,'sub-{}_run-{}_atlas-parc+aseg_FC.csv'.format(subid, run)))
        
        # Timeseries
        ts = nib.load(join(datapath, 'xcp_d/sub-{}/func/sub-{}_task-rest_run-{}_space-fsLR_den-91k_atlas-aparc+aseg_desc-residual_bold.ptseries.nii'.format(subid, subid, run)))
        
        parcelaxis = ts.header.get_axis(1)
        labels_ts = parcelaxis.name
        labels_ts = [fs_mapper[int(c.lower().replace('label_', ''))] for c in labels_ts]
        
        tsaxis = ts.header.get_axis(0)
        time = tsaxis.time

        df_ts = pd.DataFrame(ts.get_fdata())
        df_ts.columns=labels_ts
        df_ts.index=time
        df_ts.index.name = 'time'
        df_ts = df_ts[fs_labels]
        concat_ts = concat_ts.append(df_ts)

    # Save aggregated data.
    avg_con = avg_con / 6
    avg_con.to_csv(join(fout_con,'sub-{}_run-average_atlas-parc+aseg_FC.csv'.format(subid)))
    
    concat_ts.to_csv(join(fout_ts,'sub-{}_run-all_atlas-parc+aseg_timeseries.csv'.format(subid)))
    
    # Plot.
    plt.imshow(avg_con)
    plt.title('sub-{}_run-average (BOLD FC)'.format(subid))
    plt.savefig(join(fout_con,'sub-{}_run-average_atlas-parc+aseg_FC.png'.format(subid)), dpi=300)
    plt.close()
    
    concat_ts.plot(legend=False)
    plt.title('sub-{}_run-average (BOLD timeseries)'.format(subid))
    plt.savefig(join(fout_ts,'sub-{}_run-average_atlas-parc+aseg_timeseries.png'.format(subid)), dpi=300)
    plt.close()