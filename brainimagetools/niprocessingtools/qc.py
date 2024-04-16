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

# Plotting

def plot_ts(df, xtr=100, ax=None):
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots()
        
    for c in df:
        ax.plot(df[c].values, color=LUT[c].item(), linewidth=.1, alpha=.8
                )

        ax.set_xticks(range(len(df.index))[::xtr])
        ax.set_xticklabels([round(i) for i in df.index.to_list()][::xtr], rotation=45)
        ax.set_title('sub-{}'.format(subid))
        #ax.get_legend().remove()
        ax.set_xlabel("s")

        
# Quality Check
## Visual QC
def plot_timeseries_overview(datapath):
    """
    Plots timeseries of tvb-input folder for each subject.
    """

    n_subjects = 27

    ncols = 7
    nrows = n_subjects // ncols + (n_subjects % ncols > 0)

    row=0
    col=0

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharex=True, sharey=True)

    mean_ts = []

    for n, subid in enumerate(range(1,28)):
        if col == ncols:
            col=0
            row+=1

        if subid < 10:
            subid = '0'+str(subid)

        fin_ts = join(datapath, 'tvb_input/timeseries/sub-{}'.format(subid))
        fin = join(fin_ts, 'sub-{}_run-all_atlas-parc+aseg_timeseries.csv'.format(subid))
        df = pd.read_csv(fin, index_col=0)

        # z-standardize data for comparison
        df = (df - df.mean())/df.std()

        mean_ts.append(df)

        # ax = plt.subplot(nrows, ncols, n + 1#, sharex=ax, sharey=ax)
        ax = axes[row, col]

        for c in df:
            ax.plot(df[c].values, color=LUT[c].item(), linewidth=.1, alpha=.8
                    )

            ax.set_xticks(range(len(df.index))[::200])
            ax.set_xticklabels([round(i) for i in df.index.to_list()][::200], rotation=90)

        # ax.plot(df.index, df.values, alpha=0.5)

        # chart formatting
        plt.ylim(-6, 6)
        ax.set_title('sub-{}'.format(subid))
        #ax.get_legend().remove()
        ax.set_xlabel("")

        col+=1

    # If there is space, add average timeseries.
    if 27%ncols!=0:
        mts = mean_ts[0]
        for ts in mean_ts[1:]:
            mts += ts
        mts = mts / len(mean_ts)
        for c in mts:
            axes[row, col].plot(mts[c].values, color=LUT[c].item(), linewidth=.1, alpha=.8)

        plt.ylim(-6, 6)
        axes[row, col].set_title('average timeseries')


    plt.tight_layout()
    plt.suptitle("BOLD FCs (averaged across all 6 runs)", fontsize=18, y=1.03)
    plt.savefig(join(datapath, 'tvb_input/FC/', 'BOLD_FC_overview.png'), dpi=500)