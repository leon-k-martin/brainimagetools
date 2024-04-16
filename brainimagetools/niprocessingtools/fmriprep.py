import os
import json

from os.path import join
import pandas as pd
import numpy as np
from scipy import stats

import nibabel as nib

from nilearn import plotting, image, masking, signal
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker

from bids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path

import hcp_utils as hcp
import ciftify

import matplotlib.pyplot as plt
import seaborn as sns

from tvbase import reparc, tvbase_atlas, constants

############
# Utilities#
############
labelmapper_HCP = {'L_Cerebellum':'cerebellum_left',
            'L_Thalamus':'thalamus_left',
            'L_Caudate':'caudate_left',
            'L_Putamen':'putamen_left',
            'L_Pallidum':'pallidum_left',
            'Brain-Stem':'brainStem',
            'L_Hippocampus':'hippocampus_left',
            'L_Amygdala':'amygdala_left',
            'L_Accumbens':'accumbens_left',
            'L_VentralDC':'diencephalon_left',
            'R_Cerebellum':'cerebellum_right',
            'R_Thalamus':'thalamus_right',
            'R_Caudate':'caudate_right',
            'R_Putamen':'putamen_right',
            'R_Pallidum':'pallidum_right',
            'R_Hippocampus':'hippocampus_right',
            'R_Amygdala':'amygdala_right',
            'R_Accumbens':'accumbens_right',
            'R_VentralDC':'diencephalon_right'
            }

labelmapper_tvbase = {'L_Cerebellum-Cortex':'cerebellum_left',
                     'L_Thalamus-Proper':'thalamus_left',
                     'L_Caudate':'caudate_left',
                     'L_Putamen':'putamen_left',
                     'L_Pallidum':'pallidum_left',
                     'L_Hippocampus':'hippocampus_left',
                     'L_Amygdala':'amygdala_left',
                     'L_Accumbens-area':'accumbens_left',
                     'L_VentralDC':'diencephalon_ventral_left',
                     'R_Cerebellum-Cortex':'cerebellum_right',
                     'R_Thalamus-Proper':'thalamus_right',
                     'R_Caudate':'caudate_right',
                     'R_Putamen':'putamen_right',
                     'R_Pallidum':'pallidum_right',
                     'R_Hippocampus':'hippocampus_right',
                     'R_Amygdala':'amygdala_right',
                     'R_Accumbens-area':'accumbens_right',
                     'R_VentralDC':'diencephalon_ventral_right',
                     'Brainstem':'brain_Stem'}

tvbase_labels = pd.read_csv(constants.AREA_INFO, sep='\t')['label_hemi'].to_list()[1:]
tvbase_labs = list()
tvbase_labels = [t + '.ROI' for t in tvbase_labels]
for l in tvbase_labels:
    if '_L.ROI' in l:
        l = 'L_' + l.replace('_L.ROI', '')
    if '_R.ROI' in l:
        l = 'R_' + l.replace('_R.ROI', '')
    
    if '.ROI' in l:
        l = l.replace('.ROI', '')
    tvbase_labs.append(l)

for k, v in labelmapper_tvbase.items():
    tvbase_labs = [l.replace(k, v) for l in tvbase_labs]
    
tvbase_labels_clean = [l.lower() for l in tvbase_labs]

# BIDS utils
def get_subject_list(bids_dir):
    layout = BIDSLayout(bids_dir, derivatives=True)
    subjects = layout.get_subjects()
    return subjects

def get_report(bids_dir, derivatives=False):
    report = BIDSReport(BIDSLayout(bids_dir))
    counter = report.generate()
    main_report = counter.most_common()[0][0]
    return main_report
    

def upper(df):
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]

def fractional_rescale(matrix):
        '''
        applying fractional rescaling of connectivity matrix
        following Rosen and Halgren 2021 eNeuro
        F(DTI(i,j)) := DTI(i,j) / sum(DTI(i,x))+sum(DTI(y,i)) with x!=i,y!=j
        '''
        import numpy
        colsum = numpy.nansum(matrix, axis = 0)
        _temp1 = numpy.tile(colsum,(colsum.shape[0],1))
        colsum = numpy.nansum(matrix, axis = 1)
        _temp2 = numpy.tile(colsum,(colsum.shape[0],1))
        return( matrix / ( (_temp1 + _temp2.T ) - (2*matrix)) )
    

#############
#CIFTI Based#
#############

def get_HCP_FC(datapath, subjid, run=None, save_results=True, plot=True):
    layout = BIDSLayout(datapath, derivatives=True)
    
    fout = datapath + '/derivatives/tvb-input/func-FCs_HCP/sub-{}/'.format(subjid)
    os.makedirs(fout, exist_ok=True)
    
    # Structure for BIDS build path.
    pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_parc-{parcellation}]_{suffix}.{extension}"
    entities = {
    'subject': '01',
    'run': run,
    'task': 'rest',
    'suffix': 'timeseries',
    'extension': 'csv',
    'scope': 'derivatives',
    'parc': 'HCPMMP1'
    }

    with open(layout.get(subject=subjid,
                         extension='json',
                         suffix='bold',
                         scope='derivatives',
                         return_type='filename',
                         run=run)[0]) as j:
        json_file = json.load(j)

    img = nib.load(layout.get(subject=subjid,
                              extension='dtseries.nii',
                              suffix='bold',
                              scope='derivatives',
                              return_type='filename',
                              run=run)[0])
    X = img.get_fdata()

    confound_df = pd.read_csv(layout.get(subject=subjid,
                                         extension='tsv',
                                         desc='confounds',
                                         scope='derivatives',
                                         return_type='filename',
                                         run=run)[0], sep='\t')
    
    confound_vars = ['trans_x','trans_y','trans_z',
                     'rot_x','rot_y','rot_z',
                     'global_signal',
                     'csf', 'white_matter']

    # Get derivative column names
    derivative_columns = ['{}_derivative1'.format(c) for c
                         in confound_vars]
    final_confounds = confound_vars + derivative_columns

    confound_df = confound_df[final_confounds]

    drop_confound_df = confound_df.loc[4:]
    confounds_matrix = drop_confound_df.values

    X_drop = X[4:, :]

    X_clean = signal.clean(X_drop, detrend=True, confounds=confounds_matrix, t_r=json_file['RepetitionTime'])

    ts = pd.DataFrame(hcp.parcellate(X_clean, hcp.mmp))
    ts.columns = ts.columns +1
    ts = ts.rename(hcp.mmp.labels, axis=1)
    
    entities_ts = entities.copy()    
    entities_FC = entities.copy()
    entities_FC['suffix'] = 'FC'
    
    if save_results:
        ts.transpose().to_csv(layout.build_path(entities_ts, join(fout, pattern), validate=False))
        ts.corr().to_csv(layout.build_path(entities_FC, join(fout, pattern), validate=False))

    if plot:
        entities_ts_png = entities.copy()
        entities_ts_png['extension'] = 'png'

        ts.plot(legend=False)
        # plt.title(layout.build_path(entities_ts, pattern))
        if save_results:
            plt.savefig(layout.build_path(entities_ts_png, join(fout, pattern), validate=False), dpi=300)
            plt.close()
        else:
            plt.show()

        entities_FC_png = entities.copy()
        entities_FC_png['suffix'] = 'FC'
        entities_FC_png['extension'] = 'png'
        sns.heatmap(ts.corr(), cmap='viridis', vmin=-1, vmax=1)
        # plt.imshow(ts.corr(), cmap='jet')
        # plt.title(layout.build_path(entities_FC, pattern))
        if save_results:
            plt.savefig(layout.build_path(entities_FC_png, join(fout, pattern), validate=False), dpi=300)
            plt.close()
        else:
            plt.show()
    
    return ts

#################
#Simple Approach#
#################

labels = ['ctx-lh-bankssts', 'ctx-lh-caudalanteriorcingulate',
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

tvbase_labels = pd.read_csv(constants.AREA_INFO, sep='\t')['label_hemi'].to_list()[1:]



def _load_niftis(data_path, subject, space, run=None):
    if not run:
        run_str = ''
    else:
        run_str = '_run-{}'.format(run)
        
    t1 = nib.load(join(data_path, subject, "anat", "{}_space-{}_desc-preproc_T1w.nii.gz".format(subject,space)))
    
    func_files = join(data_path, subject, 'func', subject + '_task-rest' + run_str )
    
    ref = nib.load(func_files + '_space-' + space + '_boldref.nii.gz')

    mask = nib.load(func_files + '_space-' + space + '_desc-brain_mask.nii.gz')

    aparc = nib.load(func_files + '_space-' + space + '_desc-aparcaseg_dseg.nii.gz')
    
    bold = nib.load(func_files + '_space-' + space + '_desc-preproc_bold.nii.gz')


    confounds = pd.read_csv(func_files + '_desc-confounds_timeseries.tsv', sep='\t')
    
    return t1, ref, mask, aparc, bold, confounds


#plotting.plot_anat(t1, title='T1')

def _plot_imgs(subject, ref, mask, aparc, bold, fout=None):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 5))
    fig.suptitle(subject)
    plotting.plot_anat(ref, title='boldref', axes=axs[0,0])
    plotting.plot_roi(mask, bg_img=ref, title='brain mask', axes=axs[0,1])
    plotting.plot_roi(aparc, title='aparc+aseg', bg_img=ref, axes=axs[1,0])

    mean_epi = image.image.mean_img(bold)
    plotting.plot_epi(mean_epi, title='mean epi', axes=axs[1,1])
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if fout:
        plt.savefig(fout, dpi=300)
        plt.close()
    else:
        plt.show()
        

def _extract_timeseries(labels_img, bold_ts, mask, confounds, labels=None):
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=True,
                               memory='nilearn_cache', verbose=0, mask_img=mask)

    # Here we go from nifti files to the signal time series in a numpy
    # array. Note how we give confounds to be regressed out during signal
    # extraction
    time_series = masker.fit_transform(bold_ts, confounds=confounds[['trans_x','trans_y','trans_z',
                                   'rot_x','rot_y','rot_z',
                                   'global_signal',
                                   'white_matter','csf']].replace(np.nan, 0))
    
    if isinstance(labels, type(None)):
        labels = [reparc.fs_mapper()[l] for l in masker.labels_]
    df_ts = pd.DataFrame(time_series, columns=labels)
    
    return df_ts

def _compute_fc(time_series):
    labels = time_series.columns
    # Calculate correlation matrix.
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series.values])[0]

    return pd.DataFrame(correlation_matrix, columns=labels, index=labels)

def _plot_fc(correlation_matrix, labels):
    # Make a large figure
    # Mask the main diagonal for visualization:
    #correlation_matrix = pd.DataFrame(np.fill_diagonal(correlation_matrix.values, 0), columns=correlation_matrix.columns, index=correlation_matrix.index)
    # The labels we have start with the background (0), hence we skip the
    # first label
    # matrices are ordered for block-like representation
    plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                         vmax=0.8, vmin=-0.8, reorder=False)
    plotting.show()
    
    
    
    
########## Pipeline:
def create_FC(data_path, subject, fout, space="MNI152NLin2009cAsym", run=None, parcellation=None):
    sfout = os.path.join(fout, subject)
    os.makedirs(sfout, exist_ok=True)
    
    vQC_out = os.path.join(fout, subject, 'vQC/')
    os.makedirs(vQC_out, exist_ok=True)
    
    t1, ref, mask, aparc, bold, confounds = _load_niftis(data_path, subject, space, run)
    
    if parcellation and parcellation.lower()=='tvbase':
        aparc = tvbase_atlas
        parc_str = 'tvbase'
        labels = tvbase_labels
    else:
        parc_str = 'aparc'
        labels = None
        
    vQC_img_path = os.path.join(vQC_out, '{}_run-{}_parc-{}_func.vQC.png'.format(subject, run, parc_str))
    if not os.path.exists(vQC_img_path):
        _plot_imgs(subject, ref, mask, aparc, bold, fout=vQC_img_path)

    ts_path = os.path.join(sfout, '{}_run-{}_parc-{}_time_series.csv'.format(subject, run, parc_str))
    if not os.path.exists(ts_path):
        time_series = _extract_timeseries(aparc, bold, mask, confounds, labels=labels)
        time_series.to_csv(ts_path)
    else: time_series = pd.read_csv(ts_path, index_col=0)

    if not (labels, type(None)):
        time_series.columns = labels
        
    cm_path = os.path.join(sfout, '{}_run-{}_parc-{}_FC.csv'.format(subject, run, parc_str))
    if not os.path.exists(cm_path):
        correlation_matrix = _compute_fc(time_series)
        correlation_matrix.to_csv(cm_path)
    else: correlation_matrix = pd.read_csv(cm_path, index_col=0)

    return correlation_matrix



##############################
#More sophisticated approach?#
##############################

#TODO: Make functions independent
#TODO: Define user input

def __get_files(subject_id, bids_layout, run=1):
    layout = bids_layout
    T1w_files = layout.get(subject=subject_id,
                           datatype='anat', desc='preproc',
                           space='MNI152NLin2009cAsym',
                           extension="nii.gz",
                          return_type='file')


    brainmask_files = layout.get(subject=subject_id,
                                 datatype='anat', suffix='mask',
                                 desc='brain',
                                 extension="nii.gz",
                                 space='MNI152NLin2009cAsym',
                                return_type='file')

    func_files = layout.get(subject=subject_id,
                            datatype='func', 
                            run=run,
                            desc='preproc',
                           extension="nii.gz",
                            space='MNI152NLin2009cAsym',
                           return_type='file')

    func_mask_files = layout.get(subject=subject_id,
                                 datatype='func',
                                 run=run,
                                 suffix='mask',
                                 desc='brain',
                                 extension="nii.gz",
                                 space='MNI152NLin2009cAsym',
                                return_type='file')
    
    aparc_files = layout.get(subject=subject_id,
                    datatype='func', 
                    run=run,
                    desc='aparcaseg',
                   extension="nii.gz",
                    space='MNI152NLin2009cAsym',
                   return_type='file')

    confound_files = layout.get(subject=subject_id,
                                datatype='func', 
                                run=run,
                                task='rest',
                                desc='confounds',
                               extension="tsv",
                               return_type='file')
    
    return T1w_files, brainmask_files, func_files, func_mask_files, aparc_files, confound_files

def __load_files(T1w_files, brainmask_files, func_files, func_mask_files, aparc_files):
    t1 = nib.load(T1w_files[0])
    bm = nib.load(brainmask_files[0])
    func = nib.load(func_files[0])
    func_bm = nib.load(func_mask_files[0])
    aparc = nib.load(aparc_files[0])
    
    tvbase_atlas = tvbase.constants.tvbase_atlas
    tvbase_atlas = nib.Nifti1Image(tvbase_atlas.get_fdata().round().astype(int), tvbase_atlas.affine)
    
    return t1, bm, func, func_bm, aparc, tvbase_atlas

def __resample_images(t1, bm, tvbase_atlas, func):
    # resamp_t1 = image.resample_to_img(source_img=t1, target_img=func,interpolation='continuous')
    # resamp_bm = image.resample_to_img(source_img=bm, target_img=func,interpolation='nearest')
    resamp_tvbase_atlas = image.resample_to_img(source_img=tvbase_atlas, target_img=func,interpolation='nearest')

    return resamp_tvbase_atlas #resamp_t1, resamp_bm, resamp_tvbase_atlas

def __extract_confounds(confound_tsv, confounds, dt=True):
    
    if dt:    
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names
    
    #Load in data using Pandas then extract relevant columns
    confound_df = pd.read_csv(confound_tsv,delimiter='\t') 
    confound_df = confound_df[confounds]
    
 
    #Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values 
    
    #Return confound matrix
    return confound_mat

def __tr_drop(n_tr, func, confounds):
    func = func.slicer[:,:,:,n_tr:]
    confounds = confounds[n_tr:,:] 
    
    return func, confounds


def __check_valid_regions_signal(tvbase_atlas, func, masker):
    # Get the label numbers from the atlas
    atlas_labels = np.unique(tvbase_atlas.get_fdata().astype(int))

    # Get number of labels that we have
    NUM_LABELS = len(atlas_labels)

    # Remember fMRI images are of size (x,y,z,t)
    # where t is the number of timepoints
    num_timepoints = func.shape[3]

    # Create an array of zeros that has the correct size
    final_signal = np.zeros((num_timepoints, NUM_LABELS))

    # Get regions that are kept
    regions_kept = np.array(masker.labels_)

    # Fill columns matching labels with signal values
    final_signal[:, regions_kept] = cleaned_and_averaged_time_series

    valid_regions_signal = final_signal[:, regions_kept]

    check = np.array_equal(
        valid_regions_signal,
        cleaned_and_averaged_time_series)
    return check


def __get_timeseries(masker, func, confounds):
    cleaned_and_averaged_time_series = masker.fit_transform(func, confounds)
    
    return cleaned_and_averaged_time_series

def __save_timeseries(cleaned_and_averaged_time_series, sd, atlas_name, subject_id, run):
    np.savetxt(os.path.join(sd, 'sub-{}_task-rest_run-{}_parc-{}_timeseries.txt'.format(subject_id, run, atlas_name)), cleaned_and_averaged_time_series)
    

def __get_FC(cleaned_and_averaged_time_series):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    full_correlation_matrix = correlation_measure.fit_transform([cleaned_and_averaged_time_series])
    
    return full_correlation_matrix

def __plot_FC(full_correlation_matrix, sd, atlas_name, subject_id, run):
    cm = full_correlation_matrix[0]
    np.fill_diagonal(cm, 0)

    sns.heatmap(full_correlation_matrix[0], cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('sub-{} FC ({})'.format(subject_id, atlas_name));
    plt.savefig(os.path.join(sd, 'sub-{}_task-rest_run-{}_parc-{}_FC.png'.format(subject_id, run, atlas_name)), dpi=300)
    plt.close()

def __save_FC(full_correlation_matrix, sd, atlas_name, subject_id, run):
    np.savetxt(os.path.join(sd, 'sub-{}_task-rest_run-{}_parc-{}_FC.txt'.format(subject_id, run, atlas_name)), full_correlation_matrix[0])
    
    
    
def fc_pipeline(bids_layout, subject_id, run):
    print('sub-', subject_id, 'run:', run)
    sd = 'ds002898/derivatives/tvb-input/func-FCs_new/sub-{}/func'.format(subject_id)
    os.makedirs(sd, exist_ok=True)

    # Get file paths.
    T1w_files, brainmask_files, func_files, func_mask_files, aparc_files, confound_files = __get_files(subject_id, bids_layout, run)

    # Load files.
    t1, bm, func, func_bm, aparc, tvbase_atlas = __load_files(T1w_files, brainmask_files, func_files, func_mask_files, aparc_files)

    # Resample to functional space.
    print('resampling.')
    resamp_tvbase_atlas = image.resample_to_img(source_img=tvbase_atlas, target_img=func,interpolation='nearest')

    # Set constants for confounds regression
    high_pass= 0.009
    low_pass = 0.08
    t_r = 2


    #Use the above function to pull out a confound matrix
    confounds = __extract_confounds(confound_files[0],
                                  ['trans_x','trans_y','trans_z',
                                   'rot_x','rot_y','rot_z',
                                   'global_signal',
                                   'white_matter','csf'])

    #Remove the first 4 TRs
    func, confounds = __tr_drop(4, func, confounds)

    for atlas_name, atlas in zip(['tvbase', 'aparcaseg'], [resamp_tvbase_atlas, aparc]):
        masker = input_data.NiftiLabelsMasker(labels_img=atlas,
                                              mask_img=func_bm,
                                              standardize=True,
                                              memory='nilearn_cache',
                                              verbose=1,
                                              detrend=True,
                                                low_pass = low_pass,
                                                 high_pass = high_pass,
                                                 t_r=t_r)
        print('compute FC..')
        cleaned_and_averaged_time_series = __get_timeseries(masker, func, confounds)
        __save_timeseries(cleaned_and_averaged_time_series, sd, atlas_name, subject_id, run)
        
        FC = __get_FC(cleaned_and_averaged_time_series)

        __plot_FC(FC, sd, atlas_name, subject_id, run)

        __save_FC(FC, sd, atlas_name, subject_id, run)