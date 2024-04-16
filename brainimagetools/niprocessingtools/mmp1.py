import pandas as pd

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