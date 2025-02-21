from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Effective connectivity of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################



from effconnpy import CausalityAnalyzer,create_connectivity_matrix
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels


# Generate sample time series
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_rh.label.gii').darrays[0].data + 2000
atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
func = nib.load('/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/func/desc-me_task-rest_run-2_bold/surf/sub-Pilot014_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
data = reduce_by_labels(func, yeo_surf, red_op='mean', axis=0)
print(data.shape)
#data = np.random.rand(100, 3)
analyzer = CausalityAnalyzer(data)
results = analyzer.causality_test(method='ccm')
print(results)
binary_matrix = create_connectivity_matrix(results, threshold=1, metric='p_value')
plt.imshow(binary_matrix)
plt.show()