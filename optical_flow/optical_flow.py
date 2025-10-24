from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Connectivity and communication of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_points
from brainspace.mesh import array_operations, mesh_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels


def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    fs5_lh = read_surface(micapipe + '/surfaces/fsaverage5/surf/lh.inflated', itype='fs')
    fs5_rh = read_surface(micapipe + '/surfaces/fsaverage5/surf/rh.inflated', itype='fs')

    annot_lh_fs5 = nib.freesurfer.read_annot(micapipe + '/parcellations/lh.schaefer-100_mics.annot')[0] + 1
    annot_rh_fs5 = nib.freesurfer.read_annot(micapipe + '/parcellations/rh.schaefer-100_mics.annot')[0] + 1 + annot_lh_fs5.max()
    labels = np.concatenate((annot_lh_fs5, annot_rh_fs5), axis=0)
    df_label_surf = pd.DataFrame(data={'roi': labels})

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-100_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label_surf = df_label_surf.merge(df_label, on='roi', validate="many_to_one", how='left')    

    opfl = sio.loadmat('/host/verges/tank/data/tajwar/optical_flow/derivatives/sub-PNC014/ses-01/sub-PNC014_ses-01_opfl.mat', simplify_cells=True)
    opfl_lh = opfl['us']['vf_left'][0]
    opfl_rh = opfl['us']['vf_right'][0]
    opfl_values = np.concatenate((opfl_lh, opfl_rh), axis=0).astype(float)

    df_label_surf.loc[df_label_surf.network != 'medial_wall', 'optical_flow'] = opfl_values[:,0]

    ### !!! ValueError: NumPy boolean array indexing assignment cannot assign 19360 input values to the 18741 output values where the mask is t
    plot_hemispheres(fs5_lh, fs5_rh, df_label_surf.optical_flow.values)

if __name__ == "__main__":
    main()