from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Cytoarchitecture of the salience network on 7T T1 map data
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels
from scipy import stats

from Connectome_Spatial_Smoothing import CSS as css
import bct.algorithms as bct_alg
import bct.utils as bct
from scipy.sparse.csgraph import dijkstra
import scipy

from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

from nctpy.energies import get_control_inputs, integrate_u, minimum_energy_fast
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
    expand_states
)


from sklearn.linear_model import LinearRegression

def convert_states_str2int(states_str):
    """This function takes a list of strings that designate a distinct set of binary brain states and returns
    a numpy array of integers encoding those states alongside a list of keys for those integers.

    Args:
        states_str (N, list): a list of strings that designate which regions belong to which states.
            For example, states = ['Vis', 'Vis', 'Vis', 'SomMot', 'SomMot', 'SomMot']

    Returns:
        states (N, numpy array): array of integers denoting which node belongs to which state.
        state_labels (n_states, list): list of keys corresponding to integers.
            For example, if state_labels[1] = 'SomMot' then the integer 1 in `states` corresponds to 'SomMot'.
            Together, a binary state can be extracted like so: x0 = states == state_labels.index('SomMot')

    """
    n_states = len(states_str)
    state_labels = np.unique(states_str)

    states = np.zeros(n_states)
    for i, state in enumerate(state_labels):
        for j in np.arange(n_states):
            if state == states_str[j]:
                states[j] = i

    return states.astype(float), state_labels


def build_mpc(data, parc=None, idxExclude=None):
    I = data
    szI = data.shape
    szZ = np.empty((data.shape[1], data.shape[1]))


    # Build MPC
    if np.isnan(np.sum(I)):

        # Find where are the NaNs
        is_nan = np.isnan(I[1,:])
        problemNodes = [i for i, x in enumerate(is_nan) if x]
        print("")
        print("---------------------------------------------------------------------------------")
        print("There seems to be an issue with the input data or parcellation. MPC will be NaNs!")
        print("---------------------------------------------------------------------------------")
        print("")

        # Fill matrices with NaN for return
        I = np.zeros(szI)
        I[I == 0] = np.nan
        MPC = np.zeros(szZ)
        MPC[MPC == 0] = np.nan

    else:
        problemNodes = 0

        # Calculate mean across columns, excluding mask and any excluded labels input
        I_mask = I
        NoneType = type(None)
        if type(idxExclude) != NoneType:
            for i in idxExclude:
                I_mask[:, i] = np.nan
        I_M = np.nanmean(I_mask, axis = 1)

        # Get residuals of all columns (controlling for mean)
        I_resid = np.zeros(I.shape)
        for c in range(I.shape[1]):
            y = I[:,c]
            x = I_M
            slope, intercept, _, _, _ = scipy.stats.linregress(x,y)
            y_pred = intercept + slope*x
            I_resid[:,c] = y - y_pred

        R = np.corrcoef(I_resid, rowvar=False)

        # Log transform
        MPC = 0.5 * np.log( np.divide(1 + R, 1 - R) )
        MPC[np.isnan(MPC)] = 0
        MPC[np.isinf(MPC)] = 0

        # CLEANUP: correct diagonal and round values to reduce file size
        # Replace all values in diagonal by zeros to account for floating point error
        for i in range(0,MPC.shape[0]):
                MPC[i,i] = 0
        # Replace lower triangle by zeros
        MPC = np.triu(MPC)

    # Output MPC, microstructural profiles, and problem nodes
    return (MPC, I, problemNodes)

def main():
    #### set custom plotting parameters
    params = {"ytick.color" : "w",
              "xtick.color" : "w",
              "axes.labelcolor" : "w",
              "axes.edgecolor" : "w",
              'font.size': 22}
    plt.rcParams.update(params)
    plt.style.use('dark_background')
    yeo7_colors = mp.colors.ListedColormap(np.array([
                        [0, 118, 14, 255],
                        [230, 148, 34, 255],
                        [205, 62, 78, 255],
                        [120, 18, 134, 255],
                        [220, 248, 164, 255],
                        [70, 130, 180, 255],
                        [196, 58, 250, 255]]) / 255)
    mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)

    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')

    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    # fsLR-5k mask
    mask_lh = nib.load(micapipe + '/surfaces/fsLR-5k.L.mask.shape.gii').darrays[0].data
    mask_rh = nib.load(micapipe + '/surfaces/fsLR-5k.R.mask.shape.gii').darrays[0].data
    mask_5k = np.concatenate((mask_lh, mask_rh), axis=0)


    #### T1map 
    # plot surface intensity profile
    t1map_profile = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC011/ses-03/mpc/acq-T1map/sub-PNC011_ses-03_surf-fsLR-5k_desc-intensity_profiles.shape.gii").darrays[0].data
    t1map_profile[:,(mask_5k == 0)] = np.nan
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=t1map_profile[5,:], size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(0, 0, 0, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    # # Whole brain gradient
    # t1map_profile_wo_nan = t1map_profile[:,(mask_5k == 1)]
    # t1map_mpc_wo_nan = build_mpc(t1map_profile_wo_nan)[0]
    # gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    # gm.fit(t1map_mpc_wo_nan)

    # arr = np.zeros(t1map_profile.shape[1])
    # arr[arr == 0] = np.nan
    # arr[(mask_5k == 1)] = gm.gradients_[:, 0]
    # plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    #### MT-sat
    # plot surface intensity profile
    mtsat_profile = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC011/ses-03/mpc/acq-MTSAT/sub-PNC011_ses-03_surf-fsLR-5k_desc-intensity_profiles.shape.gii").darrays[0].data
    mtsat_profile[:,(mask_5k == 0)] = np.nan
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=mtsat_profile[5,:], size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(0, 0, 0, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    # # Whole brain gradient
    # mtsat_profile_wo_nan = mtsat_profile[:,(mask_5k == 1)]
    # mtsat_mpc_wo_nan = build_mpc(mtsat_profile_wo_nan)[0]
    # gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    # gm.fit(mtsat_mpc_wo_nan)

    # arr = np.zeros(mtsat_profile.shape[1])
    # arr[arr == 0] = np.nan
    # arr[(mask_5k == 1)] = gm.gradients_[:, 0]
    # plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    #### 7T network gradient
    # load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    state_stack = np.vstack([state, salience])
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=state_stack, size=(1200, 600), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                    nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    
    ### T1 map salience gradient
    t1map_profile = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC011/ses-03/mpc/acq-T1map/sub-PNC011_ses-03_surf-fsLR-5k_desc-intensity_profiles.shape.gii").darrays[0].data
    t1map_salience_profile = t1map_profile[:,~np.isnan(salience)]
    t1map_salience_mpc = build_mpc(t1map_salience_profile)[0]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(t1map_salience_mpc)
    # Sort data based on the first gradient component
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])
    sorted_t1map_salience_profile = t1map_salience_profile[:, sorted_gradient_indx]
    plt.figure(figsize=(16, 4))
    plt.imshow(sorted_t1map_salience_profile, cmap='gray', aspect='auto')
    plt.colorbar(label="T1 Map Intensity")
    plt.xlabel("Sorted Gradient Index")
    plt.show()

    arr = np.zeros(t1map_profile.shape[1])
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = gm.gradients_[:, 0]
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    ### MT-SAT salience gradient
    mtsat_profile = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC011/ses-03/mpc/acq-MTSAT/sub-PNC011_ses-03_surf-fsLR-5k_desc-intensity_profiles.shape.gii").darrays[0].data
    mtsat_salience_profile = mtsat_profile[:,~np.isnan(salience)]
    mtsat_salience_mpc = build_mpc(mtsat_salience_profile)[0]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(mtsat_salience_mpc)
    # Sort data based on the first gradient component
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])
    sorted_mtsat_salience_profile = mtsat_salience_profile[:, sorted_gradient_indx]
    plt.figure(figsize=(16, 4))
    plt.imshow(sorted_mtsat_salience_profile, cmap='gray', aspect='auto')
    plt.colorbar(label="MT-SAT Map Intensity")
    plt.xlabel("Sorted Gradient Index")
    plt.show()

    arr = np.zeros(mtsat_profile.shape[1])
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = gm.gradients_[:, 0]
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    
    #### BigBrain network gradient
    # load yeo atlas 7 network
    surf_lh, surf_rh = load_conte69()
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    state_stack = np.vstack([state, salience])
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=state_stack, size=(1200, 600), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                    nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)



    bigbrain_profile = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
    print(bigbrain_profile.shape)
    bigbrain_salience_profile = bigbrain_profile[:,~np.isnan(salience)]
    bigbrain_salience_mpc = build_mpc(bigbrain_salience_profile)[0]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(bigbrain_salience_mpc)
    # Sort data based on the first gradient component
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])
    sorted_bigbrain_salience_profile = bigbrain_salience_profile[:, sorted_gradient_indx]
    plt.figure(figsize=(16, 4))
    plt.imshow(sorted_bigbrain_salience_profile, cmap='gray', aspect='auto')
    plt.colorbar(label="Cell-body Staining Intensity")
    plt.xlabel("Sorted Gradient Index")
    plt.show()


    arr = np.zeros(bigbrain_profile.shape[1])
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = gm.gradients_[:, 0]
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')




if __name__ == "__main__":
    main()



# wb_command -metric-resample \
# /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii \
# /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.sphere.surf.gii \
# /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-5k.R.sphere.surf.gii \
# BARYCENTRIC -largest  \
# /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii


