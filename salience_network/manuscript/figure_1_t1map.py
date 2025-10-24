from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Cytoarchitecture of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

# wb_command -surface-coordinates-to-metric /local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-bigbrain/tpl-bigbrain_hemi-R_desc-mid.surf.gii \
#     /local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-bigbrain/tpl-bigbrain_hemi-R_desc-mid_coord.func.gii

# wb_command -metric-resample /local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-bigbrain/tpl-bigbrain_hemi-R_desc-mid_coord.func.gii \
#     /local_raid/data/pbautin/software/BigBrainWarp/xfms/tpl-fs_LR_hemi-R_den-164k_desc-sphere_rsled_like_bigbrain.reg.surf.gii \
#     /local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-sphere.surf.gii \
#     BARYCENTRIC \
#     /local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-mid_coord_bigbrain.func.gii



#### imports
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations, mesh_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace.mesh import mesh_elements

from brainspace.null_models import SpinPermutations, moran
from scipy.stats import spearmanr


from brainspace.gradient import GradientMaps, kernels
import scipy
from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression
from functools import partial




#### set custom plotting parameters
params = {'font.size': 14,
          'font.family': "sans-serif",
          'font.sans-serif': 'Open Sans'}
plt.rcParams.update(params)

## Yeo 2011, 7 network colors
yeo7_rgb = np.array([
    [255, 180, 80],    # Frontoparietal (brighter orange)
    [230, 90, 100],    # Default Mode (brighter red)
    [0, 170, 50],      # Dorsal Attention (more vivid green)
    [240, 255, 200],   # Limbic (lighter yellow-green)
    [210, 100, 255],   # Ventral Attention (lighter purple)
    [100, 160, 220],   # Somatomotor (lighter blue)
    [170, 70, 200],     # Visual (brighter violet)
    [255, 255, 255],     # Visual black
], dtype=float) / 255  # Normalize to 0–1
alpha = np.ones((8, 1))  # All fully opaque
yeo7_rgba = np.hstack((yeo7_rgb, alpha))
yeo7_colors = mpl.colors.ListedColormap(yeo7_rgba)
mpl.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)

## Von econo, cortical types colors
cmap_types_rgb = np.array([
    [255, 255, 255],  # white
    [127, 140, 172],  # desaturated blue-gray
    [139, 167, 176],  # desaturated cyan-gray
    [171, 186, 162],  # muted green
    [218, 198, 153],  # dull yellow
    [253, 211, 200],  # pale coral
    [252, 229, 252],  # pale magenta
], dtype=float) / 255
cmap_types_rgb[-2:,:] = cmap_types_rgb[-2:,:] * 0.80
alpha = np.ones((7, 1))
cmap_types_rgba = np.hstack((cmap_types_rgb, alpha))
cmap_types = mpl.colors.ListedColormap(cmap_types_rgba)
mpl.colormaps.register(name='CustomCmap_type', cmap=cmap_types)


## Yeo 2011, 7 network colors
baillarger_rgb = np.array([
    [255, 255, 255],    # Frontoparietal (brighter orange)
    [102, 194, 165],
    [252, 141,  98],
    [141, 160, 203],
    [231, 138, 195],
], dtype=float) / 255  # Normalize to 0–1
alpha = np.ones((5, 1))  # All fully opaque
baillarger_rgba = np.hstack((baillarger_rgb, alpha))
baillarger_colors = mpl.colors.ListedColormap(baillarger_rgba)
mpl.colormaps.register(name='CustomCmap_baillarger', cmap=baillarger_colors)

## Yeo 2011, 7 network colors
intrusion_rgb = np.array([
    [255, 255, 255],    # Frontoparietal (brighter orange)
    [102, 194, 165],
    [252, 141,  98],
    [141, 160, 203],
], dtype=float) / 255  # Normalize to 0–1
alpha = np.ones((4, 1))  # All fully opaque
intrusion_rgba = np.hstack((intrusion_rgb, alpha))
intrusion_colors = mpl.colors.ListedColormap(intrusion_rgba)
mpl.colormaps.register(name='CustomCmap_intrusion', cmap=intrusion_colors)


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


def surf_type_isolation(surf_type_test, i):
    # Work on a copy of the input array to avoid modifying the original
    surf_type_copy = surf_type_test.copy()
    surf_type_copy[surf_type_copy != i] = np.nan
    return surf_type_copy


def partial_corr_with_covariate(X, covar):
    """
    Compute partial correlation matrix between vertices, controlling for covariate.
    
    Parameters
    ----------
    X : array, shape (n_features, n_vertices)
        Data matrix.
    covar : array, shape (n_features,)
        Covariate to control for.
    
    Returns
    -------
    R : array, shape (n_vertices, n_vertices)
        Partial correlation matrix.
    """

    n_features, n_vertices = X.shape
    # z-score each vertex across profiles
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    # z-score covariate
    covar_z = (covar - covar.mean()) / covar.std(ddof=1)
    # design matrix: intercept + covariate 
    # shape (n_profiles, 2)
    X_covar = np.column_stack([np.ones(n_features), covar_z])
    # regression for all vertices at once 
    # beta shape (2, n_vertices)
    beta, _, _, _ = np.linalg.lstsq(X_covar, X_z, rcond=None)
    # predicted values and residuals
    # shape (n_profiles, n_vertices)
    X_hat = X_covar @ beta   
    # same shape as X            
    residuals = X_z - X_hat              
    # correlation matrix of residuals across vertices
    # shape (n_vertices, n_vertices)
    R = np.corrcoef(residuals, rowvar=False) 
    # Fisher z-transform
    with np.errstate(divide='ignore', invalid='ignore'):
        # MPC = 0.5 * np.log((1 + R) / (1 - R))
        MPC = np.arctanh(R)
        MPC[np.isnan(MPC)] = 0
        MPC[np.isinf(MPC)] = 0 
    return MPC


def load_mpc(File):
     """Loads and process a MPC"""
     mpc = nib.load(File).darrays[0].data
     mpc = np.triu(mpc,1)+mpc.T
     mpc[~np.isfinite(mpc)] = np.finfo(float).eps
     mpc[mpc==0] = np.finfo(float).eps
     return(mpc)

def normalize_to_range(data, target_min, target_max):
    """
    Normalizes a NumPy array or list of numerical data to a specified target range.

    Args:
        data (np.array or list): The input numerical data.
        target_min (float): The desired minimum value of the normalized range.
        target_max (float): The desired maximum value of the normalized range.

    Returns:
        np.array: The normalized data within the target range.
    """
    data = np.array(data) # Ensure data is a NumPy array for min/max operations
    
    original_min = np.nanmin(data)
    original_max = np.nanmax(data)

    if original_min == original_max: # Handle cases where all values are the same
        return np.full_like(data, (target_min + target_max) / 2)

    # Normalize to 0-1 range first
    normalized_0_1 = (data - original_min) / (original_max - original_min)

    # Scale to the target range
    scaled_data = target_min + (normalized_0_1 * (target_max - target_min))
    return scaled_data


def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)
    surf_32k = load_conte69(join=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load(micapipe + '/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load(micapipe + '/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    df_yeo_surf = pd.DataFrame({'mics': np.concatenate([atlas_yeo_lh, atlas_yeo_rh]).astype(float)})

    #### load yeo atlas 7 network information
    df_label = pd.read_csv(micapipe + '/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    df_yeo_surf['salience_border'] = array_operations.get_labeling_border(surf_32k, df_yeo_surf['network'].eq('SalVentAttn').to_numpy())
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 1, 'salience_border'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 0, 'salience_border'] = 1
    plt_values = df_yeo_surf['network_int'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    #### load econo atlas Hardcoded based on table data in Garcia-Cabezas (2021)
    econo_surf_lh = nib.load(micapipe + '/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load(micapipe + '/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.concatenate((econo_surf_lh, econo_surf_rh), axis=0).astype(float)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])[[0] + list(range(2, 45))]
    df_yeo_surf['surf_type'] = relabel(econo_surf, econ_ctb).astype(float)
    plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(0, 0, 0, 1), cmap='CustomCmap_type', transparent_bg=True, interactive=False)
    
    #### Baillarger type
    baillarger_surf_lh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Baillarger_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    baillarger_surf_rh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Baillarger_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    baillarger_surf = np.concatenate((baillarger_surf_lh, baillarger_surf_rh), axis=0).astype(float)
    baillarger_surf[(baillarger_surf == 0) | (baillarger_surf == 1)] = 1
    print(np.unique(baillarger_surf))
    print(np.array(sns.color_palette('Set2', 5)) * 255)
    baillarger_surf = baillarger_surf * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=baillarger_surf, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_baillarger', transparent_bg=True)

    #### Intrusion type
    intrusion_surf_lh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Intrusion_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    intrusion_surf_rh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Intrusion_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    intrusion_surf = np.concatenate((intrusion_surf_lh, intrusion_surf_rh), axis=0).astype(float)
    intrusion_surf[(intrusion_surf == 0) | (intrusion_surf == 1)] = 1
    print(np.unique(intrusion_surf))
    print(np.array(sns.color_palette('Set2', 5)) * 255)
    intrusion_surf = intrusion_surf * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=intrusion_surf, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_intrusion', transparent_bg=True)


    ##########################################################################
    ####################### ANALYSIS #########################################
    ######### Part 1 -- T1 map
    ### Load the data from PNI dataset
    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'))
    print("number of files/subjects: {}".format(len(t1_files)))
    ## t1 profiles (n_subject, n_features, n_vertices)
    t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files])
    t1_salience_profiles = t1_profiles[:, :, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    # t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_profiles, axis=2))]
    # gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    # gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    # t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    # print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_salience'] = t1_gradients[:, 0]
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient2_salience'] = t1_gradients[:, 1]
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient2_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # df_yeo_surf.to_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv', index=False)

    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')
    print(df_yeo_surf)
    # t1_gradient = df_yeo_surf['t1_gradient1_salience'].dropna().values
    # print(df_yeo_surf)
    # #    Plot
    # n_plot = 30
    # step = len(t1_gradient) // n_plot
    # sorted_gradient_indx = np.argsort(t1_gradient)[::step]
    # sorted_gradient = t1_gradient[sorted_gradient_indx]
    # plt.figure(figsize=(8,8))
    # plt.imshow(t1_salience_profiles[0,: , np.argsort(t1_gradient)].T, aspect='auto')
    # plt.show()
    # norm = mpl.colors.Normalize(vmin=np.min(t1_gradient), vmax=np.max(t1_gradient))
    # cmap = mpl.colormaps.get_cmap('coolwarm')
    # colors = [cmap(norm(g)) for g in sorted_gradient]
    # plt.figure(figsize=(6, 10))
    # profile = t1_salience_profiles[0,::]
    # for idx, color in zip(sorted_gradient_indx, colors):
    #     plt.plot(profile[:, idx] / 1000, np.arange(profile.shape[0]), color=color, alpha=0.8, lw=3)
    # plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    # plt.ylabel("T1 Map Intensity")
    # plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    # plt.gca().invert_yaxis()  # pial at top
    # plt.grid(False)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.tight_layout()
    # plt.show()


    ######### Part 2 -- Contextualisation
    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_bigbrain = -data_bigbrain[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1map_salience'] = zscore(np.mean(t1_salience_profiles, axis=(0, 1)), nan_policy='omit')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1map_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'bigbrain_gradient1_salience_mean'] = zscore(np.mean(salience_bigbrain, axis=0), nan_policy='omit')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1_salience_mean'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    ### Load the data from AHEAD
    data_biel = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-Ahead-Bielschowsky_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_biel = data_biel[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'biel_gradient1_salience'] = zscore(np.mean(salience_biel, axis=0), nan_policy='omit')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['biel_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    data_parva = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-Ahead-Parvalbumin_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_parva = data_parva[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'parva_gradient1_salience'] = zscore(np.mean(salience_parva, axis=0), nan_policy='omit')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['parva_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    # Extract reference salience gradient (T1)
    x = zscore(df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_SalVentAttn'].values)

    # Define modalities to compare with T1 gradient
    modalities = [
        ("bigbrain_gradient1_salience_mean", "BigBain mean"),
        ("t1map_salience", "T1map mean"),
        ("biel_gradient1_salience", "Bielschowsky"),
        ("parva_gradient1_salience", "Parvalbumin"),
        # Add more if needed, e.g. ("other_modality", "Label")
    ]

    w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=df_yeo_surf['network'].eq('SalVentAttn').values)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=10, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)


    # Setup figure
    fig, axes = plt.subplots(len(modalities), 1, figsize=(4, 4 * len(modalities)), sharex=True, sharey=True)

    if len(modalities) == 1:
        axes = [axes]  # ensure iterable when single subplot

    for ax, (col, label) in zip(axes, modalities):
        y = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), col].values
        rand = msr.randomize(y)

        # Scatterplot with regression line
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 25, "alpha": 0.7}, line_kws={"color": "black"})


        # Correlation
        r_obs, p = spearmanr(x, y, nan_policy='omit')

        # Compute perm pval
        r_rand = np.asarray([spearmanr(x, d)[0] for d in rand])
        pv_rand = np.mean(np.abs(r_rand) >= np.abs(r_obs))
        ax.set_title(f"$r={r_obs:.2f}, p={pv_rand:.2e}$", fontsize=12)

    plt.tight_layout()
    plt.show()

    r_vals, labels = [], []
    for col, label in modalities:
        if col in df_yeo_surf.columns:
            y = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), col].values
            if len(y) > 1 and not np.all(np.isnan(y)):
                r, _ = spearmanr(x, y, nan_policy='omit')
                r_vals.append(r)
                labels.append(label)

    # Convert to numpy
    r_vals = np.array(r_vals)
    print(r_vals)

    if r_vals.size == 0:
        raise ValueError("No valid correlations could be computed. Check your modality columns.")

    # Half-circle polar coordinates
    N = len(r_vals)
    theta = np.linspace(-np.pi /2 + np.pi/N*0.8, np.pi /2, N, endpoint=False)
    radii = np.abs(r_vals)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 5))
    bars = ax.bar(theta, radii, width=np.pi/N*0.8, align="center", alpha=0.8)

    # Color by sign
    for bar, r in zip(bars, r_vals):
        bar.set_facecolor("tab:red" if r < 0 else "tab:blue")
    #plt.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()





























    
    plt_values = []
    cluster = mesh_operations.get_connected_components(surf=surf_32k, labeling=df_yeo_surf['network'].eq('SalVentAttn').to_numpy()).astype(float)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=cluster, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='viridis', transparent_bg=True)
    print(np.unique(cluster))
    for i in [7, 3, 10, 13]:
        cluster_val = cluster.copy()
        cluster_val[cluster_val != i] = np.nan 
        cluster_val[cluster_val == i] = 1
        plt_values.append(df_yeo_surf['t1_gradient1_salience'].values * cluster_val)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 1200), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    
    #    Plot
    n_plot = 30
    step = len(t1_gradients[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(t1_gradients[:, 0])[::step]
    sorted_gradient = t1_gradients[:, 0][sorted_gradient_indx]
    norm = mpl.colors.Normalize(vmin=np.min(t1_gradients[:, 0]), vmax=np.max(t1_gradients[:, 0]))
    cmap = mpl.colormaps.get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    profile = np.mean(np.array(t1_salience_profile), axis=0)
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(profile[:, idx] / 1000, np.arange(profile.shape[0]), color=color, alpha=0.8, lw=3)
    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 10))
    profile = np.mean(np.array(t1_salience_residuals), axis=0)
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(profile[:, idx], np.arange(profile.shape[0]), color=color, alpha=0.8, lw=3)
    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.show()

    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')
    df_yeo_surf['cluster'] = mesh_operations.get_connected_components(surf=surf_32k, labeling=df_yeo_surf['network'].eq('SalVentAttn').to_numpy()).astype(float)
    df_yeo_surf['cluster_str'] = 'other'
    df_yeo_surf.loc[(df_yeo_surf['cluster'] == 7) | (df_yeo_surf['cluster'] == 14), 'cluster_str'] = 'aINS'
    df_yeo_surf.loc[(df_yeo_surf['cluster'] == 3) | (df_yeo_surf['cluster'] == 4) | (df_yeo_surf['cluster'] == 11), 'cluster_str'] = 'dACC'
    df_yeo_surf.loc[(df_yeo_surf['cluster'] == 10) | (df_yeo_surf['cluster'] == 2), 'cluster_str'] = 'PCU'
    df_yeo_surf.loc[(df_yeo_surf['cluster'] == 13) | (df_yeo_surf['cluster'] == 6)  | (df_yeo_surf['cluster'] == 5), 'cluster_str'] = 'IPL'
    df_yeo_surf.loc[(df_yeo_surf['network'].eq('SalVentAttn')) & (df_yeo_surf['cluster_str'] == 'other'), 'cluster_str'] = 'other_sal'
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['cluster_str'].values == 'other_sal', size=(400, 275), zoom=1.5, color_bar='right', share='both',
    #                 nan_color=(220, 220, 220, 1), cmap='Purples', transparent_bg=True, layout_style='grid')

    print(df_yeo_surf)
    df_yeo_surf['t1_gradient1_salience'] = normalize_to_range(df_yeo_surf['t1_gradient1_salience'].values, -1, 1)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_salience'].values, size=(400, 275), zoom=1.5, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, layout_style='grid')


    ######### Part 2 -- BigBrain
    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_bigbrain = data_bigbrain[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    # mpc_bigbrain, residuals_bigbrain = build_mpc(salience_bigbrain)
    # gm_bigbrain = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
    # gm_bigbrain.fit(mpc_bigbrain, sparsity=0)
    # print(gm_bigbrain.lambdas_)
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'bigbrain_gradient1_salience'] = normalize_to_range(gm_bigbrain.gradients_[:, 0], -1, 1)
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'bigbrain_gradient1_salience'] = normalize_to_range(skew(salience_bigbrain), -1, 1)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1_salience'].values, size=(400, 275), zoom=1.5, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, layout_style='grid')

    ### Load the data from AHEAD
    data_biel = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-Ahead-Bielschowsky_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_biel = data_biel[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    # mpc_biel, residuals_biel = build_mpc(salience_biel)
    # gm_biel = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
    # gm_biel.fit(mpc_biel, sparsity=0)
    # print(gm_biel.lambdas_)
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'biel_gradient1_salience'] = normalize_to_range(gm_biel.gradients_[:, 0], -1, 1)
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'biel_gradient1_salience'] = normalize_to_range(np.mean(salience_biel, axis=0), -1, 1)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['biel_gradient1_salience'].values, size=(400, 275), zoom=1.5, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, layout_style='grid')

    data_parva = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-Ahead-Parvalbumin_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_parva = data_parva[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    # mpc_parva, residuals_parva = build_mpc(salience_parva)
    # gm_parva = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
    # gm_parva.fit(mpc_parva, sparsity=0)
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'parva_gradient1_salience'] = normalize_to_range(gm_parva.gradients_[:, 0], -1, 1)
    # print(gm_parva.lambdas_)
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'parva_gradient1_salience'] = normalize_to_range(np.mean(salience_parva, axis=0), -1, 1)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['parva_gradient1_salience'].values, size=(400, 275), zoom=1.5, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, layout_style='grid')
    print(df_yeo_surf)

    df_yeo_surf['cluster'] = mesh_operations.get_connected_components(surf=surf_32k, labeling=df_yeo_surf['network'].eq('SalVentAttn').to_numpy()).astype(float)
    df_yeo_surf['cluster_str'] = 'other'
    df_yeo_surf.loc[df_yeo_surf['cluster'] == 7, 'cluster_str'] = 'aINS'
    df_yeo_surf.loc[df_yeo_surf['cluster'] == 3, 'cluster_str'] = 'dACC'
    df_yeo_surf.loc[df_yeo_surf['cluster'] == 10, 'cluster_str'] = 'PCU'
    df_yeo_surf.loc[df_yeo_surf['cluster'] == 13, 'cluster_str'] = 'IPL'
    df_sal = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'),
                            ['t1_gradient1_salience', 
                            'bigbrain_gradient1_salience', 
                            'biel_gradient1_salience', 
                            'parva_gradient1_salience', 
                            'cluster_str', 'label']].groupby(['cluster_str', 'label']).mean().reset_index()
    print(df_sal)
    # Pairplot with hue based on cluster_str
    sns.pairplot(df_sal,
                vars=['t1_gradient1_salience', 
                    'bigbrain_gradient1_salience', 
                    'biel_gradient1_salience', 
                    'parva_gradient1_salience'],
                hue='cluster_str',
                corner=True)

    plt.show()


    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_salience'], df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'bigbrain_gradient1_salience'])
        
     # Correlation analysis
    valid_rows = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), ['t1_gradient1_salience', 'bigbrain_gradient1_salience']].dropna()
    corr_val, p_val = pearsonr(valid_rows['t1_gradient1_salience'], valid_rows['bigbrain_gradient1_salience'])

    # Regression line (based on Pearson only)
    slope, intercept, _, _, _ = linregress(valid_rows['t1_gradient1_salience'], valid_rows['bigbrain_gradient1_salience'])
    # Plot regression line
    x_vals = np.array([valid_rows['t1_gradient1_salience'].min(), valid_rows['t1_gradient1_salience'].max()])
    ax.plot(x_vals, intercept + slope * x_vals, '--', color='black', lw=1)

    # Annotate correlation results
    ax.text(
        0.02, 0.98,
        f"{'PEARSON'} r = {corr_val:.3f}, p = {p_val:.3e}",
        transform=ax.transAxes,
        verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    #ax.set_aspect('equal')
    ax.autoscale()
    plt.xlabel('t1_gradient1_salience')
    plt.ylabel('bigbrain_gradient1_salience')
    plt.title("Scatter Pie Chart with Correlation Analysis")
    plt.show()



    # Plot
    # n_plot = 30
    # step = len(gm_bigbrain.gradients_[:, 0]) // n_plot
    # sorted_gradient_indx = np.argsort(gm_bigbrain.gradients_[:, 0])[::step]
    # sorted_gradient = gm_bigbrain.gradients_[:, 0][sorted_gradient_indx]
    # norm = mpl.colors.Normalize(vmin=np.min(gm_bigbrain.gradients_[:, 0]), vmax=np.max(gm_bigbrain.gradients_[:, 0]))
    # cmap = mpl.colormaps.get_cmap('coolwarm')
    # colors = [cmap(norm(g)) for g in sorted_gradient]
    # plt.figure(figsize=(6, 10))
    # for idx, color in zip(sorted_gradient_indx, colors):
    #     plt.plot(salience_bigbrain[:, idx] / 10000, np.arange(salience_bigbrain.shape[0]), color=color, alpha=0.8, lw=3)
    # plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    # plt.ylabel("T1 Map Intensity")
    # plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    # plt.gca().invert_yaxis()  # pial at top
    # plt.grid(False)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(6, 10))
    # for idx, color in zip(sorted_gradient_indx, colors):
    #     plt.plot(residuals_bigbrain[:, idx], np.arange(residuals_bigbrain.shape[0]), color=color, alpha=0.8, lw=3)
    # plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    # plt.ylabel("T1 Map Intensity")
    # plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    # plt.gca().invert_yaxis()  # pial at top
    # plt.grid(False)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.tight_layout()
    # plt.show()


    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'bigbrain_gradient1_salience'] = gm_bigbrain.gradients_[:, 0]
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    # plt_values = []
    # cluster = mesh_operations.get_connected_components(surf=surf_32k, labeling=df_yeo_surf['network'].eq('SalVentAttn').to_numpy()).astype(float)
    # for i in [7, 3, 10, 13]:
    #     cluster_val = cluster.copy()
    #     cluster_val[cluster_val != i] = np.nan 
    #     cluster_val[cluster_val == i] = 1
    #     plt_values.append(df_yeo_surf['bigbrain_gradient1_salience'].values * cluster_val)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 1200), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    




   



    
    ###### Cortical type comparisons
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))

    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    # Compute and store results
    all_data = {}
    real_data = {}

    surf_type[np.isnan(surf_type)] = 7  # Replace NaNs with dummy label
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        # Empirical
        expected_types = np.arange(1, 8)  # Cortical types 1 to 7
        comp = surf_type[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))

        # comp = surf_type[mask] * mask[mask]
        # u, c = np.unique(comp, return_counts=True)
        # perc = (c / len(comp)) * 100
        # real_data[net_name] = dict(zip(u, perc))

        # Null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in np.unique(surf_type)}
        for n in range(n_rand):
            comp = surf_type[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df
    print(all_data)

    from scipy.stats import ks_2samp
    from sklearn.decomposition import PCA

    # Convert real_data (dict of percentages) to a DataFrame
    del real_data['medial_wall']
    networks = list(real_data.keys())
    cortical_types = np.arange(1, 8)

    # Create DataFrame with rows = networks, columns = cortical types
    real_df = pd.DataFrame.from_dict(real_data, orient='index')
    real_df.columns = [label_map[t] for t in cortical_types]

    # Compute pairwise KS test statistics
    n_nets = len(networks)
    ks_matrix = np.zeros((n_nets, n_nets))

    for i in range(n_nets):
        for j in range(n_nets):
            ks_stat, _ = ks_2samp(real_df.iloc[i], real_df.iloc[j])
            ks_matrix[i, j] = ks_stat

    # Project to first principal component for row/column ordering
    pca = PCA(n_components=1)
    pc_order = np.argsort(pca.fit_transform(real_df).ravel())
    ordered_labels = [networks[i] for i in pc_order]

    # Reorder KS matrix
    ks_matrix_ordered = ks_matrix[pc_order, :][:, pc_order]

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(ks_matrix_ordered, xticklabels=ordered_labels, yticklabels=ordered_labels,
                cmap='magma', square=True, cbar_kws={'label': 'KS Statistic'})
    plt.title('Pairwise KS Test: Cortical Type Distributions Across Networks')
    plt.tight_layout()
    plt.show()


    ######### Part 1 -- T1 map
    ### Load the data from PNI dataset
    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'))
    t1_salience_profile = [nib.load(f).darrays[0].data[:, np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] for f in t1_files]
    t1_salience_mpc = [build_mpc(f)[0] for f in t1_salience_profile]
    t1_salience_residuals = [build_mpc(f)[1] for f in t1_salience_profile]
    gm_t1 = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_t1.fit(t1_salience_mpc, sparsity=0)
    t1_gradients = np.mean(np.asarray(gm_t1.gradients_), axis=0)

    arr = np.full(64984, np.nan)
    arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = t1_gradients[:, 0]
    df_yeo_surf['t1_gradient1_salience'] = arr
    #df_yeo_salience = df_yeo_surf[df_yeo_surf['network'] == 'SalVentAttn']
    #index_type = np.argsort(df_yeo_salience['surf_type'].values)

    # from scipy.ndimage import rotate
    # surf_type_sorted = df_yeo_salience['surf_type'].values[index_type]
    # mpc_fig = t1_salience_mpc[0][index_type[:, np.newaxis], index_type].copy()
    # mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    # mpc_fig= rotate(mpc_fig, angle=-45, order=0, cval=np.nan)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # im = ax.imshow(mpc_fig, cmap='coolwarm', vmin=-2,vmax=2, origin='upper')
    # # Add colorbar
    # #cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='MPC (z-value)')
    # # Aesthetic adjustments

    # # Add surf_type labels along diagonal
    # n = len(surf_type_sorted)
    # mid = mpc_fig.shape[0] // 2
    # offset = np.linspace(0, mpc_fig.shape[0]-1, n).astype(int)

    # for i, label in enumerate(surf_type_sorted):
    #     ax.text(offset[i], offset[i], str(label), rotation=45,
    #             ha='center', va='center', fontsize=6, color='black')
    # plt.axis('off')
    # plt.title('Upper Triangle of MPC Rotated by 45°')
    # plt.tight_layout()
    # plt.show()
   
    # Plot
    n_plot = 30
    step = len(t1_gradients[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(t1_gradients[:, 0])[::step]
    sorted_gradient = t1_gradients[:, 0][sorted_gradient_indx]
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    norm = Normalize(vmin=np.min(t1_gradients[:, 0]), vmax=np.max(t1_gradients[:, 0]))
    cmap = get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(t1_salience_residuals[0][:, idx] / 1000, np.arange(t1_salience_residuals[0].shape[0]), color=color, alpha=0.8, lw=3)
    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.show()
    
    arr = np.full(64984, np.nan)
    arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = t1_gradients[:, 0]
    df_yeo_surf['t1_gradient1_salience'] = arr

    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                          nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)

    
    ######### Part 2 -- BigBrain
    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    data_bigbrain_coord_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-mid_coord_bigbrain.func.gii').darrays[1].data
    data_bigbrain_coord_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-mid_coord_bigbrain.func.gii').darrays[1].data
    coord_bigbrain_surf = np.concatenate((data_bigbrain_coord_lh, data_bigbrain_coord_rh), axis=0).astype(float)
    data_bigbrain[:, np.asarray(df_yeo_surf['network'] == 'medial_wall')] = np.nan
    y_coords = coord_bigbrain_surf.reshape(-1, 1)
    reg = LinearRegression().fit(y_coords[np.asarray(df_yeo_surf['network'] != 'medial_wall')], data_bigbrain.mean(axis=0)[np.asarray(df_yeo_surf['network'] != 'medial_wall')])
    # regress out y-axis coords if needed
    #data_bigbrain -= reg.predict(y_coords)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=data_bigbrain.mean(axis=0), size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                            nan_color=(220, 220, 220, 1), cmap='Purples', transparent_bg=True, screenshot=True, filename='/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/figures/bigbrain_average_surf_wo_y.png')
    salience_bigbrain = data_bigbrain[:, np.asarray(df_yeo_surf['network'] == 'SalVentAttn')]
    data_bigbrain[:, np.asarray(df_yeo_surf['network'] != 'SalVentAttn')] =np.nan
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=data_bigbrain.mean(axis=0), size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename='/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/figures/bigbrain_salience_average_surf_wo_y.png')
    mpc_bigbrain, residuals_bigbrain = build_mpc(salience_bigbrain)

    # from scipy.ndimage import rotate
    # mpc_fig = mpc.copy()
    # mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    # mpc_fig= rotate(mpc_fig, angle=-45, order=0, cval=np.nan)
    # plt.imshow(mpc_fig, cmap='coolwarm', vmin=-2,vmax=2, origin='upper')
    # # Add colorbar
    # #cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='MPC (z-value)')
    # # Aesthetic adjustments
    # plt.axis('off')
    # plt.title('Upper Triangle of MPC Rotated by 45°')
    # plt.tight_layout()
    # plt.show()

    gm_bigbrain = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
    gm_bigbrain.fit(mpc_bigbrain, sparsity=0)
    # Plot
    n_plot = 30
    step = len(gm_bigbrain.gradients_[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(gm_bigbrain.gradients_[:, 0])[::step]
    sorted_gradient = gm_bigbrain.gradients_[:, 0][sorted_gradient_indx]
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    # Normalize gradient values for colormap mapping
    norm = Normalize(vmin=np.min(gm_bigbrain.gradients_[:, 0]), vmax=np.max(gm_bigbrain.gradients_[:, 0]))
    cmap = get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(residuals_bigbrain[:, idx], np.arange(residuals_bigbrain.shape[0]), color=color, alpha=0.8, lw=3)

    plt.ylabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.xlabel("Profile residuals (average profile regressed out)")
    plt.title("Cortical Depth Profiles")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)

    # Colorbar for gradient values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.tight_layout()
    plt.show()

    print(gm_bigbrain.lambdas_)
    arr = np.full(64984, np.nan)
    #arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = (gm.gradients_[:, 0] - np.min(gm.gradients_[:, 0])) / (np.max(gm.gradients_[:, 0]) - np.min(gm.gradients_[:, 0]))
    arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = gm_bigbrain.gradients_[:, 0]
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #                         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename='/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/figures/bigbrain_salience_gradient1_wo_y.png')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    # # second gradient
    # arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = gm_bigbrain.gradients_[:, 1]
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                          nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    

    ############## Compare bigbrain and T1 map gradients using Moran Spectral Randomization
    from brainspace.datasets import load_mask
    #n_pts = surf_32k.n_points
    mask_salience = np.asarray(df_yeo_surf['network'] == 'SalVentAttn')

    # Keep only the temporal lobe.
    g1_bigbrain_salience = gm_bigbrain.gradients_[:, 0]
    g1_t1_salience = t1_gradients[:, 0]
    curv_salience = load_marker('curvature', join=True)[mask_salience]


    from brainspace.null_models import MoranRandomization
    from brainspace.mesh import mesh_elements as me

    # compute spatial weight matrix
    w = me.get_ring_distance(surf_32k, n_ring=1, mask=mask_salience)
    w.data **= -1

    n_rand = 1000
    msr = MoranRandomization(n_rep=n_rand, procedure='singleton', tol=1e-6,
                            random_state=0)
    msr.fit(w)
    curv_rand = msr.randomize(curv_salience)
    print(curv_rand)
    t1_rand = msr.randomize(g1_t1_salience)
    print(t1_rand)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))

    feats = {'t1': g1_t1_salience, 'curvature': g1_t1_salience}
    rand = {'t1': t1_rand, 'curvature': t1_rand}

    for k, (fn, data) in enumerate(rand.items()):
        r_obs, pv_obs = spearmanr(feats[fn], g1_bigbrain_salience, nan_policy='omit')

        # Compute perm pval
        r_rand = np.asarray([spearmanr(g1_bigbrain_salience, d)[0] for d in data])
        pv_rand = np.mean(np.abs(r_rand) >= np.abs(r_obs))

        # Plot null dist
        axs[k].hist(r_rand, bins=25, density=True, alpha=0.5, color=(.8, .8, .8))
        axs[k].axvline(r_obs, lw=2, ls='--', color='k')
        axs[k].set_xlabel(f'Correlation with {fn}')
        if k == 0:
            axs[k].set_ylabel('Density')

        print(f'{fn.capitalize()}:\n r_Obs  : {r_obs:.5e}\n pval_Obs  : {pv_obs:.5e}\n pval_Moran: {pv_rand:.5e}\n')

    fig.tight_layout()
    plt.show()
    


    ######### Part 1
    ### Load the data from the specified text file AHEAD
    data_ahead_parva = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-Ahead-Parvalbumin_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_ahead_parva = data_ahead_parva[:, np.asarray(df_yeo_surf['network'] == 'SalVentAttn')]
    mpc = build_mpc(salience_ahead_parva)[0]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(mpc, sparsity=0)

    n_plot = 50
    step = len(gm.gradients_[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])[::step]
    sorted_gradient = gm.gradients_[:, 0][sorted_gradient_indx]


    # Plot
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    # Normalize gradient values for colormap mapping
    norm = Normalize(vmin=np.min(gm.gradients_[:, 0]), vmax=np.max(gm.gradients_[:, 0]))
    cmap = get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(salience_ahead_parva[:, idx] / 10000, np.arange(salience_ahead_parva.shape[0]), color=color, alpha=0.8, lw=3)

    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)

    # Colorbar for gradient values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.tight_layout()
    plt.show()

    print(gm.lambdas_)
    arr = np.zeros(64984)
    arr[arr == 0] = np.nan
    arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = (gm.gradients_[:, 0] - np.min(gm.gradients_[:, 0])) / (np.max(gm.gradients_[:, 0]) - np.min(gm.gradients_[:, 0]))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    # second gradient
    arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = (gm.gradients_[:, 1] - np.min(gm.gradients_[:, 1])) / (np.max(gm.gradients_[:, 1]) - np.min(gm.gradients_[:, 1]))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)



    #### tractogram
    # load the conte69 hemisphere surfaces and spheres
    from scipy.spatial import KDTree
    fsLR_lh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    fsLR_rh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    coords = np.concatenate([fsLR_lh, fsLR_rh])
    tree = KDTree(coords)

    tractogram_f = nib.streamlines.load('/local_raid/data/pbautin/data/output_tractogram_with_dps.trk')
    tractogram = tractogram_f.streamlines
    endpoints = [(sl[0], sl[-1]) for sl in tractogram]

    vertex_values = surf_type.copy()
    vertex_values[np.isnan(salience)] = np.nan
    streamline_values = []  # List to store values per streamline, e.g., tuple (start_val, end_val)
    for start_pt, end_pt in endpoints:
        dist_start, idx_start = tree.query(start_pt)
        dist_end, idx_end = tree.query(end_pt)

        # Optionally, define a cutoff distance to ensure endpoint is "close enough" to surface
        # For example, if we only trust assignments when endpoint is within 2 mm of surface:
        cutoff = 2.0
        if dist_start <= cutoff:
            start_val = vertex_values[idx_start]
        else:
            start_val = np.nan  # or some default value

        if dist_end <= cutoff:
            end_val = vertex_values[idx_end]
        else:
            end_val = np.nan

        streamline_values.append(np.nanmax([start_val, end_val]))

    num_streamlines = len(tractogram)
    streamline_arr = np.zeros(num_streamlines)
    streamline_values = np.asarray(streamline_values)
    streamline_values[np.isnan(streamline_values)] = 0
    streamline_arr = streamline_values
    mask = (streamline_arr != 0)
    streamline_arr = (streamline_arr - np.min(streamline_arr)) / (np.max(streamline_arr) - np.min(streamline_arr))
    filtered_tractogram = [tractogram[i] for i in range(len(tractogram)) if mask[i]]
    filtered_streamline_arr = streamline_arr[mask]
    new_tractogram = nib.streamlines.Tractogram(filtered_tractogram, affine_to_rasmm=tractogram_f.tractogram.affine_to_rasmm)
    #cmap = mp.colormaps['coolwarm']
    color = (cmap_types(filtered_streamline_arr)[:, 0:3] * 255).astype(np.uint8)
    print(color)
    tmp = [np.tile(color[i], (len(new_tractogram.streamlines[i]), 1))
           for i in range(len(new_tractogram.streamlines))]
    new_tractogram.data_per_point['color'] = tmp
    trk_file = nib.streamlines.TrkFile(new_tractogram)
    trk_file.save('/local_raid/data/pbautin/data/output_tractogram_with_dps_2.trk')


    ###### Cortical type comparisons
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))

    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    # Compute and store results
    all_data = {}
    real_data = {}

    surf_type[np.isnan(surf_type)] = 7  # Replace NaNs with dummy label
    state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        # Empirical
        expected_types = np.arange(1, 8)  # Cortical types 1 to 7
        comp = surf_type[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))

        # comp = surf_type[mask] * mask[mask]
        # u, c = np.unique(comp, return_counts=True)
        # perc = (c / len(comp)) * 100
        # real_data[net_name] = dict(zip(u, perc))

        # Null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in np.unique(surf_type)}
        for n in range(n_rand):
            comp = surf_type[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df

    # --- Plotting ---

    # Setup: Salience in full column
    n_total = len(all_data)
    n_cols = 4
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)

    # Plot Salience in full column
    ax_sal = fig.add_subplot(gs[:, 0])  # full height first column
    df = all_data["SalVentAttn"]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)

    # Plot other networks
    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        sns.barplot(data=df, ax=ax, color='lightgrey')
        rdict = {label_map.get(k, k): v for k, v in real_data[net_name].items()}
        sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()