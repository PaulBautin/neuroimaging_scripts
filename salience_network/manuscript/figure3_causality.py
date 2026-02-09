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


import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
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
from matplotlib.cm import get_cmap

from effconnpy import CausalityAnalyzer,create_connectivity_matrix, MultivariateGrangerCausality

from pydcm.models import pdcm




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
    #df_yeo_surf['salience_border'] = array_operations.get_labeling_border(surf_32k, df_yeo_surf['network'].eq('SalVentAttn').to_numpy())
    #df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 1, 'salience_border'] = np.nan
    #df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 0, 'salience_border'] = 1
    #plt_values = df_yeo_surf['network_int'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['network_int'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    

    ##########################################################################
    ####################### ANALYSIS #########################################
    ######### Part 1 -- T1 map
    ### Load the data from PNI dataset
    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'))
    print("number of files/subjects: {}".format(len(t1_files)))
    ## t1 profiles (n_subject, n_features, n_vertices)
    # t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files])
    # t1_profiles_net = {}
    # for net in np.unique(df_yeo_surf['network'].values):
    #     t1_profiles_net = t1_profiles[:, :, df_yeo_surf['network'].eq(net).to_numpy()]
    #     t1_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_profiles_net[:, :, :], np.nanmean(t1_profiles, axis=2))]
    #     gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    #     gm_t1.fit(t1_mpc, sparsity=0.9)
    #     t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    #     df_yeo_surf.loc[df_yeo_surf['network'].eq(net), 't1_gradient1_' + net] = t1_gradients[:, 0]
        # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_' + net].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_salience'] = t1_gradients[:, 0]
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient2_salience'] = t1_gradients[:, 1]

    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient2_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # df_yeo_surf.to_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv', index=False)

    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    # Compute quantile thresholds
    low_thresh = np.nanquantile(df_yeo_surf['t1_gradient1_SalVentAttn'].values, 0.25)
    high_thresh = np.nanquantile(df_yeo_surf['t1_gradient1_SalVentAttn'].values, 1 - 0.25)
    # Identify vertex indices for extremes
    df_yeo_surf.loc[np.where(df_yeo_surf['t1_gradient1_SalVentAttn'].values <= low_thresh)[0], 'quantile_idx'] = -1
    df_yeo_surf.loc[np.where(df_yeo_surf['t1_gradient1_SalVentAttn'].values >= high_thresh)[0], 'quantile_idx'] = 1
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['quantile_idx'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    func_files = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
    func = np.vstack([zscore(nib.load(f).darrays[0].data, axis=0) for f in func_files[:5] if nib.load(f).darrays[0].data.shape[0] == 275])
    func_bottom = zscore(func[:, df_yeo_surf['quantile_idx'].eq(-1)].mean(axis=1), axis=0)
    func_top = zscore(func[:, df_yeo_surf['quantile_idx'].eq(1)].mean(axis=1), axis=0)
    func_sal = zscore(func[:, df_yeo_surf['network'].eq('SalVentAttn')].mean(axis=1), axis=0)
    corr_bottom = np.arctanh(np.dot(func_bottom, func) / (len(func_bottom) - 1))
    corr_top = np.arctanh(np.dot(func_top, func) / (len(func_top) - 1))
    corr = np.abs(corr_top) - np.abs(corr_bottom)
    corr[df_yeo_surf['network'].eq('medial_wall')] = np.nan
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=zscore(corr, nan_policy='omit'), size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #     nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    func_network = np.array([zscore(func[:, df_yeo_surf['network'].eq(network)].mean(axis=1), axis=0) for network in df_yeo_surf['network'].unique()]).T
    print(func_network.shape)

    #analyzer = CausalityAnalyzer(func_network)
    #results = analyzer.causality_test(method='granger')
    analyzer = MultivariateGrangerCausality(func_network)
    results = analyzer.multivariate_granger_causality()
    print(results)
    binary_matrix =  create_connectivity_matrix(results, method = 'granger', threshold=0.05) 
    plt.imshow(binary_matrix)
    plt.show()


if __name__ == "__main__":
    main()