from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Dynamics of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


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

from brainspace.null_models import SpinPermutations
from scipy.stats import spearmanr


from brainspace.gradient import GradientMaps, kernels
import scipy
from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression
from functools import partial

from scipy.spatial import KDTree


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


def project_streamline_values_to_surface(streamlines, values, surf_vertices, sigma=1):
    """
    Project streamline values onto cortical surface by nearest endpoint mapping 
    with Gaussian weighting over nearby vertices.

    Parameters
    ----------
    streamlines : list of (N_i, 3) arrays
        List of streamlines, each array contains 3D coordinates.
    values : np.ndarray
        Array of shape (n_streamlines,) with per-streamline values.
    surf_vertices : np.ndarray
        Array of shape (n_vertices, 3) with surface vertex coordinates.
    sigma : float
        Standard deviation for Gaussian weighting in mm.

    Returns
    -------
    vertex_map : np.ndarray
        Shape (n_vertices,), aggregated streamline values.
    """
    tree = KDTree(surf_vertices)
    vertex_map = np.zeros(len(surf_vertices))
    weight_sum = np.zeros(len(surf_vertices))  # to normalize weighted contributions

    for sl, val in zip(streamlines, values):
        for endpoint in [sl[0], sl[-1]]:  # project both ends
            # Find k nearest vertices for smooth weighting (e.g., k=5)
            dists, idx = tree.query(endpoint, k=10)
            dists = np.atleast_1d(dists)
            idx = np.atleast_1d(idx)
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            weights /= weights.sum()

            vertex_map[idx] += weights * val
    return vertex_map


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
    # plt_values = df_yeo_surf['network_int'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    #### load econo atlas Hardcoded based on table data in Garcia-Cabezas (2021)
    econo_surf_lh = nib.load(micapipe + '/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load(micapipe + '/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.concatenate((econo_surf_lh, econo_surf_rh), axis=0).astype(float)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])[[0] + list(range(2, 45))]
    df_yeo_surf['surf_type'] = relabel(econo_surf, econ_ctb).astype(float)
    plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #             nan_color=(0, 0, 0, 1), cmap='CustomCmap_type', transparent_bg=True)


        #### tractogram
    # load the conte69 hemisphere surfaces and spheres

    fsLR_lh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    fsLR_rh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    coords = np.concatenate([fsLR_lh, fsLR_rh])
    tree = KDTree(coords)

    # --- Load tractogram ---
    tractogram_f = nib.streamlines.load('/local_raid/data/pbautin/data/output_tractogram_salience.trk')
    tractogram = tractogram_f.streamlines
    endpoints = [(sl[0], sl[-1]) for sl in tractogram]

    # --- Vertex labels ---
    g1_bigbrain_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    g1_bigbrain_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    g1_bigbrain = np.concatenate((g1_bigbrain_lh, g1_bigbrain_rh), axis=0).astype(float)
    df_yeo_surf['g1_bigbrain'] = normalize_to_range(g1_bigbrain, -1, 1)
    vertex_values = df_yeo_surf['network_int'].values
    #vertex_values = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')['t1_gradient1_salience'].values
    vertex_values = df_yeo_surf['g1_bigbrain'].values
    #print(df)
    salience_mask = df_yeo_surf.network == 'SalVentAttn'
    vertex_values[salience_mask] = np.nan

    # --- Assign streamline labels based on endpoints ---
    cutoff = 2.0  # mm tolerance for endpoint assignment

    # Extract start and end points
    starts = np.array([pt[0] for pt in endpoints])
    ends   = np.array([pt[1] for pt in endpoints])

    streamline_values = []

    for start_pt, end_pt in zip(starts, ends):
        # Find nearest surface vertices
        dist_start, idx_start = tree.query(start_pt)
        dist_end, idx_end     = tree.query(end_pt)

        # Assign vertex values if within cutoff
        start_val = vertex_values[idx_start] if dist_start <= cutoff else np.nan
        end_val   = vertex_values[idx_end]   if dist_end   <= cutoff else np.nan

        # Keep only streamlines with at least one endpoint in salience network (label 4)
        values_arr = np.array([start_val, end_val])
        if ~np.isnan(values_arr).all():
            val = np.nanmax(values_arr)
        else:
            val = np.nan
        streamline_values.append(val)

    # --- Filter streamlines ---
    streamline_values = np.asarray(streamline_values)
    surf_proj = project_streamline_values_to_surface(tractogram, streamline_values, coords, sigma=1)
    surf_proj[df_yeo_surf.network != 'SalVentAttn'] = np.nan
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=surf_proj, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    surf_proj = project_streamline_values_to_surface(tractogram, streamline_values, coords, sigma=0.5)
    surf_proj[df_yeo_surf.network != 'SalVentAttn'] = np.nan
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=surf_proj, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)

    print(surf_proj.shape)

    mask = ~np.isnan(streamline_values)
    filtered_streamlines = [s for s, keep in zip(tractogram, mask) if keep]
    filtered_values = streamline_values[mask]

    # --- Normalize values for colormap ---
    norm = plt.Normalize(vmin=np.min(filtered_values), vmax=np.max(filtered_values))
    cmap = mpl.colormaps['coolwarm']
    colors_val = [np.array(cmap(norm(i))[:3]) * 255 for i in filtered_values]  # use RGB only (NiBabel expects 3 columns)
    print(colors_val)

    # --- Build per-point color arrays ---
    data_per_point = {
        "color": [np.tile(c, (len(sl), 1)) for sl, c in zip(filtered_streamlines, colors_val)]
    }

    # --- Build new tractogram ---
    new_tractogram = nib.streamlines.Tractogram(
        filtered_streamlines,
        data_per_point=data_per_point,
        affine_to_rasmm=tractogram_f.tractogram.affine_to_rasmm
    )

    # --- Save to file ---
    trk_file = nib.streamlines.TrkFile(new_tractogram, tractogram_f.header)
    nib.streamlines.save(trk_file, "/local_raid/data/pbautin/data/output_tractogram_salience_gradient.trk")





if __name__ == "__main__":
    main()