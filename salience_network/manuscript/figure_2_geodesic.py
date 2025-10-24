from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Ooverlapping community detection algorithms (OCDAs) code
# conda activate env_ocda
# 
# Changes to make in github code:
# AttributeError: 'dict' object has no attribute 'iteritems': change .iteritems for .items
# AttributeError: module 'itertools' has no attribute 'izip_longest': change izip_longest for zip_longest
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
from argparse import Namespace
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh
from brainspace.gradient import GradientMaps, kernels

from lapy import Solver, TriaMesh, heat
from lapy import diffgeo
import matplotlib as mp

from scipy.stats import pearsonr, spearmanr, linregress
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path, dijkstra

import bct.algorithms as bct_alg
import bct.utils as bct
import networkx as nx

## Yeo 2011, 7 network colors
yeo7_rgb = np.array([
    [255, 180, 80],    # Frontoparietal (brighter orange)
    [230, 90, 100],    # Default Mode (brighter red)
    [0, 170, 50],      # Dorsal Attention (more vivid green)
    [250, 224, 51],   # Limbic (lighter yellow-green)
    [210, 100, 255],   # Ventral Attention (lighter purple)
    [100, 160, 220],   # Somatomotor (lighter blue)
    [170, 70, 200],     # Visual (brighter violet)
    [0, 0, 0],     # Visual black
], dtype=float) / 255  # Normalize to 0–1
# Optional alpha channel for transparency
alpha = np.ones((8, 1))  # All fully opaque
yeo7_rgba = np.hstack((yeo7_rgb, alpha))
yeo7_colors = mp.colors.ListedColormap(yeo7_rgba)

def compute_surface_eigenmodes(mesh: TriaMesh, k: int = 200):
    """
    Compute first k non-trivial Laplace–Beltrami modes and eigenvalues.
    Returns (evals [k], evecs [V,k]), and mass matrix diag [V].
    """
    solver = Solver(mesh)        # builds stiffness C and mass M (FEM)
    evals, evecs = solver.eigs(k=k+1)  # includes the constant mode (~0)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    # Drop the first (constant) eigenpair
    evals, evecs = evals[1:k+1], evecs[:, 1:k+1]
    return evals, evecs


def eigen_decomposition(data, eigenmodes, method='matrix'):
    """
    Decompose data using eigenmodes and calculate the coefficient of 
    contribution of each vector
    
    Parameters:
    -----------
    data : np.ndarray of shape (n_vertices, 3)
        N = number of vertices, P = columns of independent data
    eigenmodes : np.ndarray of shape (n_vertices, M)
        N = number of vertices, M = number of eigenmodes
    method : string
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'
    
    Returns:
    -------
    coeffs : numpy array of shape (N, 3)
     coefficient values
    
    """
    
    if data.ndim > 1:
        N, P = data.shape
    else:
        P = 1
    
    _, M = eigenmodes.shape
    
    if method == 'matrix':
        #print("Using matrix decomposition to reconstruct data")
        coeffs = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data))
    elif method == 'matrix_separate':
        coeffs = np.zeros((M, P))
        if P > 1:
            for p in range(P):
                coeffs[:, p] = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data[:, p]))
    elif method == 'regression':
        #print("Using regression method to reconstruct data")
        coeffs = np.zeros((M, P))
        if P > 1:
            for p in range(P):
                coeffs[:, p] = np.linalg.lstsq(eigenmodes, data[:, p], rcond=None)[0]
        else:
            coeffs = np.linalg.lstsq(eigenmodes, data, rcond=None)[0]
            
    else:
        raise ValueError("Accepted methods for decomposition are 'matrix', and 'regression'")
                
    return coeffs


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
    
    original_min = np.min(data)
    original_max = np.max(data)

    if original_min == original_max: # Handle cases where all values are the same
        return np.full_like(data, (target_min + target_max) / 2)

    # Normalize to 0-1 range first
    normalized_0_1 = (data - original_min) / (original_max - original_min)

    # Scale to the target range
    scaled_data = target_min + (normalized_0_1 * (target_max - target_min))
    return scaled_data

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


def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    #### load yeo atlas 7 network fslr5k
    atlas_yeo_lh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh_5k[atlas_yeo_rh_5k == 1800] = 2000
    yeo_surf_5k = np.concatenate((atlas_yeo_lh_5k, atlas_yeo_rh_5k), axis=0).astype(float)
    df_yeo_surf_5k = pd.DataFrame(data={'mics': yeo_surf_5k})

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    df_yeo_surf_5k = df_yeo_surf_5k.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf['salience_border'] = salience_border == 1


    ##########################################################################
    ####################### ANALYSIS #########################################
    ######### Part 1 -- T1 map
    ### Load the data from PNI dataset
    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-5k_desc-intensity_profiles.shape.gii'))
    print("number of files/subjects: {}".format(len(t1_files[:])))
    ## t1 profiles (n_subject, n_features, n_vertices)
    # t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files[:10]])
    # t1_salience_profiles = t1_profiles[:, :, df_yeo_surf_5k['network'].eq('SalVentAttn').to_numpy()]
    # t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_profiles, axis=2))]
    # gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    # gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    # t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    # print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    # df_yeo_surf_5k.loc[df_yeo_surf_5k['network'].eq('SalVentAttn'), 't1_gradient1_salience'] = normalize_to_range(t1_gradients[:, 0], -1, 1)
    # df_yeo_surf_5k.loc[df_yeo_surf_5k['network'].eq('SalVentAttn'), 't1_gradient2_salience'] = normalize_to_range(t1_gradients[:, 1], -1, 1)
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['t1_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['t1_gradient2_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    
    # A_vertex = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    # A_vertex = np.average(np.array([nib.load(f).darrays[0].data for f in A_vertex[:]]), axis=0)
    # A_vertex = np.log(np.triu(A_vertex,1) + A_vertex.T + 1)
    # A_vertex = np.delete(np.delete(A_vertex, (df_yeo_surf_5k.network == 'medial_wall') | (df_yeo_surf_5k.network.isna()), axis=0), (df_yeo_surf_5k.network == 'medial_wall') | (df_yeo_surf_5k.network.isna()), axis=1)
    # df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall', 'A_sup05'] = np.nanmean(A_vertex[df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall','t1_gradient1_salience'].values > np.nanpercentile(df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall','t1_gradient1_salience'].values, 75),:], axis=0)
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['A_sup05'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='Purples')
    # df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall', 'A_inf05'] = np.nanmean(A_vertex[df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall','t1_gradient1_salience'].values < np.nanpercentile(df_yeo_surf_5k.loc[df_yeo_surf_5k.network != 'medial_wall','t1_gradient1_salience'].values, 25),:], axis=0)
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['A_inf05'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='Purples')




    # df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')
    print(df_yeo_surf)

    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    g1_bigbrain_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    g1_bigbrain_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    g1_bigbrain = np.concatenate((g1_bigbrain_lh, g1_bigbrain_rh), axis=0).astype(float)
    df_yeo_surf['g1_bigbrain'] = normalize_to_range(g1_bigbrain, -1, 1)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['g1_bigbrain'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #                 nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    #### Load connectivity matrix
    A = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A = np.average(np.array([nib.load(f).darrays[0].data for f in A[:]]), axis=0)
    A = np.log(np.triu(A,1) + A.T + 1)
    #A = np.delete(np.delete(A, df_label.hemisphere.isna(), axis=0), df_label.hemisphere.isna(), axis=1)
    A = np.delete(np.delete(A, (df_label.network == 'medial_wall') | (df_label.network.isna()), axis=0), (df_label.network == 'medial_wall') | (df_label.network.isna()), axis=1)
    np.fill_diagonal(A, 0)
    print(A.shape)
    A[A != 0] = 1 / A[A != 0]
    #A = bct_alg.distance_wei(A)[0]
    #A = csr_array(A)
    #mask = (df_label[(df_label.network != 'medial_wall') & (~df_label.network.isna())].network == 'SalVentAttn') & (df_label[(df_label.network != 'medial_wall') & (~df_label.network.isna())].hemisphere == 'RH')
    A = dijkstra(A, directed=False)
    A[A != 0] = 1 / A[A != 0]
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G)
    #threshold = 2  # keep only edges >= threshold
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 7])
    print(df_label)
    #df_label.loc[df_label.network.isna(), 'network'] = 'NaN'
    colors = [yeo7_rgba[i.astype(int)] for i in convert_states_str2int(df_label.loc[(df_label.network != 'medial_wall') & (~df_label.network.isna()), 'network'].values)[0]]
    print(np.unique(np.array(colors)))
    nx.draw(G, pos, node_color=colors, node_size=50, edge_color='gray')
    plt.title("Spring Layout Graph from Connectivity Matrix")
    plt.axis('scaled')
    plt.show()

    A_vertex = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A_vertex = np.average(np.array([nib.load(f).darrays[0].data for f in A_vertex[:]]), axis=0)
    A_vertex = np.log(np.triu(A_vertex,1) + A_vertex.T + 1)
    A_vertex = np.delete(np.delete(A_vertex, (df_yeo_surf_5k.network == 'medial_wall') | (df_yeo_surf_5k.network.isna()), axis=0), (df_yeo_surf_5k.network == 'medial_wall') | (df_yeo_surf_5k.network.isna()), axis=1)
    # plt.imshow(A_vertex)
    # plt.show()
    A_vertex[A_vertex != 0] = 1 / A_vertex[A_vertex != 0]
    mask = (df_yeo_surf_5k[(df_yeo_surf_5k.network != 'medial_wall') & (~df_yeo_surf_5k.network.isna())].network == 'SalVentAttn') & (df_yeo_surf_5k[(df_yeo_surf_5k.network != 'medial_wall') & (~df_yeo_surf_5k.network.isna())].hemisphere == 'RH')
    A_vertex = dijkstra(A_vertex, directed=False, indices=np.where(mask))[0,::].T
    print(np.unique(~np.isinf(A_vertex).any(axis=1)))
    A_vertex = A_vertex[~np.isinf(A_vertex).any(axis=1),:]
    plt.imshow(A_vertex)
    plt.show()
    
    # n_feature x n_regions
    #A_sal = A[(df_label[(df_label.network != 'medial_wall') & (~df_label.network.isna())].network == 'SalVentAttn') & (df_label[(df_label.network != 'medial_wall') & (~df_label.network.isna())].hemisphere == 'RH'),:][:,df_label[(df_label.network != 'medial_wall') & (~df_label.network.isna())].network != 'medial_wall'].T
    #A_sal = A_vertex[(df_yeo_surf_5k[(df_yeo_surf_5k.network != 'medial_wall') & (~df_yeo_surf_5k.network.isna())].network == 'SalVentAttn') & (df_yeo_surf_5k[(df_yeo_surf_5k.network != 'medial_wall') & (~df_yeo_surf_5k.network.isna())].hemisphere == 'RH'),:][:,df_yeo_surf_5k[(df_yeo_surf_5k.network != 'medial_wall') & (~df_yeo_surf_5k.network.isna())].network != 'medial_wall'].T
    sc_cov = partial_corr_with_covariate(A, np.nanmean(A, axis=1))
    gm_sc_cov = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_sc_cov.fit(sc_cov, sparsity=0.9)
    df_label.loc[(df_label.network == 'SalVentAttn') & (df_label.hemisphere == 'RH'), 'sal_sc_gradient1'] = normalize_to_range(gm_sc_cov.gradients_[:, 0], -1, 1).astype(float)
    df_label.loc[(df_label.network == 'SalVentAttn') & (df_label.hemisphere == 'RH'), 'sal_sc_gradient2'] = normalize_to_range(gm_sc_cov.gradients_[:, 1], -1, 1).astype(float)
    df_label.loc[(df_label.network == 'SalVentAttn') & (df_label.hemisphere == 'RH'), 'sal_sc_gradient3'] = normalize_to_range(gm_sc_cov.gradients_[:, 2], -1, 1).astype(float)
    df_yeo_surf = df_yeo_surf.merge(df_label.loc[df_label.network != 'medial_wall', ['label', 'sal_sc_gradient1', 'sal_sc_gradient2', 'sal_sc_gradient3']], on='label', validate="many_to_one", how='left')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient2'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient3'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    
    sc_cov = partial_corr_with_covariate(A_vertex, np.nanmean(A_vertex, axis=1))
    gm_sc_cov = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_sc_cov.fit(sc_cov, sparsity=0.9)
    df_yeo_surf_5k.loc[(df_yeo_surf_5k.network == 'SalVentAttn') & (df_yeo_surf_5k.hemisphere == 'RH'), 'sal_sc_gradient1'] = normalize_to_range(gm_sc_cov.gradients_[:, 0], -1, 1).astype(float)
    df_yeo_surf_5k.loc[(df_yeo_surf_5k.network == 'SalVentAttn') & (df_yeo_surf_5k.hemisphere == 'RH'), 'sal_sc_gradient2'] = normalize_to_range(gm_sc_cov.gradients_[:, 1], -1, 1).astype(float)
    df_yeo_surf_5k.loc[(df_yeo_surf_5k.network == 'SalVentAttn') & (df_yeo_surf_5k.hemisphere == 'RH'), 'sal_sc_gradient3'] = normalize_to_range(gm_sc_cov.gradients_[:, 2], -1, 1).astype(float)
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['sal_sc_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['sal_sc_gradient2'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['sal_sc_gradient3'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')

    print(df_yeo_surf)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient2'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_sc_gradient3'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')

    for name in df_label[df_label.network == 'SalVentAttn'].label:
        df_label.loc[df_label.network != 'medial_wall', 'sal_dist_' + name] = A[df_label[df_label.network != 'medial_wall'].label == name, df_label[df_label.network != 'medial_wall'].network != 'medial_wall']
    print(df_label)
    df_label.loc[df_label.network != 'medial_wall', 'sal_dist'] = np.nanmean(A[df_label[df_label.network != 'medial_wall'].network == 'SalVentAttn', :], axis=0)
    df_yeo_surf = df_yeo_surf.merge(df_label.loc[df_label.network != 'medial_wall', ['label', 'sal_dist']], on='label', validate="many_to_one", how='left')
    print(df_yeo_surf)

    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == True, 'salience_border'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == False, 'salience_border'] = 1
    df_yeo_surf['sal_dist'] = np.nan_to_num(df_yeo_surf.sal_dist, nan=0) * df_yeo_surf['salience_border'].astype(float)
    df_yeo_surf.loc[df_yeo_surf.hemisphere != 'RH', 'sal_dist'] = 0
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sal_dist'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                    nan_color=(0, 0, 0, 1), cmap='Purples', transparent_bg=True)

    


    ### Right hemisphere surface eigenmodes
    df_yeo_surf.loc[:, 'SalVentAttn_mask'] = df_yeo_surf.network == 'SalVentAttn'
    surf_32k = mesh.mesh_operations.mask_points(surf_32k, np.array(df_yeo_surf.hemisphere == 'RH'))
    tria_32k = TriaMesh(mesh.mesh_elements.get_points(surf_32k), mesh.mesh_elements.get_cells(surf_32k))
    heat_f = heat.diffusion(tria_32k, df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', 'SalVentAttn_mask'].values, m=2)
    geodesic = diffgeo.compute_geodesic_f(tria_32k, heat_f)
    df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', 'geodesic'] = geodesic

    # df_yeo_surf.loc[df_yeo_surf['salience_border'].values == True, 'salience_border'] = np.nan
    # df_yeo_surf.loc[df_yeo_surf['salience_border'].values == False, 'salience_border'] = 1
    # df_yeo_surf['geodesic'] = np.nan_to_num(df_yeo_surf.geodesic, nan=0) * df_yeo_surf['salience_border'].astype(float)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['geodesic'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #                     nan_color=(0, 0, 0, 1), cmap='Purples', transparent_bg=True)

    #evals, evecs = compute_surface_eigenmodes(tria_32k, k=200)
    df_yeo_surf.loc[(df_yeo_surf.hemisphere == 'RH') & (df_yeo_surf.network != 'medial_wall'), 'g1_bigbrain'] = normalize_to_range(df_yeo_surf.loc[(df_yeo_surf.hemisphere == 'RH') & (df_yeo_surf.network != 'medial_wall'), 'g1_bigbrain'], -1, 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, net in enumerate(np.unique(df_yeo_surf.network.values)[:-1]):
        print(net)
        plt.scatter(df_yeo_surf.loc[(df_yeo_surf.hemisphere == 'RH') & (df_yeo_surf.network == net)].groupby(['label']).mean(numeric_only=True).g1_bigbrain.values, 
                    df_yeo_surf.loc[(df_yeo_surf.hemisphere == 'RH') & (df_yeo_surf.network == net)].groupby(['label']).mean(numeric_only=True).sal_dist.values,
                    color=yeo7_colors.colors[i])
        
        # Correlation analysis
    valid_rows = df_yeo_surf[df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True)[['g1_bigbrain', 'sal_dist']].dropna()
    corr_val, p_val = pearsonr(valid_rows['g1_bigbrain'], valid_rows['sal_dist'])

    # Regression line (based on Pearson only)
    slope, intercept, _, _, _ = linregress(valid_rows['g1_bigbrain'], valid_rows['sal_dist'])
    # Plot regression line
    x_vals = np.array([valid_rows['g1_bigbrain'].min(), valid_rows['g1_bigbrain'].max()])
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
    plt.xlabel('g1_bigbrain')
    plt.ylabel('sal_dist')
    plt.title("Scatter Pie Chart with Correlation Analysis")
    plt.show()

if __name__ == "__main__":
    main()