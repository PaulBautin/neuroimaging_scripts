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

from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path, dijkstra, laplacian

import bct.algorithms as bct_alg
import bct.utils as bct
import networkx as nx
from sklearn.manifold import SpectralEmbedding
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from pathlib import Path
from scipy.sparse.linalg import cg,splu,lsqr, cgs, gmres, minres
from scipy.sparse import csc_matrix


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
    
    original_min = np.nanmin(data)
    original_max = np.nanmax(data)

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


def plot_spring_graph(G, node_size_by=None, community_labels=None, edge_alpha=0.5, ax=None):
    """Plot a force-directed layout"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12))
    pos = nx.spring_layout(G, weight='weight', seed=42)  # FR layout
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 7])
    # sizes and colors
    if node_size_by is None:
        sizes = 40
    else:
        sizes = [node_size_by.get(i, 0.0)*200 for i in G.nodes()]
    if community_labels is not None:
        colors = [community_labels[i] for i in G.nodes()]
    else:
        colors = 'gray'
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, ax=ax)
    ax.set_axis_off()
    return ax


def run_mapper(A, filter_fn='pca', n_components=2, cover_params=(10, 0.3)):
    """Run a simple Mapper on node feature matrix derived from adjacency A."""
    # feature matrix: rows = nodes, columns = connectivity profile (row of A)
    X = A.copy()
    mapper = km.KeplerMapper(verbose=1)
    if filter_fn == 'pca':
        lens = PCA(n_components=n_components).fit_transform(X)
    else:
        lens = X.mean(axis=1, keepdims=True)  # trivial lens
    cover_n_cubes, cover_overlap = cover_params
    graph = mapper.map(lens, X,
                       cover=km.Cover(n_cubes=cover_n_cubes, perc_overlap=cover_overlap),
                       clusterer=km.cluster.DBSCAN(eps=0.5, min_samples=3))
    return mapper, graph


def plot_force_graph():
    G = nx.from_numpy_array(A[df_valid.hemisphere == "RH", :][:, df_valid.hemisphere == "RH"])  # create undirected graph
    pos = nx.forceatlas2_layout(G, weight='weight', seed=42)
    #pos = nx.spring_layout(G, weight=None, seed=42)
    node_sizes = [G.degree[i] for i in range(len(G.nodes()))]  # scale for plotting

    # Example: color nodes by network label if available
    # Assume df_valid.network aligns with nodes in A
    network_colors = {
        'SalVentAttn': np.array([210, 100, 255]) / 255,
        'Default': np.array([230, 90, 100]) / 255,
        'DorsAttn': np.array([0, 170, 50]) / 255,
        'Limbic': np.array([250, 224, 51]) / 255,
        'Cont': np.array([255, 180, 80]) / 255,
        'Vis': np.array([170, 70, 200]) / 255,
        'SomMot': np.array([100, 160, 220]) / 255
    }
    node_colors = [network_colors.get(n, 'grey') for n in df_valid[df_valid.hemisphere == "RH"].network]

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(10, 10))
    nx.draw(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='lightgrey',
        alpha=0.7,
        width=1
    )
    plt.title("Force-Directed Network from Connectivity Matrix")
    plt.show()



def propagate_dirichlet(L, sn_values, sn_idx):
    """
    Propagate SN values across graph using Laplace equation with Dirichlet condition.

    Parameters
    ----------
    L : (N, N) ndarray
        Graph Laplacian.
    sn_values : (k,) ndarray
        Gradient values on salience nodes.
    sn_idx : list[int]
        Indices of salience nodes.

    Returns
    -------
    u : (N,) ndarray
        Propagated values on all nodes (SN fixed).
    """
    N = L.shape[0]
    all_idx = np.arange(N)
    free_idx = np.setdiff1d(all_idx, sn_idx)

    # Partition Laplacian
    L_ff = L[np.ix_(free_idx, free_idx)]
    L_fb = L[np.ix_(free_idx, sn_idx)]
    
    # RHS = -L_fb * u_b
    rhs = -L_fb @ sn_values
    
    # Solve sparse system
    L_ff_sparse = csc_matrix(L_ff, dtype=np.float64)
    #lu = splu(L_ff_sparse)
    #u_f = lu.solve(rhs.astype(np.float64))
    u_f = lsqr(L_ff_sparse, rhs)[0]

    # Assemble final vector
    u = np.zeros(N)
    u[sn_idx] = sn_values
    u[free_idx] = u_f
    return u


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

    # #### load yeo atlas 7 network fslr5k
    # atlas_yeo_lh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    # atlas_yeo_rh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    # atlas_yeo_rh_5k[atlas_yeo_rh_5k == 1800] = 2000
    # yeo_surf_5k = np.concatenate((atlas_yeo_lh_5k, atlas_yeo_rh_5k), axis=0).astype(float)
    # df_yeo_surf_5k = pd.DataFrame(data={'mics': yeo_surf_5k})

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    # df_yeo_surf_5k = df_yeo_surf_5k.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf['salience_border'] = salience_border == 1
    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv')
    df_label = df_label.merge(df_yeo_surf[['label', 't1_gradient1_salience']].groupby('label').mean(), on='label', how='left')


    # ##### FEM solve laplace equation on surface
    # df_yeo_surf['t1_gradient1_salience'] = normalize_to_range(df_yeo_surf['t1_gradient1_salience'].values, -1, 1)
    # df_yeo_surf.loc[:, 'SalVentAttn_mask'] = df_yeo_surf.network == 'SalVentAttn'
    # # rh
    # mask_rh = (df_yeo_surf.hemisphere == 'RH') & (df_yeo_surf.network != 'medial_wall')
    # mask_rh_sal = mask_rh & (df_yeo_surf.network == 'SalVentAttn')
    # surf_32k_rh = mesh.mesh_operations.mask_points(surf_32k, np.array(mask_rh))
    # tria_32k_rh = TriaMesh(mesh.mesh_elements.get_points(surf_32k_rh), mesh.mesh_elements.get_cells(surf_32k_rh))
    # d_values_rh = df_yeo_surf.loc[mask_rh_sal, 't1_gradient1_salience'].values
    # d_idx_rh = np.where(df_yeo_surf.loc[mask_rh, 'SalVentAttn_mask'].values)[0]
    # u_t_rh = Solver(tria_32k_rh, lump=False).poisson(dtup=(d_idx_rh, d_values_rh))
    # df_yeo_surf.loc[mask_rh, 'heat_f'] = u_t_rh * (1 / np.nanmax(np.abs(u_t_rh)))

    # # lh
    # mask_lh = (df_yeo_surf.hemisphere == 'LH') & (df_yeo_surf.network != 'medial_wall')
    # mask_lh_sal = mask_lh & (df_yeo_surf.network == 'SalVentAttn')
    # surf_32k_lh = mesh.mesh_operations.mask_points(surf_32k, np.array(mask_lh))
    # tria_32k_lh = TriaMesh(mesh.mesh_elements.get_points(surf_32k_lh), mesh.mesh_elements.get_cells(surf_32k_lh))
    # d_values_lh = df_yeo_surf.loc[mask_lh_sal, 't1_gradient1_salience'].values
    # d_idx_lh = np.where(df_yeo_surf.loc[mask_lh, 'SalVentAttn_mask'].values)[0]
    # u_t_lh = Solver(tria_32k_lh, lump=False).poisson(dtup=(d_idx_lh, d_values_lh))
    # df_yeo_surf.loc[mask_lh, 'heat_f'] = u_t_lh

    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['heat_f'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm')
    

    ##### Solve laplace equation on graph
    files = sorted(Path("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0").glob(
        "sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii"))
    A = np.mean([nib.load(f).darrays[0].data for f in files[:10]], axis=0)
    A = np.log((np.triu(A, 1) + np.triu(A, 1).T) + 1)
    # Remove medial wall / NaN labels
    mask_invalid = (df_label.network == "medial_wall") | (df_label.network.isna())
    A = A[~mask_invalid][:, ~mask_invalid]
    df_valid = df_label.loc[~mask_invalid].copy()

    ##### Compute graph theory metrics
    L = bct.weight_conversion(A, 'lengths')
    between_A = bct_alg.centrality.betweenness_wei(L)
    df_valid["between_A"] = between_A
    df_yeo_surf = df_yeo_surf.merge(df_valid.loc[df_label.network != 'medial_wall', ['label', 'between_A']], on='label', validate="many_to_one", how='left')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['between_A'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm')
    
    ##### Solve laplace equation on graph
    files = sorted(Path("/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0").glob(
        "sub-HC*/ses-01/dwi/connectomes/sub-HC*_ses-01_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii"))
    A = np.mean([nib.load(f).darrays[0].data for f in files[:2]], axis=0)
    A = np.log((np.triu(A, 1) + np.triu(A, 1).T) + 1)
    # Remove medial wall / NaN labels
    mask_invalid = (df_label.network == "medial_wall") | (df_label.network.isna())
    A = A[~mask_invalid][:, ~mask_invalid]
    df_valid = df_label.loc[~mask_invalid].copy()

    ##### Compute graph theory metrics
    A = bct.autofix(A)
    L = bct.weight_conversion(A, 'lengths')
    between_mics = bct_alg.centrality.betweenness_wei(L)
    df_valid["between_mics"] = between_mics
    df_yeo_surf = df_yeo_surf.merge(df_valid.loc[df_label.network != 'medial_wall', ['label', 'between_mics']], on='label', validate="many_to_one", how='left')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['between_mics'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm')



    v0 = normalize_to_range(df_valid["t1_gradient1_salience"].values, -1,1)
    df_valid["v0"] = v0
    d_values = v0[df_valid.network == 'SalVentAttn']
    d_idx = np.where((df_valid.network == 'SalVentAttn').values)[0]
    L = laplacian(A)
    v1 = propagate_dirichlet(L, d_values, d_idx)
    #df_valid['v1'] = v1
    df_valid.loc[np.isnan(v0), "v1"] = v1[np.isnan(v0)]

    df_yeo_surf = df_yeo_surf.merge(df_valid.loc[df_label.network != 'medial_wall', ['label', 'v0', 'v1']], on='label', validate="many_to_one", how='left')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['v0'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range='sym')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['v1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')



    ######### Part 1 -- T1 map
    ### Load the data from PNI dataset
    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-5k_desc-intensity_profiles.shape.gii'))
    print("number of files/subjects: {}".format(len(t1_files[:])))
    # t1 profiles (n_subject, n_features, n_vertices)
    t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files[:10]])
    t1_salience_profiles = t1_profiles[:, :, df_yeo_surf_5k['network'].eq('SalVentAttn').to_numpy()]
    t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_profiles, axis=2))]
    gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    df_yeo_surf_5k.loc[df_yeo_surf_5k['network'].eq('SalVentAttn'), 't1_gradient1_salience'] = normalize_to_range(t1_gradients[:, 0], -1, 1)
    df_yeo_surf_5k['index1'] = df_yeo_surf_5k.index

    ###########################

    files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii'))
    A_vertex = np.mean([nib.load(f).darrays[0].data for f in files[:]], axis=0)
    A_vertex = np.log((np.triu(A_vertex, 1) + np.triu(A_vertex, 1).T) + 1)
    # Remove medial wall / NaN labels
    mask_invalid = (df_yeo_surf_5k.network == "medial_wall") | (df_yeo_surf_5k.network.isna())
    A_vertex = A_vertex[~mask_invalid][:, ~mask_invalid]
    df_valid = df_yeo_surf_5k.loc[~mask_invalid].copy()
    
    v0 = df_valid["t1_gradient1_salience"].values
    v0 = normalize_to_range(v0, -1, 1)
    df_valid["v0"] = v0
    d_values = v0[df_valid.network == 'SalVentAttn']
    d_idx = np.where((df_valid.network == 'SalVentAttn').values)[0]
    L_vertex = laplacian(A_vertex)
    v1 = propagate_dirichlet(L_vertex, d_values, d_idx)
    print(v1)
    df_valid.loc[df_valid.network != 'SalVentAttn', 'v1'] = v1[df_valid.network != 'SalVentAttn']
    df_yeo_surf_5k = df_yeo_surf_5k.merge(df_valid[['index1', 'v0', 'v1']], on='index1', how='left', validate="one_to_one")
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['v0'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['v1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')





    # ---------------------
    # Structural gradients
    # ---------------------
    A_rh = A_dij[df_valid.hemisphere == "RH", :][:, df_valid.hemisphere == "RH"]

    sc_cov = partial_corr_with_covariate(A_rh, np.nanmean(A_rh, axis=1))
    gm_sc_cov = GradientMaps(
        n_components=2, approach="dm", kernel="normalized_angle", alignment="procrustes"
    )
    gm_sc_cov.fit(sc_cov, sparsity=0.9)

    df_valid.loc[df_valid.hemisphere == "RH", "sc_gradient1"] = normalize_to_range(
        gm_sc_cov.gradients_[:, 0], -1, 1
    )
    df_valid.loc[df_valid.hemisphere == "RH", "sc_gradient2"] = normalize_to_range(
        gm_sc_cov.gradients_[:, 1], -1, 1
    )
    df_valid.loc[df_valid.hemisphere == "RH", "degree"] = normalize_to_range(
        np.nanmean(A_rh, axis=0), 0, 100
    )

    # ---------------------
    # Plotting helper
    # ---------------------
    def scatter_panel(ax, x, y, sizes, colors, title, cmap=None, norm=None, alpha=1.0):
        sc = ax.scatter(x, y, c=colors, s=sizes, cmap=cmap, norm=norm, alpha=alpha, plotnonfinite=False)
        ax.set_title(title)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        return sc

    # Common variables
    x = df_valid.loc[df_valid.hemisphere == "RH", "sc_gradient1"].values
    y = df_valid.loc[df_valid.hemisphere == "RH", "sc_gradient2"].values
    sizes = df_valid.loc[df_valid.hemisphere == "RH", "degree"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    # ---------------------
    # Panel 1: Network colors with Salience highlighted
    # ---------------------
    colors_net = np.array([
        yeo7_rgba[i.astype(int)]
        for i in convert_states_str2int(df_valid.loc[df_valid.hemisphere == "RH", "network"].values)[0]
    ])
    scatter_panel(axes[0], x, y, sizes, colors_net, "Yeo7 Networks", alpha=0.3)
    colors_sal = colors_net.copy()
    colors_sal[df_valid.loc[df_valid.hemisphere == "RH", "network"].values != "SalVentAttn",-1:] = 0
    scatter_panel(axes[0], x, y, sizes, colors_sal, "Yeo7 Networks", alpha=None)

    # # ---------------------
    # # Panel 2: Average shortest path to Salience
    # # ---------------------
    # mask_sal_RH = (df_valid.network == "SalVentAttn") & (df_valid.hemisphere == "RH")
    # A_sal_dist = np.nanmean(A_dij[np.ix_((df_valid.hemisphere == "RH"), mask_sal_RH.values)], axis=1)
    # scatter_panel(
    #     axes[1], x, y, sizes, A_sal_dist, "Salience Distance",
    #     cmap="Purples", norm=mp.colors.Normalize(vmin=np.nanmin(A_sal_dist), vmax=np.nanmax(A_sal_dist))
    # )

    # ---------------------
    # Panel 3: Propagated microstructure
    # ---------------------
    v0 = df_valid["t1_gradient1_salience"].values
    colors_micro = v0[df_valid.hemisphere == "RH"]
    #colors_micro = df_valid.loc[df_valid.hemisphere == "RH","v1"].values
    axes[1].scatter(x, y, c="grey", s=sizes, alpha=0.3)
    scatter_panel(
        axes[1], x, y, sizes, colors_micro, "Propagated Salience Microstructure",
        cmap="coolwarm", norm=mp.colors.Normalize(vmin=np.nanmin(colors_micro), vmax=np.nanmax(colors_micro))
    )

    # ---------------------
    # Panel 3: Propagated microstructure
    # ---------------------
    v0 = df_valid["t1_gradient1_salience"].values
    colors_micro = v0[df_valid.hemisphere == "RH"]
    colors_micro = df_valid.loc[df_valid.hemisphere == "RH","v1"].values
    #axes[1].scatter(x, y, c="grey", s=sizes, alpha=0.3)
    scatter_panel(
        axes[2], x, y, sizes, colors_micro, "Propagated Salience Microstructure",
        cmap="coolwarm", norm=mp.colors.Normalize(vmin=np.nanmin(colors_micro), vmax=np.nanmax(colors_micro))
    )

    plt.tight_layout()
    plt.locator_params(nbins=3)
    plt.show()


    df_yeo_surf = df_yeo_surf.merge(df_valid.loc[df_label.network != 'medial_wall', ['label', 'sc_gradient1', 'sc_gradient2', 'degree']], on='label', validate="many_to_one", how='left')


    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sc_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['sc_gradient2'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['degree'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm')
    
    X = A.copy()
    X = StandardScaler().fit_transform(X)      # normalize features
    X = PCA(n_components=20).fit_transform(X)  # reduce to 20D
    lens = X[:, [0]]   # just the first PCA component as lens
    mapper = km.KeplerMapper(verbose=1)
    lens = X[:, 0].reshape(-1,1)
    mapper = km.KeplerMapper(verbose=1)
    graph = mapper.map(
        lens,
        X,
        cover=km.Cover(n_cubes=10, perc_overlap=0.3),
        clusterer=KMeans(n_clusters=3, random_state=42)
    )
    mapper.visualize(graph, path_html="/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/mapper_salience.html")

    #### spring embedding
    G = nx.from_numpy_array(A)
    colors = [yeo7_rgba[i.astype(int)] for i in convert_states_str2int(df_label.loc[(df_label.network != 'medial_wall') & (~df_label.network.isna()), 'network'].values)[0]]
    plot_spring_graph(G, node_size_by=None, community_labels=colors, edge_alpha=0.5, ax=None)
    plt.show()

if __name__ == "__main__":
    main()