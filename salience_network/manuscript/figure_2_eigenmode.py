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
from brainspace.null_models.spin import spin_permutations

from lapy import Solver, TriaMesh
import matplotlib as mp

## Yeo 2011, 7 network colors
yeo7_rgb = np.array([
    [255, 180, 80],    # Frontoparietal (brighter orange)
    [230, 90, 100],    # Default Mode (brighter red)
    [0, 170, 50],      # Dorsal Attention (more vivid green)
    [240, 255, 200],   # Limbic (lighter yellow-green)
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


def build_mpc(data):
    """
    Compute Microstructural Profile Covariance (MPC) from input data.

    Parameters:
        data : np.ndarray of shape (features, nodes)
            Microstructural profiles matrix.

    Returns:
        MPC : np.ndarray of shape (nodes, nodes)
            Fisher z-transformed correlation matrix of residualized profiles.
    """
    X = data.copy()  # (features, nodes)
    # Compute covariate: mean profile across nodes
    covar = np.nanmean(X, axis=1, keepdims=True)  # (features, 1)
    # Z-score each node's profile (column-wise)
    X_z = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    X_z = np.nan_to_num(X_z)  # replace potential NaNs
    # Z-score the covariate
    covar_z = (covar - np.nanmean(covar)) / np.nanstd(covar)
    # Design matrix with intercept and covariate
    intercept = np.ones((covar_z.shape[0], 1))
    design_matrix = np.hstack((intercept, covar_z))  # shape: (features, 2)
    # Linear regression via least squares: solve design_matrix @ beta = X_z
    beta, _, _, _ = np.linalg.lstsq(design_matrix, X_z, rcond=None)  # shape: (2, nodes)
    X_hat = design_matrix @ beta  # predicted profiles
    residuals = X_z - X_hat       # residualized profiles
    # Correlation matrix of residuals
    R = np.corrcoef(residuals.T)  # shape: (nodes, nodes)
    # Fisher z-transform
    with np.errstate(divide='ignore', invalid='ignore'):
        MPC = 0.5 * np.log((1 + R) / (1 - R))
        MPC[np.isnan(MPC)] = 0
        MPC[np.isinf(MPC)] = 0
    # Clean up: zero diagonal and enforce symmetry
    np.fill_diagonal(MPC, 0)
    MPC = np.triu(MPC, 1) + np.triu(MPC, 1).T
    return MPC, residuals



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


def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-32k surfaces
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf['salience_border'] = salience_border == 1

    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    data_bigbrain[:, np.array(df_yeo_surf['network'] == 'medial_wall')] = np.nan
    bigbrain_perm_rh = spin_permutations(sphere32k_rh, data_bigbrain.T, n_rep=10)
    bigbrain_perm_rh = np.concatenate((bigbrain_perm_rh,bigbrain_perm_rh),axis=1)
    print(bigbrain_perm_rh.shape)

    ### Right hemisphere surface eigenmodes
    surf_32k = mesh.mesh_operations.mask_points(surf_32k, df_yeo_surf.hemisphere.values == 'RH')
    tria_32k = TriaMesh(mesh.mesh_elements.get_points(surf_32k), mesh.mesh_elements.get_cells(surf_32k))
    evals, evecs = compute_surface_eigenmodes(tria_32k, k=200)

    fig, axes = plt.subplots(1,2,figsize=(12, 6), sharex=True, sharey=True)
    power = {}
    offset = 0.15
    power_net_null = np.zeros((bigbrain_perm_rh.shape[0], 200))
    
    for i, net in enumerate(np.unique(df_yeo_surf.network.values)[:-1]):
        net_bigbrain = data_bigbrain[:, np.asarray(df_yeo_surf['network'] == net)]
        mpc_bigbrain, residuals_bigbrain = build_mpc(net_bigbrain)
        gm_bigbrain = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
        gm_bigbrain.fit(mpc_bigbrain, sparsity=0)
        print(gm_bigbrain.lambdas_)
        df_yeo_surf.loc[df_yeo_surf.network == net, net + 'bigbrain_gradient1'] = normalize_to_range(gm_bigbrain.gradients_[:, 0], -1, 1)
        df_yeo_surf.loc[df_yeo_surf.network == net, net + '_mask'] = 1
        df_yeo_surf[net + 'bigbrain_gradient1'] = np.nan_to_num(df_yeo_surf[net + 'bigbrain_gradient1'].values, nan=0)
        # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[net + 'bigbrain_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        #                 nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
        evecs_net = evecs * np.nan_to_num(df_yeo_surf[df_yeo_surf.hemisphere == 'RH'][net + '_mask'].values[:, np.newaxis], nan=0)
        coeffs_net = eigen_decomposition(df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', net + 'bigbrain_gradient1'], evecs_net, method='matrix')
        power_net = coeffs_net**2
        power_net /= power_net.sum()
        power[net] = power_net
        modes = np.arange(1, len(power_net) + 1)
        y_vals = power_net + i * offset
        axes[0].fill_between(modes, i * offset, y_vals, color=yeo7_colors.colors[i], alpha=0.7)
        axes[0].plot(modes, y_vals, color=yeo7_colors.colors[i], lw=2, label=net)
        axes[0].axhline(y=i * offset, xmin=modes[0], xmax=modes[-1])
        axes[1].scatter(np.average(modes, weights=power_net), i * offset, color=yeo7_colors.colors[i])
        for j in range(bigbrain_perm_rh.shape[0]):
            net_bigbrain = bigbrain_perm_rh[j, np.asarray(df_yeo_surf['network'] == net), :].T
            mpc_bigbrain, residuals_bigbrain = build_mpc(net_bigbrain)
            gm_bigbrain = GradientMaps(n_components=3, random_state=12, approach='dm', kernel='normalized_angle')
            gm_bigbrain.fit(mpc_bigbrain, sparsity=0)
            print(gm_bigbrain.lambdas_)
            df_yeo_surf.loc[df_yeo_surf.network == net, net + 'bigbrain_gradient1'] = normalize_to_range(gm_bigbrain.gradients_[:, 0], -1, 1)
            df_yeo_surf.loc[df_yeo_surf.network == net, net + '_mask'] = 1
            df_yeo_surf[net + 'bigbrain_gradient1'] = np.nan_to_num(df_yeo_surf[net + 'bigbrain_gradient1'].values, nan=0)
            # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[net + 'bigbrain_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            #                 nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
            evecs_net = evecs * np.nan_to_num(df_yeo_surf[df_yeo_surf.hemisphere == 'RH'][net + '_mask'].values[:, np.newaxis], nan=0)
            coeffs_net = eigen_decomposition(df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', net + 'bigbrain_gradient1'], evecs_net, method='matrix')
            power_net = coeffs_net**2
            power_net /= power_net.sum()
            power[net] = power_net
            power_net_null[j,:] = power_net
        power_net = np.mean(power_net_null, axis=0)
        modes = np.arange(1, len(power_net) + 1)
        y_vals = power_net + i * offset
        axes[1].fill_between(modes, i * offset, y_vals, color='grey', alpha=0.3)
        axes[1].plot(modes, y_vals, color='grey', lw=2, label=net)
        axes[1].boxplot(np.average(np.tile(modes, (bigbrain_perm_rh.shape[0], 1)), weights=power_net_null, axis=1), vert=False, positions=[i * offset], widths=0.05)

    axes[0].set_axis_off()
    axes[1].set_axis_off()
    plt.tight_layout()
    plt.show()




    df_yeo_surf.loc[np.isnan(df_yeo_surf['network_int'].values),'network_int'] = 0
    print(df_yeo_surf)
    df_yeo_surf.loc[df_yeo_surf['network_int'].values == 1,'network_int'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['network_int'].values == 0,'network_int'] = 1
    df_yeo_surf['bigbrain_gradient1_mean'] = np.nan_to_num(df_yeo_surf['bigbrain_gradient1_mean'].values, nan=0) * df_yeo_surf['network_int']
    df_yeo_surf['bigbrain_gradient2_mean'] = np.nan_to_num(df_yeo_surf['bigbrain_gradient2_mean'].values, nan=0) * df_yeo_surf['network_int']
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1_mean'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                            nan_color=(0, 0, 0, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient1'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    # # second gradient
    # arr[np.asarray(df_yeo_surf['network'] == 'SalVentAttn')] = gm_bigbrain.gradients_[:, 1]
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bigbrain_gradient2_mean'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(0, 0, 0, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)



    

if __name__ == "__main__":
    main()