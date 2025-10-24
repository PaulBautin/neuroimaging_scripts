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
from tqdm import tqdm
import seaborn as sns
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh
from brainspace.gradient import GradientMaps, kernels

from nctpy.energies import integrate_u, get_control_inputs
from nctpy.pipelines import ComputeControlEnergy, ComputeOptimizedControlEnergy
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    convert_states_str2int,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
)
from nctpy.plotting import roi_to_vtx, null_plot, surface_plot, add_module_lines, set_plotting_params
from nctpy.metrics import ave_control, modal_control

from scipy.integrate import simpson as simps
import scipy as sp


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

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_label = df_label[df_label.network != 'medial_wall']
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf.loc[salience_border == 1, 'network_int'] = 1
    centers = df_label[['label', 'coor.x', 'coor.y', 'coor.z']].values
    np.savetxt("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/centres.txt", centers, delimiter="\t", fmt="%s\t%.6f\t%.6f\t%.6f")

    #### Load connectivity matrix
    A = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A = np.average(np.array([nib.load(f).darrays[0].data for f in A[:2]]), axis=0)
    A = np.log(np.triu(A,1) + A.T + 1)
    A = A[49:, 49:]
    A = np.delete(np.delete(A, 200, axis=0), 200, axis=1)
    np.savetxt("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/weights.txt", A, delimiter="\t")

    dist = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii')
    distance_matrix = np.average(np.array([nib.load(f).darrays[0].data for f in dist[:2]]), axis=0)
    distance_matrix = distance_matrix[49:, 49:]
    distance_matrix = np.delete(np.delete(distance_matrix, 200, axis=0), 200, axis=1)
    np.savetxt("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/tract_lengths.txt", A, delimiter="\t")


    A_norm = matrix_normalization(A, system='continuous', c=1)

    from scipy.ndimage import rotate
    # --- Sort nodes by network ---
    node_networks = df_label['network'].values
    sort_idx = np.argsort(node_networks)
    print(node_networks[sort_idx])
    A_sorted = A_norm[sort_idx][:, sort_idx]
    sorted_networks = node_networks[sort_idx]
    import matplotlib.patches as patches

    # --- Find network boundaries ---
    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries,0,0)

    mpc_fig = A_sorted.copy()
    mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)
    fig, ax = plt.subplots()
    # Overlay borders
    b_ext = np.append(boundaries,400)
    for i, b in enumerate(boundaries):
        rect = patches.Rectangle((200 * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_colors.colors[i], facecolor='none', angle=45)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.imshow(mpc_fig, cmap='Blues', origin='upper')
    plt.axis('off')
    plt.title('Upper Triangle of MPC Rotated by 45°')
    plt.tight_layout()
    plt.show()

    dist = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC001/ses-01/dwi/connectomes/sub-PNC001_ses-01_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii')
    distance_matrix = np.average(np.array([nib.load(f).darrays[0].data for f in dist[:2]]), axis=0)
    distance_matrix = distance_matrix[49:, 49:]
    distance_matrix = np.delete(np.delete(distance_matrix, 200, axis=0), 200, axis=1)

    # define time system
    system = 'continuous'
    ac = ave_control(A_norm, system=system)  # get average controllability
    df_label['ac'] = normalize_to_range(ac, 0, 2)
    #df_label.loc[df_label.network != 'SalVentAttn', 'ac'] = np.nan
    mc = modal_control(A_norm)  # get modal controllability
    df_label['mc'] = normalize_to_range(mc, 0, 2)
    #df_label.loc[df_label.network != 'SalVentAttn', 'mc'] = np.nan

    df_yeo_surf = df_yeo_surf.merge(df_label[['label', 'ac','mc']], on='label', validate="many_to_one", how='left')
    df_yeo_surf.loc[np.isnan(df_yeo_surf['network_int'].values),'network_int'] = 0
    print(df_yeo_surf)
    df_yeo_surf.loc[df_yeo_surf['network_int'].values == 1,'network_int'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['network_int'].values == 0,'network_int'] = 1
    df_yeo_surf['ac'] = np.nan_to_num(df_yeo_surf['ac'].values, nan=0) * df_yeo_surf['network_int']
    df_yeo_surf['mc'] = np.nan_to_num(df_yeo_surf['mc'].values, nan=0) * df_yeo_surf['network_int']
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['ac'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                        nan_color=(0, 0, 0, 1), cmap='Purples', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['mc'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
                    nan_color=(0, 0, 0, 1), cmap='Purples', transparent_bg=True)
    
        #df = pd.read_json('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/output_clusters.json', orient='records')
    clusters = pd.read_json("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/output_clusters.json")

    # Flatten the "nodes" list into rows
    df_cluster = (clusters
        .explode("nodes")  # one row per node
        .assign(label=lambda d: d["nodes"].apply(lambda x: x["id"]))[["id", "label"]]
        .rename(columns={"id": "community"})
        .groupby("label")["community"]
        .apply(list)
        .reset_index())

    for n_max in range(1, df_cluster["community"].explode().max() + 2):
        df_cluster["community_" + str(n_max)] = df_cluster["community"].apply(lambda x: n_max if isinstance(x, (list, tuple)) and n_max-1 in x else np.nan)

    print(df_cluster)


    #df_label = df_label.drop(columns='network').merge(df_cluster, on='label', how='left')
    df_yeo_surf = df_yeo_surf.merge(df_cluster, on='label', validate="many_to_one", how='left')
    
    # print(df_yeo_surf)
    # values = np.array([df_yeo_surf[col].values for col in df_yeo_surf.columns if col.startswith('community_')])
    # values[np.tile(df_yeo_surf.network_int.values, (df_cluster["community"].explode().max()+1,1)) == 1] = 0
    # print(values)
    # cmap = plt.cm.tab10.copy()
    # cmap.set_under('black')
    # # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=values, size=(600, 600), zoom=1.3, color_bar='bottom', share='both',
    # #         nan_color=(220, 220, 220, 1), cmap='inferno', transparent_bg=True)
    


    

    ######### Part 2 -- BigBrain
    ### Load the data from BigBrain 
    data_bigbrain = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    data_bigbrain_coord_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-mid_coord_bigbrain.func.gii').darrays[1].data
    data_bigbrain_coord_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-mid_coord_bigbrain.func.gii').darrays[1].data
    data_bigbrain[:, np.asarray(df_yeo_surf['network'] == 'medial_wall')] = np.nan
    salience_bigbrain = data_bigbrain[:, np.asarray(df_yeo_surf['network'] == 'SalVentAttn')]
    data_bigbrain[:, np.asarray(df_yeo_surf['network'] != 'SalVentAttn')] =np.nan
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=data_bigbrain.mean(axis=0), size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #                         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
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
    print(gm_bigbrain.lambdas_)

    df_yeo_surf.loc[df_yeo_surf.network == 'SalVentAttn', 'bigbrain_gradient1'] = normalize_to_range(gm_bigbrain.gradients_[:, 0], -1, 1)
    df_yeo_surf.loc[df_yeo_surf.network == 'SalVentAttn', 'bigbrain_gradient2'] = normalize_to_range(gm_bigbrain.gradients_[:, 1], -1, 1)
    df_yeo_surf['bigbrain_gradient1_mean'] = df_yeo_surf.groupby('label')['bigbrain_gradient1'].transform('mean')
    df_yeo_surf['bigbrain_gradient2_mean'] = df_yeo_surf.groupby('label')['bigbrain_gradient2'].transform('mean')
    print(df_yeo_surf)


    from matplotlib.patches import Wedge
    import re
    import matplotlib.colors as mcolors
    from scipy.stats import pearsonr, spearmanr, linregress

    def community_key(col_name):
        """Extract numeric suffix from community column for sorting."""
        match = re.search(r'community_(\d+)', col_name)
        return int(match.group(1)) if match else -1

    def scatter_pie_chart(df, x_col, y_col, radius=None, cmap=plt.cm.inferno, corr_method='pearson'):
        """
        Plot a scatter pie chart for community columns in df, with correlation analysis and regression line.
        
        Args:
            df (pd.DataFrame): DataFrame with x, y, and community_* columns
            x_col (str): Name of column for x-axis
            y_col (str): Name of column for y-axis
            radius (float): Pie chart radius. Auto-scaled if None
            cmap (matplotlib colormap): Colormap for wedges
            corr_method (str): 'pearson' or 'spearman' correlation
        """
        # Get all community columns
        comm_cols = sorted(
            [col for col in df.columns if col.startswith('community_')],
            key=lambda c: int(c.split('_')[1])  # assumes 'community_N' format
        )
        if not comm_cols:
            raise ValueError("No community_* columns found in DataFrame.")

        # Check coordinate columns
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found.")

        # Auto radius if not set
        if radius is None:
            x_range = df[x_col].max() - df[x_col].min()
            y_range = df[y_col].max() - df[y_col].min()
            radius = 0.02 * max(x_range, y_range)  # 2% of plot size

        # Correlation analysis
        valid_rows = df[[x_col, y_col]].dropna()
        if corr_method == 'pearson':
            corr_val, p_val = pearsonr(valid_rows[x_col], valid_rows[y_col])
        elif corr_method == 'spearman':
            corr_val, p_val = spearmanr(valid_rows[x_col], valid_rows[y_col])
        else:
            raise ValueError("corr_method must be 'pearson' or 'spearman'")

        # Regression line (based on Pearson only)
        slope, intercept, _, _, _ = linregress(valid_rows[x_col], valid_rows[y_col])

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, row in df.iterrows():
            x, y = row[x_col], row[y_col]
            values = np.array([row[c] for c in comm_cols], dtype=float)
            values = values[~np.isnan(values)]
            values_bin = (values > 0).astype(int)

            total = values_bin.sum()
            if total <= 0:
                continue  # skip empty points

            proportions = values_bin / total
            start_angle = 90
            for i, prop in enumerate(proportions):
                if prop > 0:
                    end_angle = start_angle + 360 * prop
                    wedge = Wedge(
                        center=(x, y), r=radius,
                        theta1=start_angle, theta2=end_angle,
                        facecolor=cmap(values[i] / len(comm_cols)),
                        edgecolor='white', lw=0.5
                    )
                    ax.add_patch(wedge)
                    start_angle = end_angle

        # Plot regression line
        x_vals = np.array([df[x_col].min(), df[x_col].max()])
        ax.plot(x_vals, intercept + slope * x_vals, '--', color='black', lw=1)

        # Annotate correlation results
        ax.text(
            0.02, 0.98,
            f"{corr_method.capitalize()} r = {corr_val:.3f}, p = {p_val:.3e}",
            transform=ax.transAxes,
            verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        ax.set_aspect('equal')
        ax.autoscale()
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Scatter Pie Chart with Correlation Analysis")
        plt.show()




    # Example usage
    scatter_pie_chart(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'][df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True), x_col='bigbrain_gradient1', y_col='ac')
    scatter_pie_chart(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'][df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True), x_col='bigbrain_gradient1', y_col='mc')
    scatter_pie_chart(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'][df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True), x_col='bigbrain_gradient2', y_col='ac')
    scatter_pie_chart(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'][df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True), x_col='bigbrain_gradient2', y_col='mc')




    



    ##### Network control theory
    n_nodes = A.shape[0]
    control_tasks = []
    control_set = np.eye(n_nodes)
    trajectory_constraints = np.eye(n_nodes)
    rho = 1

    initial_state = normalize_state(states == 1)
    target_state = normalize_state(states == 0)
    
    state_trajectory, u, _ = get_control_inputs(A_norm=A_norm, T=1, B=control_set,
                                x0=initial_state, xf=target_state, system="continuous",
                                rho=rho, S=trajectory_constraints, xr="zero")
    
    energy = np.sum(integrate_u(u))
    
    from netneurotools.networks import strength_preserving_rand_sa, strength_preserving_rand_sa_mse_opt, randmio_und, match_length_degree_distribution


    # run permutation
    n_perms = 30  # number of permutations
    energy_null = np.zeros(n_perms)

    for perm in tqdm(np.arange(n_perms)):
        # rewire adjacency matrix
        #A_null = A
        _, A_null, _ = match_length_degree_distribution(A, distance_matrix)

        # compute control energy for Wsp
        A_norm_null = matrix_normalization(A_null, system='continuous', c=1)
        _, control_signals, _ = get_control_inputs(
            A_norm=A_norm_null,
            T=1,
            B=control_set,
            x0=initial_state,
            xf=target_state,
            system='continuous',
            rho=rho,
            S=trajectory_constraints,
        )
        node_energy = integrate_u(control_signals)
        energy_null[perm] = np.sum(node_energy)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 3))  # ax is a single object, not an array
    null_plot(
        observed=energy,
        null=energy_null,
        xlabel="Strength-preserving null",
        ax=ax
    )
    ax.set_title("Energy vs Null Distribution", fontsize=12, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)  # Light dashed grid
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    plt.show()

    


    # plt.plot(state_trajectory)
    # plt.show()
    # timepoints_to_plot = np.arange(0, state_trajectory.shape[0], int(state_trajectory.shape[0] / 5))

    # for timepoint in timepoints_to_plot:
    #     df_label['timepoint'] = state_trajectory[timepoint]
    #     plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')['timepoint'].values,
    #             layout_style='grid', size=(600, 600), zoom=1.3,
    #             color_bar='bottom', share='both',
    #             nan_color=(220, 220, 220, 1),
    #             cmap="Purples", transparent_bg=True)
        
    # print(df_yeo_surf)



    n_states = len(state_labels)
    for initial_idx in np.arange(n_states):
        initial_state = normalize_state(states == initial_idx)  # initial state
        for target_idx in np.arange(n_states):
            target_state = normalize_state(states == target_idx)  # target state
            control_task = dict()  # initialize dict
            control_task["x0"] = initial_state  # store initial state
            control_task["xf"] = target_state  # store target state
            control_task["B"] = control_set  # store control set
            control_task["S"] = trajectory_constraints  # store state trajectory constraints
            control_task["rho"] = rho  # store rho
            control_tasks.append(control_task)
    compute_control_energy = ComputeControlEnergy(A=A_norm, control_tasks=control_tasks, system="continuous", c=1, T=1)
    compute_control_energy.run()
    energy_matrix = np.reshape(compute_control_energy.E, (n_states, n_states))

    from netneurotools.networks import strength_preserving_rand_sa
    
    ##### Network control theory
    A_norm[df_label.network == 'SalVentAttn'][:, df_label.network == 'SalVentAttn'], min_energy = strength_preserving_rand_sa(A_norm[df_label.network == 'SalVentAttn'][:, df_label.network == 'SalVentAttn'])
    print(min_energy)
    n_nodes = A.shape[0]
    control_tasks = []
    control_set = np.eye(n_nodes)
    trajectory_constraints = np.eye(n_nodes)
    rho = 1
    n_states = len(state_labels)
    for initial_idx in np.arange(n_states):
        initial_state = normalize_state(states == initial_idx)  # initial state
        for target_idx in np.arange(n_states):
            target_state = normalize_state(states == target_idx)  # target state
            control_task = dict()  # initialize dict
            control_task["x0"] = initial_state  # store initial state
            control_task["xf"] = target_state  # store target state
            control_task["B"] = control_set  # store control set
            control_task["S"] = trajectory_constraints  # store state trajectory constraints
            control_task["rho"] = rho  # store rho
            control_tasks.append(control_task)
    compute_control_energy = ComputeControlEnergy(A=A_norm, control_tasks=control_tasks, system="continuous", c=1, T=1)
    compute_control_energy.run()
    energy_matrix_rewired = np.reshape(compute_control_energy.E, (n_states, n_states))    
    energy_matrix_delta = energy_matrix - energy_matrix_rewired

    f, ax = plt.subplots(1, 3, figsize=(14, 8))
    mask = np.zeros_like(energy_matrix)
    mask[np.eye(n_states) == 1] = True
    sns.heatmap(
        energy_matrix,
        ax=ax[0],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy", "shrink": 0.25},
        mask=mask,
    )
    # plot without self-transitions
    sns.heatmap(
        energy_matrix_rewired,
        ax=ax[1],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy", "shrink": 0.25},
        mask=mask,
    )
    # plot energy asymmetries
    mask = np.triu(np.ones_like(energy_matrix, dtype=bool))
    sns.heatmap(
        energy_matrix_delta,
        ax=ax[2],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy (delta)", "shrink": 0.25},
        mask=mask,
        cmap="RdBu_r",
        center=0,
    )
    for cax in ax:
        cax.set_ylabel("initial state (x0)")
        cax.set_xlabel("target state (xf)")
        cax.set_yticklabels(state_labels, rotation=0, size=6)
        cax.set_xticklabels(state_labels, rotation=90, size=6)
    f.tight_layout()
    plt.show()





    


if __name__ == "__main__":
    main()