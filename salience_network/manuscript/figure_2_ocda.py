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
import oslom
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh
from brainspace.gradient import GradientMaps, kernels


def connectivity_to_oslom_edges(conn_matrix: np.ndarray, node_labels: list, output_file: str, weight_threshold: float = 0.0):
    """
    Convert a NumPy connectivity matrix to OSLOM-compatible edge list format.
    
    Parameters
    ----------
    conn_matrix : np.ndarray
        Square matrix (N x N) of connectivity values.
    node_labels : list
        List of node identifiers (strings or integers) of length N.
    output_file : str
        Path to save the edges file.
    weight_threshold : float, optional
        Minimum absolute weight to include an edge (default=0.0).
    """
    if conn_matrix.shape[0] != conn_matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    if len(node_labels) != conn_matrix.shape[0]:
        raise ValueError("Number of node labels must match matrix size.")
    
    # Create edge list (i, j, weight) excluding diagonal
    edges = []
    for i in range(conn_matrix.shape[0]):
        for j in range(i+1, conn_matrix.shape[1]):  # upper triangular to avoid duplicates
            weight = conn_matrix[i, j]
            if abs(weight) > weight_threshold:
                edges.append((node_labels[i], node_labels[j], weight))
    
    # Save as TAB-separated file
    df_edges = pd.DataFrame(edges)
    df_edges.to_csv(output_file, sep='\t', header=False, index=False)

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

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf.loc[salience_border == 1, 'network_int'] = 1
    print(df_yeo_surf)


    #### Load connectivity matrix
    A = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A = glob.glob('/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC*/ses-01/dwi/connectomes/sub-HC*_ses-01_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A = np.array([nib.load(f).darrays[0].data for f in A[:]])
    A = np.log(A + 1)
    A = np.average(A, axis=0)
    A = np.triu(A,1) + A.T
    A = A[49:, 49:]
    A = np.delete(np.delete(A, 200, axis=0), 200, axis=1)
    A = A[200:, 200:]
    print(A.shape)
    plt.imshow(A)
    plt.show()

    df_label = df_label[df_label.network != 'medial_wall']
    node_labels = df_label[df_label.hemisphere != 'LH']['label'].values

    connectivity_to_oslom_edges(A, node_labels, "/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/avg_edges.tsv")

    # run OSLOM with files already on disk
    args = Namespace()
    args.edges = "/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/avg_edges.tsv"
    args.output_clusters = "/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/output_clusters.json"
    args.oslom_output = "/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/oslom_aux_files"
    args.oslom_exec = "/local_raid/data/pbautin/software/OSLOM2/OSLOM2/oslom_undir"
    args.min_cluster_size = 0
    args.oslom_args = ["-w", "-r", "100", "-hr", "50", "-t", "0.3"]
    oslom.run(args)

    # 
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
    
    print(df_yeo_surf)
    values = np.array([df_yeo_surf[col].values for col in df_yeo_surf.columns if col.startswith('community_')])
    if values.shape[0] % 2 != 0:
        values = np.vstack((values, np.zeros(values.shape[1])))
    #values[np.tile(df_yeo_surf.network_int.values, (df_cluster["community"].explode().max()+1,1)) == 1] = 0
    print(values.shape)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=values, size=(600, 300), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap='inferno')
    


    

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

    def community_key(col_name):
        """Extract numeric suffix from community column for sorting."""
        match = re.search(r'community_(\d+)', col_name)
        return int(match.group(1)) if match else -1

    def scatter_pie_chart(df, x_col, y_col, radius=None, cmap=plt.cm.inferno):
        """
        Plot a scatter pie chart for community columns in df.
        Automatically scales radius if not provided.
        """
        # Get all community columns
        comm_cols = sorted(
            [col for col in df.columns if col.startswith('community_')],
            key=community_key
        )
        if not comm_cols:
            raise ValueError("No community_* columns found in DataFrame.")

        # Check coordinates
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found.")

        # Auto radius if not set
        if radius is None:
            x_range = df[x_col].max() - df[x_col].min()
            y_range = df[y_col].max() - df[y_col].min()
            radius = 0.02 * max(x_range, y_range)  # 2% of plot size

        fig, ax = plt.subplots(figsize=(8, 6))

        for idx, row in df.iterrows():
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
                        facecolor=cmap(values[i] / 6), edgecolor='white', lw=0.5
                    )
                    ax.add_patch(wedge)
                    start_angle = end_angle

        ax.set_aspect('equal')
        ax.autoscale()
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()



    # Example usage
    scatter_pie_chart(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'][df_yeo_surf.hemisphere == 'RH'].groupby(['label']).mean(numeric_only=True), x_col='bigbrain_gradient1', y_col='bigbrain_gradient2')

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