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

from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as snsct

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


# def compute_weighted_matrix(fmri_data, structural_connections):
#     """
#     Compute a weighted connectivity matrix using a regression-based framework.
#
#     Args:
#         fmri_data (numpy.ndarray): A 2D array of shape (n_regions, n_timepoints)
#                                    containing the fMRI BOLD activity for each region over time.
#         structural_connections (numpy.ndarray): A binary structural connectivity matrix
#                                                 of shape (n_regions, n_regions)
#                                                 where an entry (i, j) is 1 if regions i and j are connected, else 0.
#
#     Returns:
#         numpy.ndarray: A weighted connectivity matrix of shape (n_regions, n_regions),
#                        where weights are assigned using regression.
#     """
#     n_regions, n_timepoints = fmri_data.shape
#     weighted_matrix = np.zeros_like(structural_connections, dtype=float)
#     fmri_data = (fmri_data - np.mean(fmri_data, axis=1, keepdims=True)) / np.std(fmri_data, axis=1, keepdims=True, ddof=1)
#
#     binary_structural_connections = structural_connections > 0
#
#     # Prepare the regression inputs
#     X = []
#     for n in range(n_regions):
#         neighbours_n = np.where(binary_structural_connections[n] == 1)[0]
#         X.append(np.sum(fmri_data[neighbours_n, :-1], axis=0))  # Neighbors' states at time t-1
#     X = np.array(X).T
#     print(X.shape)
#     y = fmri_data[:, 1:].T  # Region i's state at time t
#
#     # Perform linear regression
#     model = LinearRegression(fit_intercept=True)
#     model.fit(X, y)
#     print(model.coef_.shape)
#
#
#     # Assign weights to the connectivity matrix
#     #weighted_matrix[:, i] = model.coef_ + model.intercept_
#     #print(weighted_matrix.shape)
#     return model.coef_ + model.intercept_

def compute_weighted_matrix(fmri_data, structural_connections):
    """
    Compute a weighted connectivity matrix using a regression-based framework.

    Args:
        fmri_data (numpy.ndarray): A 2D array of shape (n_regions, n_timepoints)
                                   containing the fMRI BOLD activity for each region over time.
        structural_connections (numpy.ndarray): A binary structural connectivity matrix
                                                of shape (n_regions, n_regions)
                                                where an entry (i, j) is 1 if regions i and j are connected, else 0.

    Returns:
        numpy.ndarray: A weighted connectivity matrix of shape (n_regions, n_regions),
                       where weights are assigned using regression.
    """
    n_regions, n_timepoints = fmri_data.shape
    weighted_matrix = np.zeros_like(structural_connections, dtype=float)

    # Z-score fMRI data per region
    # fmri_data = (fmri_data - np.mean(fmri_data, axis=1, keepdims=True)) / \
    #             np.std(fmri_data, axis=1, keepdims=True, ddof=1)

    binary_structural_connections = structural_connections > 0

    # Fit regression for each region
    for i in range(n_regions):
        neighbours_i = np.where(binary_structural_connections[i] == 1)[0]
        if len(neighbours_i) == 0:
            continue  # Skip regions with no neighbors

        # Neighbors' states (features)
        X_i = np.sum(fmri_data[neighbours_i, :-1], axis=0).T  # Sum neighbors' states at time t-1
        print(X_i.reshape(-1, 1).shape)

        # Current region's state (target)
        y_i = fmri_data[i, 1:]  # Shape: (n_timepoints - 1,)
        print(y_i.shape)

        # Perform regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X_i.reshape(-1, 1), y_i)  # Reshape X_i to 2D as expected by sklearn

        # Assign weights to the connectivity matrix
        weighted_matrix[neighbours_i, i] = model.coef_

    return weighted_matrix


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

    return states.astype(int), state_labels




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
    cmap_types = mp.colors.ListedColormap(np.array([
                            [127, 140, 172, 255],
                            [139, 167, 176, 255],
                            [171, 186, 162, 255],
                            [218, 198, 153, 255],
                            [253, 211, 200, 255],
                            [252, 229, 252, 255],
                            [252, 229, 252, 255]])/255)
    mp.colormaps.register(name='CustomCmap', cmap=cmap_types)


    # load the conte69 hemisphere surfaces and spheres
    surf_lh, surf_rh = load_conte69()
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    fsLR_lh = '/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii'
    fsLR_rh = '/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii'

    in_tractogram='/local_raid/data/pbautin/data/dTOR_full_tractogram_full.tck'
    cifti_file='/local_raid/data/pbautin/downloads/ones.dscalar.nii'

    cifti_atlas_dscalar = '/local_raid/data/pbautin/downloads/Schaefer2018_400Parcels_7Networks_order.dscalar.nii'
    cifti_atlas = '/local_raid/data/pbautin/downloads/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'

    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 2000
    yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)

    func = nib.load('/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/func/desc-me_task-rest_run-2_bold/surf/sub-Pilot014_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
    func_label = reduce_by_labels(func, yeo_surf, red_op='mean', axis=0)


    # connectome = css.map_high_resolution_structural_connectivity(in_tractogram, fsLR_lh, fsLR_rh, threshold=5, cifti_file=cifti_atlas)
    # connectome = css.map_high_resolution_structural_connectivity(in_tractogram, fsLR_lh, fsLR_rh, threshold=5, cifti_file=cifti_atlas_dscalar)
    # conn_labeled = np.log(reduce_by_labels(reduce_by_labels(connectome.toarray(), yeo_surf, red_op='sum'), yeo_surf, red_op='sum', axis=1)+1)
    # np.save('/local_raid/data/pbautin/data/conn_labeled.npy', conn_labeled)
    conn_labeled = np.load('/local_raid/data/pbautin/data/conn_labeled.npy')
    print(conn_labeled.shape)
    A_norm = matrix_normalization(conn_labeled, system='continuous', c=1)

    weighted_matrix = compute_weighted_matrix(func_label.T, conn_labeled)
    weighted_matrix = weighted_matrix
    print(weighted_matrix)
    max_val = np.max(np.abs(weighted_matrix))
    plt.imshow(weighted_matrix, cmap='coolwarm', aspect='auto', vmin=-max_val, vmax=max_val)
    plt.colorbar(label="Weight")
    plt.title("Asymmetric, Weighted, and Signed Connectivity Matrix")
    plt.xlabel("Target Region")
    plt.ylabel("Source Region")
    plt.show()

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
    df_label.fillna('other', inplace=True)




    ############
    D = bct_alg.distance_wei(conn_labeled)[0]
    plt.imshow(D[np.where(df_label.label.values ==  'SalVentAttn')[0], :])
    df_label['shortest_path'] = np.mean(D[np.where(df_label.label.values ==  'SalVentAttn')[0], :], axis=0)
    df_label.to_csv('/local_raid/data/pbautin/data/out.csv')
    dist_surf = map_to_labels(np.mean(D[np.where(df_label.label.values ==  'SalVentAttn')[0], :], axis=0), yeo_surf)
    plot_hemispheres(surf_lh, surf_rh, array_name=dist_surf, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='Purples', transparent_bg=True)

    # Load and preprocess data
    data = df_label.drop('mics', axis=1).set_index('label').drop('other', axis=0).reset_index()

    # # Calculate mean for ordering
    # data['shortest_path_mean'] = data.groupby('label')['shortest_path'].transform('mean')
    #
    # # Sort labels by mean for consistent order
    # data['label'] = pd.Categorical(data['label'],
    #                                categories=data.groupby('label')['shortest_path'].mean().sort_values().index,
    #                                ordered=True)

    # Create the ridge plot on the same axes
    plt.figure(figsize=(12, 8))

    palette = sns.color_palette('CustomCmap_yeo', n_colors=len(data['label'].unique()))

    for i, (label, subset) in enumerate(data.groupby('label')):
        sns.kdeplot(
            data=subset,
            x='shortest_path',
            fill=True,
            bw_adjust=1,
            alpha=0.7,
            color=palette[i],
            label=label  # Add label for legend
        )

    # Customize the plot
    plt.xlabel("Shortest Path")
    plt.yticks(range(len(data['label'].unique())), data['label'].unique())
    plt.ylabel("Label")
    plt.title("Ridge Plots Overlayed on Same Axes")

    # Add legend
    plt.legend(title="Labels", loc='upper right', bbox_to_anchor=(1, 1))

    sns.despine(left=True, bottom=True)
    plt.show()



    # Load the data from the specified text file
    data = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
    base = np.ones((50, 64984))
    data_entropy = stats.entropy(data, base, axis=0)

    ############ Minimum State-to-state energy
    T = 1   # time horizon
    n_nodes = A_norm.shape[0]
    B = np.eye(n_nodes) # set all nodes to control nodes
    print(df_label)
    x0_mat = normalize_state(df_label.label.values ==  'SalVentAttn')

    dict_state_energy = {}
    for name in np.unique(df_label.label.values)[:-1]:
        print(name)
        xf_mat = normalize_state(df_label.label.values ==  name)
        #x, u, error = get_control_inputs(A_norm, T, B, x0_mat, xf_mat, system='continuous', rho=1, S='identity', xr='zero', expm_version='scipy')
        e_fast = minimum_energy_fast(A_norm=A_norm, T=T, B=B, x0=x0_mat, xf=xf_mat)[...,0]
        e_fast = np.sum(e_fast)  # sum over nodes
        dict_state_energy.update({name:[e_fast]})

    print(dict_state_energy)
    df_state_energy = pd.DataFrame.from_dict(dict_state_energy, orient='columns')
    print(df_state_energy)
    sns.barplot(data=df_state_energy, palette='CustomCmap_yeo')
    plt.show()



    #################################
    # load econo atlas
    econo_surf_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.hstack(np.concatenate((econo_surf_lh, econo_surf_rh), axis=0)).astype(float)
    # Hardcoded based on table data in Garcia-Cabezas (2021)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])
    econ_ctb = econ_ctb[[0] + list(range(2, 45))]
    #mask = econo_surf != 0
    surf_type = map_to_labels(econ_ctb.astype(float), econo_surf)#, mask=mask, fill=np.nan
    surf_type[surf_type == 0] = np.nan
    print(np.unique(surf_type))
    surf_type_lh = surf_type[:32492]
    surf_type_rh = surf_type[32492:]


    # surf_type_arr = [surf_type[surf_type != i] == 0 for i in np.unique(surf_type)]
    # surf_type_arr = np.array(surf_type_arr[0])
    # print(surf_type_arr.shape)
    def surf_type_isolation(surf_type_test, i):
        # Work on a copy of the input array to avoid modifying the original
        surf_type_copy = surf_type_test.copy()
        surf_type_copy[surf_type_copy != i] = np.nan
        return surf_type_copy

    # Generate a new array with isolated surf types
    print(np.unique(surf_type))
    surf_type_arr = np.array([surf_type_isolation(surf_type, i) for i in np.unique(surf_type)])
    print(np.unique(surf_type_arr))

    # plot_hemispheres(surf_lh, surf_rh, array_name=surf_type_arr[:6], size=(800, 1400), share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='CustomCmap', transparent_bg=True)
    # plot_hemispheres(surf_lh, surf_rh, array_name=surf_type, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='CustomCmap', transparent_bg=True)

    #####################################33
    # load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 2000
    yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
    print(np.unique(yeo_surf))
    # Concatenate micro gradient and shaeffer parecellation
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
    df_label.fillna('other', inplace=True)
    print(df_label)
    df = pd.DataFrame(data={'mics': yeo_surf})
    df = df.merge(df_label, on='mics',validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df['label'].values)
    print(np.unique(state_name))
    state = state.astype(float)

    # network composition
    surf_type[np.isnan(surf_type)] = 7
    unique_vals = np.unique(surf_type)
    salience_composition = {val: [] for val in unique_vals}
    dict_state_composition = {}
    dict_state = {}
    for s_name in state_name:
        dict_value ={}
        # isolate state
        val = state == np.where(state_name == s_name)[0][0]
        val = val.astype(float)
        val[val == 0] = np.nan
        dict_state.update({s_name:val})
        # find composition
        composition = surf_type[~np.isnan(val)] * val[~np.isnan(val)]
        unique, counts = np.unique(composition, return_counts=True)
        counts = (counts / len(composition)) * 100
        counts_dict = dict(zip(unique, counts))
        for value in unique_vals:
            dict_value.update({value:counts_dict.get(value, 0)})
            dict_state_composition.update({s_name:dict_value})

    print("hererr")
    print("dict {}".format(dict_state_composition))

    # composition = surf_type[~np.isnan(salience)] * salience[~np.isnan(salience)]
    # unique, counts = np.unique(composition, return_counts=True)
    # counts = (counts / len(composition)) * 100
    # counts_dict_real = dict(zip(unique, counts))


    salience = state == np.where(state_name == 'SalVentAttn')[0][0]
    salience = salience.astype(float)
    salience[salience == 0] = np.nan
    salience_lh = salience.astype(float)[:32492]
    salience_rh = salience.astype(float)[32492:]
    # salience = salience.astype(float) *  np.where(state_name == 'SalVentAttn')[0][0]
    # salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    print(salience)
    # state[state == 7] = np.nan
    # print(np.unique(state))
    # #print(np.where(state_name == 'SalVentAttn'))
    # #salience = df['label'].values == 'SalVentAttn'
    # #salience = salience.astype(float)
    #
    # # print(salience.shape)


    composition = data[:,~np.isnan(salience)] * salience[~np.isnan(salience)]
    print(composition.shape)
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(composition.T)
    print(gm.gradients_[:, 0].shape)
    colors = plt.cm.coolwarm((gm.gradients_[:, 0] + np.abs(np.min(gm.gradients_[:, 0]))) / (np.max(gm.gradients_[:, 0]) + np.abs(np.min(gm.gradients_[:, 0]))))
    for i in range(len(composition[0,:])):
        plt.plot(range(0,50), composition[:,i], color=colors[i])
    plt.show()


    arr = np.zeros(64984)
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = gm.gradients_[:, 0] / 0.00187
    plot_hemispheres(surf_lh, surf_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

    #
    # yeo7_colors = mp.colors.ListedColormap(np.array([
    #                         [0, 118, 14, 255],
    #                         [230, 148, 34, 255],
    #                         [205, 62, 78, 255],
    #                         [120, 18, 134, 255],
    #                         [220, 248, 164, 255],
    #                         [70, 130, 180, 255],
    #                         [196, 58, 250, 255]]) / 255)
    # mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)
    # state = np.vstack([state, salience])
    # plot_hemispheres(surf_lh, surf_rh, array_name=state, size=(1200, 600), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)
    # plot_hemispheres(surf_lh, surf_rh, array_name=salience, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                          nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)



    # Let's create some rotations
    n_rand = 10

    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere_lh, points_rh=sphere_rh)

    surf_type[np.isnan(surf_type)] = 7
    salience_rotated = np.hstack(sp.randomize(salience_lh, salience_rh))


    composition = surf_type[~np.isnan(salience)] * salience[~np.isnan(salience)]
    unique, counts = np.unique(composition, return_counts=True)
    counts = (counts / len(composition)) * 100
    counts_dict_real = dict(zip(unique, counts))


    unique_vals = np.unique(surf_type)
    #unique_vals = unique_vals[~np.isnan(unique_vals)]  # Remove NaN values from unique keys
    salience_composition = {val: [] for val in unique_vals}

    for n in range(n_rand):
        composition = surf_type[~np.isnan(salience_rotated[n])] * salience_rotated[n][~np.isnan(salience_rotated[n])]
        unique, counts = np.unique(composition, return_counts=True)
        counts = (counts / len(composition)) * 100

        counts_dict = dict(zip(unique, counts))
        for val in unique_vals:
            salience_composition[val].append(counts_dict.get(val, 0))


    # Convert dictionary to DataFrame for plotting
    data = pd.DataFrame(salience_composition)
    # Melt the DataFrame to long format
    data_melted = data.melt(var_name='Key', value_name='Count')
    print(data_melted)
    #x_arr = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag']

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Key', y='Count', data=data_melted, color='white',fill=False)#,vert=False
    sns.scatterplot(x=range(0,7), y=counts_dict_real.values(), color=cmap_types.colors)
    plt.title('Salience Network Composition')
    plt.xlabel('Cortical Types')
    plt.ylabel('Percentage')
    plt.show()



if __name__ == "__main__":
    main()
