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

    return states.astype(float), state_labels


def ridge_plot(data_dict):
    # 1. Filter out invalid keys (e.g., NaN) or empty arrays
    filtered_dict = {
        k: v for k, v in data_dict.items()
        if not pd.isna(k) and len(v) > 0}

    # 2. Manually build a long-form DataFrame
    #    Each row will have two columns: 'category' and 'value'
    long_dataframes = []
    for key, array_vals in filtered_dict.items():
        # Create a temporary DataFrame for this key
        temp_df = pd.DataFrame({
            'category': [key] * len(array_vals),
            'value': array_vals
        })
        long_dataframes.append(temp_df)

    long_df = pd.concat(long_dataframes, axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Gather sorted categories and assign colors
    unique_cats = sorted(long_df['category'].unique())
    colors = sns.color_palette('CustomCmap_type', n_colors=len(unique_cats))

    # ------------------------------------------------------------------------------
    # 4. Compute and plot each category's KDE on the same axis (no offset).
    # ------------------------------------------------------------------------------
    for i, cat in enumerate(unique_cats):
        subset = long_df.loc[long_df['category'] == cat, 'value'].dropna()
        if len(subset) == 0:
            continue

        # Compute the kernel density estimate (KDE)
        kde = stats.gaussian_kde(subset)
        x_vals = np.linspace(subset.min(), subset.max(), 200)
        y_vals = kde(x_vals)

        # Overlay the distribution on the same axis
        # Use fill_between with alpha to visualize overlapping areas
        ax.fill_between(
            x_vals,
            y_vals,
            color=colors[i],
            alpha=0.7,
            label=f"{cat}"
        )

    # ------------------------------------------------------------------------------
    # 5. Tidy up the plot.
    # ------------------------------------------------------------------------------
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(title="Category")
    plt.tight_layout()
    plt.show()


def violin_plot(data_dict):
    # ---------------------------------------------------------------------
    # 1. Flatten the nested dictionary into a long-form DataFrame.
    # ---------------------------------------------------------------------
    # records = []
    # for outer_key, sub_dict in data_dict.items():
    #     for network_key, arr in sub_dict.items():
    #         # Convert to array in case it's not already
    #         arr = np.asarray(arr, dtype=float)
    #         # Remove NaNs from this array
    #         arr = arr[~np.isnan(arr)]
    #
    #         # Skip empty arrays
    #         if len(arr) == 0:
    #             continue
    #
    #         # Append the group (outer_key), network (inner_key), and each value
    #         for val in arr:
    #             records.append((outer_key, network_key, val))
    #
    # df = pd.DataFrame(records, columns=["group", "network", "value"])
    #
    # # Optional: filter out any "NaN" group or "medial_wall" network if unwanted
    # df = df[(df["group"] != "NaN") & (df["network"] != "medial_wall")]
    df = data_dict
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="value",
        y="network_1",
        hue="network_2",
        errorbar='sd',
        palette='CustomCmap_type',
        orient='h'
    )

    plt.xlabel("Network")
    plt.ylabel("Distance (Mean ± SD)")
    plt.title("")
    plt.legend(title="Cortical types", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()



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
    mp.colormaps.register(name='CustomCmap_type', cmap=cmap_types)

    #### load the conte69 hemisphere surfaces and spheres
    surf_lh, surf_rh = load_conte69()

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    #### load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    network_labels = ['Vis','Default','Cont','DorsAttn','Limbic','SalVentAttn','SomMot','medial_wall']
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    #print(np.unique(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'].label.str.strip('7Networks_LRH_SalVentAttn_').values))
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    state_stack = np.vstack([state, salience])
    # plot_hemispheres(surf_lh, surf_rh, array_name=state_stack, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                 nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    #### load econo atlas
    econo_surf_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.concatenate((econo_surf_lh, econo_surf_rh), axis=0).astype(float)
    # Hardcoded based on table data in Garcia-Cabezas (2021)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])
    econ_types = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'NaN']
    econ_ctb = econ_ctb[[0] + list(range(2, 45))]
    #mask = econo_surf != 0
    surf_type = map_to_labels(econ_ctb.astype(float), econo_surf)#, mask=mask, fill=np.nan
    surf_type[surf_type == 0] = np.nan
    # surf_type_lh = surf_type[:32492]
    # surf_type_rh = surf_type[32492:]
    # plot_hemispheres(surf_lh, surf_rh, array_name=surf_type, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                         nan_color=(250, 250, 250, 1), cmap='CustomCmap_type', transparent_bg=True)

    #### upload connectome
    connectome = nib.load('/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/dwi/connectomes/sub-Pilot014_ses-01_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii').darrays[0].data[48:, 48:]
    connectome = np.triu(connectome, 1) + connectome.T
    connectome_dist = bct_alg.distance_wei_floyd(connectome,transform='inv')[0]
    dist_network = {label:np.nanmean(connectome_dist[np.where(df_label['network'].values == label)[0],:], axis=0) for label in network_labels}
    for label in network_labels:
        df_label['dist_'+label] = dist_network[label]
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    network_dist = [df_yeo_surf['dist_'+label].values for label in network_labels[:-1]]
    print(network_labels[:-1])
    plot_hemispheres(surf_lh, surf_rh, array_name=network_dist, size=(1200, 1200), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                    nan_color=(250, 250, 250, 1), transparent_bg=True, cmap='Purples')


    surf_type[np.isnan(surf_type)] = 7
    unique_vals = np.unique(surf_type).astype(int)
    print(unique_vals)
    salience_dist_cyto = {econ_types[val-1]: {label:df_yeo_surf['dist_'+label].values[np.where(surf_type == val)] for label in network_labels} for val in unique_vals}
    print([np.where(state == np.where(state_name == label)[0][0]) for label in state_name])
    salience_dist_network = {label_1: {label:df_yeo_surf['dist_'+label].values[np.where(state == np.where(state_name == label)[0][0])] for label in state_name} for label_1 in state_name}
    df_dist_network = pd.DataFrame.from_dict(salience_dist_network)
    print(df_dist_network)
    violin_plot(salience_dist_network)
    ridge_plot(salience_dist_cyto)






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
