from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Effective connectivity of the salience network
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
import nibabel as nib
import pandas as pd
import seaborn as sns


from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels

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


# Generate sample time series
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_rh.label.gii').darrays[0].data + 1950
atlas_yeo_rh[atlas_yeo_rh == 1950] = 2000
yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
print(np.unique(df_yeo_surf['mics'].values))

surf_lh, surf_rh = load_conte69()


# load .csv associated with schaefer 400
df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-100_mics.csv')
print(np.unique(df_label['mics'].values))
df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
print(df_label)
df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
print(df_yeo_surf)
state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
salience = state.copy()
salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
state_stack = np.vstack([state, salience])
#plot_hemispheres(surf_lh, surf_rh, array_name=state_stack, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)



func = nib.load('/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/func/desc-me_task-rest_run-2_bold/surf/sub-Pilot014_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
# Assign transposed time series (converting NumPy arrays to lists if necessary)
df_yeo_surf['timecourse'] = list(func.T)
print(df_yeo_surf)

# Group by 'network' and aggregate time series (e.g., averaging per network)
df_grouped = df_yeo_surf.groupby('network')['timecourse'].apply(lambda x: np.mean(np.vstack(x), axis=0))
print(df_grouped)

# Define networks and spacing
networks = df_grouped.index[::-1]  # Reverse order for better stacking
offset = 6  # Vertical spacing between ridges

# Define a color palette
colors = sns.color_palette("CustomCmap_yeo", len(networks))  # Choose a color palette

# Set up figure
plt.figure(figsize=(12, 6))
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})  # Transparent background

# Loop over networks and plot each with an offset
for i, network in enumerate(networks):
    timepoints = np.arange(len(df_grouped[network]))  # Get time axis
    signal = df_grouped[network] + i * offset  # Apply vertical offset

    # Use color from the palette for the current network
    color = colors[i]

    sns.lineplot(x=timepoints, y=signal, label=network, lw=1.5, color=color)  # Line plot

    # Fill under the curve for joyplot effect
    plt.fill_between(timepoints, signal, i * offset, color=color, alpha=0.3)

    # Add network label at the right end of the plot
    plt.text(timepoints.max() + 2, i * offset, network, ha="left", fontsize=12, color=color)

# Aesthetic adjustments
plt.xlabel("Timepoints")
plt.ylabel("Signal Intensity (Offset)")
plt.yticks([])  # Hide y-axis labels
plt.title("Joyplot of Average Time Courses per Network", fontsize=14)
plt.legend([], [], frameon=False)  # Hide legend (labels are already in plot)
plt.grid(False)
plt.show()






    # # load .csv associated with schaefer 400
    # df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-100_mics.csv')
    # df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    # df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    # #print(np.unique(df_yeo_surf[df_yeo_surf.network == 'SalVentAttn'].label.str.strip('7Networks_LRH_SalVentAttn_').values))
    # state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    # state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    # salience = state.copy()
    # salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    # state_stack = np.vstack([state, salience])
    # plot_hemispheres(surf_lh, surf_rh, array_name=state_stack, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                 nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)