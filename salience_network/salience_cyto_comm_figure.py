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

params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')


# Load and preprocess data
df = pd.read_csv('/local_raid/data/pbautin/data/out.csv', index_col=[0]).drop('mics', axis=1).set_index('label').drop('other', axis=0).reset_index()

# Melt data for ridge plots
data = df#.reset_index().melt(id_vars='index', var_name="label", value_name="shortest_path")

# Calculate mean for ordering
data['shortest_path_mean'] = data.groupby('label')['shortest_path'].transform('mean')

# Sort labels by mean for consistent order
data['label'] = pd.Categorical(data['label'],
                               categories=data.groupby('label')['shortest_path'].mean().sort_values().index,
                               ordered=True)

# Create the ridge plot on the same axes
plt.figure(figsize=(12, 8))
yeo7_colors = mp.colors.ListedColormap(np.array([
                        [0, 118, 14, 255],
                        [230, 148, 34, 255],
                        [205, 62, 78, 255],
                        [120, 18, 134, 255],
                        [220, 248, 164, 255],
                        [70, 130, 180, 255],
                        [196, 58, 250, 255]]) / 255)
mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)
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
