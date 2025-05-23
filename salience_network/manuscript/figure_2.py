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
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation

from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

from brainspace.gradient import GradientMaps, kernels
import scipy
from joblib import Parallel, delayed


#### set custom plotting parameters
params = {'font.size': 14}
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
    [0, 0, 0],     # Visual black
], dtype=float) / 255  # Normalize to 0–1
# Optional alpha channel for transparency
alpha = np.ones((8, 1))  # All fully opaque
yeo7_rgba = np.hstack((yeo7_rgb, alpha))
yeo7_colors = mp.colors.ListedColormap(yeo7_rgba)
mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)

## Von econo, cortical types colors
cmap_types_rgb = np.array([
    [127, 140, 172],  # desaturated blue-gray
    [139, 167, 176],  # desaturated cyan-gray
    [171, 186, 162],  # muted green
    [218, 198, 153],  # dull yellow
    [253, 211, 200],  # pale coral
    [252, 229, 252],  # pale magenta
    [0, 0, 0],     # Visual black
], dtype=float) / 255
# Optional alpha channel for transparency
alpha = np.ones((7, 1))
cmap_types_rgba = np.hstack((cmap_types_rgb, alpha))
cmap_types = mp.colors.ListedColormap(cmap_types_rgba)
mp.colormaps.register(name='CustomCmap_type', cmap=cmap_types)

## Von econo, cortical types colors with medial wall
cmap_types_rgb_mw = np.array([
    [127, 140, 172],  # desaturated blue-gray
    [139, 167, 176],  # desaturated cyan-gray
    [171, 186, 162],  # muted green
    [218, 198, 153],  # dull yellow
    [253, 211, 200],  # pale coral
    [252, 229, 252],  # pale magenta
    [220, 220, 220],  # pale gray
], dtype=float) / 255
# Optional alpha channel for transparency
alpha = np.ones((7, 1))
cmap_types_rgba_mw = np.hstack((cmap_types_rgb_mw, alpha))
cmap_types_mw = mp.colors.ListedColormap(cmap_types_rgba_mw)


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


def main():
    #### load the conte69 hemisphere surfaces and spheres
    surf_32k = load_conte69(join=True)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')


    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = 0
    salience_border = array_operations.get_labeling_border(surf_32k, salience)
    state[salience_border == 1] = 7
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=state, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                    nan_color=(220, 220, 220, 1), cmap='CustomCmap_yeo', transparent_bg=True)

    #### load econo atlas
    econo_surf_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.hstack(np.concatenate((econo_surf_lh, econo_surf_rh), axis=0)).astype(float)
    # Hardcoded based on table data in Garcia-Cabezas (2021)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])
    econ_ctb = econ_ctb[[0] + list(range(2, 45))]
    surf_type = relabel(econo_surf, econ_ctb)
    surf_type[surf_type == 0] = np.nan
    surf_type[salience_border == 1] = 7
    print(np.unique(surf_type))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=surf_type, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 1), cmap='CustomCmap_type', transparent_bg=True)
    surf_type_lh = surf_type[:32492]
    surf_type_rh = surf_type[32492:]

    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))


    
    #### Load the data from the specified text file T1 map
     #### 7T network gradient
    # load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan



if __name__ == "__main__":
    main()