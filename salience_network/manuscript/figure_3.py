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
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh

from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

from brainspace.gradient import GradientMaps, kernels
import scipy
from joblib import Parallel, delayed

from scipy.spatial import cKDTree


from brainspace.vtk_interface import wrap_vtk
from brainspace.plotting.base import Plotter
from vtkmodules.vtkFiltersSources import vtkSphereSource

from scipy.signal import welch

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

import bct.algorithms as bct_alg
import bct.utils as bct


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
], dtype=float) / 255  # Normalize to 0–1
# Optional alpha channel for transparency
alpha = np.ones((7, 1))  # All fully opaque
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


def main():
    #### load the conte69 hemisphere surfaces and spheres
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)
    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    # Load fsLR-5k inflated surface
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.midthickness.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.midthickness.surf.gii', itype='gii')


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
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=state, size=(600, 600), zoom=1.3, color_bar='bottom', share='both',
    #             nan_color=(220, 220, 220, 1), cmap='CustomCmap_yeo', transparent_bg=True, background=(0,0,0), layout_style='grid')
    
    #### Load connectivity matrix
    C = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    C = np.average(np.array([nib.load(f).darrays[0].data for f in C[:2]]), axis=0)
    C = np.log(np.triu(C,1)+C.T + 1)
    C = C[49:, 49:]
    C = np.delete(np.delete(C, 200, axis=0), 200, axis=1)
    N = C.shape[0] # Number of nodes
    C[np.eye(N, dtype=bool)] = 0
    C = C / np.mean(C[~np.eye(N, dtype=bool)])

    #### tractogram
    # load the conte69 hemisphere surfaces and spheres
    from scipy.spatial import KDTree
    fsLR_lh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    fsLR_rh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    coords = np.concatenate([fsLR_lh, fsLR_rh])
    tree = KDTree(coords)
    tractogram_f = nib.streamlines.load('/local_raid/data/pbautin/data/dTOR_full_tractogram.trk')
    tractogram = tractogram_f.streamlines
    endpoints = [(sl[0], sl[-1]) for sl in tractogram]

    vertex_values = state.copy()
    vertex_values[salience != 0] = np.nan
    cutoff = 5.0
    n_vertices = len(salience)
    vertex_val = np.zeros(n_vertices)

    streamline_values = []
    for start_pt, end_pt in endpoints:
        dist_start, idx_start = tree.query(start_pt)
        dist_end, idx_end = tree.query(end_pt)
        start_val = vertex_values[idx_start] if dist_start <= cutoff else np.nan
        end_val = vertex_values[idx_end] if dist_end <= cutoff else np.nan
        sl_val = np.nanmax([start_val, end_val])
        streamline_values.append(sl_val)
        # Map onto salience vertices if either endpoint maps to them
        if dist_start <= cutoff and salience[idx_start] != 0 and ~np.isnan(sl_val):
            vertex_values[idx_start] = sl_val
        if dist_end <= cutoff and salience[idx_end] != 0 and ~np.isnan(sl_val):
            vertex_values[idx_end] = sl_val
    
    # Normalize: average per vertex
    vertex_val[vertex_val == 0] = np.nan
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=vertex_values, size=(600, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap='CustomCmap_yeo', transparent_bg=True, background=(0,0,0), layout_style='grid')

    streamline_values = np.asarray(streamline_values)
    streamline_values[np.isnan(streamline_values)] = 0
    mask = (streamline_values != 0)
    streamline_arr = (streamline_values - np.min(streamline_values)) / (np.max(streamline_values) - np.min(streamline_values))
    filtered_tractogram = [tractogram[i] for i in range(len(tractogram)) if mask[i]]
    filtered_streamline_arr = streamline_arr[mask]
    new_tractogram = nib.streamlines.Tractogram(filtered_tractogram, affine_to_rasmm=tractogram_f.tractogram.affine_to_rasmm)
    color = (yeo7_colors(filtered_streamline_arr)[:, 0:3] * 255).astype(np.uint8)
    tmp = [np.tile(color[i], (len(new_tractogram.streamlines[i]), 1))
           for i in range(len(new_tractogram.streamlines))]
    new_tractogram.data_per_point['color'] = tmp
    trk_file = nib.streamlines.TrkFile(new_tractogram)
    trk_file.save('/local_raid/data/pbautin/data/output_tractogram_with_dps_3.trk')


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
    # load connectivty matrix 5k
    C_5k = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    C_5k = np.average(np.array([nib.load(f).darrays[0].data for f in C_5k[:2]]), axis=0)
    C_5k = np.log(np.triu(C_5k,1)+C_5k.T + 1)
    N = C_5k.shape[0] # Number of nodes
    C_5k[np.eye(N, dtype=bool)] = 0
    C_5k = C_5k / np.mean(C_5k[~np.eye(N, dtype=bool)])

    

    fsl32k_bigbrain_g1_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G1.shape.gii').darrays[0].data
    fsl32k_bigbrain_g1_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G1.shape.gii').darrays[0].data
    fsl32k_bigbrain_g1 = np.concatenate((fsl32k_bigbrain_g1_lh, fsl32k_bigbrain_g1_rh), axis=0).astype(float)
    fsl32k_bigbrain_g1[np.isnan(state)] = np.nan
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=fsl32k_bigbrain_g1, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
    #                         nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    
    df_yeo_surf['g1_position'] = fsl32k_bigbrain_g1


    ######### Comunication model ############
    from matplotlib.collections import LineCollection
    D = bct_alg.distance_wei(C)[0]
    df_label = df_label[df_label.network != 'medial_wall']
    df_label['g1_position'] = df_yeo_surf.groupby('mics')['g1_position'].mean().drop(1000).drop(2000).values
    df_label['shortest_path_salience'] = np.mean(D[np.where(df_label.network.values ==  'SalVentAttn')[0], :], axis=0)
    # Get network list and assign colors
    print(state_name)
    color_dict = dict(zip(state_name[:-1], yeo7_rgb))

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    x_pos = np.linspace(np.min(df_label['g1_position'].values), np.max(df_label['g1_position'].values),len(np.unique(df_label.network.values)))
    print(x_pos.shape)

    for nb, net in enumerate(state_name[:-1]):
        df_net = df_label[df_label['network'] == net].sort_values('g1_position')
        x = df_net['g1_position'].values
        y = df_net['shortest_path_salience'].values
        
        # Create line segments from (0,0) to each (x,y) point
        segments = [[[x_pos[nb], 0.02], [x[i], y[i]]] for i in range(len(x))]
        line_collection = LineCollection(segments, colors=color_dict[net], linewidths=0.8, alpha=0.2)
        ax.add_collection(line_collection)
        
        # Scatter points
        ax.scatter(x, y, color=color_dict[net], edgecolor='black', s=35, label=net, zorder=3)

    # Aesthetics
    ax.set_xlabel('Cortical Gradient Position (g1)', fontsize=12)
    ax.set_ylabel('Shortest Path to Salience Network', fontsize=12)
    ax.set_title('Shortest Path Length to Salience Network Across Functional Networks', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Adjust axes to include all
    x_all = df_label['g1_position'].values
    y_all = df_label['shortest_path_salience'].values
    ax.set_xlim(min(0, x_all.min()) - 0.1, x_all.max() + 0.1)
    ax.set_ylim(min(0.02, y_all.min()), y_all.max() + 0.01)

    # Legend
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    



    ######### Network control theory ############

    ######### Kuramoto model ############


if __name__ == "__main__":
    main()