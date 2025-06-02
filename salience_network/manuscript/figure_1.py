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


def build_mpc(data, parc=None, idxExclude=None):
    # If no parcellation is provided, MPC will be computed vertexwise
    if parc is None:
        downsample = 0
    else:
        downsample = 1


    # Parcellate input data according to parcellation scheme provided by user
    if downsample == 1:
        uparcel = np.unique(parc)
        I = np.zeros([data.shape[0], len(uparcel)])

        # Parcellate data by averaging profiles within nodes
        for (ii, _) in enumerate(uparcel):

            # Get vertices within parcel
            thisparcel = uparcel[ii]
            tmpData = data[:, parc == thisparcel]
            tmpData[:,np.mean(tmpData) == 0] = 0

            # Define function to find outliers: Return index of values above three scaled median absolute deviations of input data
            # https://www.mathworks.com/help/matlab/ref/isoutlier.html
            def find_outliers(data_vector):
                c = -1 / (np.sqrt(2) * scipy.special.erfcinv(3/2))
                scaled_MAD = c * np.median(np.abs(data_vector - np.median(data_vector)))
                is_outlier = np.greater(data_vector, (3 * scaled_MAD) + np.median(data_vector))
                idx_outlier = [i for i, x in enumerate(is_outlier) if x]
                return idx_outlier

            # Find if there are any outliers in vertex-wise average profile within given parcel
            idx = find_outliers(np.mean(tmpData, axis = 0))
            if len(idx) > 0:
                tmpData[:,idx] = np.nan

            # Average profiles within parcels
            I[:,ii] = np.nanmean(tmpData, axis = 1)

        # Get matrix sizes
        szI = I.shape
        szZ = [len(uparcel), len(uparcel)]

    else:
        I = data
        szI = data.shape
        szZ = np.empty((data.shape[1], data.shape[1]))


    # Build MPC
    if np.isnan(np.sum(I)):

        # Find where are the NaNs
        is_nan = np.isnan(I[1,:])
        problemNodes = [i for i, x in enumerate(is_nan) if x]
        print("")
        print("---------------------------------------------------------------------------------")
        print("There seems to be an issue with the input data or parcellation. MPC will be NaNs!")
        print("---------------------------------------------------------------------------------")
        print("")

        # Fill matrices with NaN for return
        I = np.zeros(szI)
        I[I == 0] = np.nan
        MPC = np.zeros(szZ)
        MPC[MPC == 0] = np.nan

    else:
        problemNodes = 0

        # Calculate mean across columns, excluding mask and any excluded labels input
        I_mask = I
        NoneType = type(None)
        if type(idxExclude) != NoneType:
            for i in idxExclude:
                I_mask[:, i] = np.nan
        I_M = np.nanmean(I_mask, axis = 1)

        # Get residuals of all columns (controlling for mean)
        I_resid = np.zeros(I.shape)
        for c in range(I.shape[1]):
            y = I[:,c]
            x = I_M
            slope, intercept, _, _, _ = scipy.stats.linregress(x,y)
            y_pred = intercept + slope*x
            I_resid[:,c] = y - y_pred

        R = np.corrcoef(I_resid, rowvar=False)

        # Log transform
        MPC = 0.5 * np.log( np.divide(1 + R, 1 - R) )
        MPC[np.isnan(MPC)] = 0
        MPC[np.isinf(MPC)] = 0

        # CLEANUP: correct diagonal and round values to reduce file size
        # Replace all values in diagonal by zeros to account for floating point error
        for i in range(0,MPC.shape[0]):
                MPC[i,i] = 0
        # Replace lower triangle by zeros
        MPC = np.triu(MPC)

    # Output MPC, microstructural profiles, and problem nodes
    return (MPC, I, problemNodes)


def load_mpc(File):
     """Loads and process a MPC"""
     mpc = nib.load(File).darrays[0].data
     mpc = np.triu(mpc,1)+mpc.T
     mpc[~np.isfinite(mpc)] = np.finfo(float).eps
     mpc[mpc==0] = np.finfo(float).eps
     return(mpc)


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
    #surf_type[salience_border == 1] = 7
    print(np.unique(surf_type))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=surf_type, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 1), cmap='CustomCmap_type', transparent_bg=True)
    surf_type_lh = surf_type[:32492]
    surf_type_rh = surf_type[32492:]

    #### tractogram
    # load the conte69 hemisphere surfaces and spheres
    from scipy.spatial import KDTree
    fsLR_lh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    fsLR_rh = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii').darrays[0].data
    coords = np.concatenate([fsLR_lh, fsLR_rh])
    tree = KDTree(coords)

    tractogram_f = nib.streamlines.load('/local_raid/data/pbautin/data/output_tractogram_with_dps.trk')
    tractogram = tractogram_f.streamlines
    endpoints = [(sl[0], sl[-1]) for sl in tractogram]

    vertex_values = surf_type.copy()
    vertex_values[np.isnan(salience)] = np.nan
    streamline_values = []  # List to store values per streamline, e.g., tuple (start_val, end_val)
    for start_pt, end_pt in endpoints:
        dist_start, idx_start = tree.query(start_pt)
        dist_end, idx_end = tree.query(end_pt)

        # Optionally, define a cutoff distance to ensure endpoint is "close enough" to surface
        # For example, if we only trust assignments when endpoint is within 2 mm of surface:
        cutoff = 2.0
        if dist_start <= cutoff:
            start_val = vertex_values[idx_start]
        else:
            start_val = np.nan  # or some default value

        if dist_end <= cutoff:
            end_val = vertex_values[idx_end]
        else:
            end_val = np.nan

        streamline_values.append(np.nanmax([start_val, end_val]))

    num_streamlines = len(tractogram)
    streamline_arr = np.zeros(num_streamlines)
    streamline_values = np.asarray(streamline_values)
    streamline_values[np.isnan(streamline_values)] = 0
    streamline_arr = streamline_values
    mask = (streamline_arr != 0)
    streamline_arr = (streamline_arr - np.min(streamline_arr)) / (np.max(streamline_arr) - np.min(streamline_arr))
    filtered_tractogram = [tractogram[i] for i in range(len(tractogram)) if mask[i]]
    filtered_streamline_arr = streamline_arr[mask]
    new_tractogram = nib.streamlines.Tractogram(filtered_tractogram, affine_to_rasmm=tractogram_f.tractogram.affine_to_rasmm)
    #cmap = mp.colormaps['coolwarm']
    color = (cmap_types(filtered_streamline_arr)[:, 0:3] * 255).astype(np.uint8)
    print(color)
    tmp = [np.tile(color[i], (len(new_tractogram.streamlines[i]), 1))
           for i in range(len(new_tractogram.streamlines))]
    new_tractogram.data_per_point['color'] = tmp
    trk_file = nib.streamlines.TrkFile(new_tractogram)
    trk_file.save('/local_raid/data/pbautin/data/output_tractogram_with_dps_2.trk')


    ###### Cortical type comparisons
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))

    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere_lh, points_rh=sphere_rh)

    # Compute and store results
    all_data = {}
    real_data = {}

    surf_type[np.isnan(surf_type)] = 7  # Replace NaNs with dummy label
    state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        # Empirical
        expected_types = np.arange(1, 8)  # Cortical types 1 to 7
        comp = surf_type[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))

        # comp = surf_type[mask] * mask[mask]
        # u, c = np.unique(comp, return_counts=True)
        # perc = (c / len(comp)) * 100
        # real_data[net_name] = dict(zip(u, perc))

        # Null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in np.unique(surf_type)}
        for n in range(n_rand):
            comp = surf_type[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df

    # --- Plotting ---

    # Setup: Salience in full column
    n_total = len(all_data)
    n_cols = 4
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)

    # Plot Salience in full column
    ax_sal = fig.add_subplot(gs[:, 0])  # full height first column
    df = all_data["SalVentAttn"]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)

    # Plot other networks
    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        sns.barplot(data=df, ax=ax, color='lightgrey')
        rdict = {label_map.get(k, k): v for k, v in real_data[net_name].items()}
        sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()


    ######### Part 2
    ### Load the data from the specified text file BigBrain
    data_bigbrain = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
    salience_bigbrain = data_bigbrain[:,~np.isnan(salience)]
    mpc = build_mpc(salience_bigbrain)[0]
    mpc = np.triu(mpc,1)+mpc.T
    mpc[~np.isfinite(mpc)] = np.finfo(float).eps
    mpc[mpc==0] = np.finfo(float).eps
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(mpc, sparsity=0)

    n_plot = 30
    step = len(gm.gradients_[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])[::step]
    sorted_gradient = gm.gradients_[:, 0][sorted_gradient_indx]


    # Plot
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    # Normalize gradient values for colormap mapping
    norm = Normalize(vmin=np.min(gm.gradients_[:, 0]), vmax=np.max(gm.gradients_[:, 0]))
    cmap = get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(salience_bigbrain[:, idx] / 10000, np.arange(salience_bigbrain.shape[0]), color=color, alpha=0.8, lw=3)

    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)

    # Colorbar for gradient values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.tight_layout()
    plt.show()

    print(gm.lambdas_)
    arr = np.zeros(64984)
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = (gm.gradients_[:, 0] - np.min(gm.gradients_[:, 0])) / (np.max(gm.gradients_[:, 0]) - np.min(gm.gradients_[:, 0]))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)

    arr[~np.isnan(salience)] = (gm.gradients_[:, 1] - np.min(gm.gradients_[:, 1])) / (np.max(gm.gradients_[:, 1]) - np.min(gm.gradients_[:, 1]))
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)
    
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

    def preproc_profile(f):
        mpc = build_mpc(nib.load(f).darrays[0].data[:, ~np.isnan(salience)])[0]
        mpc = np.triu(mpc,1)+mpc.T
        mpc[~np.isfinite(mpc)] = np.finfo(float).eps
        mpc[mpc==0] = np.finfo(float).eps
        return mpc

    t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-5k_desc-intensity_profiles.shape.gii'))
    t1_salience_profile = np.array([nib.load(f).darrays[0].data[:, ~np.isnan(salience)] for f in t1_files])
    t1_salience_profile_avg = np.mean(t1_salience_profile, axis=0)
    t1_salience_mpc = [preproc_profile(f) for f in t1_files]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm.fit(t1_salience_mpc, sparsity=0)
    gm.gradients_ = np.mean(np.stack(gm.gradients_), axis=0)
    print(gm.lambdas_)
   

    n_plot = 30
    step = len(gm.gradients_[:, 0]) // n_plot
    sorted_gradient_indx = np.argsort(gm.gradients_[:, 0])[::step]
    sorted_gradient = gm.gradients_[:, 0][sorted_gradient_indx]


    # Plot
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap
    # Normalize gradient values for colormap mapping
    norm = Normalize(vmin=np.min(gm.gradients_[:, 0]), vmax=np.max(gm.gradients_[:, 0]))
    cmap = get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(t1_salience_profile_avg[:, idx] / 1000, np.arange(t1_salience_profile_avg.shape[0]), color=color, alpha=0.8, lw=3)

    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)

    # Colorbar for gradient values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.tight_layout()
    plt.show()
    

    arr = np.zeros(9684)
    arr[arr == 0] = np.nan
    arr[~np.isnan(salience)] = (gm.gradients_[:, 0] - np.min(gm.gradients_[:, 0])) / (np.max(gm.gradients_[:, 0]) - np.min(gm.gradients_[:, 0]))
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)

    arr[~np.isnan(salience)] = (gm.gradients_[:, 1] - np.min(gm.gradients_[:, 1])) / (np.max(gm.gradients_[:, 1]) - np.min(gm.gradients_[:, 1]))
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                             nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf5k_lh, surf5k_rh, array_name=arr, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)




if __name__ == "__main__":
    main()