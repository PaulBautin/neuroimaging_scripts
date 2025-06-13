from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Dynamics of the salience network
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
import pycatch22 as catch22


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


def plot_surface_nodes(surf, data):
    # Create custom colormap
    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4

    channel_type_raw = data['ChannelType']
    channel_type_flat = [item[0][0] for item in channel_type_raw]
    channel_type_mapping_int = {
        'D': 1,  # Dixi intracerebral electrodes
        'M': 2,  # Homemade MNI intracerebral electrodes
        'A': 3,  # AdTech intracerebral electrodes
        'G': 4   # AdTech subdural strips and grids
    }
    channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])

    ##### Plotting #####
    p = Plotter(nrow=1, ncol=2, size=(1600, 800))
    ren = p.AddRenderer(row=0, col=0, background=(1, 1, 1))
    # Add brain surface
    actor_surf = ren.AddActor()
    mapper_surf = actor_surf.SetMapper()
    mapper_surf.SetInputData(surf)
    actor_surf.GetProperty().SetColor(0.85,0.85,0.85)
    actor_surf.GetProperty().SetOpacity(1)
    # Add colored spheres
    for i, pos in enumerate(data['ChannelPosition']):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = ren.AddActor()
        mapper = actor.SetMapper()
        mapper.SetInputData(sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
    # Camera setup
    camera = ren.GetActiveCamera()
    camera.Azimuth(-90)
    camera.Elevation(0)
    camera.Roll(90)
    camera.Dolly(0.002 *  1.2)
    ren.ResetCameraClippingRange()

    ren = p.AddRenderer(row=0, col=1, background=(1, 1, 1))
    # Add brain surface
    actor_surf = ren.AddActor()
    mapper_surf = actor_surf.SetMapper()
    mapper_surf.SetInputData(surf)
    actor_surf.GetProperty().SetColor(0.85,0.85,0.85)
    actor_surf.GetProperty().SetOpacity(1)
    # Add colored spheres
    for i, pos in enumerate(data['ChannelPosition']):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = ren.AddActor()
        mapper = actor.SetMapper()
        mapper.SetInputData(sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
    # Camera setup
    camera = ren.GetActiveCamera()
    camera.Azimuth(90)
    camera.Elevation(0)
    camera.Roll(-90)
    camera.Dolly(0.002 * 1.2)
    ren.ResetCameraClippingRange()
    p.show()

def smooth_euclidean(plot_values, vertices, radius=10.0, sigma=3.0):
    tree = cKDTree(vertices)
    smoothed = np.zeros_like(plot_values)

    for i, vtx in enumerate(vertices):
        neighbor_ids = tree.query_ball_point(vtx, r=radius)
        if len(neighbor_ids) == 0:
            continue
        distances = np.linalg.norm(vertices[neighbor_ids] - vtx, axis=1)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        values = plot_values[neighbor_ids]
        smoothed[i] = np.average(values, weights=weights)

    return smoothed

def compute_psd(fs, data_w):
    """
    Compute the Power Spectral Density (PSD) for each channel.

    Parameters:
    - fs: int
        Sampling frequency (Hz).
    - data_w: np.ndarray
        Time-series data for each channel (channels x samples).

    Returns:
    - pxx_log: np.ndarray
        Log-transformed, normalized PSD matrix (frequencies x channels).
    """
    # Compute the PSD for frequencies between 0.5 and 80 Hz
    f, pxx = welch(data_w, fs=fs)
    # freq_mask = (f >= 0.5) & (f <= 80)
    # f = f[freq_mask]
    # pxx = pxx[:, freq_mask]
    # Normalize PSD to total power = 1
    pxx = pxx / np.sum(pxx, axis=1, keepdims=True)
    # Log-transform and normalize the PSD
    pxx_log = np.log(pxx)
    return f, pxx_log


# Compute the catch22 features
def compute_features(x):
    res = catch22.catch22_all(x, catch24 = True)
    return res['values'] # just return the values 


def extract_band_power(pxx_log: np.ndarray, freq: np.ndarray, band: tuple) -> np.ndarray:
    """Compute z-scored power for a given frequency band."""
    band_idx = np.where((freq >= band[0]) & (freq < band[1]))[0]
    band_power = np.mean(pxx_log[:, band_idx], axis=1)
    zscore = (band_power - np.mean(band_power)) / np.std(band_power)
    return zscore


def frequency_band_analysis(data, surf32k_lh_infl, surf32k_rh_infl, state, state_name, indices_32k):
    ##### Frequency band analysis
    FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 80)}
    # Define band order and colors
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    vertices_32k_infl = np.vstack((surf32k_lh_infl.GetPoints(), surf32k_rh_infl.GetPoints()))
    data_w = data['Data_W'].T
    f, pxx_log = compute_psd(data['SamplingFrequency'], data_w)
    freq = f[0, :]
    band_maps = {}
    for band_name, band_range in FREQ_BANDS.items():
        zscore = extract_band_power(pxx_log, freq, band_range)
        plot_values = np.zeros(vertices_32k_infl.shape[0])
        plot_values[indices_32k] = zscore
        smoothed = smooth_euclidean(plot_values, vertices_32k_infl, radius=10.0, sigma=3.0)
        smoothed[:32492] = smoothed[32492:]
        band_maps[band_name] = smoothed / np.max(smoothed)
    # Stack all frequency band maps: shape (vertices, n_bands)
    type_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_array = np.stack([band_maps[band] for band in type_labels], axis=1)  # normalized maps
    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    sp.fit(sphere_lh, points_rh=sphere_rh)
    # Initialize result containers
    real_data = {}
    all_data = {}
    # Iterate over each network
    for net_idx, net_name in enumerate(state_name):
        if net_name == 'medial_wall':
            continue
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]
        # Real (empirical) composition: mean normalized power per band
        band_means = np.mean(band_array[mask], axis=0)
        #percentages = (band_means / np.sum(band_means)) * 100
        percentages = band_means
        real_data[net_name] = dict(zip(type_labels, percentages))
        # Null distribution via spin permutation
        comp_dict = {band: [] for band in type_labels}
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        for n in range(n_rand):
            rotated_mask = net_rot[n].astype(bool)
            band_means = np.mean(band_array[rotated_mask], axis=0)
            #percentages = (band_means / np.sum(band_means)) * 100
            percentages = band_means
            for i, band in enumerate(type_labels):
                comp_dict[band].append(percentages[i])
        # Store as dataframe
        all_data[net_name] = pd.DataFrame(comp_dict)
    # --- Plotting ---
    n_total = len(all_data)
    n_cols = 4
    # Get index and names
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)
    # --- Plot Salience in full first column ---
    ax_sal = fig.add_subplot(gs[:, 0])
    df = all_data["SalVentAttn"]
    df_melt = df[band_order].melt(var_name='Band', value_name='Average z-score')
    sns.barplot(data=df_melt, x='Band', y='Average z-score', ax=ax_sal, color='lightgrey')
    real_vals = [real_data["SalVentAttn"][band] for band in band_order]
    sns.scatterplot(x=band_order, y=real_vals, color=band_colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(-0.2, 0.2)
    ax_sal.tick_params(axis='x', labelrotation=90)
    # --- Plot all other networks ---
    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        df_melt = df[band_order].melt(var_name='Band', value_name='Average z-score')
        sns.barplot(data=df_melt, x='Band', y='Average z-score', ax=ax, color='lightgrey')
        real_vals = [real_data[net_name][band] for band in band_order]
        sns.scatterplot(x=band_order, y=real_vals, color=band_colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(-0.2, 0.2)
        ax.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()
    # Plot band frequency spatial distribution on surface
    surf32k_rh_infl.append_array(band_maps['delta'][32492:], name="overlay2")
    surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
        nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, return_plotter=True)
    p.show()
    # plot PSD
    plt.plot(freq, pxx_log[89, :], color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Log(Normalized PSD)')
    plt.title('Power Spectral Density')
    plt.axvline(x=0.5, color='grey', linestyle='--', label='δ (0.5-4Hz)',alpha=0.5)
    plt.axvline(x=4, color='grey', linestyle='--',alpha=0.5)
    plt.axvline(x=8, color='grey', linestyle='--', label='θ (4-8Hz)',alpha=0.5)
    plt.axvline(x=13, color='grey', linestyle='--', label='α (8-13Hz)',alpha=0.5)
    plt.axvline(x=30, color='grey', linestyle='--', label='β (13-30Hz)',alpha=0.5)
    plt.axvline(x=80, color='grey', linestyle='--', label='γ (30-80Hz)',alpha=0.5)
    plt.legend()
    plt.show()


def preproc_profile(f):
    mpc = build_mpc(f.T)[0]
    mpc = np.triu(mpc,1)+mpc.T
    mpc[~np.isfinite(mpc)] = np.finfo(float).eps
    mpc[mpc==0] = np.finfo(float).eps
    return mpc



def main():
    ##### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-5k inflated surface
    surf5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    # Load fsLR-32k inflated surface
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    # Load fsLR-32k midthickness surface
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.midthickness.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.midthickness.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    # Load fsLR-32k sphere surface
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    ##### load yeo atlas 7 network
    atlas_yeo_lh = nib.load(micapipe + '/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load(micapipe + '/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    # load .csv associated with schaefer 400
    df_label = pd.read_csv(micapipe + '/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
    salience = state.copy()
    salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = 0
    salience_border = array_operations.get_labeling_border(surf_32k, salience)
    #state[salience_border == 1] = 7

    ##### Extract the data #####
    data = scipy.io.loadmat("/local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat") # dict: data.keys()
    channel_name = [item[0][0] for item in data['ChannelName']]
    # Data_W: matrix with one column per channel, and 13600 samples containing all the signals for wakefulness
    data_w = data['Data_W'].T
    channel_type_raw = data['ChannelType']
    channel_type_flat = [item[0][0] for item in channel_type_raw]
    channel_type_mapping_int = {
        'D': 1,  # Dixi intracerebral electrodes
        'M': 2,  # Homemade MNI intracerebral electrodes
        'A': 3,  # AdTech intracerebral electrodes
        'G': 4}  # AdTech subdural strips and grids
    channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])
    # Create custom colormap
    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4
    # Create surface polydata objects
    surf_lh = mesh.mesh_creation.build_polydata(points=data['NodesLeft'], cells=data['FacesLeft'] - 1)
    surf_rh = mesh.mesh_creation.build_polydata(points=data['NodesRight'], cells=data['FacesRight'] - 1)
    mesh.mesh_io.write_surface(surf_lh, '/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/surf_lh_ieeg_atlas.surf.gii')
    mesh.mesh_io.write_surface(surf_rh, '/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/surf_rh_ieeg_atlas.surf.gii')

    ###### electrode projection on cortical surface
    data['ChannelPosition'][:,0] = np.abs(data['ChannelPosition'][:,0])
    vertices = np.vstack((data['NodesLeft'], data['NodesRight']))
    tree = cKDTree(vertices)
    indices_surf = tree.query(data['ChannelPosition'])[1]
    data['ChannelPosition'] = vertices[indices_surf]
    ## electrode projection on registered (to template) cortical surface
    surf_reg_lh = read_surface('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/L.anat.reg.surf.gii', itype='gii')
    surf_reg_rh = read_surface('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/R.anat.reg.surf.gii', itype='gii')
    vertices_surf_reg = np.vstack((surf_reg_lh.GetPoints(), surf_reg_rh.GetPoints()))
    data['ChannelPosition'] = vertices_surf_reg[indices_surf]
    ## electrode projection on template cortical surface
    vertices_32k = np.vstack((surf32k_lh.GetPoints(), surf32k_rh.GetPoints()))
    vertices_32k_infl = np.vstack((surf32k_lh_infl.GetPoints(), surf32k_rh_infl.GetPoints()))
    tree = cKDTree(vertices_32k)
    channel_indices_32k = tree.query(data['ChannelPosition'])[1]
    data['ChannelPosition'] = vertices_32k_infl[channel_indices_32k]
    indices_in_salience_mask = salience[channel_indices_32k] != 0
    indices_32k_salience = channel_indices_32k[indices_in_salience_mask]
    #data['ChannelPosition'] = data['ChannelPosition'][indices_in_salience_mask,:]

    ##### Compute the features #####
    threads_to_use = os.cpu_count()
    data_salience = data_w[indices_in_salience_mask,:]
    # results_rad = np.asarray(Parallel(n_jobs=threads_to_use)(
    #     delayed(RAD)(data_w[i]) for i in range(data_w.shape[0])))
    results = np.asarray(Parallel(n_jobs=threads_to_use)(
        delayed(compute_features)(data_salience[i]) for i in range(data_salience.shape[0])))
    results = (results - results.mean(axis=0, keepdims=True)) / results.std(axis=0, keepdims=True)
    names = catch22.catch22_all(data_salience[0], catch24 = True)['names']
    mpc = build_mpc(results.T)[0]
    mpc = np.triu(mpc,1)+mpc.T
    mpc[~np.isfinite(mpc)] = np.finfo(float).eps
    mpc[mpc==0] = np.finfo(float).eps
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(mpc, sparsity=0)
    print(gm.lambdas_)

    ##### Load the data from the specified text file BigBrain to sort ieeg gradient
    data_bigbrain = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
    salience_bigbrain = data_bigbrain[:,indices_32k_salience]
    print(salience_bigbrain.shape)
    mpc_bigbrain = build_mpc(salience_bigbrain)[0]
    mpc_bigbrain = np.triu(mpc_bigbrain,1)+mpc_bigbrain.T
    mpc_bigbrain[~np.isfinite(mpc_bigbrain)] = np.finfo(float).eps
    mpc_bigbrain[mpc_bigbrain==0] = np.finfo(float).eps
    gm_bigbrain = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm_bigbrain.fit(mpc_bigbrain, sparsity=0)

    n_plot = 20
    step = len(gm_bigbrain.gradients_[:, 0]) // n_plot
    sorted_gradient_indx_bigbrain = np.argsort(gm_bigbrain.gradients_[:, 0])[::step]
    sorted_results = results[sorted_gradient_indx_bigbrain]
    plt.figure(figsize=(8, 10))
    im = plt.imshow(sorted_results, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)  # change cmap as needed
    plt.colorbar(im, label='Feature value')
    plt.xticks(ticks=np.arange(len(names)), labels=names, rotation=90)
    plt.tight_layout()
    plt.show()

    plot_values_gradient = np.zeros(vertices_32k_infl.shape[0])
    plot_values_gradient[indices_32k_salience] = gm.gradients_[:, 0]
    smoothed_values_gradient = smooth_euclidean(plot_values_gradient, vertices_32k_infl, radius=10.0, sigma=3.0)
    smoothed_values_gradient[salience == 0] = np.nan

    salience_border = salience_border.astype(float)

    plot_values_density = np.zeros(vertices_32k_infl.shape[0])
    plot_values_density[channel_indices_32k] = 1
    smoothed_values_density = smooth_euclidean(plot_values_density, vertices_32k_infl, radius=10.0, sigma=3.0)
    smoothed_values_density = smoothed_values_density / np.max(smoothed_values_density)
    salience_border = salience_border.astype(float)
    smoothed_values_density[salience_border == 1] = np.nan

    # Append to surface
    surf32k_rh_infl.append_array(smoothed_values_gradient[32492:], name="overlay1")
    surf32k_rh_infl.append_array(salience_border[32492:], name="overlay2")
    surf32k_rh_infl.append_array(smoothed_values_density[32492:], name="overlay3")
    surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]

    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay1", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 1), cmap="coolwarm", transparent_bg=True, return_plotter=True)
    p.show()
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay1", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 0), cmap="coolwarm", transparent_bg=True, return_plotter=True)
    p.show()
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(0, 0, 0, 1), cmap="Purples", color_range=(0,1), transparent_bg=True, return_plotter=True)
    p.show()
    smoothed_values_density[salience == 0] = np.nan
    surf32k_rh_infl.append_array(smoothed_values_density[32492:], name="overlay3")
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 0), cmap="Purples", color_range=(0,1), transparent_bg=True, return_plotter=True)
    p.show()
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="Greys", color_range=(0,1), transparent_bg=True, return_plotter=True)
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=np.min(gm.gradients_[:, 0]), vmax=np.max(gm.gradients_[:, 0]))
    for i, pos in enumerate(data['ChannelPosition'][indices_in_salience_mask,:]):
        val = gm.gradients_[:, 0]
        rgba = custom_cmap(norm(val[i]))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(data['ChannelPosition'][indices_in_salience_mask,:]):
        val = gm.gradients_[:, 0]
        rgba = custom_cmap(norm(val[i]))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    p.show()

    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="Greys", color_range=(0,1), transparent_bg=True, return_plotter=True)
    # Add colored spheres
    # salience channel index: 89
    for i, pos in enumerate(data['ChannelPosition'][:,:]):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(data['ChannelPosition'][:,:]):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    p.show()




    ################### BOLD data ##################
    # Load data for multiple subjects
    func_32k = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
    # Load the data for each subject
    func_salience = np.array([nib.load(f).darrays[0].data[:,salience!=0] for f in func_32k])
    print(func_salience.shape)

    threads_to_use = os.cpu_count()
    names = catch22.catch22_all(func_salience[0][:,0], catch24 = True, short_names=True)['names']
    results = Parallel(n_jobs=threads_to_use)(
        delayed(lambda sub: [
            compute_features(func_salience[sub][:, i]) for i in range(func_salience[0].shape[1])
        ])(sub) for sub in range(len(func_salience)))
    results = np.array(results)
    results = np.array([(results[i,:,:] - results[i,:,:].mean(axis=0, keepdims=True)) / results[i,:,:].std(axis=0, keepdims=True) for i in range(results.shape[0])])
    mpc = [preproc_profile(results[i,:,:]) for i in range(results.shape[0])]
    gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm.fit(mpc, sparsity=0)
    gm.gradients_ = np.mean(np.stack(gm.gradients_), axis=0)
    print(gm.lambdas_)  

    ##### Load the data from the specified text file BigBrain to sort ieeg gradient
    data_bigbrain = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
    salience_bigbrain = data_bigbrain[:,salience!=0]
    print(salience_bigbrain.shape)
    mpc_bigbrain = build_mpc(salience_bigbrain)[0]
    mpc_bigbrain = np.triu(mpc_bigbrain,1)+mpc_bigbrain.T
    mpc_bigbrain[~np.isfinite(mpc_bigbrain)] = np.finfo(float).eps
    mpc_bigbrain[mpc_bigbrain==0] = np.finfo(float).eps
    gm_bigbrain = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
    gm_bigbrain.fit(mpc_bigbrain, sparsity=0)
    # plot heatmap of features in function of position of MPC bigbrainn gradient
    n_plot = 20
    step = len(gm_bigbrain.gradients_[:, 0]) // n_plot
    sorted_gradient_indx_bigbrain = np.argsort(gm_bigbrain.gradients_[:, 0])[::step]
    sorted_results = results.mean(axis=0)[sorted_gradient_indx_bigbrain]
    plt.figure(figsize=(8, 10))
    im = plt.imshow(sorted_results, aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)  # change cmap as needed
    plt.colorbar(im, label='Feature value')
    plt.xticks(ticks=np.arange(len(names)), labels=names, rotation=90)
    plt.tight_layout()
    plt.show()

    # Append to surface
    plot_values = np.zeros(salience.shape[0])
    plot_values[salience != 0] = gm.gradients_[:, 0]
    plot_values[plot_values == 0] = np.nan
    surf32k_rh_infl.append_array(plot_values[32492:], name="overlay1")
    surf32k_rh_infl.append_array(salience_border[32492:], name="overlay2")
    surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]

    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay1", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 1), cmap="coolwarm", transparent_bg=True, return_plotter=True)
    p.show()
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay1", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 0), cmap="coolwarm", transparent_bg=True, return_plotter=True)
    p.show()













    # output_txt = "/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/electrodes_foci.txt"
    # with open(output_txt, "w") as f:
    #     for name, color, coord in zip(channel_name, custom_cmap(norm(channel_integers)) * 255, data['ChannelPosition']):
    #         f.write(f"{name}\n")
    #         f.write(f"{color[0]:.0f} {color[1]:.0f} {color[2]:.0f} {coord[0]:.3f} {coord[1]:.3f} {coord[2]:.3f}\n")
    










    # plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=10, color_bar='bottom', share='both',
    #     nan_color=(220, 220, 220, 1), cmap='Purples', transparent_bg=True)

    # # --- Distance-weighted smoothing ---
    # smoothed_values = np.zeros_like(plot_values)
    # smoothed_values[smoothed_values == 0] = np.nan
    # for vtx_idx, vtx in enumerate(vertices):
    #     # Find neighbors within radius
    #     neighbor_ids = tree.query_ball_point(vtx, r=5)
    #     distances = np.linalg.norm(vertices[neighbor_ids] - vtx, axis=1)
    #     weights = np.exp(-distances**2 / (2 * 1**2))
    #     values = plot_values[neighbor_ids]
    #     smoothed_values[vtx_idx] = np.average(values, weights=weights)

    # plot_hemispheres(surf_lh, surf_rh, array_name=smoothed_values, size=(1200, 300), zoom=10, color_bar='bottom', share='both',
    #        nan_color=(220, 220, 220, 1), cmap='Purples', transparent_bg=True)
    




if __name__ == "__main__":
    main()