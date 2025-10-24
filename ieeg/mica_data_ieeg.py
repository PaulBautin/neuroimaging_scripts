from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# The maps are only for 32k surfaces and fsaverage5. The 5k surfaces are not 
# able to properly depict the sensitivity, which varies quickly in space. 
# fsnative surfaces have triangles that vary widely in size (1000 times), 
# which leads to some numerical issues, I might see if I can fix the code in the future
#
# database 
# BIDS_ieeg: 30 sessions from 29 patients
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import scipy.io as sio
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pycatch22 as catch22
from joblib import Parallel, delayed
import os
from brainspace import mesh
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation

from brainspace.mesh.array_operations import get_parcellation_centroids
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from scipy.signal import welch
from scipy.signal import hilbert, butter, filtfilt
import nibabel as nib
import glob

from brainspace.mesh.mesh_elements import get_points
from brainspace.plotting import plot_surf
from vtkmodules.vtkFiltersSources import vtkSphereSource
from matplotlib import cm
from matplotlib.colors import to_rgb
import re
import matplotlib as mp


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
    [255, 255, 255],     # Visual black
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

def load_centroids_with_map_index(all_channel_maps, surface, reflect_x=False):
    """
    Compute centroids from nonzero-labeled vertices
    """
    coords = []
    colors = []
    n_maps = all_channel_maps.shape[0]
    cmap = cm.get_cmap('tab20', n_maps)
    points = get_points(surface)
    
    for i, labels in enumerate(all_channel_maps):
        mask = labels != 0
        if np.any(mask):
            centroid_idx = get_parcellation_centroids(surface, labels, mask=mask)
            centroid_xyz = points[centroid_idx != 0]
            if reflect_x:
                centroid_xyz[:, 0] = np.abs(centroid_xyz[:, 0])
            coords.append(centroid_xyz)
            colors.extend([to_rgb(cmap(i))] * centroid_xyz.shape[0])

    return np.vstack(coords), np.array(colors)


def add_colored_spheres(renderer, coords, radius=1.5, rotation=[-90, 0, 90]):
    for pos in coords:
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(radius)
        sphere.Update()
        actor = renderer.AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor([0,0,0])
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(rotation[0])
        actor.RotateY(rotation[1])
        actor.RotateZ(rotation[2])


def load_original_data_files():
    # /rawdata  : edf files with recording montage
    # /original : minimally processed data. Mat file, power line interference filtered out, bipolar montage
    # Precompile regex pattern once
    pattern = re.compile(r'sub-(PX\d+)/ses-(\d+)')
    # Get list of .mat files
    ieeg_files = glob.glob('/local_raid/data/pbautin/data/ieeg_mica/original/sub-PX*/ses-*/sub-PX*_ses-*_stage-W.mat')
    # One-pass list comprehension with minimal lookups
    df_ieeg = pd.DataFrame([
        {
            'Subject'     : m.group(1),
            'Session'     : m.group(2),
            'ChannelName' : ch,
            'SamplingRate': d['SamplingRate'],
            'Data'        : d['Data'][:, i]
        }
        for f in ieeg_files
        if (m := pattern.search(f))                         # Match subject/session
        and (d := sio.loadmat(f, simplify_cells=True))      # Load .mat file
        and 'ChannelName' in d and 'Data' in d              # Ensure required keys
        for i, ch in enumerate(d['ChannelName'])            # Loop channels
    ])
    df_ieeg['ChannelName_ref'] = df_ieeg['ChannelName'].str.split('-').str[0]
    df_ieeg['ChannelName_ref'] = df_ieeg['ChannelName_ref'].str.upper()
    return df_ieeg

def load_electrode_data_files():
    """
    Load and parse electrode channel information from multiple subjects and sessions.
    
    Returns:
        pd.DataFrame: Consolidated DataFrame with channel name, coordinates, subject, and session info.
    """
    channel_info_files = glob.glob(
        '/local_raid/data/pbautin/data/ieeg_mica/derivatives/sub-PX*/ses-*/ibis/*_channel_info.mat'
    )

    df_elecs = pd.concat([
        (
            pd.read_csv(
                csv_path := os.path.join(os.path.dirname(mat_path), rel_path),
                header=None, delimiter=',', names=list(range(14))
            )
            .pipe(lambda df: df.set_axis(df.iloc[-1], axis=1))                     # Set column names using last row
            .pipe(lambda df: df[df['[order]'] == '[channel]'])                     # Keep only rows for channels
            [['ChannelName', 'indexChannel', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']]  # Select required columns
            .assign(
                ChannelName=lambda s: s['ChannelName'].str.replace(
                    r'^([A-Za-z]+)(\d+)-(\d+)$',
                    lambda m: f"{m.group(1)}{m.group(2)}-{m.group(1)}{m.group(3)}",
                    regex=True
                ).str.upper(),
                Subject=m.group(1),  # extracted via regex match from file path
                Session=m.group(2)
            )
        )
        for mat_path in channel_info_files
        if (m := re.search(r'sub-(PX\d+)/ses-(\d+)', mat_path))                         # Extract subject/session
        for rel_path in sio.loadmat(mat_path, simplify_cells=True).get('elec', [])     # Get relative CSV path(s)
        if os.path.exists(os.path.join(os.path.dirname(mat_path), rel_path))           # Check file existence
    ], ignore_index=True)

    return df_elecs

def load_sensitivity_data_files():
    """
    Load iEEG sensitivity maps from preprocessed .mat files.
    Each row corresponds to a channel's sensitivity map in one hemisphere.
    """
    # Precompile regex
    pattern_sub  = re.compile(r'sub-(PX\d+)')
    pattern_ses  = re.compile(r'ses-(\d+)')
    pattern_hemi = re.compile(r'hemi-([LRlr])')

    # Collect all matching sensitivity map files
    sensitivity_files = glob.glob('/local_raid/data/pbautin/data/ieeg_mica/derivatives/sub-PX*/ses-*/maps/*_sensitivty-map_hemi-*_space-nativepro_surf-fsLR-32k_label-midthickness.mat')

    # Single-pass construction via list comprehension
    df_sensitivity = pd.concat([
        pd.DataFrame({
            'ChannelName_ref'       : d['ContactName'],
            'ContactOutsideBrain'   : d['ContactOutsideBrain'],
            'ContactSensitivityMap' : list(map(list, d['ContactSensitivityMap'])),
            'Subject'               : pattern_sub.search(f).group(1),
            'Session'               : pattern_ses.search(f).group(1),
            'Hemisphere'            : pattern_hemi.search(f).group(1).lower() + 'h'
        })
        for f in sensitivity_files
        if os.path.exists(f)
        and (d := sio.loadmat(f, simplify_cells=True))
        and 'ContactName' in d and 'ContactSensitivityMap' in d
    ], ignore_index=True)
    df_sensitivity_lh = df_sensitivity[df_sensitivity['Hemisphere'] == 'lh'].drop(columns='Hemisphere').rename(columns={"ContactSensitivityMap": "ContactSensitivityMap_lh",'ContactOutsideBrain':'ContactOutsideBrain_lh'})
    df_sensitivity_rh = df_sensitivity[df_sensitivity['Hemisphere'] == 'rh'].drop(columns='Hemisphere').rename(columns={"ContactSensitivityMap": "ContactSensitivityMap_rh",'ContactOutsideBrain':'ContactOutsideBrain_rh'})
    df_sensitivity = df_sensitivity_rh.merge(df_sensitivity_lh, on=['Subject', 'Session', 'ChannelName_ref'], how='left')
    df_sensitivity['ChannelName_ref'] = df_sensitivity['ChannelName_ref'].str.capitalize()
    return df_sensitivity


def load_channel_data_files():
    """
    Load iEEG sensitivity maps from preprocessed .mat files.
    Each row corresponds to a channel's sensitivity map in one hemisphere.
    """
    # Precompile regex
    pattern_sub  = re.compile(r'sub-(PX\d+)')
    pattern_ses  = re.compile(r'ses-(\d+)')
    pattern_hemi = re.compile(r'hemi-([LRlr])')

    # Collect all matching sensitivity map files
    channel_files = glob.glob('/host/verges/tank/data/BIDS_iEEG/derivatives/sub-PX*/ses-*/maps/*_channel-map_hemi-*_space-nativepro_surf-fsLR-32k_label-midthickness.mat')
    print(sio.loadmat(channel_files[0], simplify_cells=True))

    # Single-pass construction via list comprehension
    df_channel = pd.concat([
        pd.DataFrame({
            'ChannelName'           : d['ChannelName'],
            'ChannelMap'            : list(map(list, np.tile(d['ChannelMap'], (len(d['ChannelName']), 1)))),
            'Subject'               : pattern_sub.search(f).group(1),
            'Session'               : pattern_ses.search(f).group(1),
            'Hemisphere'            : pattern_hemi.search(f).group(1).lower() + 'h'
        })
        for f in channel_files
        if os.path.exists(f)
        and (d := sio.loadmat(f, simplify_cells=True))
        and 'ChannelName' in d and 'ChannelMap' in d
    ], ignore_index=True)
    print(df_channel)

    df_channel_lh = df_channel[df_channel['Hemisphere'] == 'lh'].drop(columns='Hemisphere').rename(columns={"ChannelMap": "ChannelMap_lh"})
    df_channel_rh = df_channel[df_channel['Hemisphere'] == 'rh'].drop(columns='Hemisphere').rename(columns={"ChannelMap": "ChannelMap_rh"})
    df_channel = df_channel_rh.merge(df_channel_lh, on=['Subject', 'Session', 'ChannelName'], how='left')
    df_channel['ChannelName'] = df_channel['ChannelName'].str.capitalize()
    return df_channel


def add_nativepro_coords(df_ieeg: pd.DataFrame) -> pd.DataFrame:
    """Append X/Y/Z_nativepro columns to df_ieeg for each Subject–Session.
    # https://github.com/ANTsX/ANTs/wiki/ITK-affine-transform-conversion
    """
    flip_lps_to_ras = np.array([
        [ 1, -1,  1,  1],
        [-1,  1,  1,  1],
        [ 1,  1,  1, -1],
        [ 1,  1,  1,  1]
    ])

    native_cols = ['X_nativepro', 'Y_nativepro', 'Z_nativepro']
    df_ieeg[native_cols] = np.nan                                      # pre‑allocate

    for (subj, sess), idx in df_ieeg.groupby(['Subject', 'Session']).groups.items():
        xfm_path = (
            f'/local_raid/data/pbautin/data/ieeg_mica/derivatives/'
            f'sub-{subj}/ses-{sess}/xfm/'
            f'sub-{subj}_ses-{sess}_iEEG-volume_to_nativepro.mat'
        )
        if not os.path.exists(xfm_path):
            continue                                                    # skip if missing

        x = sio.loadmat(xfm_path, simplify_cells=True)
        A = x['AffineTransform_double_3_3']      # length‑12 vector (9 + 3)
        R = A[:9].reshape(3, 3)                  # rotation / scale
        t = A[ 9:]                               # translation
        c = x['fixed'].ravel()                   # center of rotation

        M = np.eye(4, dtype=float)
        M[:3, :3] = R
        M[:3,  3] = (t + c - R @ c).T            # ITK → homogeneous
        M *= flip_lps_to_ras                     # LPS → RAS

        # apply to all electrodes of this subject/session in one shot
        xyz = df_ieeg.loc[idx, ['X1', 'Y1', 'Z1']].astype(float).values
        native = (M[:3, :3] @ xyz.T).T + M[:3, 3]
        df_ieeg.loc[idx, native_cols] = native

    return df_ieeg

# -- Step 1: Define PSD computation --
def compute_psd(row):
    """
    Compute log-transformed relative PSD for 0.5–80 Hz using Welch's method.

    Parameters
    ----------
    row : pd.Series
        Must contain 'Data' (1D array) and 'SamplingRate' (float)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (log-relative PSD, frequency array)
    """
    signal = row['Data']
    fs = row['SamplingRate']

    # Welch parameters
    nperseg = int(fs * 2)      # 2 s segments
    noverlap = nperseg // 2    # 50% overlap

    freqs, psd = welch(
        signal,
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant"
    )

    # Restrict to 0.5–80 Hz
    freq_mask = (freqs >= 0.5) & (freqs <= 80)
    freqs = freqs[freq_mask]
    psd = psd[freq_mask]

    # Normalize and log-transform PSD
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        psd_rel = np.full_like(psd, np.nan)
    else:
        psd_rel = psd / psd_sum
    psd_log = np.log(psd_rel + np.finfo(float).eps)  # avoid log(0)

    # Optional: plot each channel
    plt.plot(freqs, psd_log, linewidth=1)

    return psd_log, freqs


def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-32k surfaces
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.midthickness.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.midthickness.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    df_yeo_surf['index'] = df_yeo_surf.index
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    df_yeo_surf.loc[df_yeo_surf['network'] == 'medial_wall', 'network_int'] = np.nan
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf.loc[salience_border == 1, 'network_int'] = 7
    
    ##### Merge iEEG + electrode geometry
    sensitivity_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_sensitivity.pkl'
    if os.path.exists(sensitivity_path):
        df_sensitivity = pd.read_pickle(sensitivity_path)
        print(f"Loaded df_ieeg from {sensitivity_path}")
        print(df_sensitivity.sort_values(by=['Subject', 'Session', 'ChannelName_ref']))
    else:
        df_sensitivity = load_sensitivity_data_files()
        df_sensitivity.to_pickle(sensitivity_path)
        print(f"Saved df_ieeg to {sensitivity_path}")
        print(df_sensitivity)


    ##### Merge iEEG + electrode geometry
    channel_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_channel.pkl'
    if os.path.exists(channel_path):
        df_channel = pd.read_pickle(channel_path)
        print(f"Loaded df_ieeg from {channel_path}")
        print(df_channel.sort_values(by=['Subject', 'Session', 'ChannelName']))
    else:
        df_channel = load_channel_data_files()
        df_channel.to_pickle(channel_path)
        print(f"Saved df_ieeg to {channel_path}")
        print(df_channel)
    df_channel = df_channel.drop(columns='ChannelName').groupby(['Subject', 'Session']).first()
    lh = np.stack(df_channel['ChannelMap_lh'].array).astype(float)  # shape: (N, 32492)
    rh = np.stack(df_channel['ChannelMap_rh'].array).astype(float)  # shape: (N, 32492)
    coord_lh, _ = load_centroids_with_map_index(lh, surf32k_lh_infl, reflect_x=False)
    coord_rh, _ = load_centroids_with_map_index(rh, surf32k_rh_infl, reflect_x=False)
    print(coord_lh)
    all = np.concatenate([np.sum(lh, axis=0), np.sum(rh, axis=0)]) # shape: (64952)
    all[all == 0] = np.nan
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, all,
                        size=(1200, 300), zoom=1.3,
                        color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1),
                        cmap="inferno", transparent_bg=True)
    
            # --- Setup surface and plot layout ---)
    surfs = {'lh1': surf32k_lh_infl, 'lh2': surf32k_lh_infl, 'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
    layout = [['lh1', 'lh2', 'rh1', 'rh2']]
    view = [['lateral', 'medial', 'lateral', 'medial']]

    p = plot_surf(
        surfs, layout=layout, view=view, size=(1200, 300), zoom=1.3,
        color_bar='bottom', share='both', nan_color=(220, 220, 220, 1),
        cmap="CustomCmap_yeo", transparent_bg=True, return_plotter=True
    )
        # --- Helper: add colored spheres ---
    def add_colored_spheres(renderer, coords, radius=1.5, rotation=[-90, 0, 90]):
        for pos in coords:
            sphere = vtkSphereSource()
            sphere.SetCenter(*pos)
            sphere.SetRadius(radius)
            sphere.Update()
            actor = renderer.AddActor()
            actor.SetMapper(inputData=sphere.GetOutput())
            actor.GetProperty().SetColor([0,0,0])
            actor.GetProperty().SetOpacity(1.0)
            actor.RotateX(rotation[0])
            actor.RotateY(rotation[1])
            actor.RotateZ(rotation[2])
    # --- Extract transformed coordinates ---
    coords = np.vstack([coord_lh, coord_rh])
    # --- Render all hemispheres/views ---
    for i, rot in zip(range(4), [[-90,0,90], [-90,0,270]]*2):
        add_colored_spheres(p.renderers[i][0], coords, radius=1.5, rotation=rot)
    p.show()



    ieeg_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_ieeg.pkl'
    if os.path.exists(ieeg_path):
        df_ieeg = pd.read_pickle(ieeg_path)
        print(f"Loaded df_ieeg from {ieeg_path}")
        print(df_ieeg.sort_values(by=['Subject', 'Session', 'ChannelName']))
    else:
        df_ieeg = load_original_data_files()
        df_ieeg.to_pickle(ieeg_path)
        print(f"Saved df_ieeg to {ieeg_path}")
        print(df_ieeg)

    electrodes_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_electrodes.pkl'
    if os.path.exists(electrodes_path):
        df_electrodes = pd.read_pickle(electrodes_path)
        print(f"Loaded df_ieeg from {electrodes_path}")
        print(df_electrodes.sort_values(by=['Subject', 'Session', 'ChannelName']))
    else:
        df_electrodes = load_electrode_data_files()
        df_electrodes = add_nativepro_coords(df_electrodes)
        df_electrodes = df_electrodes.drop(columns=['indexChannel', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'])
        df_electrodes.to_pickle(electrodes_path)
        print(f"Saved df_ieeg to {electrodes_path}")
        print(df_electrodes)

    # Merge dataframes    
    df_ieeg = df_ieeg.merge(df_sensitivity, on=['Subject', 'Session', 'ChannelName_ref'], how='left')
    df_ieeg = df_ieeg.merge(df_electrodes, on=['Subject', 'Session', 'ChannelName'], how='left')
    df_ieeg = df_ieeg.dropna()
    print(df_ieeg)

    # Show all sensitivity maps
    subject = 'PX001'
    df_ieeg = df_ieeg[df_ieeg['Subject'] == subject]
    lh = np.stack(df_ieeg['ContactSensitivityMap_lh'].array)  # shape: (N, 32492)
    rh = np.stack(df_ieeg['ContactSensitivityMap_rh'].array)  # shape: (N, 32492)

    all = np.concatenate([lh, rh]) # shape: (N, 64952)
    all_sum = np.sum(all, axis=0)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, all_sum,
                        size=(1200, 300), zoom=1.3,
                        color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1),
                        cmap="coolwarm", transparent_bg=True, color_range='sym')
    


    # --- Load surfaces (you may want to loop over subjects/sessions here if needed) ---

    data_matrix = np.stack(df_ieeg['Data'].values).T
    conn_matrix = pd.DataFrame(data_matrix, columns=df_ieeg['ChannelName']).corr(method='pearson')
    print(conn_matrix)
    import seaborn as sns
    sns.heatmap(conn_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Pearson Correlation Connectivity")
    plt.show()

    surf_lh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/sub-'+subject+'/ses-01/surf/sub-'+subject+'_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii', itype='gii')
    surf_rh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/sub-'+subject+'/ses-01/surf/sub-'+subject+'_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii', itype='gii')
    lh = np.stack(df_ieeg['ContactSensitivityMap_lh'].array)  # shape: (N, 32492)
    rh = np.stack(df_ieeg['ContactSensitivityMap_rh'].array)  # shape: (N, 32492)
    all = np.concatenate([np.sum(lh, axis=0), np.sum(rh, axis=0)]) # shape: (64952)

    # --- Setup surface and plot layout ---
    surf_lh.append_array(np.sum(lh, axis=0), name="overlay1")
    surf_rh.append_array(np.sum(rh, axis=0), name="overlay2")
    surf_lh.append_array(df_yeo_surf['network_int'].values[:32492], name="overlay1_yeo")
    surf_rh.append_array(df_yeo_surf['network_int'].values[32492:], name="overlay2_yeo")
    surfs = {'lh1': surf_lh, 'lh2': surf_lh, 'rh1': surf_rh, 'rh2': surf_rh}
    layout = [['lh1', 'lh2', 'rh1', 'rh2']]
    view = [['lateral', 'medial', 'lateral', 'medial']]

    p = plot_surf(
        surfs, layout=layout, view=view, array_name=["overlay1_yeo","overlay1_yeo","overlay2_yeo","overlay2_yeo"], size=(1200, 300), zoom=1.3,
        color_bar='bottom', share='both', nan_color=(220, 220, 220, 1),
        cmap="CustomCmap_yeo", transparent_bg=True, return_plotter=True
    )

    # --- Helper: add colored spheres ---
    def add_colored_spheres(renderer, coords, radius=1.5, rotation=[-90, 0, 90]):
        for pos in coords:
            sphere = vtkSphereSource()
            sphere.SetCenter(*pos)
            sphere.SetRadius(radius)
            sphere.Update()
            actor = renderer.AddActor()
            actor.SetMapper(inputData=sphere.GetOutput())
            actor.GetProperty().SetColor([0,0,0])
            actor.GetProperty().SetOpacity(1.0)
            actor.RotateX(rotation[0])
            actor.RotateY(rotation[1])
            actor.RotateZ(rotation[2])
    # --- Extract transformed coordinates ---
    coords = df_ieeg[['X_nativepro', 'Y_nativepro', 'Z_nativepro']].to_numpy()
    # --- Render all hemispheres/views ---
    for i, rot in zip(range(4), [[-90,0,90], [-90,0,270]]*2):
        add_colored_spheres(p.renderers[i][0], coords, radius=1.5, rotation=rot)
    p.show()


    


    # ##### Merge iEEG + electrode geometry
    # merged_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_ieeg.pkl'
    # if os.path.exists(merged_path):
    #     df_ieeg = pd.read_pickle(merged_path).drop(columns=['indexChannel', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'])
    #     default_map = np.zeros(32492)
    #     # contact sensitivity maps are in Volts/(Ampere/m)
    #     df_ieeg['ContactSensitivityMap_lh'] = df_ieeg['ContactSensitivityMap_lh'].apply(lambda x: x if ~np.isnan(np.sum(x)) else default_map)
    #     df_ieeg['ContactSensitivityMap_rh'] = df_ieeg['ContactSensitivityMap_rh'].apply(lambda x: x if ~np.isnan(np.sum(x)) else default_map)
    #     df_ieeg['ContactOutsideBrain'] = df_ieeg['ContactOutsideBrain'].fillna(0)
    #     df_ieeg = df_ieeg.dropna()
    #     print(f"Loaded df_ieeg from {merged_path}")
    #     print(df_ieeg)
    # else:
    #     df_sensitivity = load_sensitivity_data_files()
    #     df_ieeg.to_pickle('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_sensitivity.pkl')
    #     print(df_sensitivity)
    #     df_ieeg = load_original_data_files()
    #     print(df_ieeg)
    #     df_elecs = load_electrode_data_files()
    #     print(df_elecs)
    #     df_ieeg = df_ieeg.merge(df_elecs, on=['Subject', 'Session', 'ChannelName'], how='left')
             

    #     
    #     df_ieeg.to_pickle(merged_path)
    #     print(f"Saved df_ieeg to {merged_path}")

    ################ plotting
    #subject = 'PX004'
    #df_ieeg = df_ieeg[df_ieeg['Subject'] == subject]
    
    #df_ieeg = add_nativepro_coords(df_ieeg)

    from scipy.signal import welch
    from scipy.stats import zscore



    # -- Step 2: Apply PSD computation to DataFrame --
    df_ieeg[['psd', 'freqs']] = df_ieeg.apply(compute_psd, axis=1, result_type='expand')
    # -- Step 3: Compute band power and z-score by frequency band --
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 80),
    }
    pxx_log = np.vstack(df_ieeg['psd'].to_numpy())
    freqs_ref = df_ieeg['freqs'].iloc[0]  # All freqs are assumed identical
    band_powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        band_idx = (freqs_ref >= fmin) & (freqs_ref < fmax)
        band_power = pxx_log[:, band_idx].mean(axis=1)
        band_powers[band_name] = zscore(band_power, nan_policy='omit')
    df_ieeg = pd.concat([df_ieeg, pd.DataFrame(band_powers, index=df_ieeg.index)], axis=1)
    # -- Step 4: Plot all PSDs --
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log-Relative PSD (a.u.)")
    plt.title("Channel-wise Normalized PSD (0.5–80 Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    lh = np.stack(df_ieeg['ContactSensitivityMap_lh'].array)  # shape: (N, 32492)
    rh = np.stack(df_ieeg['ContactSensitivityMap_rh'].array)  # shape: (N, 32492)
    alpha = df_ieeg['alpha'].values.reshape(-1, 1)            # shape: (N, 1)

    # Element-wise alpha weighting
    alpha_lh = lh * alpha
    alpha_rh = rh * alpha

    # Combine hemispheres and compute max projection across electrodes
    alpha_map = np.max(np.hstack([alpha_lh, alpha_rh]), axis=0)  # shape: (2×32492,) → concatenated hemispheres

    # Plot
    plot_hemispheres(
        surf32k_lh_infl, surf32k_rh_infl, alpha_map,
        size=(1200, 600), zoom=1.3,
        color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1),
        cmap="viridis", transparent_bg=True
    )

    # --- Create color mapping across all unique channels ---
    unique_channels = df_ieeg['ChannelName'].unique()
    cmap = cm.get_cmap("magma", len(unique_channels))
    channel_to_color = {ch: to_rgb(cmap(i)) for i, ch in enumerate(unique_channels)}
    colors = np.array([channel_to_color[ch] for ch in df_ieeg['ChannelName']])



if __name__ == "__main__":
    main()