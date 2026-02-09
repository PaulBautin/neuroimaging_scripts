import scipy.io as sio
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
import os
from brainspace import mesh
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.datasets import load_conte69
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from scipy.signal import welch
from scipy.signal import hilbert, butter, filtfilt
import nibabel as nib
import subprocess



def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data: np.ndarray
        Input signal.
    - fs: int
        Sampling frequency.
    - lowcut: float
        Low cutoff frequency.
    - highcut: float
        High cutoff frequency.
    - order: int
        Filter order.

    Returns:
    - filtered_data: np.ndarray
        Bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_phase_amplitude_coupling(data, fs, low_freq_band, high_freq_band):
    """
    Compute cross-frequency coupling using phase-amplitude coupling (PAC).

    Parameters:
    - data: np.ndarray
        Input signal (1D array).
    - fs: int
        Sampling frequency.
    - low_freq_band: tuple
        Low-frequency band (e.g., (4, 8) for theta).
    - high_freq_band: tuple
        High-frequency band (e.g., (30, 80) for gamma).

    Returns:
    - pac: float
        Modulation Index (MI) as a measure of PAC.
    """
    # Bandpass filter for low-frequency and high-frequency bands
    low_freq_signal = bandpass_filter(data, fs, low_freq_band[0], low_freq_band[1])
    high_freq_signal = bandpass_filter(data, fs, high_freq_band[0], high_freq_band[1])

    # Extract phase of low-frequency signal
    low_freq_phase = np.angle(hilbert(low_freq_signal))

    # Extract amplitude envelope of high-frequency signal
    high_freq_amplitude = np.abs(hilbert(high_freq_signal))

    # Compute phase-amplitude coupling (Modulation Index)
    n_bins = 18  # Number of phase bins
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    amplitude_means = np.zeros(n_bins)

    for i in range(n_bins):
        # Find indices where phase is within the current bin
        indices = np.where((low_freq_phase >= phase_bins[i]) & (low_freq_phase < phase_bins[i + 1]))[0]
        # Compute mean amplitude for the current phase bin
        amplitude_means[i] = np.mean(high_freq_amplitude[indices])

    # Normalize amplitude means
    amplitude_means /= np.sum(amplitude_means)

    # Compute Modulation Index (MI)
    pac = -np.sum(amplitude_means * np.log(amplitude_means + 1e-10)) / np.log(n_bins)

    return pac, phase_bins, amplitude_means

def plot_phase_amplitude_coupling(phase_bins, amplitude_means):
    """
    Plot phase-amplitude coupling.

    Parameters:
    - phase_bins: np.ndarray
        Phase bins (edges).
    - amplitude_means: np.ndarray
        Mean amplitude for each phase bin.
    """
    # Convert phase bins to centers
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(phase_centers, amplitude_means, width=2 * np.pi / len(phase_bins), align='center', edgecolor='k')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Phase-Amplitude Coupling')
    plt.show()


def RAD(x, centre=False, tau=1):
    """
    Harris, Gollo, & Fulcher's rescaled auto-density (RAD) noise-insensitive
    metric for inferring the distance to criticality.

    Parameters
    ----------
    x: array
        A time-series input vector

    centre : boolean
        Whether to centre the time series and take absolute values

    tau: integer
        The embedding and differencing delay in units of the timestep

    Returns
    -------
    The RAD feature value.
    """

    # ensure that x is in the form of a numpy array
    x = np.array(x)
    
    # if specified: centre the time series and take the absolute value
    if centre:
        x = x - np.median(x)
        x = np.abs(x)

    # Delay embed at interval tau
    y = x[tau:]
    x = x[:-tau]

    # Median split
    subMedians = x < np.median(x)
    superMedianSD = np.std(x[~subMedians], ddof=1)
    subMedianSD = np.std(x[subMedians], ddof=1)

    # Properties of the auto-density
    sigma_dx = np.std(y - x)
    densityDifference = (1/superMedianSD) - (1/subMedianSD)

    # return RAD
    return sigma_dx * densityDifference


# Compute the catch22 features
def compute_features(x):
    res = catch22.catch22_all(x, catch24 = True)
    return res['values'] # just return the values 



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



def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_sphere = read_surface(micapipe + '/surfaces/fsLR-5k.L.sphere.surf.gii', itype='gii')
    surf5k_rh_sphere = read_surface(micapipe + '/surfaces/fsLR-5k.R.sphere.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})

    #### load yeo atlas 7 network fslr5k
    atlas_yeo_lh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh_5k = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh_5k[atlas_yeo_rh_5k == 1800] = 2000
    yeo_surf_5k = np.concatenate((atlas_yeo_lh_5k, atlas_yeo_rh_5k), axis=0).astype(float)
    df_yeo_surf_5k = pd.DataFrame(data={'mics': yeo_surf_5k})

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    df_yeo_surf_5k = df_yeo_surf_5k.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf['salience_border'] = salience_border == 1

    ##### Extract the data #####
    data = sio.loadmat("/local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat")
    print(data.keys())
    # Data_W: matrix with one column per channel, and 13600 samples containing all the signals for wakefulness
    data_w = data['Data_W'].T
    channel_type_raw = data['ChannelType']
    channel_type_flat = [item[0][0] for item in channel_type_raw]
    channel_type_mapping_int = {
        'D': 1,  # Dixi intracerebral electrodes
        'M': 2,  # Homemade MNI intracerebral electrodes
        'A': 3,  # AdTech intracerebral electrodes
        'G': 4   # AdTech subdural strips and grids
    }
    channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])
    data['ChannelPosition'][:,0] = np.abs(data['ChannelPosition'][:,0])

    # Create custom colormap
    colors = ['white', 'purple', 'orange', 'red', 'blue']  # white for 0 (no channel)
    custom_cmap = ListedColormap(colors)

    # Create surface polydata objects
    surf_lh = mesh.mesh_creation.build_polydata(points=data['NodesLeft'], cells=data['FacesLeft'] - 1)
    surf_rh = mesh.mesh_creation.build_polydata(points=data['NodesRight'], cells=data['FacesRight'] - 1)
    mesh.mesh_io.write_surface(surf_lh, '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_lh.surf.gii')
    mesh.mesh_io.write_surface(surf_rh, '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_rh.surf.gii')
    print(data['NodesLeft'].shape)
    plot_hemispheres(surf_lh, surf_rh)



    ##### Cross-frequency coupling #####
    #low_freq_band = (4, 8)  # Theta band
    low_freq_band = (8, 13)  # Alpha band
    #low_freq_band = (13, 30)  # Beta band
    high_freq_band = (30, 80)  # Gamma band
    fs = data['SamplingFrequency'][0][0]
    # Compute PAC for each channel
    pac_values = np.zeros(data_w.shape[0])
    for i in range(data_w.shape[0]):
        pac, phase_bins, amplitude_means = compute_phase_amplitude_coupling(data_w[i], fs, low_freq_band, high_freq_band)
        pac_values[i] = pac

    print(pac_values.shape)

    ##### surface data #####
    # Surface plot with 3mm radius
    all_nodes = np.vstack((data['NodesLeft'], data['NodesRight']))
    tree = cKDTree(all_nodes)
    plot_values = np.zeros(all_nodes.shape[0])
    radius = 3  # 3mm radius
    # Update plot_values with integer mapping
    for i, channel_pos in enumerate(data['ChannelPosition']):
        nearby_indices = tree.query_ball_point(channel_pos, radius)
        plot_values[nearby_indices] = channel_integers[i]
        plot_values[nearby_indices] = pac_values[i]
    print(plot_values.shape)

    plot_values_lh = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/L.transformed_and_reprojected.func.gii').darrays[0].data
    plot_values_rh = nib.load('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/R.transformed_and_reprojected.func.gii').darrays[0].data
    #combine the two hemispheres
    plot_values = np.hstack(np.concatenate((plot_values_lh, plot_values_rh), axis=0)).astype(float)
    print(plot_values)
    print(plot_values.shape)
    
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plot_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #     nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    





if __name__ == "__main__":
    main()