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
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.datasets import load_conte69
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from scipy.signal import welch
from scipy.signal import hilbert, butter, filtfilt

params = {"ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
            'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')


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


import subprocess



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

plot_values[plot_values == 0] = np.nan 

# Plot surface with custom colormap
plot_hemispheres(surf_lh, surf_rh, 
                array_name=plot_values, 
                size=(1200, 300), 
                zoom=1.25, 
                color_bar='bottom', 
                share='both', 
                background=(0,0,0),
                nan_color=(250, 250, 250, 1), 
                cmap='hot',
                interactive=False)

# Plot surface with custom colormap
plot_hemispheres(surf_lh, surf_rh, 
                array_name=plot_values, 
                size=(1200, 300), 
                zoom=1.25, 
                color_bar='bottom', 
                share='both', 
                background=(0,0,0),
                nan_color=(250, 250, 250, 1), 
                cmap=custom_cmap,
                interactive=False)

f, pxx_log = compute_psd(data['SamplingFrequency'], data['Data_W'].T)
# plot frequency vs power
plt.plot(f[0, :], pxx_log[0, :])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Log(Normalized PSD)')
plt.title('Power Spectral Density')

# Add lines to delimitate frequency bands
plt.axvline(x=0.5, color='cyan', linestyle='--', label='δ (0.5-4Hz)')
plt.axvline(x=4, color='cyan', linestyle='--')
plt.axvline(x=8, color='green', linestyle='--', label='θ (4-8Hz)')
plt.axvline(x=13, color='yellow', linestyle='--', label='α (8-13Hz)')
plt.axvline(x=30, color='orange', linestyle='--', label='β (13-30Hz)')
plt.axvline(x=80, color='red', linestyle='--', label='γ (30-80Hz)')

plt.legend()
plt.show()

##### Compute the features #####
threads_to_use = os.cpu_count()
results_rad = np.asarray(Parallel(n_jobs=threads_to_use)(
    delayed(RAD)(data_w[i]) for i in range(data_w.shape[0])))
results = np.asarray(Parallel(n_jobs=threads_to_use)(
    delayed(compute_features)(data_w[i]) for i in range(data_w.shape[0])))
print(results.shape)
feature_index = np.argwhere(np.asarray(catch22.catch22_all(data_w[0], catch24 = True)['names']) == 'FC_LocalSimple_mean3_stderr')[0][0]

# Surface plot with 3mm radius
all_nodes = np.vstack((data['NodesLeft'], data['NodesRight']))
tree = cKDTree(all_nodes)
plot_values = np.zeros(all_nodes.shape[0])
radius = 3  # 3mm radius

# Apply radius to each channel
for i, channel_pos in enumerate(data['ChannelPosition']):
    nearby_indices = tree.query_ball_point(channel_pos, radius)
    plot_values[nearby_indices] = results[i,feature_index]
    #plot_values[nearby_indices] = results_rad[i]

plot_values[plot_values == 0] = np.nan 

# Plot surface
plot_hemispheres(surf_lh, surf_rh, 
                array_name=plot_values, 
                size=(1200, 300), 
                zoom=1.25, 
                color_bar='bottom', 
                share='both', 
                background=(0,0,0),
                nan_color=(250, 250, 250, 1), 
                cmap='hot',
                interactive=False)


# 
# results_list = Parallel(n_jobs=threads_to_use)(
#     delayed(compute_features)(data_w[i]) for i in range(len(data_w))
# )
# results = np.asarray(results_list)

# ##### Plot the data #####
# # Surface plot
# plot_values = np.zeros(all_nodes.shape[0])
# plot_values[indices] = feature_rad
# plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                              nan_color=(250, 250, 250, 1), cmap='Purples', interactive=False)

# # Create figure with multiple subplots (1 row, 3 columns)
# fig, axes = plt.subplots(1, 3, figsize=(27, 9), subplot_kw={'projection': '3d'})
# view_angles = [(0, 0), (0, -90), (0, 180)]
# for ax, (elev, azim) in zip(axes, view_angles):
#     sc = ax.scatter(
#         channel_position[:, 0], 
#         channel_position[:, 1], 
#         channel_position[:, 2], 
#         c=results[-2],  # Use the assigned colors
#         s=10, 
#         alpha=0.8
#     )
#     ax.view_init(elev, azim)  # Set view angles
#     ax.axis('equal')
#     ax.axis('off')
# plt.show()