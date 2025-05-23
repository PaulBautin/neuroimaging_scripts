import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import tqdm
import time
import nibabel as nib
from scipy.signal import welch

from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69
import glob
from sklearn.decomposition import PCA, FastICA
import os

import pycatch22 as catch22
from joblib import Parallel, delayed
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def Kuramoto_Delays_Run(C, D, f, K, MD, tmax=2.0):
    """
    Efficient simulation of delayed Kuramoto dynamics using a circular buffer.
    """
    # --- Simulation parameters ---
    dt = 1e-4            # Integration resolution (seconds)
    t_prev = 0.0         # Preview time before saving (seconds)
    dt_save = 2e-3       # Resolution of saved simulations (seconds)
    noise = 0.0          # Noise level (standard deviation)

    # --- Derived parameters ---
    N = C.shape[0]                                        # Number of oscillators
    Omega = 2 * np.pi * f * dt * np.ones(N)               # Oscillator angular step in rad
    kC = K * C * dt                                       # Pre-scaled coupling matrix
    dsig = np.sqrt(dt) * noise                            # Noise term scaled with dt

    # --- Delay matrix computation ---
    if MD == 0:
        Delays = np.zeros((N, N), dtype=np.int16)  # All delays are zero
    else:
        D[C > 0] = D[C > 0] / D[C > 0].mean()
        Delays = np.round(D * MD / dt).astype(np.int16)  # Delays in time steps
    Max_History = Delays.max() + 1
    Delay_Index = Max_History - Delays
    print(Delay_Index)

    # Phase history as circular buffer
    Phase_History = 2 * np.pi * np.random.rand(N, Max_History)
    pointer = 0  # Current write index

    num_save_points = int(tmax / dt_save)
    Phases_Save = np.zeros((N, num_save_points))
    save_interval = int(dt_save / dt)

    print(f"Running Kuramoto with K = {K}, Mean Delay = {MD * 1e3:.1f} ms")
    print(f"Max history length: {Max_History} steps")

    tic = time.time()
    total_steps = int((t_prev + tmax) / dt)
    save_counter = 0

    row_idx = np.arange(N)[:, None]

    for i_t in tqdm.tqdm(range(total_steps), desc="Simulating"):
        # Current phase at write pointer
        Phase_Now = Phase_History[:, pointer]

        # Compute delayed indices with circular buffer
        delayed_indices = (pointer - Delay_Index) % Max_History
        Delayed_Phases = Phase_History[row_idx, delayed_indices]  # (N, N)

        Phase_Diff = Delayed_Phases - Phase_Now[:, None]
        sumz = np.sum(kC * np.sin(Phase_Diff), axis=1)

        noise_term = dsig * np.random.randn(N) if noise > 0 else 0.0
        next_phase = (Phase_Now + Omega + sumz + noise_term) % (2 * np.pi)

        # Advance circular buffer
        pointer = (pointer + 1) % Max_History
        Phase_History[:, pointer] = next_phase

        if i_t % save_interval == 0 and i_t * dt > t_prev:
            Phases_Save[:, save_counter] = next_phase
            save_counter += 1

    toc = time.time() - tic
    sim_duration = t_prev + tmax
    print(f"Finished simulation in {toc:.2f} s (real time)")
    print(f"Simulated {sim_duration:.2f} s at speed ratio {toc / sim_duration:.2f}")

    return Phases_Save[:, :save_counter], dt_save


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

def plot_ica_component(component_idx, A_, df_label, yeo_surf, surf_lh, surf_rh):
    """
    Plot a single ICA component on the cortical surface.
    """
    plot_values = np.full(df_label.label.shape, np.nan)
    mask = df_label.label.values != 'other'
    plot_values[mask] = A_[:, component_idx]

    plot_values = map_to_labels(plot_values, yeo_surf)

    plot_hemispheres(
        surf_lh, surf_rh, array_name=plot_values,
        size=(1200, 300), zoom=1.25, color_bar='bottom',
        share='both', background=(0, 0, 0), nan_color=(250, 250, 250, 1),
        cmap='coolwarm', transparent_bg=True, color_range='sym'
    )


def plot_kuramoto_results(Phases_Save, dt_save, f, K, MD):
    """
    Plot Kuramoto simulation results: time series, synchrony, and power spectra.
    """
    N_time = Phases_Save.shape[1]
    tmax = N_time * dt_save
    time = np.linspace(0, tmax, N_time)

    fbins = 5000
    freqZ = np.fft.fftfreq(fbins, d=dt_save)
    freqZ = freqZ[:fbins//2]

    # Order parameter (synchrony)
    OP = np.abs(np.mean(np.exp(1j * Phases_Save), axis=0))

    # Compute power spectrum of phases
    phase_signal = np.sin(Phases_Save)
    fft_phases = np.fft.fft(phase_signal, n=fbins, axis=-1)
    PSD_phases = np.abs(fft_phases[:, :fbins//2])**2
    PSD_phases_avg = PSD_phases.mean(axis=0)

    # Compute power spectrum of order parameter
    fft_op = np.fft.fft(OP - OP.mean(), n=fbins)
    PSD_op = np.abs(fft_op[:fbins//2])**2
    PSD_op /= PSD_op.sum()

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # Time series
    axes[0, 0].plot(time, np.sin(Phases_Save.T), lw=0.5, alpha=0.7)
    axes[0, 0].set_ylabel(r'$\sin(\theta)$')
    axes[0, 0].set_title('Kuramoto Oscillators')

    # Synchrony time series
    axes[1, 0].plot(time, OP, color='C0')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Order parameter (R(t))')
    axes[1, 0].set_ylim([0, 1])

    # Power spectrum of sin(theta)
    axes[0, 1].plot(freqZ, PSD_phases_avg, color='C0')
    axes[0, 1].set_xlim([0, 100])
    axes[0, 1].set_title('Power Spectrum: sin($\\theta$)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_xlabel('Frequency (Hz)')

    # Power spectrum of order parameter
    axes[1, 1].plot(freqZ, PSD_op, color='C0')
    axes[1, 1].set_xlim([0, 10])
    axes[1, 1].set_title('Power Spectrum: Order Parameter')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Normalized Power')

    plt.suptitle(f'Kuramoto Model Results (f={f}Hz, K={K}, MD={MD*1e3:.1f} ms)', fontsize=16, y=1.02)
    plt.show()


def plot_matrices(C, D, Phases_Save, Order=None, cmap='viridis'):
    """
    Plot coupling (C), distance (D), and functional correlation matrices.
    Parameters:
        C (ndarray): Coupling matrix
        D (ndarray): Distance matrix
        Phases_Save (ndarray): Time series of oscillator phases
        Order (ndarray): Optional node reordering array
        cmap (str): Colormap to use
    """
    if Order is None:
        Order = np.arange(C.shape[0])
    
    # Compute functional correlation matrix
    PC = np.cos(Phases_Save[:, None, :] - Phases_Save[None, :, :])
    FC = np.mean(PC, axis=-1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Plot Coupling Matrix
    im0 = axs[0].imshow(C[Order][:, Order], cmap=cmap)
    axs[0].set_title('Structural Coupling (C)')
    axs[0].set_xlabel('Node index')
    axs[0].set_ylabel('Node index')
    axs[0].set_aspect('equal')
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    # Plot Distance Matrix
    im1 = axs[1].imshow(D[Order][:, Order], cmap='plasma')
    axs[1].set_title('Distance Matrix (D)')
    axs[1].set_xlabel('Node index')
    axs[1].set_ylabel('Node index')
    axs[1].set_aspect('equal')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    # Plot Functional Correlation Matrix
    im2 = axs[2].imshow(FC[Order][:, Order], cmap='coolwarm', vmin=-1, vmax=1)
    axs[2].set_title('Phase Correlation (FC)')
    axs[2].set_xlabel('Node index')
    axs[2].set_ylabel('Node index')
    axs[2].set_aspect('equal')
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    fig.suptitle('Connectivity, Distance, and Functional Correlation Matrices', fontsize=14, y=1.02)
    plt.show()


def compute_CAPS(Phases_Save, n_states=4, threshold=0.7, seed=0):
    """
    Compute CAPS (Co-Activation Patterns) from Kuramoto simulation.

    Parameters:
        Phases_Save : np.ndarray
            Phase time series (N nodes × T time points).
        n_states : int
            Number of coactivation patterns (clusters).
        threshold : float
            Global synchrony threshold (0 < threshold < 1).
        seed : int
            Random seed for KMeans reproducibility.

    Returns:
        caps : np.ndarray
            Cluster centers of CAPS (n_states × N).
        labels : np.ndarray
            Cluster labels over time (T_caps,).
        R_t : np.ndarray
            Global synchrony over time (T,).
    """
    # Global synchrony R(t)
    R_t = np.abs(np.mean(np.exp(1j * Phases_Save), axis=0))

    # Use only timepoints above synchrony threshold
    mask = R_t > threshold
    if mask.sum() == 0:
        raise ValueError("No timepoints exceed the given synchrony threshold.")

    # Extract instantaneous signals (e.g., sin phase)
    X = np.sin(Phases_Save[:, mask]).T  # Shape: (T_masked, N)

    # K-means clustering on co-activation vectors
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    caps = kmeans.cluster_centers_  # Shape: (n_states, N)

    return caps, labels, R_t


def main():
    params = {"ytick.color" : "w",
                "xtick.color" : "w",
                "axes.labelcolor" : "w",
                "axes.edgecolor" : "w",
                'font.size': 22}
    plt.rcParams.update(params)
    plt.style.use('dark_background')


    # Load the data
    surf_lh, surf_rh = load_conte69()
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)

    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
    df_label.fillna('other', inplace=True)
    df = pd.DataFrame(data={'mics': yeo_surf})
    df = df.merge(df_label, on='mics',validate="many_to_one", how='left')
    state, state_name = convert_states_str2int(df['label'].values)
    state = state.astype(float)
    # salience
    salience = state == np.where(state_name == 'SalVentAttn')[0][0]

    # Connectivity matrix
    C = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    C = np.average(np.array([nib.load(f).darrays[0].data for f in C]), axis=0)
    C = np.log(np.triu(C,1)+C.T + 1)
    C = C[49:, 49:]
    C = np.delete(np.delete(C, 200, axis=0), 200, axis=1)
    N = C.shape[0] # Number of nodes
    C[np.eye(N, dtype=bool)] = 0
    C = C / np.mean(C[~np.eye(N, dtype=bool)])

    # Distance matrix 
    D = glob.glob("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii")
    D = np.average(np.array([nib.load(f).darrays[0].data for f in D]), axis=0)
    D = (np.triu(D,1)+D.T) / 1000 # convert to m
    D = D[49:, 49:]
    D = np.delete(np.delete(D, 200, axis=0), 200, axis=1)
    D[np.eye(N, dtype=bool)] = 0


    # --- Simulate Kuramoto model ---
    f = 40        # Node natural frequency (Hz)
    MD = 20e-3    # Mean delay (s)
    K = 10         # Global coupling strength
    
    MD_list = [5e-3, 10e-3, 15e-3, 25e-3, 30e-3, 35e-3, 40e-3]
    MD_list = [20e-3]
    #K_list = [5, 10, 15, 20, 25, 30, 35, 40]
    # Prepare figure
    #plt.figure(figsize=(12, 6))
    for MD in MD_list:
        # Define output file
        kuramoto_file = f"/local_raid/data/pbautin/software/neuroimaging_scripts/functional/kuramoto_f{f}_K{K}_MD{int(MD*1e3)}ms.npz"

        # Load if file exists, otherwise simulate and save
        if os.path.exists(kuramoto_file):
            data = np.load(kuramoto_file)
            Phases_Save = data["Phases_Save"]
            dt_save = data["dt_save"]
        else:
            Phases_Save, dt_save = Kuramoto_Delays_Run(C, D, f=f, K=K, MD=MD, tmax=20)
            np.savez(kuramoto_file, Phases_Save=Phases_Save, dt_save=dt_save)
            # Compute global synchrony R(t)
    #     R_t = np.abs(np.mean(np.exp(1j * Phases_Save), axis=0))
    #     time = np.linspace(0, 20, Phases_Save.shape[1])

    #     # Plot R(t)
    #     plt.plot(time, R_t, label=f"K = {K}")

    # # Finalize plot
    # plt.xlabel("Time (s)")
    # plt.ylabel("Synchrony R(t)")
    # plt.title(f"Kuramoto Synchrony R(t) for f = {f}Hz and MD = {int(MD*1e3)} ms")
    # plt.ylim(0, 0.2)
    # #plt.legend(title="Average time delay (MD)")
    # plt.legend(title="Coupling strength K")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    caps, labels, R_t = compute_CAPS(Phases_Save, n_states=3, threshold=0.0)
    for cap in caps:
        # Assign CAP values to cortical labels
        plot_values = np.full(df_label.label.shape, np.nan)
        plot_values[df_label.label.values != 'other'] = cap
        plot_values = map_to_labels(plot_values, yeo_surf)

        # Visualize on cortical surface
        plot_hemispheres(
            surf_lh, surf_rh, array_name=plot_values,
            size=(1200, 300), zoom=1.25, color_bar='bottom',
            share='both', background=(0, 0, 0), nan_color=(250, 250, 250, 1),
            cmap='coolwarm', transparent_bg=True, color_range='sym'
        )

    plot_kuramoto_results(Phases_Save, dt_save, f, K, MD)
    plot_matrices(C, D, Phases_Save, Order=None)

    # --- ICA decomposition ---
    ica = FastICA(n_components=3, whiten="arbitrary-variance", random_state=0)
    S_ = ica.fit_transform(np.sin(Phases_Save.T))  # ICA on sine-transformed phases
    A_ = ica.mixing_

    # --- Plot all ICA components ---
    for idx in range(3):
        plot_ica_component(idx, A_, df_label, yeo_surf, surf_lh, surf_rh)




    def compute_features(x):
        res = catch22.catch22_all(x, catch24=True)
        return res['values']

    print(f"Number of cores available: {os.cpu_count()}")
    threads_to_use = max(1, os.cpu_count() // 2)  # Use half the cores, ensuring at least 1 thread
    results_list = Parallel(n_jobs=threads_to_use)(
        delayed(compute_features)(np.sin(Phases_Save.T)[:, i])  for i in range(Phases_Save.T.shape[1]))

    # Reshape the results into a 3D matrix: Subjects x Regions x Features
    results = results_list

    # compute and plot the region per region similarity
    # compute the correlation matrix
    print(np.mean(results, axis=0).shape)
    corr_matrix = np.corrcoef(results)
    plt.figure(figsize=(10, 10))
    plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Cortical regions')
    plt.ylabel('Cortical regions')
    plt.show()

    # plot the first two PCA components on surfaces
    pca = PCA(n_components=10)
    pca.fit_transform(zscore(results).T)
    # scatter plot of variance explained
    plt.figure(figsize=(10, 10))
    plt.scatter(np.arange(0,10), pca.explained_variance_ratio_)
    plt.xlabel('PCA component')
    plt.ylabel('Variance explained')
    plt.show()

    plot_values = np.zeros(df_label.label.values.shape)
    plot_values[df_label.label.values != 'other'] = pca.components_[0]
    plot_values[df_label.label.values == 'other'] = np.nan
    mask = (yeo_surf != 1000) & (yeo_surf != 2000)
    plot_values = map_to_labels(plot_values, yeo_surf)
    # mask = salience
    # When salience true multiply by plot_values
    # plot_values = np.zeros(salience.shape)
    # plot_values[salience] = pca.components_[0]
    # plot_values[plot_values == 0] = np.nan
    # print(plot_values)

    plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                                nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


if __name__ == "__main__":
    main()











# # --- ICA decomposition ---
# ica = FastICA(n_components=3, whiten="arbitrary-variance", random_state=0)
# S_ = ica.fit_transform(np.sin(Phases_Save.T))  # ICA on sine-transformed phases
# A_ = ica.mixing_

# # --- Plot all ICA components ---
# for idx in range(3):
#     plot_ica_component(idx, A_, df_label, yeo_surf, surf_lh, surf_rh)






df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
df_label.fillna('other', inplace=True)
print(df_label)

# Define SN regions (Desikan labels)
salience_labels = ['SalVentAttn']

# Load region labels in order (must match your connectivity matrix)
region_labels = df_label.label.values
print(region_labels)

# Get indices of SN nodes
sn_indices = [i for i, label in enumerate(region_labels) if any(s in label for s in salience_labels)]
print(sn_indices)







T = 100
dt = 0.1
time = np.arange(0, T, dt)

# Random initial phases and natural frequencies
theta0 = 2 * np.pi * np.random.rand(n_nodes)
omega = 2 * np.pi * (np.random.rand(n_nodes) * 0.5 + 0.5)

# Define perturbation
def sn_perturb(t):
    return 4.0 if 20 <= t <= 40 else 0.0

# Run simulation
from scipy.integrate import odeint
theta = odeint(kuramoto_empirical, theta0, time, args=(omega, 2.0, W, sn_indices, sn_perturb))

# Global synchrony R(t)
R = np.abs(np.mean(np.exp(1j * theta), axis=1))

# Salience-specific synchrony
R_sn = np.abs(np.mean(np.exp(1j * theta[:, sn_indices]), axis=1))

plt.figure(figsize=(10, 4))
plt.plot(time, R, label='Global Synchrony')
plt.plot(time, R_sn, label='SN Synchrony', linestyle='--')
plt.axvspan(20, 40, color='red', alpha=0.2, label='SN Perturbation')
plt.xlabel('Time')
plt.ylabel('Synchrony')
plt.title('Global and Salience Network Synchrony Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Choose a few regions for clarity (e.g., SN and key non-SN nodes)
region_indices = sn_indices + [10, 20, 30]  # adjust as needed

plt.figure(figsize=(12, 5))
for idx in region_indices:
    plt.plot(time, theta[:, idx], label=region_labels[idx])
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')
plt.title('Individual Phase Trajectories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def compute_plv(phases):
    n = phases.shape[1]
    plv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            delta_phase = phases[:, i] - phases[:, j]
            plv[i, j] = np.abs(np.mean(np.exp(1j * delta_phase)))
    return plv


window_size = int(5 / dt)  # 5-second window
step_size = int(1 / dt)    # 1-second step

plv_series = []
times = []

for start in range(0, len(time) - window_size, step_size):
    end = start + window_size
    window_phases = theta[start:end]
    plv_matrix = compute_plv(window_phases)
    plv_series.append(plv_matrix)
    times.append(time[start + window_size // 2])


import matplotlib.colors as mcolors

# Compute average PLV within SN across time
sn_fc = []
for plv in plv_series:
    sn_sub = plv[np.ix_(sn_indices, sn_indices)]
    sn_fc.append(np.mean(sn_sub[np.triu_indices(len(sn_indices), k=1)]))

plt.figure(figsize=(10, 4))
plt.plot(times, sn_fc, label='Mean SN FC (PLV)', color='purple')
plt.axvspan(20, 40, color='red', alpha=0.2, label='SN Perturbation')
plt.xlabel('Time (s)')
plt.ylabel('Mean PLV')
plt.title('Salience Network Functional Connectivity Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from matplotlib import cm

selected_times = [10, 30, 60]  # seconds
for t_sel in selected_times:
    idx = np.argmin(np.abs(np.array(times) - t_sel))
    plt.figure(figsize=(6, 5))
    plt.imshow(plv_series[idx], cmap='viridis', vmin=0, vmax=1)
    plt.title(f"PLV Matrix at t = {times[idx]:.1f} s")
    plt.colorbar(label='PLV')
    plt.tight_layout()
    plt.show()


plv_sn_sn = [plv[np.ix_(sn_indices, sn_indices)] for plv in plv_series]
plv_sn_sn = np.stack(plv_sn_sn)  # shape: (T, SN, SN)

# Flatten upper triangle to 1D series per edge
from itertools import combinations
sn_pairs = list(combinations(range(len(sn_indices)), 2))
fc_edges = np.array([
    [plv_sn_sn[:, i, j] for i, j in sn_pairs]
]).squeeze().T  # shape: (T, #edges)

plt.figure(figsize=(12, 5))
for i in range(fc_edges.shape[1]):
    plt.plot(times, fc_edges[:, i], alpha=0.7)
plt.axvspan(20, 40, color='red', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('PLV')
plt.title('Edgewise FC Dynamics within Salience Network')
plt.grid(True)
plt.tight_layout()
plt.show()