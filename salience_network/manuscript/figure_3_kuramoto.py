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
from tqdm import tqdm
import seaborn as sns
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh

from nctpy.energies import integrate_u, get_control_inputs
from nctpy.pipelines import ComputeControlEnergy, ComputeOptimizedControlEnergy
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    convert_states_str2int,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
)
from nctpy.plotting import roi_to_vtx, null_plot, surface_plot, add_module_lines, set_plotting_params

from scipy.integrate import simpson as simps
import scipy as sp
import time
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

    for i_t in tqdm(range(total_steps), desc="Simulating"):
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


from sklearn.metrics import silhouette_score

def compute_CAPS(Phases_Save: np.ndarray, 
                 n_states: int = None, 
                 threshold: float = 0.1, 
                 seed: int = 0, 
                 min_states: int = 2, 
                 max_states: int = 10, 
                 return_best_n_states: bool = False):
    """
    Compute Co-Activation Patterns (CAPS) from Kuramoto simulation using KMeans clustering.

    Parameters:
        Phases_Save : np.ndarray
            Phase time series (N nodes × T time points).
        n_states : int or None
            If provided, number of CAPS (clusters); otherwise determined automatically.
        threshold : float
            Global synchrony threshold (0 < threshold < 1).
        seed : int
            Random seed for reproducibility.
        min_states : int
            Minimum number of clusters to try if n_states is None.
        max_states : int
            Maximum number of clusters to try if n_states is None.
        return_best_n_states : bool
            If True, return the number of selected clusters.

    Returns:
        caps : np.ndarray
            Cluster centers of CAPS (n_states × N).
        labels : np.ndarray
            Cluster labels over time (T_caps,).
        R_t : np.ndarray
            Global synchrony over time (T,).
        best_n_states (optional) : int
            Optimal number of clusters (only if return_best_n_states=True).
    """
    # Compute global synchrony R(t)
    R_t = np.abs(np.mean(np.exp(1j * Phases_Save), axis=0))

    # Select timepoints above the synchrony threshold
    mask = R_t > threshold
    if np.sum(mask) == 0:
        raise ValueError("No timepoints exceed the given synchrony threshold.")

    X = np.sin(Phases_Save[:, mask]).T  # shape: (T_selected, N)
    n_samples, n_features = X.shape

    if n_samples < min_states:
        raise ValueError("Too few timepoints above threshold to perform clustering.")

    # Determine optimal number of clusters
    if n_states is None:
        best_score = -np.inf
        best_k = min_states
        for k in range(min_states, min(max_states + 1, n_samples)):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels_k = kmeans.fit_predict(X)
            try:
                score = silhouette_score(X, labels_k)
            except ValueError:
                continue  # skip if silhouette fails (e.g., due to singleton clusters)
            if score > best_score:
                best_score = score
                best_k = k
        n_states = best_k

    # Final clustering with selected number of states
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    caps = kmeans.cluster_centers_

    if return_best_n_states:
        return caps, labels, R_t, n_states
    return caps, labels, R_t


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
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-100_conte69_rh.label.gii').darrays[0].data + 1950
    atlas_yeo_rh[atlas_yeo_rh == 1950] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-100_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    #df_label = df_label[df_label.network != 'medial_wall']
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    states, state_labels = convert_states_str2int(df_label['network'].values)

    #### Load connectivity matrix
    C = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-100_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    C = np.average(np.array([nib.load(f).darrays[0].data for f in C]), axis=0)
    C = np.log(np.triu(C,1)+C.T + 1)
    C = C[49:, 49:]
    C = np.delete(np.delete(C, 50, axis=0), 50, axis=1)
    N = C.shape[0] # Number of nodes
    C[np.eye(N, dtype=bool)] = 0
    C = C / np.mean(C[~np.eye(N, dtype=bool)])

    # Distance matrix 
    D = glob.glob("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-100_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii")
    D = np.average(np.array([nib.load(f).darrays[0].data for f in D]), axis=0)
    D = (np.triu(D,1)+D.T) / 1000 # convert to m
    D = D[49:, 49:]
    D = np.delete(np.delete(D, 50, axis=0), 50, axis=1)
    D[np.eye(N, dtype=bool)] = 0


    # --- Simulate Kuramoto model ---
    f = 40        # Node natural frequency (Hz)
    MD = 20e-3    # Mean delay (s)
    K = 8        # Global coupling strength
    Phases_Save, dt_save = Kuramoto_Delays_Run(C=C.copy(), D=D.copy(), f=f, K=K, MD=MD, tmax=200)
    plot_kuramoto_results(Phases_Save, dt_save, f, K, MD)
    #plot_matrices(C, D, Phases_Save, Order=None)

    caps, labels, R_t = compute_CAPS(Phases_Save, threshold=0.1)
    plot_values = np.full((caps.shape[0], df_label.label.shape[0]), fill_value=np.nan)
    plot_values[:, df_label.label.values != 'medial_wall'] = caps
    plot_values = map_to_labels(plot_values, yeo_surf, mask=df_yeo_surf.network.values != 'medial_wall')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plot_values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                            nan_color=(220, 220, 220, 0), cmap='coolwarm', transparent_bg=True)





    


if __name__ == "__main__":
    main()