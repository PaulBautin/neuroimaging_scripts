import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import tqdm
import time
import nibabel as nib

def Kuramoto_Delays_Run_AAL(C, D, f, K, MD):
    """
    Simulate delayed Kuramoto model dynamics with empirically-derived coupling and delays.
    """

    # Integration and simulation parameters
    dt = 1e-4
    tmax = 1.0
    t_prev = 0.0
    dt_save = 2e-3
    noise = 0.0

    N = C.shape[0]
    Omega = 2 * np.pi * f * dt * np.ones((N, 1))  # shape (N, 1)
    kC = K * C * dt
    dsig = np.sqrt(dt) * noise

    # Delay matrix: steps as integers
    if MD == 0:
        Delays = np.zeros_like(C, dtype=np.int32)
    else:
        avg_dist = D[C > 0].mean()
        Delays = (D / avg_dist * MD / dt).astype(np.int32)
    Delays[C == 0] = 0

    Max_History = Delays.max() + 1
    Delay_Index = Max_History - Delays - 1

    # Initialize phase history
    Phases_History = 2 * np.pi * np.random.rand(N, Max_History)
    Phases_History += Omega @ np.ones((1, Max_History))
    Phases_History %= 2 * np.pi

    num_save_points = int(tmax / dt_save)
    Phases_Save = np.zeros((N, num_save_points))
    save_interval = int(dt_save / dt)

    print(f"Running Kuramoto with K = {K}, Mean Delay = {MD * 1e3:.1f} ms")
    print(f"Max history length: {Max_History} steps")

    tic = time.time()
    nt = 0
    total_steps = int((t_prev + tmax) / dt)

    for i_t in tqdm.tqdm(range(total_steps), desc="Simulating"):
        Phase_Now = Phases_History[:, -1]  # shape (N,)

        # Compute delayed phases: gather using Delay_Index
        Delayed_Phases = Phases_History[
            np.arange(N)[:, None],
            Delay_Index
        ]  # shape (N, N)

        # Phase differences: delayed - current (broadcast Phase_Now over rows)
        Phase_Diff = Delayed_Phases - Phase_Now[:, None]  # shape (N, N)

        # Coupling input
        sumz = np.sum(kC * np.sin(Phase_Diff), axis=1)  # shape (N,)

        # Slide phase history
        if MD > 0:
            Phases_History[:, :-1] = Phases_History[:, 1:]

        # Update phases
        noise_term = dsig * np.random.randn(N) if noise > 0 else 0.0
        Phases_History[:, -1] = (Phase_Now + Omega[:, 0] + sumz + noise_term) % (2 * np.pi)

        # Save
        if i_t % save_interval == 0 and i_t * dt > t_prev:
            Phases_Save[:, nt] = Phases_History[:, -1]
            nt += 1

    toc = time.time() - tic
    sim_duration = t_prev + tmax
    print(f"Finished simulation in {toc:.2f} s (real time)")
    print(f"Simulated {sim_duration:.2f} s at speed ratio {toc / sim_duration:.2f}")

    return Phases_Save[:, :nt], dt_save

# Connectivity matrix
#C = np.load('/local_raid/data/pbautin/data/conn_labeled.npy')
C = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/dwi/connectomes/sub-PNC032_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii").darrays[0].data
C = np.triu(C,1)+C.T
C = C[49:, 49:]
C = np.delete(np.delete(C, 200, axis=0), 200, axis=1)
N = C.shape[0]
C[np.diag(np.ones(N))==0] /= C[np.diag(np.ones(N))==0].mean()

# Distance matrix 
#D = np.abs(np.eye(C.shape[0]) - 1)
D = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/dwi/connectomes/sub-PNC032_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii").darrays[0].data
D = np.triu(D,1)+D.T
D = D[49:, 49:]
D = np.delete(np.delete(D, 200, axis=0), 200, axis=1)
# Define here the parameters of the network model to be manipulated
# Node natural frequency in Hz
f = 40    # (i.e. f=40)):
# Mean Delay in seconds
MD = 0.021 # (i.e. MD = 0.01)
# Global Coupling strength
K = 4 

Phases_Save, dt_save = Kuramoto_Delays_Run_AAL(C, D, f, K, MD)

N_time = Phases_Save.shape[1] 
tmax = N_time * dt_save 
time = np.linspace(0, tmax, N_time)

# Plot simulated time series
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.plot(time, np.sin(Phases_Save.T))
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('sin(\theta)')
ax.set_title(['Coupled ' + str(f) + 'Hz oscillators with for K= ' + str(K) + ' and mean delay ' + str(MD*1e3) + ' ms'])
fig.suptitle(' K= ' + str(K) + ' and mean delay ' + str(MD*1e3) + ' ms')
plt.show()


# Power Spectrum
fig, axs = plt.subplots(1, 3, figsize=(15, 8))
fbins = 5000
freqZ = np.arange(fbins)/(dt_save*fbins)
ind100Hz = np.argmin(freqZ==100)
Fourier = np.fft.fft(np.sin(Phases_Save), n=fbins, axis=-1)
PSD = np.abs(Fourier)**2
ax = axs[0]
ax.plot(freqZ[:fbins//2], PSD[:, :fbins//2].mean(axis=0))
ax.set_xlim([0, 100])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')

# Plot the Order Parameter over time
ax = axs[1]
OP = np.abs(np.mean(np.exp(1j*Phases_Save), axis=0))
ax.plot(time, OP)
ax.set_ylim([0, 1])
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Order Parameter')

# Power Spectrum of the Order Parameter
ax = axs[2]
Fourier = np.fft.fft(OP-OP.mean(), n=fbins, axis=0)
PSD = np.abs(Fourier)**2
PSD = PSD[:fbins//2]
PSD /= PSD.sum()
ax.plot(freqZ[:fbins//2], PSD)
ax.set_xlim([0, 10])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')
plt.show()


# Plot Connectivity and Distance Matrices and mean Phase Coherence

# Change the order of units for visualization (optional)
Order = np.arange(0, N, 2) # This sorts the AAL areas into left and right hemispheres
Order = np.hstack((Order, np.arange(N-1, 0, -2)))

fig, axs = plt.subplots(1, 3, figsize=(15, 8))
ax = axs[0]
# colormap(jet)
ax.pcolormesh(C[Order, :][:, Order])

ax.set_xlabel('Nodes')
ax.set_ylabel('Nodes')
ax.set_title('Coupling Matrix')
ax.axis('square')
#plt.colorbar()

ax = axs[1]
ax.pcolormesh(D[Order, :][:, Order])
ax.set_xlabel('Nodes')
ax.set_ylabel('Nodes')
ax.set_title('Distance Matrix')
ax.axis('square')
#plt.colorbar()

ax = axs[2]
X = np.sin(Phases_Save)
FC = (X[:, None, :] * X[None, :, :]).mean(axis=-1)
ax.pcolormesh(FC[Order, :][:, Order])
ax.set_xlabel('Nodes')
ax.set_ylabel('Nodes')
ax.set_title('Correlation Matrix')
ax.axis('square')
plt.show()


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