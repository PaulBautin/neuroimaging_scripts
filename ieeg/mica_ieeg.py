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

params = {"ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
            'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')


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

def register_surface_with_msm(freesurfer_surface, hemisphere, output_dir, fsLR_template_dir, msm_config):
    """
    Perform surface registration using FSL MSM.

    Parameters:
    - freesurfer_surface: str
        Path to the FreeSurfer surface file (e.g., lh.pial or rh.pial).
    - freesurfer_sphere: str
        Path to the FreeSurfer sphere file (e.g., lh.sphere.reg or rh.sphere.reg).
    - hemisphere: str
        Hemisphere ('L' for left, 'R' for right).
    - output_dir: str
        Directory to save the output registered surface.
    - fsLR_template_dir: str
        Directory containing fsLR template surfaces.
    - msm_config: str
        Path to the MSM configuration file.

    Returns:
    - output_file: str
        Path to the registered surface.
    """
    # Convert FreeSurfer surface and sphere to GIFTI
    gifti_surface = freesurfer_surface

    # Define fsLR template sphere
    ref_surface = f"{fsLR_template_dir}/fsLR-32k.{hemisphere}.surf.gii"
    output_prefix = f"{output_dir}/{hemisphere.lower()}.fsLR-registered"

    # Run MSM
    subprocess.run([
        "msm",
        f"--inmesh={gifti_surface}",
        f"--refmesh={ref_surface}",
        f"--indata={gifti_surface}",
        f"--out={output_prefix}",
        f"--conf={msm_config}"
    ], check=True)

    return f"{output_prefix}.gii"

# Example usage
output_dir = "/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg"
fsLR_template_dir = "/local_raid/data/pbautin/software/micapipe/surfaces"
msm_config = "/path/to/msm_config"
lh_surface = "/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_lh.surf.gii"
rh_surface = "/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_rh.surf.gii"

lh_registered = register_surface_with_msm(lh_surface, lh_sphere, "L", output_dir, fsLR_template_dir, msm_config)
rh_registered = register_surface_with_msm(rh_surface, rh_sphere, "R", output_dir, fsLR_template_dir, msm_config)

print(f"Left hemisphere registered to fsLR: {lh_registered}")
print(f"Right hemisphere registered to fsLR: {rh_registered}")



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

#plot_values[plot_values == 0] = np.nan 

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