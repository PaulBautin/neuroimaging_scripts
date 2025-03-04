from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Effective connectivity of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

import numpy as np
import networkx as nx
from scipy.linalg import expm

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


import scipy.linalg

import bct.algorithms as bct_alg
import bct.utils as bct

def mean_first_passage_time(W, tol=1e-3):
    """
    Calculate mean first passage time.

    The first passage time from i to j is the expected number of steps it takes
    a random walker starting at node i to arrive for the first time at node j.
    The mean first passage time is not a symmetric measure: `mfpt(i,j)` may be
    different from `mfpt(j,i)`.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) ndarray
        Weighted/unweighted, direct/undirected connection weight/length array
    tol : float, optional
        Tolerance for eigenvalue of 1. Default: 1e-3

    Returns
    -------
    mfpt : (N x N) ndarray
        Pairwise mean first passage time array

    References
    ----------
    .. [1] Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N.,
       van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013). Exploring the
       morphospace of communication efficiency in complex networks. PLoS One,
       8(3), e58070.
    """
    P = W / np.sum(W, axis=1)[:, None]  # transition matrix
    n = len(P)
    D, V = np.linalg.eig(P.T)
    D_minidx = np.argmin(np.abs(D - 1))

    if D[D_minidx] > 1 + tol:
        raise ValueError(
            f"Cannot find eigenvalue of 1. Minimum eigenvalue is larger than {tol}."
        )

    w = V[:, D_minidx][None, :]
    w /= np.sum(w)
    W_prob = np.real(np.repeat(w, n, 0))
    Z = np.linalg.inv(np.eye(n) - P + W_prob)  # fundamental matrix
    mfpt = (np.repeat(np.diag(Z)[None, :], n, 0) - Z) / W_prob
    return mfpt


def diffusion_efficiency(W):
    """
    Calculate diffusion efficiency.

    The diffusion efficiency between nodes i and j is the inverse of the
    mean first passage time from i to j, that is the expected number of
    steps it takes a random walker starting at node i to arrive for the
    first time at node j. Note that the mean first passage time is not a
    symmetric measure -- mfpt(i,j) may be different from mfpt(j,i) -- and
    the pair-wise diffusion efficiency matrix is hence also not symmetric.

    This function is adapted and optimized from the Brain Connectivity Toolbox.

    .. warning::
       Test before use.

    Parameters
    ----------
    W : (N x N) ndarray
        Weighted/unweighted, direct/undirected connection weight/length array

    Returns
    -------
    GE_diff : float
        Global diffusion efficiency
    E_diff : (N x N) ndarray
        Pair-wise diffusion efficiency array

    References
    ----------
    .. [1] Goñi, J., Avena-Koenigsberger, A., Velez de Mendizabal, N.,
       van den Heuvel, M. P., Betzel, R. F., & Sporns, O. (2013). Exploring the
       morphospace of communication efficiency in complex networks. PLoS One,
       8(3), e58070.
    """
    n = W.shape[0]
    mfpt = mean_first_passage_time(W)
    E_diff = np.divide(1, mfpt)
    np.fill_diagonal(E_diff, 0.0)
    GE_diff = np.sum(E_diff) / (n * (n - 1))
    return GE_diff, E_diff

def weight_to_length(W, epsilon=1e-6):
    """
    Converts a connectivity weight matrix to a length matrix.

    Parameters:
    - W (np.array): NxN connectivity weight matrix.
    - epsilon (float): Small value to avoid division by zero.

    Returns:
    - L (np.array): NxN connection length matrix.
    """
    W = np.array(W)  # Ensure it's a NumPy array
    W_safe = np.where(W > 0, W, epsilon)  # Avoid division by zero
    L = 1.0 / W_safe  # Convert to length
    L[W == 0] = 100000  # Set zero-weight connections to infinity
    return L


#### load yeo atlas 7 network
atlas_yeo_lh = nib.load('/home/pabaua/dev_mni/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/home/pabaua/dev_mni/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)


df_schaefer_400 = pd.read_csv('/home/pabaua/Downloads/region_info_Schaefer400.csv', index_col=0)
df_schaefer_400_brainstem = df_schaefer_400[df_schaefer_400.structure == "brainstem"]
print(df_schaefer_400)


# Load functional matrix
FC = np.load('/home/pabaua/Downloads/brainstemfc_mean_corrcoeff_full_Schaefer400.npy')
print(FC)


# Initialize arousal at brainstem node (e.g., first node)
LC_r = df_schaefer_400.labels == "LC_r"
LC_l = df_schaefer_400.labels == "LC_l"
LC = LC_r + LC_l

df_schaefer_400['init_arousal'] = LC

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(df_schaefer_400.x, df_schaefer_400.y, df_schaefer_400.z, cmap='viridis', s=50, c=df_schaefer_400['init_arousal'])
ax.axis('equal')
ax.view_init(0,-90,0)
ax.axis('off')
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
plt.show()

# Simulate arousal wave propagation
FC = weight_to_length(FC)
_, diff_eff = diffusion_efficiency(FC)
diff_eff = mean_first_passage_time(FC)
active_indices = np.where(df_schaefer_400['init_arousal'] == 1)[0]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(df_schaefer_400.x, df_schaefer_400.y, df_schaefer_400.z, cmap='viridis', s=50, c=np.mean(np.log(diff_eff[active_indices,:]), axis=0))
ax.axis('equal')
ax.view_init(0,-90,0)
ax.axis('off')
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
plt.show()



# Iterate over selected time points and map the arousal wave to the cortical regions
time_steps = [1, 3, 5, 7, 9]
mapped_waves = []
for t in time_steps:
    mapped_wave = map_to_labels(arousal_wave[t, 81:], yeo_surf) 
    #mapped_wave[np.isnan(mapped_wave)] = 0
    mapped_wave = mapped_wave / np.max(map_to_labels(arousal_wave[t, 81:], yeo_surf))  # Map to cortical surface
    mapped_waves.append(mapped_wave)  # Append the mapped wave to the list
mapped_waves_stack = np.stack(mapped_waves, axis=0)
print(mapped_waves_stack.shape)

# Plot the stacked waves for each time point
surf_lh, surf_rh = load_conte69()
plot_hemispheres(
    surf_lh, surf_rh, array_name=mapped_waves_stack,
    size=(1200, 300), zoom=1.25, color_bar='bottom',
    share='both', background=(0, 0, 0),
    nan_color=(250, 250, 250, 1), transparent_bg=True
)

