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
    L[W == 0] = 1000000  # Set zero-weight connections to infinity
    return L


params = {"ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
            'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')

#### load yeo atlas 7 network
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')

df_schaefer_400 = pd.read_csv('/local_raid/data/pbautin/data/region_info_Schaefer400.csv', index_col=0)
df_schaefer_400.rename(columns={"labels": "label"}, inplace=True)

# Load functional matrix
FC = np.load('/local_raid/data/pbautin/data/brainstemfc_mean_corrcoeff_full_Schaefer400.npy')


# Initialize arousal at brainstem node (e.g., first node)
LC_r = df_schaefer_400.label == "LC_r"
LC_l = df_schaefer_400.label == "LC_l"
LC = LC_r + LC_l

df_schaefer_400['init_arousal'] = LC
df_schaefer_400_brainstem = df_schaefer_400[df_schaefer_400.structure == "brainstem"]

# Create figure with multiple subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(27, 9), subplot_kw={'projection': '3d'})

# Define different view angles
view_angles = [(0, 0), (0, -90), (0, 180)]

for ax, (elev, azim) in zip(axes, view_angles):
    sc = ax.scatter(df_schaefer_400_brainstem.x, df_schaefer_400_brainstem.y, df_schaefer_400_brainstem.z, 
                    cmap='Purples', s=100, c=df_schaefer_400_brainstem['init_arousal'], alpha=0.8)
    ax.view_init(elev, azim)  # Set view angles
    ax.axis('equal')
    ax.axis('off')

# Add colorbar to the right of the figure
#cbar = fig.colorbar(sc, ax=axes, shrink=1, aspect=100, location='bottom')
plt.show()


# Simulate arousal wave propagation
#FC = weight_to_length(FC)
#_, FC = diffusion_efficiency(FC)
#FC = mean_first_passage_time(FC)
active_indices = np.where(df_schaefer_400['init_arousal'] == 1)[0]


df_schaefer_400['lc_conn'] = np.mean(FC[active_indices,:], axis=0)
df_yeo_surf = df_yeo_surf.merge(df_schaefer_400, on='label', validate="many_to_one", how='left')
print(df_yeo_surf)
#surf_map = map_to_labels(np.mean(diff_eff[active_indices,:], axis=0), yeo_surf) 

# Plot the stacked waves for each time point
surf_lh, surf_rh = load_conte69()
plot_hemispheres(
    surf_lh, surf_rh, array_name=df_yeo_surf.lc_conn.values,
    size=(1200, 300), zoom=1.25, color_bar='bottom',
    share='both', background=(0, 0, 0),
    nan_color=(250, 250, 250, 1), transparent_bg=True, cmap="Purples"
)




