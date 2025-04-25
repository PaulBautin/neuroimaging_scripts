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



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import nibabel as nib
import pandas as pd
import seaborn as sns


from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels


def run_LFA(data_ts: np.ndarray, n_lag: int = 10, exp_var_lim: float = 95.0):
    """
    Perform linear forecasting analysis (LFA) on time series data.

    Parameters:
    data_ts : np.ndarray
        2D array (time x variable)
    n_lag : int, optional
        Number of future timepoints to predict (default: 10)
    exp_var_lim : float, optional
        Percentage of explained variance to retain (default: 95.0)

    Returns:
    lmse : np.ndarray
        Mean squared error of linear model predictions
    msd : np.ndarray
        Mean squared deviation of autocorrelation
    """
    regions_ts = data_ts
    data_ts = data_ts[:,~np.isnan(data_ts).any(axis=0)]
    # Ensure correct shape unpacking
    n_time, _ = data_ts.shape  

    # Define X (current state) and Y (next state)
    X, Y = data_ts[:-1, :], data_ts[1:, :]

    # Perform SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance and select number of principal components
    exp_var = 100 * (S**2) / np.sum(S**2)
    n_pcs = max(1, np.argmax(np.cumsum(exp_var) >= exp_var_lim) + 1)
    print(n_pcs)

    # Reduce rank
    U, S, Vt = U[:, :n_pcs], np.diag(S[:n_pcs]), Vt[:n_pcs, :]

    # Time evolution of principal components (n_time x n_PCs)
    X_svd = U @ S
    regions_ts[0,~np.isnan(regions_ts).any(axis=0)] = np.diag(S[:1]) @ Vt[:1, :]
    pc1 = regions_ts[0,:]

    # Estimate linear propagator A_tilde
    A_tilde = U.T @ Y @ Vt.T @ np.linalg.inv(S)

    # Estimate DMD
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

    #print(eigenvalues.shape)
    #print(eigenvectors.shape)
    DMD_modes = Y @ Vt.T @ np.linalg.inv(S) @ eigenvectors
    print(DMD_modes.shape)
    # DMD_mode_1 = DMD_modes[:,:1] @ Vt[:1, :]
    # print(DMD_mode_1.shape)
    # DMD_mode_1 = eigenvectors[:1,:] @ Vt
    # print(DMD_mode_1.shape)
    # regions_ts[0,~np.isnan(regions_ts).any(axis=0)] = eigenvectors[:1,:] @ Vt
    # dmd1 = regions_ts[0,:]
    

    # Initialize LMSE and MSD
    lmse = np.zeros((n_time - n_lag, n_lag))
    msd = np.zeros((n_time - n_lag, n_lag))

    # Forecast linear model
    for ss in range(n_time - n_lag - 1):
        X_pred = np.zeros((A_tilde.shape[0], n_lag))
        X_pred[:, 0] = X_svd[ss, :]  # Initial condition
        lmse[ss, 0] = np.mean((X_pred[:, 0] - X_svd[ss, :]) ** 2)
        msd[ss, 0] = np.mean((X_svd[ss, :] - X_svd[ss, :]) ** 2)

        # Predict future time steps
        for ll in range(n_lag - 1):
            X_pred[:, ll + 1] = A_tilde @ X_pred[:, ll]
            lmse[ss, ll + 1] = np.mean((X_pred[:, ll + 1] - X_svd[ss + ll + 1, :]) ** 2)
            msd[ss, ll + 1] = np.mean((X_svd[ss, :] - X_svd[ss + ll + 1, :]) ** 2)

    return lmse, msd, pc1, dmd1


def main():
    #### set custom plotting parameters
    params = {"ytick.color" : "w",
                "xtick.color" : "w",
                "axes.labelcolor" : "w",
                "axes.edgecolor" : "w",
                'font.size': 22}
    plt.rcParams.update(params)
    plt.style.use('dark_background')

    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')

    # Load fsLR-32k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    # fsLR-5k mask
    mask_lh = nib.load(micapipe + '/surfaces/fsLR-5k.L.mask.shape.gii').darrays[0].data
    mask_rh = nib.load(micapipe + '/surfaces/fsLR-5k.R.mask.shape.gii').darrays[0].data
    mask_5k = np.concatenate((mask_lh, mask_rh), axis=0)

    # load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)

    # Load functional data
    func_32k = nib.load('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC032_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
    func_32k[:, yeo_surf == 1000] = np.nan
    func_32k[:, yeo_surf == 2000] = np.nan
    func_32k = (func_32k - np.nanmean(func_32k, axis=0)) / np.nanstd(func_32k, axis=0)
    func_32k[func_32k > 3] = 3
    func_32k[func_32k < -3] = -3
    print(func_32k.shape)

    # func_lh = nib.load('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC032_ses-a1_hemi-L_surf-fsLR-5k.func.gii')
    # func_lh = np.vstack(np.array([darray.data for darray in func_lh.darrays]))
    # func_rh = nib.load('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC032_ses-a1_hemi-R_surf-fsLR-5k.func.gii')
    # func_rh = np.vstack(np.array([darray.data for darray in func_rh.darrays]))
    # func = np.concatenate((func_lh, func_rh), axis=1).astype(float)
    # func[:, mask_5k == 0] = np.nan
    # func[:, mask_5k == 0] = np.nan
    # func = (func - np.nanmean(func, axis=0)) / np.nanstd(func, axis=0)
    # func[func > 3] = 3
    # func[func < -3] = -3
    # plot_hemispheres(surf5k_lh, surf5k_rh, array_name=func[10,:], size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
    #                              nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    # Run LFA analysis
    n_lag = 10
    lmse, msd, pc1, dmd1 = run_LFA(func_32k, n_lag=n_lag)
    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=dmd1, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                            nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    plt.imshow(lmse)
    plt.show()

    # Get dimensions
    n_time, n_lags = lmse.shape

    # Create a meshgrid for time and lag
    time = np.arange(n_time)  # Time axis
    lags = np.arange(n_lags)  # Lag axis
    T, L = np.meshgrid(time, lags)  # Grid of time vs lag

    # Transpose lmse to match meshgrid dimensions
    Z = lmse.T  # Shape (n_lags, n_time)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Surface plot
    surf = ax.plot_surface(T, L, Z, cmap="coolwarm", edgecolor="none", alpha=0.95)

    # Labels and title
    ax.set_xlabel("Time", labelpad=20)
    ax.set_ylabel("Lag", labelpad=20)
    ax.set_zlabel("MSE", labelpad=40)
    ax.set_title("3D Horizon Plot of LMSE Over Time")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)  # Colorbar for scale
    plt.show()

    print(lmse.shape)
    w_average = lmse[:, 2::2].T @ func_32k[:-n_lag, :]

    plot_hemispheres(surf32k_lh, surf32k_rh, array_name=w_average, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                                nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    print(w_average.shape)
    print(lmse.shape)
    print(msd.shape)

if __name__ == "__main__":
    main()
