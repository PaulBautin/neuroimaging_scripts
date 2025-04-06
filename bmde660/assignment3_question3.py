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
import scipy.io as sio
import matplotlib.pyplot as plt

params = {'font.size': 22}
plt.rcParams.update(params)

# Load data structure
data = sio.loadmat("/home/pabaua/Downloads/VFAdata.mat")

# Define parameters
alpha1 = np.deg2rad(20)  # Convert degrees to radians
alpha2 = np.deg2rad(3)   # Convert degrees to radians
TR = 15e-3  # (s)

# Extract data
S_T1w = data['t1w']
S_PD = data['pdw']
B1 = data['B1map']
mask = data['mask'].astype(bool)  # Ensure mask is boolean

# Apply brain mask
S_T1w[~mask] = 0
S_PD[~mask] = 0

# Compute intermediate values
Y1 = S_T1w / np.sin(alpha1)
Y2 = S_PD / np.sin(alpha2)
X1 = S_T1w / np.tan(alpha1)
X2 = S_PD / np.tan(alpha2)

# T1 map without B1 correction
m = (Y2 - Y1) / (X2 - X1)
T1 = -TR / np.log(m)

# B1 correction
Y1_corr = S_T1w / np.sin(alpha1 * B1)
Y2_corr = S_PD / np.sin(alpha2 * B1)
X1_corr = S_T1w / np.tan(alpha1 * B1)
X2_corr = S_PD / np.tan(alpha2 * B1)

m_corr = (Y2_corr - Y1_corr) / (X2_corr - X1_corr)
T1_corr = -TR / np.log(m_corr)

# Mask B1 to brain region
B1_brain = np.where(mask, B1, 0)

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display T1 map without B1 correction
im1 = axes[0].imshow(T1.T * 1000, cmap='jet', vmin=0, vmax=3000, origin="lower")
fig.colorbar(im1, ax=axes[0], label='T1 (ms)')
axes[0].set_title('T1 w/o B1 corr.')

# Display T1 map with B1 correction
im2 = axes[1].imshow(T1_corr.T * 1000, cmap='jet', vmin=0, vmax=3000, origin="lower")
fig.colorbar(im2, ax=axes[1], label='T1 (ms)')
axes[1].set_title('T1 w/ B1 corr.')

# Display B1 map
im3 = axes[2].imshow(B1.T, cmap='jet', origin="lower")
fig.colorbar(im3, ax=axes[2])
axes[2].set_title('B1 map')

# Adjust layout
plt.tight_layout()
plt.show()
