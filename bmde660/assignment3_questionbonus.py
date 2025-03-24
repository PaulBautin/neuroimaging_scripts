from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Assignement 3 question Bonus
#
# ---------------------------------------------------------------------------------------
# Author: Paul Bautin with help from ChatGPT 3
#
#########################################################################################


import numpy as np
import scipy.io as sio
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from numpy.fft import ifft2, fft2, ifftshift
from scipy.sparse.linalg import LinearOperator

def generate_image_grid(N):
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, N, endpoint=False),
                        np.linspace(-0.5, 0.5, N, endpoint=False))
    return np.column_stack((x.ravel(), y.ravel()))

def generate_epi_trajectory(N):
    kx, ky = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2))
    return np.column_stack((kx.ravel(), ky.ravel()))

def generate_encoding_matrix(N, grid, trajectory, field_map=None, echo_spacing=None):
    if field_map is not None:
        x, _ = np.meshgrid(np.linspace(0, N * echo_spacing, N),
                    np.linspace(0, N * echo_spacing, N))
        phase_shift = (field_map * x).flatten()[np.newaxis,:]
        phase_shift_sq =  phase_shift.T @ np.ones(phase_shift.shape)
        plt.imshow(phase_shift_sq)
        plt.show()
        E = np.exp(1j * 2 * np.pi * np.dot(grid, trajectory.T) + phase_shift_sq)
        return E
    else:
        return np.exp(1j * 2 * np.pi * np.dot(grid, trajectory.T))
    
def generate_encoding_matrix(N, grid, trajectory, field_map=None, echo_spacing=None):
    """
    Generate an encoding matrix for MRI reconstruction.

    Parameters:
    - N (int): Number of spatial points
    - grid (ndarray): Grid coordinates (Nx2 or Nx3 array)
    - trajectory (ndarray): K-space trajectory (Mx2 or Mx3 array)
    - field_map (ndarray, optional): Field map for off-resonance correction (NxN array)
    - echo_spacing (float, optional): Echo spacing for field-dependent phase shift
    
    Returns:
    - E (ndarray): Encoding matrix
    """
    # Compute base encoding matrix
    E = np.exp(1j * 2 * np.pi * np.dot(grid, trajectory.T))

    if field_map is not None and echo_spacing is not None:
        # Compute phase shift based on field map
        time_axis = np.linspace(0, N * echo_spacing, N).reshape(-1, 1)
        phase_shift = (field_map * time_axis).flatten()
        
        # Apply phase shift to encoding matrix
        E *= np.exp(1j * 2 * np.pi * phase_shift[:, np.newaxis])

    return E


def epi_reconstruction(mat_data, field_map=None, echo_spacing=None):
    N = mat_data.shape[0]
    s = mat_data.flatten()
    
    # Generate components
    grid = generate_image_grid(N)
    trajectory = generate_epi_trajectory(N)
    E = generate_encoding_matrix(N=N, grid=grid, trajectory=trajectory, field_map=field_map, echo_spacing=echo_spacing)

    # Define encoding and decoding operators
    Em, cEm = lambda m: E @ m, lambda s: E.conj().T @ s.flatten()
    
    # Solve pseudo-inverse problem using conjugate gradient method
    A = LinearOperator((N*N, N*N), matvec=lambda m: cEm(Em(m)), dtype=np.complex128)
    m, _ = spla.cg(A, cEm(s))
    img_forward_model = np.abs(m.reshape((N, N)))
    return img_forward_model


params = {'font.size': 22}
plt.rcParams.update(params)

########### Load the .mat file
mat_file = '/local_raid/data/pbautin/downloads/forward_model_bonus_question/question/s.mat'
mat_data = sio.loadmat(mat_file)['s']

mat_file_corr = '/local_raid/data/pbautin/downloads/forward_model_bonus_question/question/s_corrupted.mat'
mat_data_corr = sio.loadmat(mat_file_corr)['s']

mat_file_fm = '/local_raid/data/pbautin/downloads/forward_model_bonus_question/question/fieldMap.mat'
mat_data_fm = sio.loadmat(mat_file_fm)['fieldMap']

########### Question a
img_ifft = np.abs(ifftshift(ifft2(mat_data)))
img_forward_model = epi_reconstruction(mat_data)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].imshow(img_ifft.T, cmap='gray')
axes[0].set_title('Direct IFFT Reconstruction')
axes[0].axis('off')

axes[1].imshow(img_forward_model.T, cmap='gray', origin='lower')
axes[1].set_title('Forward Model Reconstruction')
axes[1].axis('off')

plt.show()


########### Question b
img_ifft = np.abs(ifftshift(ifft2(mat_data_corr)))
img_forward_model = epi_reconstruction(mat_data_corr)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].imshow(img_ifft.T, cmap='gray')
axes[0].set_title('Direct IFFT Reconstruction')
axes[0].axis('off')

axes[1].imshow(img_forward_model.T, cmap='gray', origin='lower')
axes[1].set_title('Forward Model Reconstruction')
axes[1].axis('off')

plt.show()


########### Question c
img_ifft = np.abs(ifftshift(ifft2(mat_data_corr)))
img_forward_model = epi_reconstruction(mat_data=mat_data_corr, field_map=mat_data_fm, echo_spacing=500E-6)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].imshow(img_ifft.T, cmap='gray')
axes[0].set_title('Direct IFFT Reconstruction')
axes[0].axis('off')

axes[1].imshow(img_forward_model.T, cmap='gray', origin='lower')
axes[1].set_title('Forward Model Reconstruction')
axes[1].axis('off')

plt.show()