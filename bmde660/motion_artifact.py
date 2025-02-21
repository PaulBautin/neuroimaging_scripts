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
import nibabel as nib


import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from itertools import product
import numpy as np
from scipy import ndimage as ndi
import nibabel as nb

plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["xtick.bottom"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["ytick.labelleft"] = False

def get_centers(brain_slice):
    samples_x = np.arange(6, brain_slice.shape[1] - 3, step=12).astype(int)
    samples_y = np.arange(6, brain_slice.shape[0] - 3, step=12).astype(int)
    return zip(*product(samples_x, samples_y))


def plot_brain(brain_slice, brain_cmap="magma", grid=False, voxel_centers_c=None):
    fig, ax = plt.subplots()

    # Plot image
    ax.imshow(brain_slice, cmap=brain_cmap, origin="lower");

    # # Generate focus axes
    # axins = inset_axes(
    #     ax,
    #     width="200%",
    #     height="100%",
    #     bbox_to_anchor=(1, .6, .5, .4),
    #     bbox_transform=ax.transAxes,
    #     loc=2,
    # )
    # axins.set_aspect("auto")

    # # sub region of the original image
    # x1, x2 = (np.array((0, 48)) + (z_s.shape[1] - 1) * 0.5).astype(int)
    # y1, y2 = np.round(np.array((-15, 15)) + (z_s.shape[0] - 1) * 0.70).astype(int)

    # axins.imshow(brain_slice[y1:y2, x1:x2], extent=(x1, x2, y1, y2), cmap=brain_cmap, origin="lower");
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    # ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1, linewidth=1.5);


    if grid:
        params = {}
        if voxel_centers_c is not None:
            params["norm"] = mpl.colors.CenteredNorm()
            params["c"] = voxel_centers_c
            params["cmap"] = "seismic"
        elif voxel_centers_c is None:
            params['c'] = 'black'

        # Voxel centers
        ax.scatter(*get_centers(brain_slice), s=10, **params)
        # axins.scatter(*get_centers(brain_slice), s=80, **params)

        # Voxel edges
        e_i = np.arange(0, z_slice.shape[1], step=12).astype(int)
        e_j = np.arange(0, z_slice.shape[0], step=12).astype(int)

        # Plot grid
        ax.plot([e_i[1:-1]] * len(e_j), e_j, c='k', lw=1, alpha=0.3);
        ax.plot(e_i, [e_j[1:-1]] * len(e_i), c='k', lw=1, alpha=0.3);
        # axins.plot([e_i[1:-1]] * len(e_j), e_j, c='k', lw=1, alpha=0.3);
        # axins.plot(e_i, [e_j[1:-1]] * len(e_i), c='k', lw=1, alpha=0.3);

    return fig, ax#, axins

def axis_cbar(ax):
    cbar = inset_axes(
        ax,
        width="100%",
        height="10%",
        bbox_to_anchor=(0, -0.15, 1, 0.5),
        bbox_transform=ax.transAxes,
        loc='lower center',
    )
    cbar.set_aspect("auto")
    return cbar



def simulate_motion(image: np.ndarray, motion_type: str = "random", strength: float = 0.1) -> np.ndarray:
    """
    Simulates subject movement by introducing phase perturbations in k-space.
    
    Parameters:
    image (np.ndarray): The input raw image data.
    motion_type (str): Type of motion ("random", "shift", "rotation").
    strength (float): The magnitude of motion effects.
    
    Returns:
    np.ndarray: The k-space data with simulated motion.
    """

    if motion_type == "shift":
        # Simulate translational motion by shifting k-space lines
        shift_amount = int(strength * 10)
        image_motion = np.roll(image, shift_amount, axis=0)


    elif motion_type == "rotation":
        # Simulate rotational motion by rotating the image in spatial domain before FFT
        from scipy.ndimage import rotate
        image_motion = rotate(image, angle=strength * 10, reshape=False)
    
    return image_motion

# orignal image and k-space
brain_img = nib.load("/local_raid/data/pbautin/software/micapipe/MNI152Volumes/MNI152_T1_1mm_brain.nii.gz")
brain_data = np.swapaxes(brain_img.get_fdata()[:,:,90], 0, 1)
print(brain_img.shape)
plt.imshow(brain_data, origin="lower", cmap="magma")
plt.show()

# original kspace
kspace = np.fft.fftshift(np.fft.fft2(brain_data))
plt.imshow(np.log(np.abs(kspace) + 1), origin="lower", cmap="magma")
plt.show()

# image with motion
image_motion = simulate_motion(brain_data, "rotation", 2)
plt.imshow(image_motion, origin="lower", cmap="magma")
plt.show()

# k_space with motioin
kspace_motion = np.fft.fftshift(np.fft.fft2(image_motion))
plt.imshow(np.log(np.abs(kspace_motion) + 1), origin="lower", cmap="magma")
plt.show()

# k-space combination
kspace_combination = np.copy(kspace)
kspace_combination[90:92, :] = kspace_motion[90:92, :]
plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_combination))), origin="lower", cmap="magma")
plt.show()






# z_slice = np.swapaxes(brain_data[..., 90], 0, 1).astype("float32")
# kspace = get_kspace(z_slice)
# z_s = z_slice.copy()
# z_slice[z_slice == 0] = np.nan

# # Display k-space magnitude
# plot_brain(np.log(np.abs(kspace) + 1))
# plt.show()
# plot_brain(np.ones_like(z_slice) * np.nan, grid=True)
# plt.show()

# plot_brain(np.log(np.abs(kspace) + 1), grid=True)
# plt.show()




