#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# python /local_raid/data/pbautin/software/twixtools/utils/convert_to_cfl.py /local_raid/data/pbautin/data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat /local_raid/data/pbautin/data/TWIX2/proc_data/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9  --type image --remove_os

# import h5py
import numpy as np
import os
# import ismrmrd
from bart import bart
import cfl
import twixtools
import matplotlib.pyplot as plt

def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))



# functions to calculate & apply phase-correction:

def calc_pc_corr(sig):
    ncol = sig.shape[-1]

    # ifft col dim.
    pc = ifftnd(sig, [-1])

    # calculate phase slope from autocorrelation (for both readout polarities separately - each in its own dim)
    slope = np.angle((np.conj(pc[...,1:]) * pc[...,:-1]).sum(-1, keepdims=True).sum(-2, keepdims=True))
    x = np.arange(ncol) - ncol//2

    return np.exp(1j * slope * x)


def apply_pc_corr_and_ifft(sig, pc_corr):

    # ifft col dim.
    sig = ifftnd(sig, [-1])

    # apply phase-correction slope
    sig *= pc_corr

    # remove segment dim
    sig = sig.sum(0).squeeze()

    # ifft lin dim.
    sig = ifftnd(sig, [0])

    return sig

os.environ["BART_TOOLBOX_PATH"] = '/local_raid/data/pbautin/downloads/bart-0.9.00'

#### read files
A = cfl.readcfl('/local_raid/data/pbautin/data/TWIX2/proc_data/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9_image')
ref = cfl.readcfl('/local_raid/data/pbautin/data/TWIX2/proc_data/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9_refscan')
ref = bart(1, 'resize -c 1 192', ref)
ksp = A + ref
#ksp = np.average(A, axis=10)
#print(ksp.shape)
ksp = ksp[:,:, :, :, 0, 0, 0, 0, 0, 0,0, 0,0, 100]
#ksp_ref = bart(1, 'flip 1', ksp_ref)
#ksp = bart(1, 'fftmod 3', ksp)
print(ksp.shape)


#### coil compression (cc)
num_vcoils=8
nc=8
cc_matrix = bart(1, 'cc -M', ksp)
ksp = bart(1, 'ccapply -p {}'.format(num_vcoils), ksp, cc_matrix)
print(ksp.shape)

cimg = bart(1, 'fft -iu 3', ksp)
sens, ev_maps= bart(2, 'ecalib -a -m 1 -I -S', ksp)

# Create a figure with three rows: one for coil images, one for k-space images, and one for sensitivity images
plt.figure(figsize=(16, 15))  # Adjust figure size as needed

# First row for coil images
for i in range(nc):
    plt.subplot(3, nc, i + 1)  # 3 rows, nc columns, position i+1
    plt.imshow(abs(cimg[:, :, 0, i]).squeeze(), cmap='gray')
    plt.title('Coil image {}'.format(i))
    plt.axis('off')  # Optionally remove axes for cleaner appearance

# Second row for k-space images
for i in range(nc):
    plt.subplot(3, nc, nc + i + 1)  # Start positions at nc + 1
    plt.imshow(abs(ksp[:, :, 0, i]).squeeze() ** 0.2, cmap='gray')
    plt.title('k-space image {}'.format(i))
    plt.axis('off')  # Optionally remove axes for cleaner appearance

# Third row for sensitivity images
for i in range(nc):
    plt.subplot(3, nc, 2 * nc + i + 1)  # Start positions at 2*nc + 1
    plt.imshow(abs(sens[:, :, 0, i]).squeeze(), cmap='gray')
    plt.title('Sensitivity image {}'.format(i))
    plt.axis('off')  # Optionally remove axes for cleaner appearance

# Adjust the layout to avoid overlapping titles and ensure the images are neatly arranged
plt.tight_layout()
plt.show()

#data = bart(1, 'fft -i -u 3', ksp)
rss_ref = bart(1, 'rss 8', cimg)
plt.title('root of sum of squares image')
plt.imshow(abs(rss_ref[:,:]) ** 0.5, cmap='gray', origin='lower')
plt.show()
