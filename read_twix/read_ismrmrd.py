#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import ismrmrd
from bart import bart
import cfl
import twixtools
from twixtools
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


# parse the twix file
#twix = twixtools.read_twix('/media/pabaua/MyPassport/siemens_data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat', include_scans=[0])

twixtools.convert_to_cfl('/media/pabaua/MyPassport/siemens_data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat','/media/pabaua/MyPassport/siemens_data/ismrmd_data/output')

# map the twix data to twix_array objects
mapped = twixtools.map_twix(twix)
im_data = mapped[-1]['image']

# make sure that we later squeeze the right dimensions:
print(im_data.non_singleton_dims)

# the twix_array object makes it easy to remove the 2x oversampling in read direction
im_data.flags['remove_os'] = True

# read the data (array-slicing is also supported)
data = im_data[:].squeeze()

plt.figure(figsize=[12,8])
plt.subplot(121)
plt.title('k-space')
plt.imshow(abs(data[:,0])**0.2, cmap='gray', origin='lower')

image = ifftnd(data, [0,-1])
image = rms_comb(image)
plt.subplot(122)
plt.title('image')
plt.imshow(abs(image), cmap='gray', origin='lower')
plt.show()
