#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# python /local_raid/data/pbautin/software/twixtools/utils/convert_to_cfl.py /local_raid/data/pbautin/data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat /local_raid/data/pbautin/data/TWIX2/proc_data/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9  --type image --remove_os

# import h5py
import numpy as np
# import ismrmrd
from bart import bart
#import cfl
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

def resize_array(arr, target_shape, center=False):
    """Resize a NumPy array along specified dimensions by truncating or zero-padding.

    Parameters:
        arr (np.ndarray): Input array to be resized.
        target_shape (tuple): Desired shape of the output array.
        center (bool): If True, center the input array when padding.

    Returns:
        np.ndarray: Resized array.
    """
    current_shape = arr.shape
    slices = []
    padding = []

    for dim, (curr, target) in enumerate(zip(current_shape, target_shape)):
        if curr > target:
            # Truncate if the current dimension is larger
            if center:
                start = (curr - target) // 2
                end = start + target
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, target))
            padding.append((0, 0))
        else:
            # Zero-padding if the current dimension is smaller
            if center:
                pad_before = (target - curr) // 2
                pad_after = target - curr - pad_before
            else:
                pad_before, pad_after = 0, target - curr
            slices.append(slice(0, curr))
            padding.append((pad_before, pad_after))

    # Apply slicing and padding
    arr = arr[tuple(slices)]
    arr = np.pad(arr, padding, mode='constant')

    return arr

filename = '/local_raid/data/pbautin/data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat'
twix = twixtools.map_twix(filename)[-1]
print(list(twix.keys()))
im_array = twix['image']
refscan = twix['refscan']
pc_array = twix['phasecorr']
noise_array = twix['noise']

# phasecorr.flags['remove_os'] = True
# phasecorr.flags['average']['Rep'] = True  # average all repetitions
# phasecorr.flags['zf_missing_lines'] = True
# phasecorr.flags['squeeze_singletons'] = True  # squeezes singleton dims
# print(phasecorr.shape)

# for phase-correction, we need to keep the individual segments (which indicate the readout's polarity)
im_array.flags['remove_os'] = True
im_array.flags['average']['Seg'] = False
#im_array.flags['skip_empty_lead'] = True
im_array.flags['zf_missing_lines'] = True
im_array.flags['squeeze_singletons'] = True  # squeezes singleton dims
im_array.flags['regrid'] = True  # activate ramp sampling regridding
print("image array: {}".format(im_array.shape))

# the twix_array object makes it easy to remove the 2x oversampling in read direction
refscan.flags['remove_os'] = True  # activate automatic os removal
refscan.flags['average']['Seg'] = False
refscan.flags['squeeze_singletons'] = True  # squeezes singleton dims
#refscan.flags['skip_empty_lead'] = True
refscan.flags['zf_missing_lines'] = True
refscan.flags['regrid'] = True  # activate ramp sampling regridding
print("ref array: {}".format(refscan.shape))

# the twix_array object makes it easy to remove the 2x oversampling in read direction
noise_array.flags['remove_os'] = True  # activate automatic os removal
noise_array.flags['average']['Seg'] = False
noise_array.flags['squeeze_singletons'] = True  # squeezes singleton dims
#refscan.flags['skip_empty_lead'] = True
noise_array.flags['zf_missing_lines'] = True
noise_array.flags['regrid'] = True  # activate ramp sampling regridding
print("noise array: {}".format(refscan.shape))

# read the data
data = im_array[:,0,100,...]
data_ref = refscan[100,...]
data_noise = noise_array[:]
data_ref = data_ref[np.newaxis,...]
data = data + data_ref
print("data array: {}".format(data.shape))
print("noise array: {}".format(data_noise.shape))

# reconstruct the data
image = ifftnd(data[0,...], [0,-1])
image = rms_comb(image)
print(image.shape)

# plot the data
plt.figure(figsize=[12,6])
plt.subplot(141)
plt.title('k-space')
plt.imshow(abs(data_ref[0,:,0,:])**0.2, cmap='gray', origin='lower')

plt.subplot(142)
plt.title('k-space')
plt.imshow(abs(data[0,:,0,:])**0.2, cmap='gray', origin='lower')

plt.subplot(143)
plt.title('image')
plt.imshow(image, cmap='gray', origin='lower')

plt.subplot(144)
plt.title('image')
plt.imshow(abs(data_noise[:,3,:])**0.2, cmap='gray', origin='lower')
plt.show()

pc_array.flags['remove_os'] = True
pc_array.flags['average']['Seg'] = False
pc_array.flags['squeeze_singletons'] = True  # squeezes singleton dims
pc_array.flags['zf_missing_lines'] = True
print("pc array: {}".format(pc_array.shape))

data_pc = pc_array[0,0,100,...]
print(data_pc.shape)

# calculate phase-correction
pc_corr = calc_pc_corr(data_pc)
print(pc_corr.shape)

# apply phase-correction
print("data array: {}".format(data.shape))
image_pc = apply_pc_corr_and_ifft(data, pc_corr)
print(image_pc.shape)

# RMS coil combination# plot results
image_pc = rms_comb(image_pc)
print(image_pc.shape)


plt.figure(figsize=[8,6])
plt.subplot(121)
plt.title('image')
plt.imshow(image, cmap='gray', origin='lower')
plt.axis('off')
plt.subplot(122)
plt.title('after phase-correction')
plt.imshow(image_pc, cmap='gray', origin='lower')
plt.axis('off')
plt.show()















#####################################
#
# # the twix_array object makes it easy to remove the 2x oversampling in read direction
# refscan.flags['remove_os'] = True  # activate automatic os removal
# refscan.flags['average']['Rep'] = True  # average all repetitions
# refscan.flags['regrid'] = True  # activate ramp sampling regridding
# refscan.flags['squeeze_singletons'] = True  # squeezes singleton dims
# refscan.flags['zf_missing_lines'] = True
# print(refscan.shape)
#
# # the twix_array object makes it easy to remove the 2x oversampling in read direction
# im_data.flags['remove_os'] = True  # activate automatic os removal
# im_data.flags['average']['Rep'] = True  # average all repetitions
# im_data.flags['regrid'] = True  # activate ramp sampling regridding
# im_data.flags['squeeze_singletons'] = True  # squeezes singleton dims
# im_data.flags['zf_missing_lines'] = True
# print(im_data.shape)
# data = im_data[100,...] + refscan[100,...]
# # apply phase-correction
# image_pc = apply_pc_corr_and_ifft(data, phasecorr)
#
#
# plt.figure(figsize=[12,8])
# plt.subplot(121)
# plt.title('k-space')
# plt.imshow(abs(data[:,10,:])**0.2, cmap='gray', origin='lower')
# plt.axis('off')
#
# image = ifftnd(data, [0,-1])
# image = rms_comb(image)
# print(image.shape)
# plt.subplot(122)
# plt.title('image')
# plt.imshow(abs(image), cmap='gray', origin='lower')
# plt.show()
