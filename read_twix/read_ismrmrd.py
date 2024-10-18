#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# python /local_raid/data/pbautin/software/twixtools/utils/convert_to_cfl.py /local_raid/data/pbautin/data/TWIX2/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9.dat /local_raid/data/pbautin/data/TWIX2/proc_data/meas_MID00060_FID02675_dwi_acq_multib_38dir_AP_acc9  --type image --remove_os

# import h5py
import numpy as np
# import ismrmrd
from bart import bart
import cfl
import twixtools
import matplotlib.pyplot as plt

###### noise
nc = 8
noise = cfl.readcfl('/local_raid/data/pbautin/data/TWIX2/proc_data/twix_tool_noise')

#### coil compression (cc)
num_vcoils=8
cc_matrix = bart(1, 'cc -M', noise)
noise = bart(1, 'ccapply -p {}'.format(num_vcoils), noise, cc_matrix)
print(noise.shape)

noise_flat = np.reshape(noise, (-1, nc))
cov = np.dot(np.conj(noise_flat).T, noise_flat)
noise_white = bart(1, 'whiten', noise_flat[:,None,None,:], noise_flat[:,None,None,:]).reshape((-1, nc))
cov_white = np.dot(np.conj(noise_white).T, noise_white)
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(abs(cov.squeeze()))
plt.title('Covariance matrix before pre-whitening')
plt.subplot(1, 2, 2)
plt.imshow(abs(cov_white.squeeze()))
plt.title('Covariance matrix after pre-whitening')
plt.show()


#### read files
A = cfl.readcfl('/local_raid/data/pbautin/data/TWIX2/proc_data/twix_tool_image')
ref = cfl.readcfl('/local_raid/data/pbautin/data/TWIX2/proc_data/twix_tool_refscan')
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



############### Reconstruction
ksp_white = bart(1, 'whiten', ksp[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
sens_white = bart(1, 'whiten', sens[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
cimg_white = bart(1, 'fft -iu 3', ksp_white).squeeze()


plt.figure(figsize=(16, 8))
plt.subplot(1, 4, 1)
reco = bart(1, 'pics  -l1 -r 0.001 -S', ksp[:,:,:,:], sens[:,:,:,:])
plt.imshow(abs(reco[:,:]).squeeze() ** 0.5, cmap='gray')
plt.title('Reconstruction l1 regularisation')

plt.subplot(1, 4, 2)
reco_l2 = bart(1, 'pics -l2 -r 0.001 -S', ksp[:,:,:], sens[:,:,:])
plt.imshow(abs(reco_l2[:,:]).squeeze() ** 0.5, cmap='gray')
plt.title('Reconstruction l2 regularisation')

data = bart(1, 'fft -i -u 3', ksp)
rss_ref = bart(1, 'rss 8', data)
plt.subplot(1, 4, 3)
plt.title('root of sum of squares image')
plt.imshow(abs(rss_ref[:,:]) ** 0.5, cmap='gray', origin='lower',)


reco_sense = bart(1, 'pocsense -l 2', ksp[:,:,:], sens[:,:,:])
plt.subplot(1, 4, 4)
plt.title('sense')
plt.imshow(abs(rss_ref[:,:]) ** 0.5, cmap='gray', origin='lower',)
plt.show()



###### get the scale factor from the reference data
scale_factor = np.percentile(abs(rss_ref), 99)
