#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to denoise a dataset with the Non Local Means algorithm.

Formerly: scil_run_nlmeans.py
"""

import argparse
import logging
import warnings

from dipy.denoise.non_local_means import non_local_means
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.noise_estimate import estimate_sigma
import nibabel as nib
import numpy as np
logger = logging.getLogger(__name__)


def get_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_image',
                   help='Path of the image file to denoise.')
    p.add_argument('out_image',
                   help='Path to save the denoised image file.')
    p.add_argument('number_coils', type=int,
                   help='Number of receiver coils of the scanner.\nUse '
                        'number_coils=1 in the case of a SENSE (GE, Philips) '
                        'reconstruction and \nnumber_coils >= 1 for GRAPPA '
                        'reconstruction (Siemens). number_coils=4 works well '
                        'for the 1.5T\n in Sherbrooke. Use number_coils=0 if '
                        'the noise is considered Gaussian distributed.')

    p.add_argument('--mask', metavar='',
                   help='Path to a binary mask. Only the data inside the mask'
                        ' will be used for computations')
    p.add_argument('--sigma', metavar='float', type=float,
                   help='The standard deviation of the noise to use instead '
                        'of computing  it automatically.')
    p.add_argument('--log', dest="logfile",
                   help='If supplied, name of the text file to store '
                        'the logs.')
    return p


def main():
    parser = get_parser()
    arguments = parser.parse_args()

    vol = nib.load(arguments.in_image)
    data = vol.get_fdata(dtype=np.float32)
    if arguments.mask is None:
        mask = np.zeros(data.shape[0:3], dtype=bool)
        if data.ndim == 4:
            mask[np.sum(data, axis=-1) > 0] = 1
        else:
            mask[data > 0] = 1
    else:
        mask = get_data_as_mask(nib.load(arguments.mask), dtype=bool)

    sigma = arguments.sigma

    if sigma is not None:
        logger.info('User supplied noise standard deviation is: {}'.format(sigma))
        # Broadcast the single value to a whole 3D volume for nlmeans
        sigma = np.ones(data.shape[:3]) * sigma
    else:
        logger.info('Estimating noise')
        sigma = estimate_sigma(vol.get_fdata(dtype=np.float32), N=arguments.number_coils)

    logger.info('Denoise small patch')
    data_denoised_small = non_local_means(
        data, sigma, mask=mask, rician=arguments.number_coils > 0, patch_radius=1, block_radius=1)
    logger.info('Denoise large patch')
    data_denoised_large = non_local_means(
        data, sigma, mask=mask, rician=arguments.number_coils > 0, patch_radius=2, block_radius=1)
    logger.info('Denoise adaptive soft matching')
    data_denoised = adaptive_soft_matching(data, data_denoised_small, data_denoised_large, sigma[0])

    nib.save(nib.Nifti1Image(data_denoised, vol.affine, header=vol.header), arguments.out_image)


if __name__ == "__main__":
    main()
