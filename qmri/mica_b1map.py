from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
# Compute MT saturation map and T1 map from a PD-weighted, a T1-weighted, and MT-weighted FLASH images
#
# Reference paper:
#    Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization transfer with inherent correction
#    for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI. Magn Reson Med 2008;60(6):1396-1407.
#
# This code is modified to remove dependencies on the Spinal Cord Toolbox (SCT).
# Original SCT code: https://github.com/neuropoly/spinalcordtoolbox
#
# example: 
# python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_b1map.py \
#   -b1_fa /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC026/ses-a2/fmap/sub-PNC026_ses-a2_acq-sfam_TB1TFL.nii.gz \
#   -b1_ref /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC026/ses-a2/fmap/sub-PNC026_ses-a2_acq-anat_TB1TFL.nii.gz \
#   -odir /local_raid/data/pbautin/results/qmri
#########################################################################################

import os
import shutil
import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import gaussian_filter
import json
import logging
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute B1 map from SIEMENS tfl_b1map. '
                    'Reference: Chung S. et al.: "Rapid B1+ Mapping Using a Preconditioning RF Pulse with TurboFLASH Readout", MRM 64:439-446 (2010).'
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-b1_fa",
        required=True,
        help="Image with flip angle",
        metavar='file',
    )
    mandatory.add_argument(
        "-b1_ref",
        required=True,
        help="Image with anatomical reference",
        metavar='file',
    )
    optional = parser.add_argument_group('OPTIONAL ARGUMENTS')
    optional.add_argument(
        "-odir",
        metavar='str',
        help="Output directory for results",
        default=".")
    optional.add_argument(
        "-smooth",
        help="Gaussian smoothing to be applied to B1 map. Default: [8,8,8]",
        default=[8,8,8],
    )
    return parser


def splitext(fname):
    if fname.endswith('.nii.gz'):
        return fname[:-7], '.nii.gz'
    else:
        return os.path.splitext(fname)


def get_json_file_name(fname, check_exist=False):
    """
    Get json file name by replacing '.nii' or '.nii.gz' extension by '.json'.
    Check if input file follows NIFTI extension rules.
    Optional: check if json file exists.
    :param fname: str: Input NIFTI file name.
    check_exist: Bool: Check if json file exists.
    :return: fname_json
    """
    list_ext = ['.nii', '.nii.gz']
    if fname.endswith('.nii.gz'):
        basename = fname[:-7]
        ext = '.nii.gz'
    elif fname.endswith('.nii'):
        basename = fname[:-4]
        ext = '.nii'
    else:
        raise ValueError(f"Problem with file: {fname}. Extension should be one of {list_ext}")
    fname_json = basename + '.json'

    if check_exist:
        if not os.path.isfile(fname_json):
            raise FileNotFoundError(f"{fname_json} not found. Either provide the file alongside {fname}, or explicitly "
                                    f"set tr and fa arguments for this image type.")
    return fname_json


def fetch_metadata(fname_json, field):
    """
    Return specific field value from json sidecar.
    :param fname_json: str: Json file
    :param field: str: Field to retrieve
    :return: value of the field.
    """
    with open(fname_json) as f:
        metadata = json.load(f)
    if field not in metadata:
        raise KeyError(f"Json file {fname_json} does not contain the field: {field}")
    else:
        return float(metadata[field])


def mask_for_B1(ref_img):
    # Placeholder function for generating a mask
    data = ref_img.get_fdata()
    mask = np.ones(data.shape, dtype=bool)
    return mask


def smoothB1(V1, B1map_norm, B1FWHM, mask):
    # Smooth the B1 map using a Gaussian filter and apply the mask
    voxel_size = V1.header.get_zooms()[:3]
    sigma = [fw / 2.355 / vs for fw, vs in zip(B1FWHM, voxel_size)]
    smoothed = gaussian_filter(B1map_norm, sigma=sigma)
    smoothed_masked = np.where(mask, smoothed, 0)
    return smoothed_masked


def calc_scaled_b1map(fa_img, ref_img, offset, scaling, smooth):
    # Get the input files data_
    fa_data = fa_img.get_fdata().astype(np.float64)
    ref_data = ref_img.get_fdata().astype(np.float64)
    # Generating the map
    B1map_norm = (np.abs(fa_data) + offset) * scaling
    # Masking
    mask = mask_for_B1(ref_img)
    # Smoothed map
    smB1map_norm = smoothB1(fa_img, B1map_norm, smooth, mask)
    # Saving image to nifti
    nii_B1map_norm = nib.Nifti1Image(smB1map_norm, fa_img.affine, fa_img.header)
    return nii_B1map_norm


def process_b1_map(b1_tfl_fa, b1_tfl_ref,smooth):
    # Processing B1 map from tfl_b1_map data
    fa_img = nib.load(b1_tfl_fa)
    ref_img = nib.load(b1_tfl_ref)

    # Get the nominal flip angle from metadata
    fa_nom = fetch_metadata(get_json_file_name(b1_tfl_fa, check_exist=True), 'FlipAngle')
    # Siemens magnitude values are stored in degrees x10
    scaling = 1 / (fa_nom * 10)
    offset = 0

    descrip = 'SIEMENS tfl_b1map protocol'

    P_trans = calc_scaled_b1map(fa_img, ref_img, offset, scaling, smooth)
    return P_trans


def main():
    parser = get_parser()
    arguments = parser.parse_args()

    logger = logging.getLogger(__name__)
    # Output directory
    output_dir = arguments.odir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scaled FA map from tfl_b1_map sequence
    b1_tfl_fa = arguments.b1_fa
    b1_tfl_ref = arguments.b1_ref

    # Copy anatomical ref image to outpath to prevent modification of original data
    ref_basename = os.path.basename(b1_tfl_ref).split('.nii')[0]
    ref_fname = os.path.join(output_dir, ref_basename + '_B1ref.nii')
    nii_B1map_ref = nib.load(b1_tfl_ref)
    nii_B1map_ref = nib.Nifti1Image(nii_B1map_ref.get_fdata(), nii_B1map_ref.affine, nii_B1map_ref.header)
    nib.save(nii_B1map_ref, ref_fname)
    # Try to copy the JSON sidecar file
    try:
        json_source = b1_tfl_ref.split('.nii')[0] + '.json'
        json_dest = ref_fname.split('.nii')[0] + '.json'
        shutil.copyfile(json_source, json_dest)
    except FileNotFoundError:
        pass

    b1_map_norm = process_b1_map(b1_tfl_fa, b1_tfl_ref, smooth=arguments.smooth)

    # Copy anatomical ref image to outpath to prevent modification of original data
    fa_basename = os.path.basename(b1_tfl_fa).split('.nii')[0]
    fa_fname = os.path.join(output_dir, fa_basename + '_B1map.nii')
    shutil.copyfile(b1_tfl_fa, fa_fname)
    # Try to copy the JSON sidecar file
    try:
        json_source = b1_tfl_fa.split('.nii')[0] + '.json'
        json_dest = fa_fname.split('.nii')[0] + '.json'
        shutil.copyfile(json_source, json_dest)
    except FileNotFoundError:
        pass

    # Output MTsat and T1 maps
    logger.info('Generate output files...')
    nib.save(b1_map_norm, fa_fname)


if __name__ == "__main__":
    main()
