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
# Original SCT code: https://github.com/neuropoly/
#
# example: 
# python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_mtsat.py \
#   -mt /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC026/ses-a2/anat/sub-PNC026_ses-a2_acq-mtw_mt-on_MTR.nii.gz \
#   -pd /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC026/ses-a2/anat/sub-PNC026_ses-a2_acq-mtw_mt-off_MTR.nii.gz \
#   -t1 /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC026/ses-a2/anat/sub-PNC026_ses-a2_acq-mtw_T1w.nii.gz \
#   -omtsat /local_raid/data/pbautin/results/qmri/MTsat.nii \
#   -ot1map /local_raid/data/pbautin/results/qmri/T1map.nii \
#   -omtr /local_raid/data/pbautin/results/qmri/MTr.nii
#
# For some cases:
#   -b1map /local_raid/data/pbautin/results/qmri/sub-PNC026_ses-a2_acq-sfam_TB1TFL_B1map.nii \
#########################################################################################

import sys
import os
import json
from typing import Sequence
import argparse
import logging
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import logging
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute MTsat and T1map. '
                    'Reference: Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization '
                    'transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH '
                    'MRI. Magn Reson Med 2008;60(6):1396-1407.'
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-mt",
        required=True,
        help="Image with MT_ON",
        metavar='file',
    )
    mandatory.add_argument(
        "-pd",
        required=True,
        help="Image PD weighted (typically, the MT_OFF)",
        metavar='file',
    )
    mandatory.add_argument(
        "-t1",
        required=True,
        help="Image T1 map",
        metavar='file',
    )

    optional = parser.add_argument_group('OPTIONAL ARGUMENTS')
    optional.add_argument(
        "-trmt",
        help="TR [in s] for the MT image (MT on). By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-trpd",
        help="TR [in s] for proton density weighted image (MT off). By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-trt1",
        help="TR [in s] for T1-weighted image. By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-famt",
        help="Flip angle [in deg] for mt image. By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-fapd",
        help="Flip angle [in deg] for pd image. By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-fat1",
        help="Flip angle [in deg] for t1 image. By default, will be fetched from the json sidecar (if it exists).",
        type=float,
        metavar='float',
    )
    optional.add_argument(
        "-b1map",
        help="B1 map",
        metavar='file',
        default=None)
    optional.add_argument(
        "-omtsat",
        metavar='str',
        help="Output file for MTsat",
        default="mtsat.nii.gz")
    optional.add_argument(
        "-ot1map",
        metavar='str',
        help="Output file for T1map",
        default="t1map.nii.gz")
    optional.add_argument(
        "-omtr",
        metavar='str',
        help="Output file for MTR",
        default="mtr.nii.gz")
    optional.add_argument(
        '-v',
        metavar='int',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
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

def divide_after_removing_zero(dividend, divisor, threshold, replacement=np.nan):
    """
    Mask zero, divide, look for numbers larger than 'threshold', and replace masked elements.

    :param dividend: np.array
    :param divisor: np.array
    :param threshold: float
    :param replacement: float: value to replace masked value with.
    :return: result_full
    """
    ind_nonzero = np.where(divisor)
    n_zero = divisor.size - len(ind_nonzero[0])
    logger.info("Found {} voxels with value=0. These will be replaced by {}.".format(n_zero, replacement))
    # divide without zero element in divisor
    result = np.true_divide(dividend[ind_nonzero], divisor[ind_nonzero])
    # find aberrant values above threshold
    logger.info("Threshold to clip values: +/- {}".format(threshold))
    np.clip(result, -threshold, threshold, out=result)
    # initiate resulting array with replacement values
    result_full = np.full_like(dividend, fill_value=replacement, dtype='float32')
    result_full[ind_nonzero] = result
    return result_full


def compute_mtr(nii_mt_off, nii_mt_on, threshold_mtr=100):
    """
    Compute Magnetization Transfer Ratio in percentage.

    :param nii_mt_off: Image object without MT pulse (MT0)
    :param nii_mt_on: Image object with MT pulse (MT1)
    :param threshold_mtr: float: value above which number will be clipped
    :return: nii_mtr
    """
    # Convert input to avoid numerical errors from int16 data
    # Related issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3636
    nii_mt_on_img = nii_mt_on
    nii_mt_on = nii_mt_on.get_fdata().astype(np.float64)
    nii_mt_off = nii_mt_off.get_fdata().astype(np.float64)

    # Initialize Image object
    nii_mtr = nii_mt_on.copy()

    # Compute MTR
    nii_mtr = divide_after_removing_zero(100 * (nii_mt_off - nii_mt_on), nii_mt_off, threshold_mtr)
    nii_mtr = nib.Nifti1Image(nii_mtr, nii_mt_on_img.affine, nii_mt_on_img.header)
    return nii_mtr


def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd, fa_t1,
                  nii_b1map=None):
    """
    Compute MTsat (in percent) and T1 map (in s) based on FLASH scans

    :param nii_mt: Image object for MTw
    :param nii_pd: Image object for PDw
    :param nii_t1: Image object for T1w
    :param tr_mt: Float: Repetition time (in s) for MTw image
    :param tr_pd: Float: Repetition time (in s) for PDw image
    :param tr_t1: Float: Repetition time (in s) for T1w image
    :param fa_mt: Float: Flip angle (in deg) for MTw image
    :param fa_pd: Float: Flip angle (in deg) for PDw image
    :param fa_t1: Float: Flip angle (in deg) for T1w image
    :param nii_b1map: Image object for B1-map (optional)
    :return: MTsat and T1map.
    """
    # params
    nii_t1map = \
        None  # it would be possible in the future to input T1 map from elsewhere (e.g. MP2RAGE). Note: this T1map
    # needs to be in s unit.
    b1correctionfactor = \
        0.4  # empirically defined in https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full#h3
    # R1 threshold, below which values will be clipped.
    r1_threshold = 0.01  # R1=0.01 s^-1 corresponds to T1=100s which is a reasonable threshold
    # Similarly, we also set a threshold for MTsat values
    mtsat_threshold = 1  # we expect MTsat to be on the order of 0.01

    # Convert flip angles into radians
    fa_mt_rad = np.radians(fa_mt)
    fa_pd_rad = np.radians(fa_pd)
    fa_t1_rad = np.radians(fa_t1)

    nii_t1 = nii_t1.get_fdata()
    nii_pd = nii_pd.get_fdata()
    nii_mt = nii_mt.get_fdata()
    if nii_b1map is not None:
            nii_b1map = nii_b1map.get_fdata()

    # ignore warnings from division by zeros (will deal with that later)
    seterr_old = np.seterr(over='ignore', divide='ignore', invalid='ignore')

    # check if a T1 map was given in input; if not, compute R1
    if nii_t1map is None:
        # compute R1
        logger.info("Compute T1 map...")
        r1map = 0.5 * np.true_divide((fa_t1_rad / tr_t1) * nii_t1 - (fa_pd_rad / tr_pd) * nii_pd,
                                     nii_pd / fa_pd_rad - nii_t1 / fa_t1_rad)
        # apply B1 correction if available
        if nii_b1map is not None:
            b1_squared = np.multiply(nii_b1map, nii_b1map)
            r1map = np.multiply(r1map, b1_squared)
        # remove nans and clip unrelistic values
        r1map = np.nan_to_num(r1map)
        ind_unrealistic = np.where(r1map < r1_threshold)
        if ind_unrealistic[0].size:
            logger.warning("R1 values were found to be lower than {}. They will be set to inf, producing T1=0 for "
                           "these voxels.".format(r1_threshold))
            r1map[ind_unrealistic] = np.inf  # set to infinity so that these values will be 0 on the T1map
        # compute T1
        nii_t1map = nii_mt.copy()
        nii_t1map = 1. / r1map
    else:
        logger.info("Use input T1 map.")
        r1map = 1. / nii_t1map

    # Compute A to compute MT-sat
    logger.info("Compute A...")
    a = (tr_pd * fa_t1_rad / fa_pd_rad - tr_t1 * fa_pd_rad / fa_t1_rad) * \
        np.true_divide(np.multiply(nii_pd, nii_t1, dtype=float),
                       tr_pd * fa_t1_rad * nii_t1 - tr_t1 * fa_pd_rad * nii_pd)
    # apply B1 correction if available
    if nii_b1map is not None:
        a = np.divide(a, nii_b1map)

    # Compute MTsat
    logger.info("Compute MTsat...")
    nii_mtsat = nii_mt.copy()
    nii_mtsat = tr_mt * np.multiply((fa_mt_rad * np.true_divide(a, nii_mt) - 1),
                                         r1map, dtype=float) - (fa_mt_rad ** 2) / 2.
    # remove nans and clip unrelistic values
    nii_mtsat = np.nan_to_num(nii_mtsat)
    ind_unrealistic = np.where(np.abs(nii_mtsat) > mtsat_threshold)
    if ind_unrealistic[0].size:
        logger.warning("MTsat values were found to be larger than {}. They will be set to zero for these voxels."
                       "".format(mtsat_threshold))
        nii_mtsat[ind_unrealistic] = 0
    # convert into percent unit (p.u.)
    nii_mtsat *= 100

    # Apply B1 correction to result
    # Weiskopf, N., Suckling, J., Williams, G., Correia, M.M., Inkster, B., Tait, R., Ooi, C., Bullmore, E.T., Lutti,
    # A., 2013. Quantitative multi-parameter mapping of R1, PD(*), MT, and R2(*) at 3T: a multi-center validation.
    # Front. Neurosci. 7, 95.
    if nii_b1map is not None:
        nii_mtsat = np.true_divide(nii_mtsat * (1 - b1correctionfactor),
                                        (1 - b1correctionfactor * nii_b1map))

    # set back old seterr settings
    np.seterr(**seterr_old)

    return nii_mtsat, nii_t1map


def resample_image(source_img, target_img):
    source_data = source_img.get_fdata()
    target_shape = target_img.shape
    source_shape = source_img.shape

    # Calculate the zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, source_shape)]

    # Resample the data
    resampled_data = zoom(source_data, zoom_factors, order=1)  # Linear interpolation

    # Create new NIfTI image with the resampled data
    resampled_img = nib.Nifti1Image(resampled_data, target_img.affine, target_img.header)
    return resampled_img


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v

    # Set up logging
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    logger.info('Load data...')
    nii_t1 = nib.load(arguments.t1)
    nii_mt = resample_image(nib.load(arguments.mt), nii_t1)
    nii_pd = resample_image(nib.load(arguments.pd), nii_t1)
    if arguments.b1map is None:
        nii_b1map = None
    else:
        nii_b1map = resample_image(nib.load(arguments.b1map), nii_t1)

    # Fetch TR and Flip Angle values, either from arguments or from json files
    if arguments.trmt is None:
        arguments.trmt = fetch_metadata(get_json_file_name(arguments.mt, check_exist=True), 'RepetitionTime')
    if arguments.trpd is None:
        arguments.trpd = fetch_metadata(get_json_file_name(arguments.pd, check_exist=True), 'RepetitionTime')
    if arguments.trt1 is None:
        arguments.trt1 = fetch_metadata(get_json_file_name(arguments.t1, check_exist=True), 'RepetitionTime')
    if arguments.famt is None:
        arguments.famt = fetch_metadata(get_json_file_name(arguments.mt, check_exist=True), 'FlipAngle')
    if arguments.fapd is None:
        arguments.fapd = fetch_metadata(get_json_file_name(arguments.pd, check_exist=True), 'FlipAngle')
    if arguments.fat1 is None:
        arguments.fat1 = fetch_metadata(get_json_file_name(arguments.t1, check_exist=True), 'FlipAngle')

    # Convert TR and Flip Angle values to float
    tr_mt = float(arguments.trmt)
    tr_pd = float(arguments.trpd)
    tr_t1 = float(arguments.trt1)
    fa_mt = float(arguments.famt)
    fa_pd = float(arguments.fapd)
    fa_t1 = float(arguments.fat1)

    # Compute MTsat
    img_mtsat, img_t1map = compute_mtsat(nii_mt, nii_pd, nii_t1,
                                         tr_mt, tr_pd, tr_t1,
                                         fa_mt, fa_pd, fa_t1,
                                         nii_b1map=nii_b1map)
    nii_mtsat = nib.Nifti1Image(img_mtsat, nii_t1.affine, nii_t1.header)
    nii_t1map = nib.Nifti1Image(img_t1map, nii_t1.affine, nii_t1.header)

    # Compute MTr
    nii_mtr = compute_mtr(nii_mt, nii_pd)

    # Output MTsat and T1 maps
    logger.info('Generate output files...')
    nib.save(nii_mtsat, arguments.omtsat)
    nib.save(nii_mtr, arguments.omtr)
    nib.save(nii_t1map, arguments.ot1map)


    logger.info(f"MTsat map saved to {arguments.omtsat}")
    logger.info(f"MTR map saved to {arguments.omtr}")
    logger.info(f"T1 map saved to {arguments.ot1map}")


if __name__ == "__main__":
    main(sys.argv[1:])