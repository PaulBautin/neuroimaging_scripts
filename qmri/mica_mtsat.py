#!/usr/bin/env python
#
# Compute MT saturation map and T1 map from a PD-weighted, a T1-weighted, and MT-weighted FLASH images
#
# Reference paper:
#    Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization transfer with inherent correction
#    for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI. Magn Reson Med 2008;60(6):1396-1407.
#
# This code is modified to remove dependencies on the Spinal Cord Toolbox (SCT).
# Original SCT code: https://github.com/neuropoly/spinalcordtoolbox

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


def compute_mtr(nii_mt1, nii_mt0, threshold_mtr=100):
    """
    Compute Magnetization Transfer Ratio in percentage.

    :param nii_mt1: Image object without MT pulse (MT0)
    :param nii_mt0: Image object with MT pulse (MT1)
    :param threshold_mtr: float: value above which number will be clipped
    :return: nii_mtr
    """
    # Convert input to avoid numerical errors from int16 data
    # Related issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3636

    nii_mt1_img = nii_mt1
    nii_mt1 = nii_mt1.get_fdata().astype(np.float64)
    nii_mt0 = nii_mt0.get_fdata().astype(np.float64)

    # Initialize Image object
    nii_mtr = nii_mt1.copy()

    # Compute MTR
    nii_mtr = divide_after_removing_zero(100 * (nii_mt0 - nii_mt1), nii_mt0, threshold_mtr)
    nii_mtr = nib.Nifti1Image(nii_mtr, nii_mt1_img.affine, nii_mt1_img.header)
    return nii_mtr


def compute_mtsat(nii_mt, nii_pd, nii_t1,
                  tr_mt, tr_pd, tr_t1,
                  fa_mt, fa_pd,
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
    :param nii_b1map: Image object for B1-map (optional)
    :return: MTsat and T1map.
    """

    # needs to be in s unit.
    b1correctionfactor = \
        0.4  # empirically defined in https://www.frontiersin.org/articles/10.3389/fnins.2013.00095/full#h3
    # R1 threshold, below which values will be clipped.
    r1_threshold = 0.01  # R1=0.01 s^-1 corresponds to T1=100s which is a reasonable threshold
    # Similarly, we also set a threshold for MTsat values
    mtsat_threshold = 1  # we expect MTsat to be on the order of 0.01

    # Convert flip angles into radians
    print(fa_mt)
    fa_mt_rad = np.radians(fa_mt)
    print(fa_mt_rad)
    fa_pd_rad = np.radians(fa_pd)
    print(tr_mt)

    nii_mt = nii_mt.get_fdata().astype(np.float64)
    nii_pd = nii_pd.get_fdata().astype(np.float64)
    nii_t1_img = nii_t1
    nii_t1map = nii_t1.get_fdata().astype(np.float64) / 1000
    nii_b1map = nii_b1map.get_fdata().astype(np.float64)


    # # Scale magnitude in nT/V
    # GAMMA = 2.675e8  # Proton's gyromagnetic ratio (rad/(T.s))
    # SATURATION_FA = 90  # Saturation flip angle hard-coded in TFL B1 mapping sequence (deg)
    #
    # nii_b1map = nii_b1map.get_fdata().astype(np.float64) / 10  # Siemens magnitude values are stored in degrees x10
    # nii_b1map[nii_b1map > 180] = 180  # Values higher than 180 degrees are due to noise
    # # Calculate B1+ efficiency (1ms, pi-pulse) and scale by the ratio of the measured FA to the saturation FA.
    # # Get the Transmission amplifier reference amplitude
    # amplifier_voltage = 256.938  # [V]
    # socket_voltage = amplifier_voltage * 10 ** -0.095  # -1.9dB voltage loss from amplifier to coil socket
    # nii_b1map = (nii_b1map / SATURATION_FA) #* (np.pi / (GAMMA * socket_voltage * 1e-3)) * 1e9  # nT/V

    # ignore warnings from division by zeros (will deal with that later)
    seterr_old = np.seterr(over='ignore', divide='ignore', invalid='ignore')


    nii_mtsat = np.abs((((nii_pd * fa_mt_rad * nii_b1map) / nii_mt) - 1) * (tr_mt / nii_t1map) - ((fa_mt_rad * nii_b1map)**2 / 2)) * 100
    nii_mtsat[nii_mtsat > 100] = 0

    # Compute A
    logger.info("Compute MTsat...")


    # set back old seterr settings
    np.seterr(**seterr_old)

    nii_mtsat = nib.Nifti1Image(nii_mtsat, nii_t1_img.affine, nii_t1_img.header)
    return nii_mtsat


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

    # Convert TR and Flip Angle values to float
    tr_mt = float(arguments.trmt)
    tr_pd = float(arguments.trpd)
    tr_t1 = float(arguments.trt1)
    fa_mt = float(arguments.famt)
    fa_pd = float(arguments.fapd)

    # Compute MTsat
    nii_mtsat = compute_mtsat(nii_mt, nii_pd, nii_t1,
                                         tr_mt, tr_pd, tr_t1,
                                         fa_mt, fa_pd,
                                         nii_b1map=nii_b1map)
    # Compute MTr
    nii_mtr = compute_mtr(nii_mt, nii_pd)

    # Output MTsat and T1 maps
    logger.info('Generate output files...')
    nib.save(nii_mtsat, arguments.omtsat)
    nib.save(nii_mtr, arguments.omtr)


    logger.info(f"MTsat map saved to {arguments.omtsat}")
    logger.info(f"MTR map saved to {arguments.omtr}")

if __name__ == "__main__":
    main(sys.argv[1:])


# python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/run_mp2rage.py -mt /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-on_MTR.nii.gz -pd /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-off_MTR.nii.gz -t1 /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-T1_0p5-T1map.nii.gz
