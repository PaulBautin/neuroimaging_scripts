#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# Script to Run COMMIT on Diffusion MRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# Structural connectomes reconstructed from dMRI have poor accuracy and can be improved
# with filtering methods.
#
# References:
# - Original Paper: https://doi.org/10.1126%2Fsciadv.aba8245
# - GitHub Repository: https://github.com/daducci/COMMIT
#
# Before Running:
# - Ensure MATLAB version 2017b or newer is installed with the Image Processing
#   Toolbox and Signal Processing Toolbox.
# - Clone the NORDIC GitHub repository.
#
# Input Data:
# - MICA-PNC Pilot014 preprocessed dMRI images
#
# Previous Implementation:
# - The tractogram data was previously filtered with SIFT
#
# ------------------------------------------------------------------------------


import numpy as np
import subprocess
import matplotlib.pyplot as plt

# Reconstruction modules
import dipy.reconst.dki as dki
import dipy.reconst.msdki as msdki

# For in-vivo data
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.io.image import save_nifti


# Function to convert neuroimaging files using mrconvert
def run_mrconvert(input_file, output_file, options=None):
    """
    Convert neuroimaging files using mrconvert.

    :param input_file: Path to the input file (e.g., .mif, .nii, .nii.gz).
    :param output_file: Path to the output file (desired format).
    :param options: List of additional command line options for mrconvert.
    """
    # Build the command as a list of strings
    command = ['mrconvert', input_file, output_file]

    # Add additional options if provided
    if options:
        command.extend(options)

    try:
        # Run the command and capture output and errors
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Output the command's response for logging
        print(result.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode('utf-8')}")



# Example usage
input_file = '/local_raid/data/pbautin/data/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-preproc_dwi.mif'  # Input neuroimaging file
output_dwi = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-preproc_dwi.nii.gz'
output_bvec = '/local_raid/data/pbautin/results/COMMIT/dwi.bvec'  # Desired output file format
output_bval = '/local_raid/data/pbautin/results/COMMIT/dwi.bval'  # Desired output file format
options = ['-export_grad_fsl', output_bvec, output_bval]  # Optional arguments (e.g., setting strides)
run_mrconvert(input_file, output_dwi, options)
run_mrconvert(input_file, output_dwi)

mask, affine = load_nifti('/local_raid/data/pbautin/data/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-brain_mask.nii.gz')

data, affine = load_nifti(output_dwi)
bvals, bvecs = read_bvals_bvecs(output_bval, output_bvec)
gtab = gradient_table(bvals, bvecs)

print("started model")
msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)

print("started fit")
msdki_fit = msdki_model.fit(data, mask=mask)

print("started metrics")
MSD = msdki_fit.msd
F = msdki_fit.smt2f
DI = msdki_fit.smt2di
uFA2 = msdki_fit.smt2uFA

save_nifti('/local_raid/data/pbautin/results/model_results/mean_signal_diffusion.nii.gz', MSD, affine)
save_nifti('/local_raid/data/pbautin/results/model_results/axonal_volume_fraction.nii.gz', F, affine)
save_nifti('/local_raid/data/pbautin/results/model_results/intrinsic_diffusivity.nii.gz', DI, affine)
save_nifti('/local_raid/data/pbautin/results/model_results/microscopic_anisotropy.nii.gz', uFA2, affine)
