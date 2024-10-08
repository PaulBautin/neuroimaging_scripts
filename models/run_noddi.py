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


import amico
from os.path import join
import numpy as np
import subprocess


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
input_file = '/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/dwi/sub-Pilot014_ses-01_space-dwi_desc-preproc_dwi.mif'  # Input neuroimaging file
output_dwi = '/local_raid/data/pbautin/results/COMMIT/sub-Pilot014_ses-01_space-dwi_desc-preproc_dwi.nii.gz'
output_bvec = '/local_raid/data/pbautin/results/COMMIT/dwi.bvec'  # Desired output file format
output_bval = '/local_raid/data/pbautin/results/COMMIT/dwi.bval'  # Desired output file format
options = ['-export_grad_fsl', output_bvec, output_bval]  # Optional arguments (e.g., setting strides)
run_mrconvert(input_file, output_dwi, options)
run_mrconvert(input_file, output_dwi)

# Class to hold all the information (data and parameters) when performing an evaluation with the AMICO framework
# lmax : Maximum SH order to use for the rotation phase
# default output path: study_path/subject/AMICO/<MODEL>
ae = amico.Evaluation(study_path='/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0', subject='sub-Pilot014', output_path='/local_raid/data/pbautin/results/model_results/noddi')
amico.setup(lmax=12)

# Prepare acquisition scheme information in FSL format
# `mrconvert -export_grad_fsl <bvec_output_file> <bval_output_file> <mrtrix_input_file> <nifti_output_file>`
#NODDI_bval = join(ae.get_config("study_path"), ae.get_config("subject"), 'ses-01/dwi/sub-PNC001_ses-01_space-dwi_desc-preproc_dwi.bval')
#NODDI_bvec = join(ae.get_config("study_path"), ae.get_config("subject"), 'ses-01/dwi/sub-PNC001_ses-01_space-dwi_desc-preproc_dwi.bvec')
out_scheme = join('/local_raid/data/pbautin/results/model_results/noddi', 'sub-Pilot014_ses-01_space-dwi_desc-preproc_dwi.scheme')
amico.util.fsl2scheme(output_bval, output_bvec, out_scheme)

# Load the diffusion signal and its corresponding acquisition scheme.
#NODDI_img = join(ae.get_config("study_path"), ae.get_config("subject"), 'ses-01/dwi/sub-Pilot014_ses-01_space-dwi_desc-preproc_dwi.nii.gz')
# NODDI_scheme = join('/local_raid/data/pbautin/results/model_results/noddi', 'sub-Pilot014_ses-01_space-dwi_desc-preproc_dwi.scheme')
brain_mask = join(ae.get_config("study_path"), ae.get_config("subject"), 'ses-01/dwi/sub-Pilot014_ses-01_space-dwi_desc-brain_mask.nii.gz')
ae.load_data(output_dwi, out_scheme, mask_filename=brain_mask, b0_thr=61)

# Set the model to use to describe the signal contributions in each voxel.
# models: ['StickZeppelinBall', 'CylinderZeppelinBall', 'NODDI', 'FreeWater', 'SANDI']
ae.set_model('NODDI')

# Define NODDI model parameters to compute each compartment response function
# para_diff is the axial diffusivity (AD) in the CC -- single fiber
para_diff=1.7E-3
# iso_diff is the mean diffusivity (MD) in ventricles.
iso_diff=3.0E-3
intra_vol_frac = np.linspace(0.1, 0.99, 12)
intra_orient_distr = np.hstack((np.array([0.03, 0.06]), np.linspace(0.09, 0.99, 10)))
ae.model.set(dPar=para_diff, dIso=iso_diff,IC_VFs=intra_vol_frac, IC_ODs=intra_orient_distr, isExvivo=False)

# Generate the high-resolution response functions for each compartment with:
# lambda1 is the first regularization parameter.
# lambda2 is the second regularization parameter.
#        StickZeppelinBall:      'set_solver()' not implemented
#        CylinderZeppelinBall:   lambda1 = 0.0, lambda2 = 4.0
#        NODDI:                  lambda1 = 5e-1, lambda2 = 1e-3
#        FreeWater:              lambda1 = 0.0, lambda2 = 1e-3
#        VolumeFractions:        'set_solver()' not implemented
#        SANDI:                  lambda1 = 0.0, lambda2 = 5e-3
ae.set_solver(lambda1=5e-1, lambda2=1e-3)
ae.generate_kernels(regenerate=True)

# Load rotated kernels and project to the specific gradient scheme of this subject.
ae.load_kernels()
# Fit the model to the data.
ae.fit()
# Save the output (directions, maps etc).
ae.save_results()
