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
# - This script is tested with a 2mm dMRI image from the Edden dataset 
#   available at: https://openneuro.org/datasets/ds004666/versions/1.0.5
# - Note: Requires dMRI magnitude and phase images.
# - Note: Not yet able to test 1.5mm dMRI images, > 16 GB RAM is required.
#
# Previous Implementation:
# - The dMRI data was previously denoised using the MPPCA (Marchenko-Pastur Principal 
#   Component Analysis) method, implemented in MRtrix3's `dwidenoise` command. 
#   Reference: https://mrtrix.readthedocs.io/en/dev/reference/commands/dwidenoise.html
#
# Questions:
# - When should NORDIC be run?. Example. before or after FSL-eddy
# - How to evaluate improvements?
# 
# Future Considerations:
# - The NORDIC method could be adapted for other MRI modalities such as fMRI 
#   or ASL.
# - Implementing NORDIC similarly to the FIX tool used in micapipe fMRI (similar Matlab requirements?).
# 
# Alternatives:
# - DIPY's patch2self https://github.com/nipreps/dmriprep/issues/132
#   Reference: https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html
# ------------------------------------------------------------------------------


# Import and configure commit package 
import commit
from commit import trk2dictionary
commit.setup()

# Import the tractogram and necessary files

trk2dictionary.run(
    filename_tractogram = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/PFT_Tracking/sub-007_ses-v1__pft_tracking_prob_wm_seed_0.trk',
    filename_peaks      = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/FODF_Metrics/sub-007_ses-v1__peaks.nii.gz',
    filename_mask       = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/Extract_B0/sub-007_ses-v1__b0_mask_resampled.nii.gz',
    fiber_shift         = 0.5,
    peaks_use_affine    = True,
    path_out            = '/home/pabaua/dev_mni/results/COMMIT'
)

# Load the diffusion data
import amico
bval_file = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/Eddy_Topup/eddy.bval'
bvec_file = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/Eddy_Topup/eddy.bvec'
out_scheme = '/home/pabaua/dev_mni/results/COMMIT/DWI.scheme'
amico.util.fsl2scheme(bval_file, bvec_file, out_scheme)
mit = commit.Evaluation()
mit.set_verbose(4)
dwi_file = '/media/pabaua/MyPassport/tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-007_ses-v1/Resample_DWI/sub-007_ses-v1__dwi_resampled.nii.gz'
mit.load_data(dwi_file, out_scheme)


# Set the forward model
mit.set_model( 'StickZeppelinBall' )
d_par       = 1.7E-3             # Parallel diffusivity [mm^2/s]
d_perps_zep = [ 0.51E-3 ]        # Perpendicular diffusivity(s) [mm^2/s]
d_isos      = [ 1.7E-3, 3.0E-3 ] # Isotropic diffusivity(s) [mm^2/s]
mit.model.set( d_par, d_perps_zep, d_isos )
mit.generate_kernels( regenerate=True )
mit.load_kernels()


# Load the sparse data-structure
mit.load_dictionary('/home/pabaua/dev_mni/results/COMMIT')


# Build the linear operator A
mit.set_threads()
mit.build_operator()


# Fit the model to the data
mit.fit( tol_fun=1e-3, max_iter=1000 )


# Save results
mit.save_results()


# If you need the estimated coefficients for all the compartments
x_ic, x_ec, x_iso = mit.get_coeffs()
print( x_ic.size ) # this shows there are 283522 streamline coefficients (one for each)











