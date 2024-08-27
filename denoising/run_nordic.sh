# ------------------------------------------------------------------------------
# Script to Run NORDIC Denoising on Diffusion MRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# 7T dMRI images SNR is poor. 
#
# References:
# - Original Paper: https://doi.org/10.1016/j.neuroimage.2020.117539
# - GitHub Repository: https://github.com/SteenMoeller/NORDIC_Raw
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


# Input and output file paths
img_magn_in='/home/pabaua/dev_mni/data/test_nordic/sub-01_ses-2mm_dir-AP_run-01_part-mag_dwi.nii.gz'
img_phase_in='/home/pabaua/dev_mni/data/test_nordic/sub-01_ses-2mm_dir-AP_run-01_part-phase_dwi.nii.gz'
img_out='sub-01_ses-2mm_dir-AP_run-01_dwi_nordic' # Do not put extension .nii
nordic_matlab_dir='/home/pabaua/dev_mni/NORDIC_Raw/'  # Directory containing NIFTI_NORDIC.m

# Arguments for the MATLAB function
ARG_temporal_phase=3
ARG_phase_filter_width=3
ARG_DIROUT='/home/pabaua/dev_mni/results/nordic/'

# Run MATLAB command with the specified arguments
matlab -nodisplay -nojvm -nosplash -nodesktop -r " \
try; \
addpath('${nordic_matlab_dir}'); \
ARG.temporal_phase = ${ARG_temporal_phase}; \
ARG.phase_filter_width = ${ARG_phase_filter_width}; \
ARG.DIROUT = '${ARG_DIROUT}'; \
NIFTI_NORDIC('${img_magn_in}', '${img_phase_in}', '${img_out}', ARG); \
end; \
quit;" \
>> ${ARG_DIROUT}/log_NORDIC_$(date '+%Y-%m-%d').txt


dwidenoise ${img_magn_in} ${ARG_DIROUT}/sub-01_ses-2mm_dir-AP_run-01_dwi_mppca.nii.gz

