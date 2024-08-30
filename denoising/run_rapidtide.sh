# ------------------------------------------------------------------------------
# Script to Run RapidTide signal cleaning on fMRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# rs-fMRI has low frequency systemic hemodynamic “Noise” due to real, 
# random fluctuations of blood oxygenation and volume 
# (both of which affect the intensity of BOLD fMRI images) 
# in the blood passing through the brain.
#
# References:
# - Original Paper: https://doi.org/10.3389/fnins.2019.00787
# - GitHub Repository: https://github.com/bbfrederick/rapidtide
#
# Before Running:
# - Ensure Python >= 3.8. 
# - Ensure installations with: conda install nibabel pyqtgraph pyfftw
# - Create virtual env: `conda create -n env_rapidtide`.
# - Upload repository: `git clone https://github.com/bbfrederick/rapidtide`
# - Install in repo: `python setup.py install`
# 
# Input Data:
# - MICA-PNC preprocessed fMRI image
#
# ------------------------------------------------------------------------------


# Input and output file paths
prepoc_bold='/home/pabaua/dev_mni/data/sub-PNC001_ses-01/sub-PNC001/ses-01/func/desc-me_task-rest_bold/volumetric/sub-PNC001_ses-01_space-func_desc-me_preproc.nii.gz'
sub='sub-PNC001_ses-01'
out_dir='/home/pabaua/dev_mni/results/rapidtide/'

rapidtide "${prepoc_bold}" "${outdir_dir}/sub_space-func_desc-me_preproc_rapidtide_" --denoising

#--filterband lfo --passes 3
# [--graymattermask MASK[:VALSPEC]] good idea

