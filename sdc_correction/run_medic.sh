# ------------------------------------------------------------------------------
# Script to Run MEDIC SDC on fMRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# 7T fMRI images SDC are particularly important. 
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
# ------------------------------------------------------------------------------

medic --magnitude --phase