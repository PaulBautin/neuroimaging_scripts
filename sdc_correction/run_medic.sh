# ------------------------------------------------------------------------------
# Script to Run MEDIC SDC on fMRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# 7T fMRI image SDC is particularly important. 
#
# References:
# - Original Paper: https://doi.org/10.1101%2F2023.11.28.568744
# - GitHub Repository: https://github.com/vanandrew/warpkit
#
# Before Running:
# - Ensure Julia is installed if not: `curl -fsSL https://install.julialang.org | sh`.
# - Create virtual env: `conda create -n env_warpkit`.
# - Install package with command: `pip install warpkit`.
# 
# Input Data:
# - This script is tested with a 2mm dMRI image from the Edden dataset 
#   available at: https://openneuro.org/datasets/ds004666/versions/1.0.5
# - Note: Requires dMRI magnitude and phase images.
# - Note: Not yet able to test 1.5mm dMRI images, > 16 GB RAM is required.
#
# ------------------------------------------------------------------------------

magnitude_data=( \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-1_part-mag_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-2_part-mag_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-3_part-mag_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-4_part-mag_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-5_part-mag_bold.nii.gz" \
)

phase_data=( \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-1_part-phase_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-2_part-phase_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-3_part-phase_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-4_part-phase_bold.nii.gz" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-5_part-phase_bold.nii.gz" \
)

metadata=( \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-1_part-mag_bold.json" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-2_part-mag_bold.json" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-3_part-mag_bold.json" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-4_part-mag_bold.json" \
    "/home/pabaua/dev_mni/data/me_fmri_data/sub-01_ses-1_task-rest_acq-MBME_run-01_echo-5_part-mag_bold.json" \
)


docker run -it --rm \
    -v /home/pabaua/dev_mni/data/me_fmri_data/:/home/pabaua/dev_mni/data/me_fmri_data/ \
    ghcr.io/vanandrew/warpkit:latest \
    --magnitude $magnitude_data --phase $phase_data --metadata $metadata