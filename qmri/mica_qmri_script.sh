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

################################################################################


# ANTS
export ANTSPATH="/data/mica1/01_programs/ants-2.3.4/bin"
# FreeSurfer
export FREESURFER_HOME="/data/mica1/01_programs/freesurfer-7.3.2"
# fastsurfer
export FASTSURFER_HOME="/data_/mica1/01_programs/fastsurfer"
export fs_licence="/data_/mica1/01_programs/freesurfer-7.3.2/license.txt"

# FreeSurfer 6.0 configuration
source "${FREESURFER_HOME}/FreeSurferEnv.sh"

##### Unable to properly register B1 anat ref to T1
# mri_synthstrip -i /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/fmap/sub-Pilot014_ses-02_acq-b1sag_run-2_fieldmap.nii.gz \
#               -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-b1sag_run-2_fieldmap_brain.nii.gz \
#               -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-b1sag_run-2_fieldmap_brain_mask.nii.gz \
#               -b 3
###########


# mri_synthstrip -i /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-T1_0p5-T1map.nii.gz \
#               -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-T1_0p5-T1map_brain.nii.gz \
#               -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-T1_0p5-T1map_brain_mask.nii.gz \
#               --no-csf
#
# mri_synthstrip -i /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-on_MTR.nii.gz \
#               -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brain.nii.gz \
#               -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brain_mask.nii.gz \
#               --no-csf
#
# mri_synthstrip -i /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-off_MTR.nii.gz \
#               -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brain.nii.gz \
#               -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brain_mask.nii.gz \
#               --no-csf
#
#
# antsRegistrationSyNQuick.sh -d 3 \
#                             -f /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-T1_0p5-T1map_brain.nii.gz \
#                             -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brain.nii.gz \
#                             -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brain \
#                             -t a
#
cp /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-off_MTR.json \
    /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brainWarped.json
# antsRegistrationSyNQuick.sh -d 3 \
#                             -f /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_acq-T1_0p5-T1map_brain.nii.gz \
#                             -m /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brain.nii.gz \
#                             -o /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brain \
#                             -t a
cp /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-on_MTR.json \
    /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brainWarped.json


python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_b1map.py \
          -b1_fa /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/fmap/sub-Pilot014_ses-02_acq-b1sag_run-1_fieldmap.nii.gz \
          -b1_ref /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/fmap/sub-Pilot014_ses-02_acq-b1sag_run-2_fieldmap.nii.gz \
          -odir /local_raid/data/pbautin/results/qmri/test \
          #-h




python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_mtsat.py \
          -mt /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-on_MTR_brainWarped.nii.gz \
          -pd /local_raid/data/pbautin/results/qmri/test/ants/sub-Pilot014_ses-02_mt-off_MTR_brainWarped.nii.gz \
          -t1 /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-T1_0p5-T1map.nii.gz \
          -b1map /local_raid/data/pbautin/results/qmri/test/sub-Pilot014_ses-02_acq-b1sag_run-1_fieldmap_B1map.nii \
          -omtsat /local_raid/data/pbautin/results/qmri/test/mtsat.nii.gz \
          #-h
