# ------------------------------------------------------------------------------
# Script to Run MicaPipe on BIDS Data
# ------------------------------------------------------------------------------
#
# Feature justification
#
#
# References:
# - Original Paper: https://doi.org/10.1016/j.neuroimage.2022.119612
# - GitHub Repository: https://github.com/MICA-MNI/micapipe/tree/master
#
# Before Running:
# export SINGULARITY_TMPDIR=/local_raid/data/pbautin/container/singularity_tmp
# export SINGULARITY_CACHEDIR=/local_raid/data/pbautin/container/singularity_cache
# singularity build micapipe-v0.2.3.simg docker://micalab/micapipe:v0.2.3
# $ export MICAPIPE=/local_raid/data/pbautin/micapipe
# $ PATH=${PATH}:${MICAPIPE}:${MICAPIPE}/functions
# $ export PATH
#
#
# Input Data:
# - MICA-PNC raw fMRI image
#
# ------------------------------------------------------------------------------


export MICAPIPE=/local_raid/data/pbautin/software/micapipe_dev024
PATH=${PATH}:${MICAPIPE}:${MICAPIPE}/functions
export PATH


#------------------------------------------------------------------------------#
# SOFTWARE CONFIGURATION for MICAPIPE
#------------------------------------------------------------------------------#
# User defined PATHS
# AFNI
export AFNIDIR="/data/mica1/01_programs/afni-20.2.06"
# ANTS
export ANTSPATH="/data/mica1/01_programs/ants-2.3.4/bin"
# Workbench
export workbench_path="/data/mica1/01_programs/workbench-1.4.2/bin_linux64"
# ICA-FIX
export FIXPATH="/data_/mica1/01_programs/fix"
# FreeSurfer
export FREESURFER_HOME="/data/mica1/01_programs/freesurfer-7.3.2"
# fastsurfer
export FASTSURFER_HOME="/data_/mica1/01_programs/fastsurfer"
export fs_licence="/data_/mica1/01_programs/freesurfer-7.3.2/license.txt"
# FSL 6.0
export FSLDIR="/data_/mica1/01_programs/fsl-6-0-3"
export FSL_DIR="/data_/mica1/01_programs/fsl-6-0-3"
export FSL_BIN="${FSLDIR}/bin"
# MRtrix3 3.0.1
export mrtrixDir="/data_/mica1/01_programs/mrtrix3-3.0.1"
# ITK utils
export itk_dir="/data_/mica1/01_programs/c3d-1.0.0-Linux-x86_64/bin"
# Python 3.7
#export PYTHON_3="/data/mica1/01_programs/micapipe-v0.2.0_conda/micapipe/bin"
# Export fs fs_licence
export fs_licence=/data_/mica1/01_programs/freesurfer-7.3.2/license.txt
# Fastsurfer singularity container
export fastsurfer_img=/data_/mica1/01_programs/fastsurfer/fastsurfer-cpu-v2.0.4.sif
# NORDIC denoising
export NORDIC_Raw="/local_raid/data/pbautin/software/NORDIC_Raw"
unset TMPDIR
# Fastsurfer conda env

#------------------------------------------------------------------------------#
# Remove any other instance from the PATH
# AFNI
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*afni*});IFS=':';echo "${p[*]}";unset IFS)
# ANTS
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*ants*});IFS=':';echo "${p[*]}";unset IFS)
# Workbench binaries
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*workbench*});IFS=':';echo "${p[*]}";unset IFS)
# FSL
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*fsl*});IFS=':';echo "${p[*]}";unset IFS)
# revome any other MRtrix3 version from path
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*mrtrix*});IFS=':';echo "${p[*]}";unset IFS)
# REMOVES any other python configuration from the PATH the conda from the PATH and LD_LIBRARY_PATH variable
PATH=$(IFS=':';p=($PATH);unset IFS;p=(${p[@]%%*conda*});IFS=':';echo "${p[*]}";unset IFS)
LD_LIBRARY_PATH=$(IFS=':';p=($LD_LIBRARY_PATH);unset IFS;p=(${p[@]%%*conda*});IFS=':';echo "${p[*]}";unset IFS)

#------------------------------------------------------------------------------#
# Software configuration
# FreeSurfer 6.0 configuration
source "${FREESURFER_HOME}/FreeSurferEnv.sh"
# FSL 6.0 configuration
source "${FSLDIR}/etc/fslconf/fsl.sh"
# PYTHON 3.7 configuration
unset PYTHONPATH
unset PYTHONHOME
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
conda3_bin=/data/mica1/01_programs/micapipe-v0.2.0_conda/conda3/bin/
conda3_bin=/local_raid/data/pbautin/software/conda/bin/
source /data/mica1/01_programs/micapipe-v0.2.0_conda/conda3/etc/profile.d/conda.sh

#------------------------------------------------------------------------------#
# Set the libraries paths for mrtrx and fsl
export LD_LIBRARY_PATH="${FSLDIR}/lib:${FSL_BIN}:${mrtrixDir}/lib"

#-----------------------------------------------------------------------------------#
# Export new PATH with al the necessary binaries
#export PATH="${AFNIDIR}:${ANTSPATH}:${workbench_path}:${FIXPATH}:${FREESURFER_HOME}/bin/:${mrtrixDir}/bin:${mrtrixDir}/lib:${FSLDIR}:${FSL_BIN}:${PYTHON_3}:${FASTSURFER_HOME}:${itk_dir}:${PATH}"
export PATH="${AFNIDIR}:${ANTSPATH}:${workbench_path}:${FIXPATH}:${FREESURFER_HOME}/bin/:${mrtrixDir}/bin:${mrtrixDir}/lib:${FSLDIR}:${FSL_BIN}:${FASTSURFER_HOME}:${itk_dir}:${conda3_bin}:${PATH}"
conda activate /data/mica1/01_programs/micapipe-v0.2.0_conda/micapipe

#bash /local_raid/data/pbautin/micapipe/functions/init.sh

export SINGULARITY_TMPDIR=/local_raid/data/pbautin/container/singularity_tmp
export SINGULARITY_CACHEDIR=/local_raid/data/pbautin/container/singularity_cache





bids='/local_raid/data/pbautin/data/3t_data'
out='/local_raid/data/pbautin/results/micapipe_commit'
fs_lic='/data_/mica1/01_programs/freesurfer-7.3.2/license.txt'
tmp='/local_raid/data/pbautin/results/micapipe_commit/tmp'
sub='HC007'
ses='01'
micapipe_img='/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif'



# Run commit
commit_tmp_dir=/local_raid/data/pbautin/results/micapipe_commit/tmp_commit
source activate env_commit

# python  "${MICAPIPE}"/functions/mica_tractogram_commit.py  \
#   $sub \
#   ${out}/micapipe_v0.2.0/sub-HC007/ses-01/dwi \
#   $commit_tmp_dir \
#   $commit_tmp_dir \
#   --tractogram /local_raid/data/pbautin/results/micapipe_commit/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography.tck \
#   --tissue /local_raid/data/pbautin/results/micapipe_commit/micapipe_v0.2.0/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-5tt.nii.gz \
#   --fodf /local_raid/data/pbautin/results/micapipe_commit/micapipe_v0.2.0/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_model-CSD_map-FOD_desc-wmNorm.nii.gz \
#   --dwi /local_raid/data/pbautin/results/micapipe_commit/micapipe_v0.2.0/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-preproc_dwi.mif



# DWI processing
micapipe \
    -bids $bids \
    -out $out \
    -fs_licence $fs_lic \
    -sub $sub \
    -ses $ses \
    -tmpDir $tmp \
    -SC \
    -tracts 1M \
    -tck /local_raid/data/pbautin/results/micapipe_commit/micapipe_v0.2.0/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography.tck \
    -nocleanup \
    -keep_tck
