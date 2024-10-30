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

export tmpDir=/local_raid/data/pbautin/results/micapipe/tmp

#bash /local_raid/data/pbautin/micapipe/functions/init.sh

export SINGULARITY_TMPDIR=/local_raid/data/pbautin/container/singularity_tmp
export SINGULARITY_CACHEDIR=/local_raid/data/pbautin/container/singularity_cache





bids='/local_raid/data/pbautin/data/3t_data'
out='/local_raid/data/pbautin/data/3t_data'
fs_lic='/data_/mica1/01_programs/freesurfer-7.3.2/license.txt'
tmp='/local_raid/data/pbautin/results/micapipe/tmp'
sub='HC007'
ses='01'
micapipe_img='/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif'

#dwi_4proc=/local_raid/data/pbautin/results/micapipe/tmp/5913_micapipe_proc-dwi_Pilot014/dwi_dns_even.mif
#dwi_dns=/local_raid/data/pbautin/results/micapipe/tmp/5913_micapipe_proc-dwi_Pilot014/sub-Pilot014_ses-02_space-dwi_desc-denoised_dwi.mif
#mrconvert "$dwi_dns" "$dwi_4proc" -coord 3 0:end -force
#dwifslpreproc /local_raid/data/pbautin/results/micapipe/tmp/5913_micapipe_proc-dwi_Pilot014/dwi_dns_even.mif /local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-02/dwi/sub-Pilot014_ses-02_space-dwi_desc-preproc_dwi.mif -#rpe_pair -align_seepi -se_epi /local_raid/data/pbautin/results/micapipe/tmp/5913_micapipe_proc-dwi_Pilot014/b0_pair.mif -pe_dir j- -readout_time 0.0357 -eddy_options " --data_is_shelled --slm=linear --repol --nvoxhp 2000 --mporder=#" -nthreads 30 -#nocleanup -scratch /local_raid/data/pbautin/results/micapipe/tmp/5913_micapipe_proc-dwi_Pilot014 -force


# singularity run --writable-tmpfs --containall \
#     -B ${bids}:/bids \
#     -B ${out}:/out \
#     -B ${tmp}:/tmp \
#     -B ${fs_lic}:/opt/licence.txt \
#     ${micapipe_img} \
#     -bids /bids \
#     -out /out \
#     -fs_licence /opt/licence.txt \
#     -sub $sub \
#     -ses $ses \
#     -proc_surf

# #structural processing
# micapipe \
#     -bids $bids \
#     -out $out \
#     -fs_licence $fs_lic \
#     -sub $sub \
#     -ses $ses \
#     -post_structural \
#     -b0thr '61' \
#     -uni -T1wStr acq-uni_0p7-T1map,acq-inv1_0p7-T1map,acq-inv2_0p7-T1map

# # DWI processing
# micapipe \
#      -bids $bids \
#      -out $out \
#      -fs_licence $fs_lic \
#      -sub $sub \
#      -ses $ses \
#      -b0thr '61' \
#      -proc_dwi \
#      -dwi_main ${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_38d_dir-AP_dwi.nii.gz,${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_70d_dir-AP_dwi.nii.gz \
#      -dwi_rpe ${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq-b0_dir-PA_run-1_epi.nii.gz \
#      -tmpDir $tmp -threads 30 \
#      -dwi_phase FALSE #${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_38d_dir-AP_part-phase_dwi.nii.gz,${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_70d_dir-AP_part-phase_dwi.nii.gz
#
#
# DWI processing
micapipe \
    -bids $bids \
    -out $out \
    -fs_licence $fs_lic \
    -sub $sub \
    -ses $ses \
    -SC -nocleanup -tracts 5M


# DWI processing
#micapipe \
#    -bids $bids \
#    -out $out \
#    -fs_licence $fs_lic \
#    -sub $sub \
#    -ses $ses \
#    -proc_func \
#    -tmpDir $tmp -threads 30 \
#    -mainScanStr task-rest_run-2_echo-1_bold,task-rest_run-2_echo-2_bold,task-rest_run-2_echo-3_bold \
#    -func_pe ${bids}/sub-${sub}/ses-${ses}/fmap/sub-${sub}_ses-${ses}_acq-fmri_dir-AP_epi.nii.gz \
#    -func_rpe ${bids}/sub-${sub}/ses-${ses}/fmap/sub-${sub}_ses-${ses}_acq-fmri_dir-PA_epi.nii.gz \
#    -mainScanRun 1 \
#    -phaseReversalRun 1
