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


export MICAPIPE=/local_raid/data/pbautin/software/micapipe
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
#conda3_bin=/local_raid/data/pbautin/software/conda/bin/
source /data/mica1/01_programs/micapipe-v0.2.0_conda/conda3/etc/profile.d/conda.sh


#------------------------------------------------------------------------------#
# Set the libraries paths for mrtrx and fsl
export LD_LIBRARY_PATH="${FSLDIR}/lib:${FSL_BIN}:${mrtrixDir}/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

#-----------------------------------------------------------------------------------#
# Export new PATH with al the necessary binaries
#export PATH="${AFNIDIR}:${ANTSPATH}:${workbench_path}:${FIXPATH}:${FREESURFER_HOME}/bin/:${mrtrixDir}/bin:${mrtrixDir}/lib:${FSLDIR}:${FSL_BIN}:${PYTHON_3}:${FASTSURFER_HOME}:${itk_dir}:${PATH}"
export PATH="${AFNIDIR}:${ANTSPATH}:${workbench_path}:${FIXPATH}:${FREESURFER_HOME}/bin/:${mrtrixDir}/bin:${mrtrixDir}/lib:${FSLDIR}:${FSL_BIN}:${FASTSURFER_HOME}:${itk_dir}:${conda3_bin}:${PATH}"
conda activate /data/mica1/01_programs/micapipe-v0.2.0_conda/micapipe

export tmpDir=/local_raid/data/pbautin/results/micapipe/tmp

#bash /local_raid/data/pbautin/micapipe/functions/init.sh

export SINGULARITY_TMPDIR=/local_raid/data/pbautin/container/singularity_tmp
export SINGULARITY_CACHEDIR=/local_raid/data/pbautin/container/singularity_cache





dataset='/local_raid/data/pbautin/data/MNI152Volumes'
output_folder='/local_raid/data/pbautin/data/MNI152Volumes/micapipe'
fs_lic='/data_/mica1/01_programs/freesurfer-7.3.2/license.txt'
tmpDir='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/tmp'
subject='mni'
session='01'
micapipe_img='/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif'


mri_synthseg --i /local_raid/data/pbautin/data/sub-AHEAD122017/anat/sub-AHEAD122017_blockface-image.nii.gz \
--o /local_raid/data/pbautin/data/sub-AHEAD122017/synthseg --robust --threads 30 --parc


# micapipe -sub ${subject} \
#     	  -bids ${dataset} \
#     	  -out ${output_folder} \
#         -fs_licence ${fs_licence} \
#         -sub ${subject} \
#     	  -ses ${session} \
#     	  -tmpDir ${tmpDir} \
#     	  -proc_structural

# micapipe -sub ${subject} \
#     	  -bids ${dataset} \
#     	  -out ${output_folder} \
#         -fs_licence ${fs_licence} \
#         -sub ${subject} \
#     	  -ses ${session} \
#     	  -tmpDir ${tmpDir} \
#     	  -proc_surf

# singularity run --writable-tmpfs --containall \
#    -B ${dataset}:/bids \
#    -B ${output_folder}:/out \
#    -B ${tmpDir}:/tmp \
#    -B ${fs_licence}:/opt/licence.txt \
#    ${micapipe_img} \
#    -bids /bids \
#    -out /out \
#    -fs_licence /opt/licence.txt \
#    -proc_surf \
#    -sub $subject \
#    -ses $session \

# singularity run --writable-tmpfs --containall \
#    -B ${dataset}:/bids \
#    -B ${output_folder}:/out \
#    -B ${tmpDir}:/tmp \
#    -B ${fs_licence}:/opt/licence.txt \
#    ${micapipe_img} \
#    -bids /bids \
#    -out /out \
#    -fs_licence /opt/licence.txt \
#    -post_structural \
#    -GD \
#    -sub $subject \
#    -ses $session \


# Remove the medial wall
# dwi_cortex='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-schaefer-200_out.nii.gz'
# dwi_subc='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-subcortical.nii.gz'
# dwi_cere='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-cerebellum.nii.gz'
# dwi_cortexSub='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-schaefer-200_sub.nii.gz'
# dwi_all='/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-schaefer-200_all.nii.gz'
#for i in 1000 2000; do fslmaths "$dwi_cortex" -thr "$i" -uthr "$i" -binv -mul "$dwi_cortex" "$dwi_cortex"; done
#fslmaths "$dwi_cortex" -binv -mul "$dwi_subc" -add "$dwi_cortex" "$dwi_cortexSub" -odt int # added the subcortical parcellation
#   fslmaths $dwi_cere -add 100 $dwi_cere
#fslmaths "$dwi_cortexSub" -binv -mul "$dwi_cere" -add "$dwi_cortexSub" "$dwi_all" -odt int # added the cerebellar parcellation


# tck='/local_raid/data/pbautin/data/dTOR_full_tractogram.tck'
#
# tck2connectome ${tck} ${dwi_all} "/local_raid/data/pbautin/data/connectome.txt"
# lut_sc='/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_subcortical-cerebellum_mics.csv'
# lut='/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-200_mics.csv'
# python "${MICAPIPE}"/functions/connectome_slicer.py --conn="/local_raid/data/pbautin/data/connectome.txt" --lut1="$lut_sc" --lut2="$lut" --mica="$MICAPIPE"



# input_t1=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w.nii.gz
# output_t1_n4=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w_n4.nii.gz
# output_t1_bias_field=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_bias_field.nii.gz

# input_t1=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w.nii.gz
# output_t1_n4=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w_n4.nii.gz
# output_t1_bias_field=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_bias_field.nii.gz

# N4BiasFieldCorrection -d 3 -i ${input_t1} \
#     -o [${output_t1_n4}, ${output_t1_bias_field}] \
#     -b [ 180, 3 ] -c [ 50x50x50x50, 0.0 ] -s 4 -v 1

# source activate env_commit
# output_t1_nlm=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w_nlm.nii.gz
# python /local_raid/data/pbautin/software/neuroimaging_scripts/denoising/mica_nlm.py \
#     ${output_t1_n4} \
#     ${output_t1_nlm} \
#     32


#structural processing
#micapipe \
#    -bids $bids \
#    -out $out \
#    -fs_licence $fs_lic \
#    -sub $sub \
#    -ses $ses \
#    -post_structural \
#    -uni -T1wStr acq-uni_0p5-T1map,acq-inv1_0p5-T1map,acq-inv2_0p5-T1map

# micapipe -sub ${subject} \
#     	  -bids ${dataset} \
#     	  -out ${output_folder} \
#         -fs_licence ${fs_licence} \
#         -sub ${subject} \
#     	  -ses ${session} \
#     	  -tmpDir ${tmpDir} \
#     	  -proc_surf

# DWI processing
# micapipe \
#     -bids $bids \
#     -out $out \
#     -fs_licence $fs_lic \
#     -sub $sub \
#     -ses $ses \
#     -b0thr '61' \
#     -proc_dwi \
#     -dwi_main ${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_38d_dir-AP_1p5iso_run-1_dwi.nii.gz,${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_70d_dir-AP_1p5iso_run-1_dwi.nii.gz \
#     -dwi_rpe ${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq-b0_dir-PA_1p5iso_run-1_epi.nii.gz \
#     -tmpDir $tmp -threads 30 \
#     -dwi_phase ${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_38d_dir-AP_1p5iso_run-2_dwi.nii.gz,${bids}/sub-${sub}/ses-${ses}/dwi/sub-${sub}_ses-${ses}_acq_multib_70d_dir-AP_1p5iso_run-2_dwi.nii.gz
#
#
# # DWI processing
 # micapipe \
 #     -bids $bids \
 #     -out $out \
 #     -fs_licence $fs_lic \
 #     -sub $sub \
 #     -ses $ses \
 #     -SC -nocleanup -tracts 5M


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

# # qMRIDWI processing
# micapipe \
#    -bids $bids \
#    -out $out \
#    -fs_licence $fs_lic \
#    -sub $sub \
#    -ses $ses \
#    -MPC -mpc_acq T1map -regSynth \
#  	 -microstructural_img ${bids}/sub-${sub}/ses-${ses}/anat/sub-${sub}_ses-${ses}_acq-T1_0p5-T1map.nii.gz \
#    -microstructural_reg ${bids}/sub-${sub}/ses-${ses}/anat/sub-${sub}_ses-${ses}_acq-inv1_0p5-T1map.nii.gz \
#    -tmpDir $tmp -threads 30 \
