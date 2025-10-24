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

version=v0.2.3

#img_singularity=/data/mica1/01_programs/micapipe-v0.2.0/micapipe_"${version}".sif
img_singularity=/data/mica1/01_programs/singularity/micapipe_v1_beta.sif

bids='/local_raid/data/pbautin/data/pilot_dataset/rawdata'
out='/local_raid/data/pbautin/data/pilot_dataset/derivatives'
fs_lic='/data_/mica1/01_programs/freesurfer-7.4.1/license.txt'
# tmpDir='/local_raid/data/pbautin/data/pilot_dataset/tmp/25959_micapipe_proc-dwi_PNC032'
# tmpDir='/local_raid/data/pbautin/data/pilot_dataset/tmp/20705_micapipe_proc-dwi_PNC032'
tmpDir='/local_raid/data/pbautin/data/pilot_dataset/tmp'
sub='sub-PNC032'
ses='ses-a1'
threads=30

# export FSLDIR=/local_raid/data/pbautin/software/fsl
# . ${FSLDIR}/etc/fslconf/fsl.sh
# $FSLDIR/bin/eddy_cuda11.0 --imain=eddy_in.nii --mask=eddy_mask.nii --acqp=eddy_config.txt --index=eddy_indices.txt --bvecs=bvecs --bvals=bvals --topup=field \
# --data_is_shelled --slm=linear --repol --niter=8 --fwhm=10,8,4,2,0,0,0,0 --repol --mporder=6 --s2v_niter=5 --s2v_lambda=1 --s2v_interp=trilinear \
# --slspec=slspec.txt --out=dwi_post_eddy --verbose --estimate_move_by_susceptibility

singularity exec --nv --env FREESURFER_HOME=/opt/freesurfer-7.4.1/freesurfer --writable-tmpfs --containall --cleanenv \
    -B "${bids}:/bids" \
    -B "${out}:/out" \
    -B "${tmpDir}:/tmp" \
    -B "${fs_lic}:/opt/license.txt" \
    -B /local_raid/data/pbautin/software/micapipe:/opt/micapipe \
    "${img_singularity}" \
    bash -c "source /opt/freesurfer-7.4.1/freesurfer/SetUpFreeSurfer.sh && \
             /opt/micapipe/micapipe \
             -bids /bids \
             -out /out \
             -fs_licence /opt/license.txt \
             -threads ${threads} \
             -sub ${sub} \
             -ses ${ses} \
             -proc_dwi \
             -dwi_main /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib38_dir-AP_dwi.nii.gz,/bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib70_dir-AP_dwi.nii.gz \
             -regSynth \
             -dwi_rpe /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-b0_dir-PA_dwi.nii.gz \
             -nocleanup"



# # Create command string
# command="singularity exec --writable-tmpfs --containall \
#         -B ${bids}:/bids \
#         -B ${out}:/out \
#         -B ${tmpDir}:/tmp \
#         -B ${fs_lic}:/opt/licence.txt \
#         -B /local_raid/data/pbautin/software/micapipe:/opt/micapipe \
#         ${img_singularity}"


# ## Run DWI
# ${command} \
#     -bids /bids -out /out -fs_licence /opt/licence.txt -threads ${threads} -sub ${sub} -ses ${ses} \
#     -proc_dwi \
#     -dwi_main /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib38_dir-AP_dwi.nii.gz,/bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib70_dir-AP_dwi.nii.gz -regSynth \
#     -dwi_rpe /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-b0_dir-PA_dwi.nii.gz \
#     -nocleanup 

    # ${command} \
#     -cleanup -sub ${sub} -out /out -bids /bids -ses ${ses} -fs_licence /opt/licence.txt -proc_dwi 



# ######## protocol for complex data denoising
# # convert phase images to mif
# mrconvert /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.nii.gz \
#     -json_import /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.json \
#     -fslgrad /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.bvec \
#     /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.bval \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.mif 

# # convert mag images to mif
# mrconvert /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.nii.gz \
#     -json_import /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.json \
#     -fslgrad /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.bvec \
#     /local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.bval \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.mif 

# # create complex data, if not siemens --> mrcalc dwi_magnitude.mif dwi_phase_rad.mif -polar dwi_complex.mif
# mrcalc /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.mif \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_part-phase_dwi.mif \
#     pi 4096 -div -mult -polar \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex.mif

# # run first-pass complex MP-PCA
# dwidenoise /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex.mif \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_MPPCA_1.mif \
#     -noise /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_MPPCA_1_noise.mif

# # correct phase of original data based on phase of first-pass MP-PCA run
# mrcalc /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex.mif \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_MPPCA_1.mif \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_MPPCA_1.mif \
#     -abs -div -conj -mult \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_pc.mif

# # denoise with complex data
# dwidenoise /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_pc.mif  \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_pc_MPPCA.mif   \
#     -nthreads 30

# # take magnitude from complex
# mrcalc /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_pc_MPPCA.mif \
#     -abs /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_pc_MPPCA_mag.mif

# ### denoise with only magnitude data
# dwidenoise /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.mif   \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi_MPPCA.mif   \
#     -nthreads 30

# ### denoise with raw complex data
# dwidenoise /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complexc.mif  \
#     /local_raid/data/pbautin/data/pilot_dataset/tmp/test/sub-PNC032_ses-a1_acq-multib70_dir-AP_complex_MPPCA.mif   \
#     -nthreads 30





# # Run pipeline FULL
# ${command} \
# -bids /bids -out /out -fs_licence /opt/licence.txt -threads ${threads} -sub ${sub} -ses ${ses} \
# -proc_structural -uni -T1wStr UNIT1,inv-1_MP2RAGE,inv-2_MP2RAGE -proc_surf -post_structural \
# -proc_dwi \
# -dwi_main /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib38_dir-AP_dwi.nii.gz,/bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-multib70_dir-AP_dwi.nii.gz -regSynth \
# -dwi_rpe /bids/${sub}/${ses}/dwi/${sub}_${ses}_acq-b0_dir-PA_dwi.nii.gz \
# -GD -proc_func \
# -mainScanStr task-cross_echo-1_bold,task-cross_echo-2_bold,task-cross_echo-3_bold \
# -func_pe /bids/${sub}/${ses}/fmap/${sub}_${ses}_dir-AP_epi.nii.gz \
# -func_rpe /bids/${sub}/${ses}/fmap/${sub}_${ses}_dir-PA_epi.nii.gz \
# -SC -tracts 40M \
# -MPC -microstructural_img /bids/${sub}/${ses}/anat/${sub}_${ses}_T1map.nii.gz \
#  -microstructural_reg FALSE -mpc_acq T1map

