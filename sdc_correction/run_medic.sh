# ------------------------------------------------------------------------------
# Script to Run MEDIC Susceptibility Distortion Correction (SDC) on fMRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# 7T fMRI image SDC is particularly important. MEDIC (Multi-Echo DIstortion Correction)
# can correct B0 distartions for each fMRI frame using the ME phase information.
#
# References:
# - Original Paper: https://doi.org/10.1101%2F2023.11.28.568744
# - GitHub Repository: https://github.com/vanandrew/warpkit
#
# Before Running:
# - Ensure Julia is installed if not: `curl -fsSL https://install.julialang.org | sh`.
# - Create virtual env: `conda create -n env_warpkit`.
# - Install package with command: `pip install warpkit`.
# - singularity build warpkit.simg docker://ghcr.io/vanandrew/warpkit:latest
#
# Input Data:
# - MICA-PNC raw fMRI image
#
# Previous Implementation:
# - The fMRI data was previously SDC using the FSL-TOPUP tool which necissates an
# image with an inverse phase acquisition direction. Moreover, changes in head position
# between fMRI frames can cause changes in B0 that cannot be addressed by TOPUP.
# MEDIC
#
# ------------------------------------------------------------------------------

export SINGULARITY_TMPDIR=/local_raid/data/pbautin/container/singularity_tmp
export SINGULARITY_CACHEDIR=/local_raid/data/pbautin/container/singularity_cache


magnitude_data=( \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-1_bold.nii.gz" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-2_bold.nii.gz" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-3_bold.nii.gz" \
)

phase_data=( \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-1_echo-1_part-phase_bold.nii.gz" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-1_echo-2_part-phase_bold.nii.gz" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-1_echo-3_part-phase_bold.nii.gz" \
)

metadata=( \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-1_bold.json" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-2_bold.json" \
    "/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-01/func/sub-Pilot014_ses-01_task-rest_run-2_echo-3_bold.json" \
)

warpkit_img='/local_raid/data/pbautin/container/warpkit.simg'
out_prefix='/local_raid/data/pbautin/results/medic/medic_'


# singularity exec --writable-tmpfs --containall  \
#     -B /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014 \
#     -B /local_raid/data/pbautin/results/medic \
#     -B /home/bic/pbautin/.julia \
#     $warpkit_img \
#     --magnitude ${magnitude_data[0]} ${magnitude_data[1]} ${magnitude_data[2]} \
#     --phase ${phase_data[0]} ${phase_data[1]} ${phase_data[2]} \
#     --metadata ${metadata[0]} ${metadata[1]} ${metadata[2]} \
#     --out_prefix $out_prefix -n 30 --debug

    # singularity exec --writable-tmpfs --containall  \
    #     -B /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014 \
    #     -B /local_raid/data/pbautin/results/medic \
    #     -B /home/bic/pbautin/.julia \
    #     $warpkit_img \
    #     extract_field_from_maps \
    #     /local_raid/data/pbautin/results/medic/medic__displacementmaps.nii \
    #     /local_raid/data/pbautin/results/medic/medic__displacementfield.nii \
    #     --frame_number 0 \
    #     --phase_encoding_axis j \
    #     --format ants

        # --magnitude ${magnitude_data[0]} ${magnitude_data[1]} ${magnitude_data[2]} \
        # --phase ${phase_data[0]} ${phase_data[1]} ${phase_data[2]} \
        # --metadata ${metadata[0]} ${metadata[1]} ${metadata[2]} \
        # --out_prefix $out_prefix -n 30 --debug

# FSL 6.0
export FSLDIR="/data_/mica1/01_programs/fsl-6-0-3"
export FSL_DIR="/data_/mica1/01_programs/fsl-6-0-3"
export FSL_BIN="${FSLDIR}/bin"
export FSLOUTPUTTYPE="NIFTI_GZ"
export PATH="${AFNIDIR}:${ANTSPATH}:${workbench_path}:${FIXPATH}:${FREESURFER_HOME}/bin/:${mrtrixDir}/bin:${mrtrixDir}/lib:${FSLDIR}:${FSL_BIN}:${FASTSURFER_HOME}:${itk_dir}:${conda3_bin}:${PATH}"
# fslroi ${out_prefix}_displacementmaps.nii ${out_prefix}_displacementmaps_1.nii.gz 0 -1 0 -1 0 -1 0 1
# fslroi ${magnitude_data[0]} ${out_prefix}_magn_1.nii.gz 0 -1 0 -1 0 -1 0 1
# applywarp -i ${out_prefix}_magn_1.nii.gz -r ${out_prefix}_magn_1.nii.gz -w /local_raid/data/pbautin/results/medic/medic__displacementmaps_1.nii.gz -o ${out_prefix}warped_img.nii.gz
# wb_command -convert-warpfield -from-fnirt /local_raid/data/pbautin/results/medic/medic__displacementmaps_1.nii.gz ${magnitude_data[0]} -to-itk ${out_prefix}itk_warp.nii.gz
#fslswapdim ${out_prefix}_displacementmaps.nii -y x z ${out_prefix}_displacementmaps_swap.nii


# Loop over the frames
for frame in {0..5}
do
  singularity exec --writable-tmpfs --containall  \
      -B /local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014 \
      -B /local_raid/data/pbautin/results/medic \
      -B /home/bic/pbautin/.julia \
      $warpkit_img \
      extract_field_from_maps \
      ${out_prefix}_displacementmaps.nii \
      ${out_prefix}_displacementfield_${frame}.nii.gz \
      --frame_number ${frame} \
      --phase_encoding_axis j \
      --format fsl
  # Extract the specific frame using fslroi
  fslroi ${magnitude_data[0]} ${out_prefix}_${frame}_img.nii.gz 0 -1 0 -1 0 -1 ${frame} 1
  applywarp -i ${out_prefix}_${frame}_img.nii.gz -r ${out_prefix}_${frame}_img.nii.gz -w ${out_prefix}_displacementfield_${frame}.nii.gz -o ${out_prefix}_warped_${frame}.nii.gz
#fslroi ${magnitude_data[0]} ${out_prefix}_${frame}_ref.nii.gz 0 -1 0 -1 0 -1 ${frame} 1
# Apply transforms using ANTs
#antsApplyTransforms -d 3 -e 3 -i ${out_prefix}_${frame}_img.nii.gz -r ${out_prefix}_${frame}_img.nii.gz -t ${out_prefix}_displacementfield_${frame}.nii -o ${out_prefix}_${frame}_warped_img.nii.gz -v
done

# Concatenate all the warped frames into a 4D NIfTI image
fslmerge -t ${out_prefix}warped_img_4D.nii.gz ${out_prefix}_warped_{0..5}.nii.gz


# out_prefix='/home/pabaua/dev_mni/results/medic_sdc/medic_'
#
# singularity run --writable-tmpfs --containall  \
#     -B /home/pabaua/dev_mni/data/me_fmri_data/:/home/pabaua/dev_mni/data/me_fmri_data/ \
#     /home/pabaua/dev_mni/software/warpkit.simg \
#     --magnitude ${magnitude_data[0]} ${magnitude_data[1]} ${magnitude_data[2]} \
#     --phase ${phase_data[0]} ${phase_data[1]} ${phase_data[2]} \
#     --metadata ${metadata[0]} ${metadata[1]} ${metadata[2]} \
#     --out_prefix $out_prefix
