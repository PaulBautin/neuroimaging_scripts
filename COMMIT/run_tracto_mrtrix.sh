# ------------------------------------------------------------------------------
# Script to Run the same Tractography as MicaPipe
# ------------------------------------------------------------------------------
#
# -select was decreased to 5M instead on 40M
#
# ------------------------------------------------------------------------------

out_folder='/local_raid/data/pbautin/results/COMMIT'
sub_folder='/local_raid/data/pbautin/data/3t_data/sub-HC007/ses-01/dwi'
sub='sub-HC007_ses-01'
tracts='1M'
tckgen \
        ${sub_folder}/${sub}_space-dwi_model-CSD_map-FOD_desc-wmNorm.nii.gz \
        ${out_folder}/${sub}_space-dwi_desc-iFOD2-${tracts}_tractography.tck \
        -act ${sub_folder}/${sub}_space-dwi_desc-5tt.nii.gz \
        -crop_at_gmwmi \
        -backtrack \
        -seed_dynamic ${sub_folder}/${sub}_space-dwi_model-CSD_map-FOD_desc-wmNorm.nii.gz \
        -algorithm iFOD2 \
        -step 0.5 \
        -angle 22.5 \
        -cutoff 0.06 \
        -maxlength 400 \
        -minlength 10 \
        -select "$tracts"
