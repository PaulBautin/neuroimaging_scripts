#########################################################################################
#
# Designer diffusion pipeline
#
# example: python nct_main.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


dwi38=/mnt/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib38_dir-AP_dwi.nii.gz
dwi70=/mnt/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.nii.gz
dwi38_phase=/mnt/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib38_dir-AP_dwi.nii.gz
dwi70_phase=/mnt/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.nii.gz
dwi_rpe=/mnt/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-b0_dir-PA_sbref.nii.gz
dwi_out=/mnt/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_space-dwi_desc-preproc_dwi.mif
dwi38_out=/mnt/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib38_space-dwi_desc-preproc_dwi.mif
dwi70_out=/mnt/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_space-dwi_desc-preproc_dwi.mif
container_sif="/local_raid/data/pbautin/container/designer2_v2.0.13.sif"

mrconf="${HOME}/.mrtrix.conf"
echo "BZeroThreshold: 61" > "$mrconf"

# singularity run --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
# ${container_sif} designer \
#     -denoise \
#     -shrinkage frob \
#     -algorithm jespersen \
#     -phase ${dwi38_phase} \
#     -degibbs -pf 0.75 -pe_dir j- \
#     -scratch /mnt/tmp/processing \
#     -nocleanup \
#     ${dwi38} ${dwi38_out}

# singularity run --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
# ${container_sif} designer \
#     -denoise \
#     -shrinkage frob \
#     -algorithm jespersen \
#     -phase ${dwi70_phase} \
#     -degibbs -pf 0.75 -pe_dir j- \
#     -scratch /mnt/tmp/processing \
#     -nocleanup \
#     ${dwi70} ${dwi70_out}

# singularity run --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
# ${container_sif} designer \
#     -denoise \
#     -shrinkage frob \
#     -algorithm jespersen \
#     -phase ${dwi38_phase},${dwi70_phase} \
#     -degibbs -pf 0.75 -pe_dir j- \
#     -scratch /mnt/tmp/processing \
#     -nocleanup \
#     ${dwi38},${dwi70} ${dwi_out}

dwi38_results=/mnt/tmp_dwi38

singularity run --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
${container_sif} tmi \
    -DTI \
    -DKI \
    ${dwi38_out} ${dwi38_results}


# singularity run --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
# ${container_sif} designer \
#     -denoise \
#     -shrinkage frob \
#     -algorithm jespersen \
#     -phase ${dwi38_phase},${dwi70_phase} \
#     -degibbs -pf 0.75 -pe_dir j- \
#     -eddy -rpe_pair ${dwi_rpe} -echo_time 79 \
#     -normalize \
#     -b1correct \
#     -scratch /mnt/tmp/processing \
#     -nocleanup \
#     ${dwi38},${dwi70} ${dwi_out}


# mrconf="${HOME}/.mrtrix.conf"
# echo "BZeroThreshold: 61" > "$mrconf"

# singularity run --nv --bind /local_raid/data/pbautin/data/pilot_dataset:/mnt \
# ${container_sif} designer \
#     -eddy -rpe_pair ${dwi_rpe}  -echo_time 79 -pf 0.75 -pe_dir j- \
#     -normalize \
#     -b1correct \
#     -scratch /mnt/tmp/processing \
#     -nocleanup \
#     /mnt/tmp/processing/designer-tmp-0IHI5C/working.nii ${dwi_out}

#        -adaptive_patch \