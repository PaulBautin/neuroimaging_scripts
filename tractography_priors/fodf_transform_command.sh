#!/bin/bash
set -euo pipefail

threads=40
subject='sub-04'

bdp_dir='/local_raid/data/pbautin/data/bradypho'
output_data='/local_raid/data/pbautin/data/bradypho/output_data'
bdp_registration_utils='/local_raid/data/pbautin/data/bradypho/bdp_registration_utils'
fodf_path='/data/mica3/BIDS_PNI/derivatives/tmp_YH/for_Paul/MICs-21sub_space-mni_fod-wm_template.nii.gz'

# 1. Initialise identity warp components in FOD space
warpinit ${fodf_path} ${output_data}/identity_warp[].nii

# 2. Apply ANTs transforms to each warp component
for i in {0..2}; do
    antsApplyTransforms -d 3 -e 0 \
        -i ${output_data}/identity_warp${i}.nii \
        -o ${output_data}/mrtrix_warp${i}.nii \
        -r ${bdp_registration_utils}/${subject}/to_specimenWarped.nii.gz \
        -t ${bdp_registration_utils}/${subject}/to_specimen1Warp.nii.gz \
        -t ${bdp_registration_utils}/${subject}/to_specimen0GenericAffine.mat \
        -n BSpline[3]
done

# 3. Correct warp field for MRtrix compatibility and apply to FOD
warpcorrect ${output_data}/mrtrix_warp[].nii ${output_data}/mrtrix_warp_corrected.mif

mrtransform ${fodf_path} \
    -warp ${output_data}/mrtrix_warp_corrected.mif \
    ${output_data}/warped_input_image.mif \
    -reorient_fod yes -nthreads ${threads} -force
