# cp /data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC*/ses-01/dwi/sub-HC*_ses-01_space-dwi_model-DTI_map-FA.nii.gz /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/

# # export FSLDIR=/local_raid/data/pbautin/software/fsl
# # . ${FSLDIR}/etc/fslconf/fsl.sh

# cd /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/

# tbss_1_preproc *.nii.gz

# tbss_2_reg -t /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_FA.nii.gz


repo="/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/FA"

# Extract unique subject names
subjects=($(find "$repo" -type f -name "sub-*_FA_to_target.nii.gz" \
  | sed -E 's|.*/(sub-[^_]+).*|\1|' | sort -u))

echo "${subjects[@]}"


############## STEP 7 ################
# cd /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/

# for subj in "${subjects[@]}"
# do

# mkdir -p ./FA_individ/${subj}/stats/
# mkdir -p ./FA_individ/${subj}/FA/

# cp ./FA/${subj}_*.nii.gz ./FA_individ/${subj}/FA/

# ####[optional/recommended]####
# fslmaths ./FA_individ/${subj}/FA/${subj}_*FA_to_target.nii.gz \
#     -mas /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_FA_mask.nii.gz \
#     ./FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz

# done

############## STEP 8 ################
cd /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/

for subj in "${subjects[@]}"
do

tbss_skeleton -i ./FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz -p 0.049 \
    /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_FA_skeleton_mask_dst \
    ${FSLDIR}/data/standard/LowerCingulum_1mm.nii.gz \
    ./FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz \
    ./FA_individ/${subj}/stats/${subj}_masked_FAskel.nii.gz \
    -s /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_FA_skeleton_mask.nii.gz

done