export FSLDIR=/local_raid/data/pbautin/software/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh

ENIGMAtemplateDirectory=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI_v2
parentDirectory=${ENIGMAtemplateDirectory}/TBSS/run_tbss
dtifit_folder=${ENIGMAtemplateDirectory}/DTIFIT
dirO1=${ENIGMAtemplateDirectory}/ENIGMA_ROI_part1
dirO2=${ENIGMAtemplateDirectory}/ENIGMA_ROI_part2

Table=${ENIGMAtemplateDirectory}/ALL_Subject_Info.txt
subjectIDcol=subjectID
subjectList=${ENIGMAtemplateDirectory}/subjectList.csv
outTable=${ENIGMAtemplateDirectory}/combinedROItable.csv
Ncov=2
covariates="Age;Sex"
Nroi="all" #2
rois="IC;EC"


#mkdir ${dtifit_folder}
# mkdir ${parentDirectory}/MD/
# mkdir ${parentDirectory}/AD/
# mkdir ${parentDirectory}/RD/
# mkdir ${parentDirectory}/FA/
# mkdir ${dirO1}
# mkdir ${dirO2}

cd $parentDirectory

readarray -t subjects < <(awk -F, '{print $1}' ${ENIGMAtemplateDirectory}/subjectList.csv)
#readarray -t subjects < <(awk -F, '{print $1}' ${parentDirectory}/MD_individ/subjectList_MD.csv)


# for subj in "${subjects[@]}"; do

# #     dwiextract "/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/${subj}/ses-01/dwi/${subj}_ses-01_space-dwi_desc-preproc_dwi.mif" \
# #     "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.mif" \
# #     -shell 0,300,700

# #     # Convert .mif to .nii.gz and export gradient files
# #     mrconvert "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.mif" \
# #         "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.nii.gz" \
# #         -export_grad_fsl \
# #         "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.bvec" \
# #         "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.bval"

# #     # Run FSL dtifit
# #     dtifit \
# #         -k "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.nii.gz" \
# #         -o "${dtifit_folder}/${subj}_ses-01_" \
# #         -m "/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/${subj}/ses-01/dwi/${subj}_ses-01_space-dwi_desc-brain_mask.nii.gz" \
# #         -r "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.bvec" \
# #         -b "${dtifit_folder}/${subj}_ses-01_space-dwi_desc-preproc_dti.bval"

# #     cp "${dtifit_folder}/${subj}_ses-01__FA.nii.gz" ${parentDirectory}/
#       # mkdir -p ${parentDirectory}/FA_individ/${subj}/stats/
#       # mkdir -p ${parentDirectory}/FA_individ/${subj}/FA/

#       # cp ${parentDirectory}/FA/${subj}_*.nii.gz ${parentDirectory}/FA_individ/${subj}/FA/

#       # ####[optional/recommended]####
#       # fslmaths ${parentDirectory}/FA_individ/${subj}/FA/${subj}_*FA_to_target.nii.gz \
#       #    -mas ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_mask.nii.gz \
#       #    ${parentDirectory}/FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz
      
#       # tbss_skeleton -i ${parentDirectory}/FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz -p 0.049 \
#       #    ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_skeleton_mask_dst \
#       #    ${FSLDIR}/data/standard/LowerCingulum_1mm.nii.gz \
#       #    ${parentDirectory}/FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz \
#       #    ${parentDirectory}/FA_individ/${subj}/stats/${subj}_masked_FAskel.nii.gz \
#       #    -s ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_skeleton_mask.nii.gz

#       # ${ENIGMAtemplateDirectory}/singleSubjROI_exe ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_look_up_table.txt \
#       #                   ${ENIGMAtemplateDirectory}/ENIGMA_targets/mean_FA_skeleton.nii.gz \
#       #                   ${ENIGMAtemplateDirectory}/ENIGMA_targets/JHU-WhiteMatter-labels-1mm.nii.gz \
#       #                   ${dirO1}/${subj}_ROIout \
#       #                   ${parentDirectory}/FA_individ/${subj}/stats/${subj}_masked_FAskel.nii.gz

#       # ${ENIGMAtemplateDirectory}/averageSubjectTracts_exe ${dirO1}/${subj}_ROIout.csv \
#       #                   ${dirO2}/${subj}_ROIout_avg.csv
#       echo ${subj},${dirO2}/${subj}_ROIout_avg.csv >> ${ENIGMAtemplateDirectory}/subjectList.csv

      
# done

# cd ${parentDirectory}/
# tbss_1_preproc *.nii.gz
# tbss_2_reg -t ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA.nii.gz
# tbss_3_postreg -S

#R --no-save --slave --args ${Table} ${subjectIDcol} ${subjectList} ${outTable} ${Ncov} ${covariates} ${Nroi} ${rois} <  ${ENIGMAtemplateDirectory}/combine_subject_tables.R 

# for subj in "${subjects[@]}"
# do
#    cp ${dtifit_folder}/${subj}*_MD.nii.gz ${parentDirectory}/MD/${subj}_MD.nii.gz
#    cp ${dtifit_folder}/${subj}*_L1.nii.gz ${parentDirectory}/AD/${subj}_AD.nii.gz
#    fslmaths ${dtifit_folder}/${subj}*_L2.nii.gz -add ${dtifit_folder}/${subj}*_L3.nii.gz -div 2 ${parentDirectory}/RD/${subj}_RD.nii.gz


#    for DIFF in MD AD RD
#    do
#    mkdir -p ${parentDirectory}/${DIFF}/origdata/
#    mkdir -p ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/
#    mkdir -p ${parentDirectory}/${DIFF}_individ/${subj}/stats/

#    fslmaths ${parentDirectory}/${DIFF}/${subj}_${DIFF}.nii.gz -mas \
#       ${parentDirectory}/FA/${subj}_ses-01__FA_FA_mask.nii.gz \
#       ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_${DIFF}

#    immv ${parentDirectory}/${DIFF}/${subj} ${parentDirectory}/${DIFF}/origdata/
#    echo "done immv"

#    applywarp -i ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_${DIFF} -o \
#       ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_${DIFF}_to_target -r \
#       $FSLDIR/data/standard/FMRIB58_FA_1mm -w ${parentDirectory}/FA/${subj}_ses-01__FA_FA_to_target_warp.nii.gz
#     echo "done applywarp"

# ##remember to change ENIGMAtemplateDirectory if you re-masked the template

#   fslmaths ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_${DIFF}_to_target -mas \
#        ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_mask.nii.gz \
#        ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_masked_${DIFF}.nii.gz

#    tbss_skeleton -i ${parentDirectory}/FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz -p 0.049 \
#        ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_skeleton_mask_dst.nii.gz $FSLDIR/data/standard/LowerCingulum_1mm.nii.gz \
#        ${parentDirectory}/FA_individ/${subj}/FA/${subj}_masked_FA.nii.gz  \
#        ${parentDirectory}/${DIFF}_individ/${subj}/stats/${subj}_masked_${DIFF}skel -a \
#        ${parentDirectory}/${DIFF}_individ/${subj}/${DIFF}/${subj}_masked_${DIFF}.nii.gz -s \
#        ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_DTI_FA_skeleton_mask.nii.gz


#    done
# done

# for DIFF in MD AD RD
# do
#    dirO1=${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part1
#    dirO2=${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part2
#    # mkdir ${dirO1}
#    # mkdir ${dirO2}
   
# #    for subj in "${subjects[@]}"; do

# #    #   ${ENIGMAtemplateDirectory}/singleSubjROI_exe ${ENIGMAtemplateDirectory}/ENIGMA_targets/ENIGMA_look_up_table.txt \
# #    #       ${ENIGMAtemplateDirectory}/ENIGMA_targets/mean_FA_skeleton.nii.gz ${ENIGMAtemplateDirectory}/ENIGMA_targets/JHU-WhiteMatter-labels-1mm.nii.gz \
# #    #       ${dirO1}/${subj}_${DIFF}_ROIout ${parentDirectory}/${DIFF}_individ/${subj}/stats/${subj}_masked_${DIFF}skel.nii.gz

# #    #   ${ENIGMAtemplateDirectory}/averageSubjectTracts_exe ${dirO1}/${subj}_${DIFF}_ROIout.csv ${dirO2}/${subj}_${DIFF}_ROIout_avg.csv

# # # can create subject list here for part 3!
# #    #   echo ${subj},${dirO2}/${subj}_${DIFF}_ROIout_avg.csv >> ${parentDirectory}/${DIFF}_individ/subjectList_${DIFF}.csv
# #    done

#    Table=${ENIGMAtemplateDirectory}/ALL_Subject_Info.txt
#    subjectIDcol=subjectID
#    subjectList=${parentDirectory}/${DIFF}_individ/subjectList_${DIFF}.csv
#    outTable=${parentDirectory}/${DIFF}_individ/combinedROItable_${DIFF}.csv
#    Ncov=2  #2 if no disease
#    covariates="Age;Sex" # Just "Age;Sex" if no disease
#    Nroi="all"
#    rois="all"

# #Run the R code
#   R --no-save --slave --args ${Table} ${subjectIDcol} ${subjectList} ${outTable} \
#          ${Ncov} ${covariates} ${Nroi} ${rois} < ${ENIGMAtemplateDirectory}/combine_subject_tables.R
# done


# mkdir ${ENIGMAtemplateDirectory}/figures/

cp ${ENIGMAtemplateDirectory}/combinedROItable.csv ${ENIGMAtemplateDirectory}/figures/

# cp /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_plots_ALL.R ${ENIGMAtemplateDirectory}/figures/

cd ${ENIGMAtemplateDirectory}/figures

# should create
# /enigmaDTI/figures/QC_ENIGMA

cohort='MICA-MICS_HC'
R --no-save --slave --args ${cohort} < ${ENIGMAtemplateDirectory}/figures/ENIGMA_DTI_plots_ALL.R


# eDTI_outliers \
#     --subjID subjectID \
#     --output ${ENIGMAtemplateDirectory}/ENIGMA_DTI_MICA-MICS/QC/QC \
#     --DTIinputs ${ENIGMAtemplateDirectory}/ENIGMA_DTI_MICA-MICS/list_files.txt
