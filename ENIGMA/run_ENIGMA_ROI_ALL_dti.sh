parentDirectory=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss
runDirectory=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI

export FSLDIR=/local_raid/data/pbautin/software/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh

# Extract unique subject names
readarray -t subjects < <(awk -F, '{print $1}' /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList_edited.csv)
echo $subjects

for DIFF in MD AD RD
do
   mkdir ${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part1
   dirO1=${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part1

   mkdir ${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part2
   dirO2=${parentDirectory}/${DIFF}_individ/${DIFF}_ENIGMA_ROI_part2

   for subject in "${subjects[@]}"
     do

     ${runDirectory}/singleSubjROI_exe ${runDirectory}/ENIGMA_look_up_table.txt \
         ${runDirectory}/mean_FA_skeleton.nii.gz ${runDirectory}/JHU-WhiteMatter-labels-1mm.nii.gz \
         ${dirO1}/${subject}_${DIFF}_ROIout ${parentDirectory}/${DIFF}_individ/${subject}/stats/${subject}_masked_${DIFF}skel.nii.gz

     ${runDirectory}/averageSubjectTracts_exe ${dirO1}/${subject}_${DIFF}_ROIout.csv ${dirO2}/${subject}_${DIFF}_ROIout_avg.csv

# can create subject list here for part 3!
     echo ${subject},${dirO2}/${subject}_${DIFF}_ROIout_avg.csv >> ${parentDirectory}/${DIFF}_individ/subjectList_${DIFF}.csv
   done

   Table=${runDirectory}/ALL_Subject_Info.txt
   subjectIDcol=subjectID
   subjectList=${parentDirectory}/${DIFF}_individ/subjectList_${DIFF}.csv
   outTable=${parentDirectory}/${DIFF}_individ/combinedROItable_${DIFF}.csv
   Ncov=2  #2 if no disease
   covariates="Age;Sex" # Just "Age;Sex" if no disease
   Nroi="all"
   rois="all"

# #location of R binary
#   Rbin=/usr/local/R-2.9.2_64bit/bin/R

#Run the R code
  R --no-save --slave --args ${Table} ${subjectIDcol} ${subjectList} ${outTable} \
         ${Ncov} ${covariates} ${Nroi} ${rois} < ${runDirectory}/combine_subject_tables.R
done