#!/bin/bash
#$ -S /bin/bash


## neda.jahanshad@ini.usc.edu ##
## ENIGMA-DTI ##

# #######
# ## part 1 - loop through all subjects to create a subject ROI file 
# #######

# #make an output directory for all files
# mkdir /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_ROI_part1
dirO1=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_ROI_part1

# Extract unique subject names
readarray -t subjects < <(awk -F, '{print $1}' /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList_edited.csv)
echo $subjects

# for subject in "${subjects[@]}"

# do
# /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/singleSubjROI_exe /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_look_up_table.txt \
#                     /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/mean_FA_skeleton.nii.gz \
#                     /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/JHU-WhiteMatter-labels-1mm.nii.gz \
#                     ${dirO1}/${subject}_ROIout \
#                     /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/TBSS/run_tbss/FA_individ/${subject}/stats/${subject}_masked_FAskel.nii.gz

# done


# #######
# ## part 2 - loop through all subjects to create ROI file 
# ##			removing ROIs not of interest and averaging others
# #######

# #make an output directory for all files
# mkdir /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_ROI_part2
# dirO2=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_ROI_part2

# # you may want to automatically create a subjectList file 
# #    in which case delete the old one
# #    and 'echo' the output files into a new name
# # rm ./subjectList.csv

# for subject in "${subjects[@]}"

# do
# /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/averageSubjectTracts_exe ${dirO1}/${subject}_ROIout.csv ${dirO2}/${subject}_ROIout_avg.csv


# # can create subject list here for part 3!
# echo ${subject},${dirO2}/${subject}_ROIout_avg.csv >> /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList.csv
# done

# # python /local_raid/data/pbautin/software/neuroimaging_scripts/ENIGMA/database_organisation.py


#######
## part 3 - combine all 
#######
Table=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ALL_Subject_Info.txt
subjectIDcol=subjectID
subjectList=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/subjectList_edited.csv
outTable=/local_raid/data/pbautin/data/ENIGMA/enigmaDTI/combinedROItable.csv
Ncov=2
covariates="Age;Sex"
Nroi="all" #2
rois="IC;EC"

#location of R binary 
#Rbin=/usr/local/lib/R

#Run the R code
R --no-save --slave --args ${Table} ${subjectIDcol} ${subjectList} ${outTable} ${Ncov} ${covariates} ${Nroi} ${rois} <  /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/combine_subject_tables.R  
