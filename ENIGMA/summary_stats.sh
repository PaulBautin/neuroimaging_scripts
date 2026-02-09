mkdir /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/figures/

cp /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/combinedROItable.csv /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/figures/

mv /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_plots_ALL.R /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/figures/

cd /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/figures

# should create
# /enigmaDTI/figures/QC_ENIGMA

cohort='MICA-MICS_HC'
R --no-save --slave --args ${cohort} < /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/figures/ENIGMA_DTI_plots_ALL.R


eDTI_outliers \
    --subjID subjectID \
    --output /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_MICA-MICS/QC/QC \
    --DTIinputs /local_raid/data/pbautin/data/ENIGMA/enigmaDTI/ENIGMA_DTI_MICA-MICS/QC/list_files.txt