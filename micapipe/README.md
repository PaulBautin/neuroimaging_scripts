# Processing MicaPipe

## Set variables
dataset='/local_raid/data/pbautin/data/pilot_dataset'
output_folder='/local_raid/data/pbautin/results/micapipe'
fs_licence='/data_/mica1/01_programs/freesurfer-7.3.2/license.txt'
tmp='/local_raid/data/pbautin/results/micapipe/tmp'
subject='Pilot014'
session='02'
micapipe_img='/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif'


## Initial structural pre-processing
```bash
mica-pipe -sub ${subject} \
    	  -bids ${dataset} \
    	  -out ${output_folder} \
        -fs_licence ${fs_licence} \
        -sub ${subject}
    	  -ses ${session} \
    	  -tmpDir ${tmpDir} \
    	  -proc_structural \
        -uni -T1wStr acq-uni_0p5-T1map,acq-inv1_0p5-T1map,acq-inv2_0p5-T1map
```
