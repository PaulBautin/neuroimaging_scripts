# Processing MicaPipe

### 1 - Set variables
```bash
dataset='/local_raid/data/pbautin/data/pilot_dataset'
output_folder='/local_raid/data/pbautin/results/micapipe'
fs_licence='/data_/mica1/01_programs/freesurfer-7.3.2/license.txt'
tmp='/local_raid/data/pbautin/results/micapipe/tmp'
subject='Pilot014'
session='02'
micapipe_img='/data_/mica1/01_programs/micapipe-v0.2.0/micapipe_v0.2.3.sif'
```

### 2 - Initial structural pre-processing
```bash
micapipe -sub ${subject} \
    	  -bids ${dataset} \
    	  -out ${output_folder} \
        -fs_licence ${fs_licence} \
        -sub ${subject} \
    	  -ses ${session} \
    	  -tmpDir ${tmpDir} \
    	  -proc_structural \
        -uni -T1wStr acq-uni_0p5-T1map,acq-inv1_0p5-T1map,acq-inv2_0p5-T1map \
        -mf 3
```

### 3 - denoising
```bash
input_t1=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w.nii.gz
output_t1_n4=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_T1w_n4.nii.gz
output_t1_bias_field=${output_folder}/micapipe_v0.2.0/sub-${subject}/ses-${session}/anat/sub-${subject}_ses-${session}_space-nativepro_bias_field.nii.gz
N4BiasFieldCorrection -i ${input_t1} \
    -o [${output_t1_n4}, ${output_t1_bias_field}] \
    -c [300x150x75x50, 1e-6] -v 1
```

### 3 - Surface reconstruction
```bash
micapipe -sub ${subject} \
    	  -bids ${dataset} \
    	  -out ${output_folder} \
        -fs_licence ${fs_licence} \
        -sub ${subject} \
    	  -ses ${session} \
    	  -tmpDir ${tmpDir} \
    	  -proc_surf
```
