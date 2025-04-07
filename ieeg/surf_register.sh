
#### left hemisphere ####
wb_command -surface-curvature /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_curvature.func.gii

wb_command -surface-curvature /home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.midthickness_curvature.func.gii

LD_PRELOAD=/home/pabaua/msm-env/lib/libstdc++.so.6:$LD_PRELOAD 
newmsm \
    --inmesh=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_sphere.surf.gii \
    --refmesh=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.sphere.surf.gii \
    --indata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_curvature.func.gii \
    --refdata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.midthickness_curvature.func.gii \
    --inanat=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
    --refanat=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
    --conf=/home/pabaua/Downloads/config_standard_MSM_strain \
    --out=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/L. \
    --verbose

--inanat=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
--refanat=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
--conf=aMSM_strainconf \


#### right hemisphere ####
mris_sphere -q /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_sphere.surf.gii

wb_command -surface-curvature /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_curvature.func.gii

wb_command -surface-curvature /home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.midthickness_curvature.func.gii

LD_PRELOAD=/home/pabaua/msm-env/lib/libstdc++.so.6:$LD_PRELOAD 
msm \
    --inmesh=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_sphere.surf.gii \
    --refmesh=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.sphere.surf.gii \
    --indata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_curvature.func.gii \
    --refdata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.midthickness_curvature.func.gii \
    --inanat=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
    --refanat=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
    --out=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/R. \
    --verbose

    --conf=/home/pabaua/Downloads/config_standard_MSM_strain \




