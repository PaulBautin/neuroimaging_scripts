
#### left hemisphere ####
# computing sphere 
mris_sphere -q /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_sphere.surf.gii

# computing curvature
wb_command -surface-curvature /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_curvature.func.gii
wb_command -surface-curvature /home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.midthickness_curvature.func.gii

# computing sulc as mm between inflated and white surfaces projected onto white normal
mris_inflate -sulc sulcus.func.gii -mm -lh \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_inflate.surf.gii
mris_inflate -sulc fslr32k_L_sulcus.func.gii -mm -lh \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness_inflate.surf.gii
wb_command -set-structure \
    /home/pabaua/dev_mni/micapipe/surfaces/lh.fslr32k_L_sulcus.func.gii \
    CORTEX_LEFT

# merging sulcus and curvature metrics
wb_command -metric-merge /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/lh.sulcus_curv.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_curvature.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/lh.sulcus.func.gii
wb_command -metric-merge /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.sulcus_curv.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.midthickness_curvature.func.gii \
-metric /home/pabaua/dev_mni/micapipe/surfaces/lh.fslr32k_L_sulcus.func.gii


wb_command -surface-curvature /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_curvature.func.gii

wb_command -surface-curvature /home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.midthickness_curvature.func.gii

mris_inflate -sulc surf_lh_sulcus.func.gii -lh \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_inflate.surf.gii

mris_inflate -sulc surf_lh_sulcus_fslr32k.func.gii -lh \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness_inflate.surf.gii

LD_PRELOAD=/home/pabaua/msm-env/lib/libstdc++.so.6:$LD_PRELOAD 
/home/pabaua/Downloads/msm_ubuntu_v3 \
    --inmesh=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh_sphere.surf.gii \
    --refmesh=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.sphere.surf.gii \
    --indata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/lh.sulcus_curv.func.gii \
    --refdata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.L.sulcus_curv.func.gii \
    --inanat=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
    --refanat=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
    --out=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/L. \
    --conf=/home/pabaua/Downloads/config_standard_MSM_strain \
    --verbose 

# wb_command -metric-resample \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/plot_values_lh.func.gii \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/L.sphere.reg.surf.gii \
#     /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.sphere.surf.gii \
#     ADAP_BARY_AREA \
#    /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/plot_values_lh_fslr32k.func.gii  \
#     -area-surfs \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_lh.surf.gii \
#     /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii 


#### right hemisphere ####
# computing sphere 
mris_sphere -q /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_sphere.surf.gii

# computing curvature
wb_command -surface-curvature /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_curvature.func.gii
wb_command -surface-curvature /home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
-mean /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.midthickness_curvature.func.gii

# computing sulc as mm between inflated and white surfaces projected onto white normal
mris_inflate -sulc sulcus.func.gii -mm -rh \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_inflate.surf.gii
mris_inflate -sulc fslr32k_R_sulcus.func.gii -mm -rh \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness_inflate.surf.gii
wb_command -set-structure \
    /home/pabaua/dev_mni/micapipe/surfaces/rh.fslr32k_R_sulcus.func.gii \
    CORTEX_RIGHT

# merging sulcus and curvature metrics
wb_command -metric-merge /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/rh.sulcus_curv.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_curvature.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/rh.sulcus.func.gii
wb_command -metric-merge /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.sulcus_curv.func.gii \
-metric /home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.midthickness_curvature.func.gii \
-metric /home/pabaua/dev_mni/micapipe/surfaces/rh.fslr32k_R_sulcus.func.gii

LD_PRELOAD=/home/pabaua/msm-env/lib/libstdc++.so.6:$LD_PRELOAD 
/home/pabaua/Downloads/msm_ubuntu_v3 \
    --inmesh=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh_sphere.surf.gii \
    --refmesh=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.sphere.surf.gii \
    --indata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/rh.sulcus_curv.func.gii \
    --refdata=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/fsLR-32k.R.sulcus_curv.func.gii \
    --inanat=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
    --refanat=/home/pabaua/dev_mni/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
    --out=/home/pabaua/dev_mni/neuroimaging_scripts/ieeg/R. \
    --conf=/home/pabaua/Downloads/config_standard_MSM_strain \
    --verbose 

# wb_command -metric-resample \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/plot_values_rh.func.gii \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/R.sphere.reg.surf.gii \
#     /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.sphere.surf.gii \
#     ADAP_BARY_AREA \
#    /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/plot_values_rh_fslr32k.func.gii  \
#     -area-surfs \
#     /local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/surf_rh.surf.gii \
#     /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
#     -largest






