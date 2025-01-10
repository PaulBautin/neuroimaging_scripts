#------------------------------------------------------------------------------#
# Create parcellation on nativepro space
FREESURFER_HOME=/export01/local/freesurfer \
/export01/local/freesurfer/bin/mris_convert \
/local_raid/data/pbautin/data/sub-AHEAD122017/fastsurfer/AHEAD-122017-0p8mm/surf/lh.white \
/local_raid/data/pbautin/data/sub-AHEAD122017/surf_proc/lh_white.surf.gii

FREESURFER_HOME=/export01/local/freesurfer \
/export01/local/freesurfer/bin/mris_convert \
/local_raid/data/pbautin/data/sub-AHEAD122017/fastsurfer/AHEAD-122017-0p8mm/surf/lh.pial \
/local_raid/data/pbautin/data/sub-AHEAD122017/surf_proc/lh_pial.surf.gii

wb_command -surface-cortex-layer \
/local_raid/data/pbautin/data/sub-AHEAD122017/surf_proc/lh_white.surf.gii \
/local_raid/data/pbautin/data/sub-AHEAD122017/surf_proc/lh_pial.surf.gii \
0.5 \
/local_raid/data/pbautin/data/sub-AHEAD122017/surf_proc/0_5_surf.surf.gii
