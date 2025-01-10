#------------------------------------------------------------------------------#
# Create parcellation on nativepro space
FREESURFER_HOME=/data/mica/mica1/01_programs/freesurfer-7.4.1 \
/data/mica/mica1/01_programs/freesurfer-7.4.1/bin/mris_smooth \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface.surf.gii \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface_smooth.surf.gii

FREESURFER_HOME=/data/mica/mica1/01_programs/freesurfer-7.4.1 \
/data/mica/mica1/01_programs/freesurfer-7.4.1/bin/mris_inflate \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface_smooth.surf.gii \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface_smooth_inflate.surf.gii

FREESURFER_HOME=/data/mica/mica1/01_programs/freesurfer-7.4.1 \
/data/mica/mica1/01_programs/freesurfer-7.4.1/bin/mris_sphere \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface_smooth_inflate.surf.gii \
/local_raid/data/pbautin/data/ahead/Ahead_brain_122017_cruise-right-wm-surface_smooth_sphere.surf.gii
