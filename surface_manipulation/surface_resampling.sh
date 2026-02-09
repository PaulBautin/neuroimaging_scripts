wb_command -label-resample \
  /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii \
  /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.sphere.reg.surf.gii \
  /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-5k.L.sphere.reg.surf.gii \
  ADAP_BARY_AREA \
  /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_lh.label.gii \
  -area-surfs /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.midthickness.surf.gii \
            /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-5k.L.surf.gii


wb_command -label-resample \
  /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii \
  /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.sphere.reg.surf.gii \
  /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-5k.R.sphere.reg.surf.gii \
  ADAP_BARY_AREA \
  /local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_fslr-5k_rh.label.gii \
  -area-surfs /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.midthickness.surf.gii \
            /local_raid/data/pbautin/software/micapipe/surfaces/fsLR-5k.R.surf.gii