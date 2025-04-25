#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/local_raid/data/pbautin/data/containers_scilus_1.6.0.sif' # or .sif
my_main_nf='/local_raid/data/pbautin/software/tractoflow/main.nf'
my_input='/local_raid/data/pbautin/data/tractoflow_test'


NXF_VER=21.10.6 nextflow run $my_main_nf --bids $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --dti_shells "0 700" --fodf_shells "0 700 2000" -profile bundling --run_gibbs_correction true 