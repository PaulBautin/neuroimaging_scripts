#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import MARSS

dwi70 = '/local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.nii.gz'
dwi70_phase = '/local_raid/data/pbautin/data/pilot_dataset/rawdata/sub-PNC032/ses-a1/dwi/sub-PNC032_ses-a1_acq-multib70_dir-AP_dwi.nii.gz'
dwi_out = '/local_raid/data/pbautin/data/pilot_dataset/tmp/test/dwi_designer_python.nii'

MARSS.MARSS_main(dwi70, MB=3, workingDir='/local_raid/data/pbautin/data/pilot_dataset/tmp/test/')#,*args)

#dwi70_id = designer_input.splitext_(dwi70)

# dwi_metadata = designer_input.get_input_info(dwi70, fslbval=None, fslbvec=None, bids=None)
# designer_func.run_mppca(args_extent, args_phase, args_shrinkage, args_algorithm, dwi_metadata)
