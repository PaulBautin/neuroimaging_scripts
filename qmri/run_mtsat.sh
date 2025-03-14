
python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_b1map.py \
-b1_fa /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC024/ses-a2/fmap/sub-PNC024_ses-a2_acq-sfam_TB1TFL.nii.gz \
-b1_ref /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC024/ses-a2/fmap/sub-PNC024_ses-a2_acq-anat_TB1TFL.nii.gz \
-odir /local_raid/data/pbautin/results/qmri/10-03-25_qmri_test



python /local_raid/data/pbautin/software/neuroimaging_scripts/qmri/mica_mtsat.py \
-mt /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC024/ses-a2/anat/sub-PNC024_ses-a2_acq-mtw_mt-on_MTR.nii.gz \
-pd /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC024/ses-a2/anat/sub-PNC024_ses-a2_acq-mtw_mt-off_MTR.nii.gz \
-t1 /data/mica/mica3/BIDS_PNI/rawdata/sub-PNC024/ses-a2/anat/sub-PNC024_ses-a2_acq-mtw_T1w.nii.gz \
-omtsat /local_raid/data/pbautin/results/qmri/10-03-25_qmri_test/mtsat.nii.gz \
-ot1map /local_raid/data/pbautin/results/qmri/10-03-25_qmri_test/t1map.nii.gz \
-omtr /local_raid/data/pbautin/results/qmri/10-03-25_qmri_test/mtr.nii.gz  \
-b1map /local_raid/data/pbautin/results/qmri/10-03-25_qmri_test/sub-PNC024_ses-a2_acq-sfam_TB1TFL_B1map.nii