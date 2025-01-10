#------------------------------------------------------------------------------#
# Create parcellation on nativepro space
Info "Surface annnot parcellations to T1-nativepro Volume"

parc_annot='/local_raid/data/pbautin/software/micapipe/parcellations/lh.schaefer-200_mics.annot';
parc_str=$(echo "${parc_annot}" | awk -F '_mics' '{print $1}')

for hemi in lh rh; do
Do_cmd mri_surf2surf --hemi "$hemi" \
            --srcsubject fsaverage5 \
            --trgsubject "$idBIDS" \
            --sval-annot "${hemi}.${parc_annot}" \
            --tval "${dir_subjsurf}/label/${hemi}.${parc_annot}" # change this to /label
done

fs_mgz="${tmp}/${parc_str}.mgz"
fs_tmp="${tmp}/${parc_str}_in_T1.mgz"
fs_nii="${tmp}/${T1str_fs}_${parc_str}.nii.gz"                   # labels in fsnative tmp dir
labels_nativepro="${dir_volum}/${T1str_nat}-${parc_str}.nii.gz"  # lables in nativepro

# Register the annot surface parcelation to the T1-surface volume
Do_cmd mri_aparc2aseg --s "$idBIDS" --o "$fs_mgz" --annot "${parc_annot/.annot/}" --new-ribbon
Do_cmd mri_label2vol --seg "$fs_mgz" --temp "$T1surf" --o "$fs_tmp" --regheader "${dir_subjsurf}/mri/aseg.mgz"
mrconvert "$fs_tmp" "$fs_nii" -force -quiet # mgz to nifti_gz
fslreorient2std "$fs_nii" "$fs_nii"         # reorient to standard
fslmaths "$fs_nii" -thr 1000 "$fs_nii"      # threshold the labels
