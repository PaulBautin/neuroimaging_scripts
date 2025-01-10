import nibabel as nib
import numpy as np

img = nib.load('/local_raid/data/pbautin/data/sub-AHEAD152017/anat/sub-AHEAD152017_blockface-image.nii.gz')
header_info = img.header
affine_info = img.affine
print(affine_info)
print(header_info['pixdim'])
affine_info[:3,:] = affine_info[:3,:] * 2
print(affine_info)
new_img = nib.nifti1.Nifti1Image(img.get_fdata(), affine=affine_info)
print(new_img.header)
nib.save(new_img, '/local_raid/data/pbautin/data/sub-AHEAD152017_blockface-image_rescale_x2.nii.gz')
