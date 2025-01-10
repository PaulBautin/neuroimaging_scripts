import nighres
import nibabel
import numpy
import os

subject='sub-AHEAD122017'

in_dir='/local_raid/data/pbautin/data/sub-AHEAD122017/'
out_dir='cortex_segmentation/'

blockface = '/local_raid/data/pbautin/data/sub-AHEAD122017/anat/sub-AHEAD122017_blockface-image.nii.gz'

# 1. an image of bright voxels is computed as 1/(1 ((img-800)/100)^2)
print('cauchy estimate')
cauchy = out_dir+subject+'_cauchy-08.nii.gz'
if not os.path.isfile(cauchy):
    img = nighres.io.load_volume(blockface)
    data = img.get_data()
    data = 1.0/(1.0 + (data-800)*(data-800)/(100*100))
    nii = nibabel.Nifti1Image(data,img.affine,img.header)
    nighres.io.save_volume(cauchy,nii)


# 2. use the results to define csf priors

dark = nighres.filtering.recursive_ridge_diffusion(input_image=blockface,ridge_filter='2D',ridge_intensities='dark',
                                            max_iter=0,save_data=True,max_scale=0,output_dir=out_dir)
light = nighres.filtering.recursive_ridge_diffusion(input_image=cauchy,ridge_filter='2D',ridge_intensities='light',
                                            max_iter=0,save_data=True,max_scale=0,output_dir=out_dir)

print('CSF partial voluming')
pvcsf = out_dir+subject'_pvcsf.nii.gz'
if not os.path.isfile(pvcsf):
    dark = nighres.io.load_volume(dark['propagation'])
    light = nighres.io.load_volume(light['propagation'])
    nii = nibabel.Nifti1Image(numpy.maximum(dark.get_fdata(), light.get_fdata()),
                               dark.affine,dark.header)
    nighres.io.save_volume(pvcsf,nii)

dark = None
light = None
nii = None

# 3. MGDM
mgdm = nighres.brain.mgdm_segmentation(contrast_image1=blockface,contrast_type1='bflidark03',
                                       contrast_image2=pvcsf,contrast_type2='PV',
                                       compute_posterior=True,n_steps=5,max_iterations=100,
                                       adjust_intensity_priors=False,
                                       normalize_qmaps=False,diffuse_probabilities=True,
                                       atlas_file='brain-atlas-insitu.txt',
                                       topology='wcs',save_data=True,overwrite=False,
                                       output_dir=out_dir)

# 4. Extract brain + CRUISE

# left cerebrum
extract = nighres.brain.extract_brain_region(extracted_region='left_cerebrum',
                                   segmentation=mgdm['segmentation'],levelset_boundary=mgdm['distance'],
                                   maximum_membership=mgdm['memberships'],maximum_label=mgdm['labels'],
                                   save_data=True,overwrite=False,output_dir=out_dir)

ctx = nighres.cortex.cruise_cortex_extraction(correct_wm_pv=True,normalize_probabilities=True,wm_dropoff_dist=3.0,
                                        init_image=extract['inside_mask'],
                                        csf_image=extract['background_proba'],
                                        gm_image=extract['region_proba'],
                                        wm_image=extract['inside_proba'],
                                        vd_image=pvcsf,data_weight=0.5,regularization_weight=0.05,
                                        save_data=True,overwrite=False,output_dir=out_dir)

nighres.surface.parcellation_to_meshes(ctx['cortex'], connectivity="18/6",
                                         spacing = 0.0, smoothing=1.0,
                                         save_data=True, overwrite=False,
                                         output_dir=out_dir, file_name=subject+'_left_cerebrum')

#
# right cerebrum
extract = nighres.brain.extract_brain_region(extracted_region='right_cerebrum',
                                   segmentation=mgdm['segmentation'],levelset_boundary=mgdm['distance'],
                                   maximum_membership=mgdm['memberships'],maximum_label=mgdm['labels'],
                                   save_data=True,overwrite=False,output_dir=out_dir)

ctx = nighres.cortex.cruise_cortex_extraction(correct_wm_pv=True,normalize_probabilities=True,wm_dropoff_dist=3.0,
                                        init_image=extract['inside_mask'],
                                        csf_image=extract['background_proba'],
                                        gm_image=extract['region_proba'],
                                        wm_image=extract['inside_proba'],
                                        vd_image=pvcsf,data_weight=0.5,regularization_weight=0.05,
                                        save_data=True,overwrite=False,output_dir=out_dir)

nighres.surface.parcellation_to_meshes(ctx['cortex'], connectivity="18/6",
                                         spacing = 0.0, smoothing=1.0,
                                         save_data=True, overwrite=False,
                                         output_dir=out_dir, file_name=subject+'_right_cerebrum')


# cerebellum
extract = nighres.brain.extract_brain_region(extracted_region='cerebellum',
                                   segmentation=mgdm['segmentation'],levelset_boundary=mgdm['distance'],
                                   maximum_membership=mgdm['memberships'],maximum_label=mgdm['labels'],
                                   save_data=True,overwrite=False,output_dir=out_dir)

ctx = nighres.cortex.cruise_cortex_extraction(correct_wm_pv=True,normalize_probabilities=True,wm_dropoff_dist=3.0,
                                        init_image=extract['inside_mask'],
                                        csf_image=extract['background_proba'],
                                        gm_image=extract['region_proba'],
                                        wm_image=extract['inside_proba'],
                                        vd_image=pvcsf,data_weight=0.5,regularization_weight=0.05,
                                        save_data=True,overwrite=False,output_dir=out_dir)

nighres.surface.parcellation_to_meshes(ctx['cortex'], connectivity="18/6",
                                         spacing = 0.0, smoothing=1.0,
                                         save_data=True, overwrite=False,
                                         output_dir=out_dir, file_name=subject+'_cerebellum')
