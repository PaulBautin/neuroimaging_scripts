
## Imports
# eODF computation
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.reconst.sh import peaks_from_sh, convert_sh_to_sf
from scilpy.reconst.bingham import bingham_fit_sh, _bingham_fit_peak, bingham_to_sf
from scilpy.tractanalysis.todi import get_sf_from_todi, TrackOrientationDensityImaging
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np
#from dipy.reconst.bingham import bingham_to_sf

# Tracking
from dipy.tracking.utils import seeds_from_mask
from dipy.data import default_sphere
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, save_tractogram
from dipy.tracking.mesh import seeds_from_surface_coordinates, random_coordinates_from_surface
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def remove_radial_peaks(fod_file, sh_basis, is_legacy, hemi='L', angle_threshold=45):
    img_sh = nib.load(fod_file)
    sh_shape = img_sh.shape
    sh_order = find_order_from_nb_coeff(sh_shape)
    sphere = get_sphere(name='repulsion724')
    mask = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/tmp_YH/for_Paul/sub-HC062_ses-01_space-dwi_swm5mm_mask.nii.gz")
    mask = nib.as_closest_canonical(mask)
  
    # Loading tractogram
    tracto_file = f"/data/mica/mica3/BIDS_PNI/derivatives/tmp_YH/for_Paul/{hemi}_fsLR32k-laplace-wm-streamlines_dwispace.tck"
    sft = load_tractogram(tracto_file, img_sh)
    sft.to_vox()
    if len(sft.streamlines) < 1:
        raise ValueError('The input bundle contains no streamline.')

    #Main computation
    #Compute TODI from streamlines
    mask_data = mask.get_fdata()
    #mask_data = np.ones(sh_shape[:3])
    todi_sf, sub_mask_3d = get_sf_from_todi(sft, mask_data, todi_sigma=1, sf_threshold=0.5, sphere='repulsion724')

    # Prior to SH
    # Memory friendly saving, as soon as possible saving then delete
    priors_3d = np.zeros(sh_shape)
    priors_3d[sub_mask_3d] = sf_to_sh(todi_sf, sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)
    priors_3d_peak_dir, priors_3d_peak_value, priors_3d_peak_idx = peaks_from_sh(priors_3d, sphere, sub_mask_3d, relative_peak_threshold=0.5, min_separation_angle=25, sh_basis_type=sh_basis, is_legacy=is_legacy, npeaks=1)


    nib.save(nib.Nifti1Image(priors_3d, img_sh.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_todi_sh_priors.nii.gz")
    nib.save(nib.Nifti1Image(sub_mask_3d.astype(np.uint8), img_sh.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_todi_mask.nii.gz")
    del priors_3d

    # Input to SF
    input_sh_3d = img_sh.get_fdata(dtype=np.float32)
    input_3d_peak_dir, input_3d_peak_value, input_3d_peak_inx = peaks_from_sh(input_sh_3d, sphere, sub_mask_3d, relative_peak_threshold=0.5, min_separation_angle=25, sh_basis_type=sh_basis, is_legacy=is_legacy)
    input_3d_sf = convert_sh_to_sf(input_sh_3d, sphere, mask=sub_mask_3d, input_basis=sh_basis, is_input_legacy=is_legacy)

    # Find peaks in the input that are close to the prior peaks (angle < threshold), and mark them for removal
    prior_sq = np.squeeze(priors_3d_peak_dir, axis=3)
    angle_to_peak = np.einsum('...pk,...k->...p', input_3d_peak_dir, prior_sq)
    angle_to_peak = np.rad2deg(np.arccos(np.clip(np.abs(angle_to_peak), 0.0, 1.0)))
    peaks_to_remove_mask = angle_to_peak < angle_threshold
    peaks_to_remove_pos = np.where(peaks_to_remove_mask)

    # 5. Bingham Fitting
    # Shape: (X, Y, Z, max_input_peaks, 7) to safely hold parameters for multiple peaks per voxel
    bingham_volume = np.zeros(input_3d_sf.shape[:3] + (input_3d_peak_dir.shape[3], 7), dtype=np.float32)
    
    for x, y, z, input_peak_idx in zip(*peaks_to_remove_pos):
        peak_dir = input_3d_peak_dir[x, y, z, input_peak_idx]
        voxel_sf = input_3d_sf[x, y, z]
        
        # Fit Bingham lobe for this specific peak
        bingham_peak = _bingham_fit_peak(voxel_sf, peak_dir, sphere, max_angle=angle_threshold).get_flatten()
        bingham_volume[x, y, z, input_peak_idx, :] = bingham_peak

    # Convert the fitted Bingham parameters back to SF
    input_peaks_to_remove_sf_all = bingham_to_sf(bingham_volume, sphere.vertices)
    
    # Sum across the peak dimension (axis=3) to combine lobes if a voxel had >1 peak removed
    total_removal_sf = np.sum(input_peaks_to_remove_sf_all, axis=3)

    # Save the removal volume in SH for debugging
    removal_sh = sf_to_sh(total_removal_sf, sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)
    nib.save(nib.Nifti1Image(removal_sh, img_sh.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_radial_peaks.nii.gz")

    # 6. Subtraction and Final Save
    # Subtract the Bingham SFs from the original SF, preventing negative values
    cleaned_3d_sf = np.maximum(input_3d_sf - total_removal_sf, 0)
    
    # Convert cleaned SF back to SH
    cleaned_sh_3d = sf_to_sh(cleaned_3d_sf, sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)
    nib.save(nib.Nifti1Image(cleaned_sh_3d, img_sh.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_fod_no_radial_peaks.nii.gz")

    return cleaned_sh_3d

    



def compute_eodf(fod_file, sh_basis, is_legacy, hemi='L'):
    img_sh = nib.load(fod_file)
    sh_shape = img_sh.shape
    sh_order = find_order_from_nb_coeff(sh_shape)
    sphere = get_sphere(name='repulsion724')
    mask = nib.load("/data/mica/mica3/BIDS_PNI/derivatives/tmp_YH/for_Paul/sub-HC062_ses-01_space-dwi_swm5mm_mask.nii.gz")

    # Loading tractogram
    tracto_file = f"/data/mica/mica3/BIDS_PNI/derivatives/tmp_YH/for_Paul/{hemi}_fsLR32k-laplace-wm-streamlines_dwispace.tck"
    sft = load_tractogram(tracto_file, mask)
    sft.to_vox()
    if len(sft.streamlines) < 1:
        raise ValueError('The input bundle contains no streamline.')

    #Main computation
    #Compute TODI from streamlines
    mask_data = mask.get_fdata()
    #mask_data = np.ones(sh_shape[:3])
    todi_sf, sub_mask_3d = get_sf_from_todi(sft, mask_data, todi_sigma=1, sf_threshold=0.2, sphere='repulsion724')

    # SF to SH
    # Memory friendly saving, as soon as possible saving then delete
    priors_3d = np.zeros(sh_shape)
    priors_3d[sub_mask_3d] = sf_to_sh(todi_sf, sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)
    nib.save(nib.Nifti1Image(priors_3d, mask.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_todi_sh_priors.nii.gz")
    nib.save(nib.Nifti1Image(sub_mask_3d.astype(np.uint8), mask.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_todi_mask.nii.gz")
    del priors_3d

    # Back to SF
    input_sh_3d = img_sh.get_fdata(dtype=np.float32)
    input_sf_1d = sh_to_sf(input_sh_3d[sub_mask_3d], sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)

    # Creation of the enhanced-FOD (direction-wise multiplication)
    mult_sf_1d = np.maximum(input_sf_1d - todi_sf, 0)
    input_max_value = np.max(input_sf_1d, axis=-1, keepdims=True)
    mult_max_value = np.max(mult_sf_1d, axis=-1, keepdims=True)
    mult_positive_mask = np.squeeze(mult_max_value > 0)
    mult_sf_1d[mult_positive_mask] = mult_sf_1d[mult_positive_mask] * (input_max_value[mult_positive_mask] / (mult_max_value[mult_positive_mask] + 1e-10))
    del todi_sf

    # And back to SH
    # Memory friendly saving
    input_sh_3d[sub_mask_3d] = sf_to_sh(mult_sf_1d, sphere, sh_order_max=sh_order, basis_type=sh_basis, legacy=is_legacy)
    nib.save(nib.Nifti1Image(input_sh_3d, mask.affine), f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_efod_sh.nii.gz")
    del input_sh_3d


def track_eodf(eodf_file, sh_basis, is_legacy, hemi='L', filter=True, filter_distance=2):
    eodf = nib.load(eodf_file)

    # Seeding from mask
    # seed_mask = nib.load("/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/dwi/sub-HC062_ses-01_space-dwi_desc-gmwmi-mask.nii.gz")
    # seeds = seeds_from_mask(seed_mask.get_fdata(), seed_mask.affine, density=1)

    # Seeding from surface
    surf = nib.load(f'/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/surf/sub-HC062_ses-01_hemi-{hemi}_space-nativepro_surf-fsLR-32k_label-white.surf.gii')
    triangles = surf.agg_data("NIFTI_INTENT_TRIANGLE")
    vts = surf.agg_data("NIFTI_INTENT_POINTSET")
    np.savetxt(f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_surface_vertices.csv", vts, delimiter=',')
    
    """
    # Here we apply ants registration to the surface vertices, to get them in dwi space

    antsApplyTransformsToPoints  \
        -d 3   \
        -i /local_raid/data/pbautin/data/tracto_eodf/R_out_surface_vertices.csv   \
        -o /local_raid/data/pbautin/data/tracto_eodf/R_out_surface_vertices_dwi_space.csv    \
        -t /data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/xfm/sub-HC062_ses-01_space-dwi_from-dwi_to-dwi_mode-image_desc-SyN_1Warp.nii.gz   \
        -t /data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/xfm/sub-HC062_ses-01_space-dwi_from-dwi_to-dwi_mode-image_desc-SyN_0GenericAffine.mat   \
        -t [/data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0/sub-HC062/ses-01/xfm/sub-HC062_ses-01_space-dwi_from-dwi_to-nativepro_mode-image_desc-affine_0GenericAffine.mat,1]
    """

    vts = np.loadtxt(f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_surface_vertices_dwi_space.csv", delimiter=',')
    nb_seeds = 1000000
    nb_triangles = len(triangles)
    tri_idx, trilin_co = random_coordinates_from_surface(nb_triangles, nb_seeds)
    seeds = seeds_from_surface_coordinates(triangles, vts, tri_idx, trilin_co)

    # Stopping criterion
    mask_data = nib.load(f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_out_todi_mask.nii.gz")
    stopping_criterion = BinaryStoppingCriterion(mask_data.get_fdata() == 1)

    streamline_generator = probabilistic_tracking(
        seed_positions=seeds,
        sc=stopping_criterion,
        affine=eodf.affine,
        sh=eodf.get_fdata(dtype=np.float32),
        basis_type=sh_basis,
        legacy=is_legacy,
        sphere=default_sphere,
        random_seed=1,
        max_angle=20,
        step_size=0.2,
    )

    streamlines = Streamlines(streamline_generator)

    if filter:
        # Remove streamlines that do not have both endpoints inside 2 mm of the surface using CKdtree
        tree = cKDTree(vts)
        valid_streamlines = []
        for sl in streamlines:
            if len(sl) < 2:
                continue
            start, end = sl[0], sl[-1]
            dist_start, _ = tree.query(start)
            dist_end, _ = tree.query(end)
            if dist_start <= filter_distance and dist_end <= filter_distance:
                valid_streamlines.append(sl)

    sft_eodf = StatefulTractogram(valid_streamlines, eodf, Space.RASMM)
    save_tractogram(sft_eodf, f"/local_raid/data/pbautin/data/tracto_eodf/{hemi}_tractogram_probabilistic_sh.tck")


def main():
    # Computing the eODF
    fod_file = "/local_raid/data/pbautin/data/tracto_eodf/sub-HC062_ses-01_space-dwi_fod_wm.nii.gz"
    sh_basis = 'tournier07'
    is_legacy = False
    remove_radial_peaks(fod_file, sh_basis, is_legacy, hemi='L')
    remove_radial_peaks(fod_file, sh_basis, is_legacy, hemi='R')
    # compute_eodf(fod_file, sh_basis, is_legacy, hemi='L')
    # compute_eodf(fod_file, sh_basis, is_legacy, hemi='R')

    # Tracking the eODF
    eodf_file = "/local_raid/data/pbautin/data/tracto_eodf/L_out_fod_no_radial_peaks.nii.gz"
    track_eodf(eodf_file, sh_basis, is_legacy, hemi='L', filter=True, filter_distance=1)

    eodf_file = "/local_raid/data/pbautin/data/tracto_eodf/R_out_fod_no_radial_peaks.nii.gz"
    track_eodf(eodf_file, sh_basis, is_legacy, hemi='R', filter=True, filter_distance=1)


if __name__ == "__main__":
    main()