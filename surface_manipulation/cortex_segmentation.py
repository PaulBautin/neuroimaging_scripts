import nighres
import nibabel
import numpy
import os

import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def recursive_ridge_diffusion(input_image, ridge_intensities, ridge_filter,
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=3,
                              diffusion_factor=1.0,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Recursive Ridge Diffusion

    Extracts planar of tubular structures across multiple scales, with an
    optional directional bias.


    Parameters
    ----------
    input_image: niimg
        Input image
    ridge_intensities: {'bright','dark','both'}
        Which intensities to consider for the filtering
    ridge_filter: {'2D','1D','0D'}
        Whether to filter for 2D ridges, 1D vessels, or 0D holes
    surface_levelset: niimg, optional
        Level set surface to restrict the orientation of the detected features
    orientation: {'undefined','parallel','orthogonal'}
        The orientation of features to keep with regard to the surface or its normal
    loc_prior: niimg, optional
        Location prior image to restrict the search for features
    min_scale: int
        Minimum scale (in voxels) to look for features (default is 0)
    max_scale: int
        Maximum scale (in voxels) to look for features (default is 3)
    diffusion_factor: float
        Scaling factor for the diffusion weighting in [0,1] (default is 1.0)
    similarity_scale: float
        Scaling of the similarity function as a factor of intensity range
    max_iter: int
        Maximum number of diffusion iterations
    max_diff: int
        Maximum difference to stop the diffusion
    threshold: float
        Detection threshold for the structures to keep (default is 0.5)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * filter (niimg): raw filter response (_rrd-filter)
        * propagation (niimg): propagated probabilistic response after diffusion (_rrd-propag)
        * scale (niimg): scale of the detection filter  (_rrd-scale)
        * ridge_dir (niimg): estimated local ridge direction (_rrd-dir)
        * ridge_pv (niimg): ridge partial volume map, taking size into account (_rrd-pv)
        * ridge_size (niimg): estimated size of each detected component (rrd-size)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Extension of the recursive ridge
    filter in [1]_.

    References
    ----------
    .. [1] Bazin et al (2016), Vessel segmentation from quantitative
           susceptibility maps for local oxygenation venography, Proc ISBI.

    """

    print('\n Recursive Ridge Diffusion')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        filter_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-filter'))

        propagation_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd-propag'))

        scale_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd-scale'))

        ridge_direction_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-dir'))

        ridge_pv_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-pv'))

        ridge_size_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-size'))

        if overwrite is False \
            and os.path.isfile(filter_file) \
            and os.path.isfile(propagation_file) \
            and os.path.isfile(scale_file) \
            and os.path.isfile(ridge_direction_file) \
            and os.path.isfile(ridge_pv_file) \
            and os.path.isfile(ridge_size_file) :

            print("skip computation (use existing results)")
            output = {'filter': filter_file,
                      'propagation': propagation_file,
                      'scale': scale_file,
                      'ridge_dir': ridge_direction_file,
                      'ridge_pv': ridge_pv_file,
                      'ridge_size': ridge_size_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = nighres.io.load_volume(input_image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)<3): dimensions = (dimensions[0], dimensions[1], 1)
    if (len(resolution)<3): resolution = [resolution[0], resolution[1], 1.0]

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    if dimensions[2]==1: rrd = nighresjava.FilterRecursiveRidgeDiffusion2D()
    else: rrd = nighresjava.FilterRecursiveRidgeDiffusion()

    # set parameters
    rrd.setRidgeIntensities(ridge_intensities)
    rrd.setRidgeFilter(ridge_filter)
    rrd.setOrientationToSurface(orientation)
    rrd.setMinimumScale(min_scale)
    rrd.setMaximumScale(max_scale)
    rrd.setDiffusionFactor(diffusion_factor)
    rrd.setSimilarityScale(similarity_scale)
    rrd.setPropagationModel("none")
    if max_iter>0: rrd.setPropagationModel("diffusion")
    rrd.setMaxIterations(max_iter)
    rrd.setMaxDifference(max_diff)
    rrd.setDetectionThreshold(threshold)
    #rrd.setUseRatio(use_ratio)

    rrd.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    rrd.setResolutions(resolution[0], resolution[1], resolution[2])

    # input input_image
    rrd.setInputImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # input surface_levelset : dirty fix for the case where surface image not input
    try:
        data = nighres.io.load_volume(surface_levelset).get_fdata()
        rrd.setSurfaceLevelSet(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    except:
        print("no surface image")

    # input location prior image : loc_prior is optional
    try:
        data = nighres.io.load_volume(loc_prior).get_fdata()
        rrd.setLocationPrior(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    except:
        print("no location prior image")


    # execute Extraction
    try:
        rrd.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    filter_data = np.reshape(np.array(rrd.getFilterResponseImage(),
                                   dtype=np.float32), dimensions, 'F')

    propagation_data = np.reshape(np.array(rrd.getPropagatedResponseImage(),
                                    dtype=np.float32), dimensions, 'F')

    scale_data = np.reshape(np.array(rrd.getDetectionScaleImage(),
                                   dtype=np.int32), dimensions, 'F')

    if dimensions[2]==1:
        ridge_direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32),
                                    (dimensions[0],dimensions[1],2),
                                    'F')
    else:
        ridge_direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32),
                                    (dimensions[0],dimensions[1],dimensions[2],3),
                                    'F')

    ridge_pv_data = np.reshape(np.array(rrd.getRidgePartialVolumeImage(),
                                   dtype=np.float32), dimensions, 'F')

    ridge_size_data = np.reshape(np.array(rrd.getRidgeSizeImage(),
                                    dtype=np.float32), dimensions, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(filter_data)
    filter_img = nb.Nifti1Image(filter_data, affine, header)

    header['cal_max'] = np.nanmax(propagation_data)
    propag_img = nb.Nifti1Image(propagation_data, affine, header)

    header['cal_max'] = np.nanmax(scale_data)
    scale_img = nb.Nifti1Image(scale_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_direction_data)
    ridge_dir_img = nb.Nifti1Image(ridge_direction_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_pv_data)
    ridge_pv_img = nb.Nifti1Image(ridge_pv_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_size_data)
    ridge_size_img = nb.Nifti1Image(ridge_size_data, affine, header)

    if save_data:
        nighres.io.save_volume(filter_file, filter_img)
        nighres.io.save_volume(propagation_file, propag_img)
        nighres.io.save_volume(scale_file, scale_img)
        nighres.io.save_volume(ridge_direction_file, ridge_dir_img)
        nighres.io.save_volume(ridge_pv_file, ridge_pv_img)
        nighres.io.save_volume(ridge_size_file, ridge_size_img)

        return {'filter': filter_file, 'propagation': propagation_file, 'scale': scale_file,
                'ridge_dir': ridge_direction_file, 'ridge_pv': ridge_pv_file,
                'ridge_size': ridge_size_file}
    else:
        return {'filter': filter_img, 'propagation': propag_img, 'scale': scale_img,
                'ridge_dir': ridge_dir_img, 'ridge_pv': ridge_pv_img,
                'ridge_size': ridge_size_img}


import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from nighres.io import load_volume, save_volume
from nighres.utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def _get_mgdm_orientation(affine, mgdm):
    '''
    Transforms nibabel affine information into
    orientation and slice order that MGDM understands
    '''
    orientation = nb.aff2axcodes(affine)
    # set mgdm slice order
    if orientation[-1] == "I" or orientation[-1] == "S":
        sliceorder = mgdm.AXIAL
    elif orientation[-1] == "L" or orientation[-1] == "R":
        sliceorder = mgdm.SAGITTAL
    else:
        sliceorder = mgdm.CORONAL

    # set mgdm orientations
    if "L" in orientation:
        LR = mgdm.R2L
    elif "R" in orientation:
        LR = mgdm.L2R  # flipLR = True
    if "A" in orientation:
        AP = mgdm.P2A  # flipAP = True
    elif "P" in orientation:
        AP = mgdm.A2P
    if "I" in orientation:
        IS = mgdm.S2I  # flipIS = True
    elif "S" in orientation:
        IS = mgdm.I2S

    return sliceorder, LR, AP, IS


def _get_mgdm_intensity_priors(atlas_file):
    """
    Returns a list of available as intensity priors
    in the MGDM atlas that you are using
    """
    priors = []
    with open(atlas_file) as fp:
        for i, line in enumerate(fp):
            if "Structures:" in line:  # this is the beginning of the LUT
                lut_idx = i
                lut_rows = list(map(int, [line.split()[1]]))[0]
            if "Intensity Prior:" in line:
                priors.append(line.split()[-1])
    return priors


def mgdm_segmentation(contrast_image1, contrast_type1,
                      contrast_image2=None, contrast_type2=None,
                      contrast_image3=None, contrast_type3=None,
                      contrast_image4=None, contrast_type4=None,
                      n_steps=5, max_iterations=800, topology='wcs',
                      atlas_file=None, topology_lut_dir=None,
                      adjust_intensity_priors=False,
                      normalize_qmaps=True,
                      compute_posterior=False, posterior_scale=5.0,
                      diffuse_probabilities=False,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ MGDM segmentation

    Estimates brain structures from an atlas for MRI data using
    a Multiple Object Geometric Deformable Model (MGDM)

    Parameters
    ----------
    contrast_image1: niimg
        First input image to perform segmentation on
    contrast_type1: str
        Contrast type of first input image, must be listed as a prior in used
        atlas(specified in atlas_file). Possible inputs by default are DWIFA3T,
        DWIMD3T, T1map9T, Mp2rage9T, T1map7T, Mp2rage7T, PV, Filters, T1pv,
        Mprage3T, T1map3T, Mp2rage3T, HCPT1w, HCPT2w, NormMPRAGE.
    contrast_image2: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type2
    contrast_type2: str, optional
        Contrast type of second input image, must be listed as a prior in used
        atlas (specified in atlas_file). Possible inputs by default are the same
        as with parameter contrast_type1 (see above).
    contrast_image3: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type3
    contrast_type3: str, optional
        Contrast type of third input image, must be listed as a prior in used
        atlas (specified in atlas_file). Possible inputs by default are the same
        as with parameter contrast_type1 (see above).
    contrast_image4: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type4
    contrast_type4: str, optional
        Contrast type of fourth input image, must be listed as a prior in used
        atlas (specified in atlas_file). Possible inputs by default are the same
        as with parameter contrast_type1 (see above).
    n_steps: int, optional
        Number of steps for MGDM (default is 5, set to 0 for quick testing of
        registration of priors, which does not perform true segmentation)
    max_iterations: int, optional
        Maximum number of iterations per step for MGDM (default is 800, set
        to 1 for quick testing of registration of priors, which does not
        perform true segmentation)
    topology: {'wcs', 'no'}, optional
        Topology setting, choose 'wcs' (well-composed surfaces) for strongest
        topology constraint, 'no' for no topology constraint (default is 'wcs')
    atlas_file: str, optional
        Path to plain text atlas file (default is stored in DEFAULT_MGDM_ATLAS)
        or atlas name to be searched in MGDM_ATLAS_DIR
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
    normalize_qmaps: bool
        Normalize quantitative maps into [0,1] (default is True)
    adjust_intensity_priors: bool
        Adjust intensity priors based on dataset (default is False)
    normalize_qmaps: bool
        Normalize quantitative maps in [0,1] (default in True, change this if using
        one of the -quant atlas text files in MGDM_ATLAS_DIR)
    compute_posterior: bool
        Compute posterior probabilities for segmented structures
        (default is False)
    posterior_scale: float
        Posterior distance scale from segmented structures to compute posteriors
        (default is 5.0 mm)
    diffuse_probabilities: bool
        Regularize probability distribution with a non-linear diffusion scheme
        (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * segmentation (niimg): Hard brain segmentation with topological
          constraints (if chosen) (_mgdm_seg)
        * labels (niimg): Maximum tissue probability labels (_mgdm_lbls)
        * memberships (niimg): Maximum tissue probability values, 4D image
          where the first dimension shows each voxel's highest probability to
          belong to a specific tissue, the second dimension shows the second
          highest probability to belong to another tissue etc. (_mgdm_mems)
        * distance (niimg): Minimum distance to a segmentation boundary
          (_mgdm_dist)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm details can be
    found in [1]_ and [2]_

    References
    ----------
    .. [1] Bazin et al. (2014). A computational framework for ultra-high
       resolution cortical segmentation at 7 Tesla.
       doi: 10.1016/j.neuroimage.2013.03.077
    .. [2] Bogovic et al. (2013). A multiple object geometric deformable model
       for image segmentation.
       doi:10.1016/j.cviu.2012.10.006.A
    """

    print('\nMGDM Segmentation')

    # check atlas_file and set default if not given
    #atlas_file = _check_mgdm_atlas_file(atlas_file)

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # find available intensity priors in selected MGDM atlas
    mgdm_intensity_priors = _get_mgdm_intensity_priors(atlas_file)

    # sanity check contrast types
    contrasts = [contrast_image1, contrast_image2,
                 contrast_image3, contrast_image4]
    ctypes = [contrast_type1, contrast_type2, contrast_type3, contrast_type4]
    for idx, ctype in enumerate(ctypes):
        if ctype is None and contrasts[idx] is not None:
            raise ValueError(("If specifying contrast_image{0}, please also "
                              "specify contrast_type{0}".format(idx+1, idx+1)))

        elif ctype is not None and ctype not in mgdm_intensity_priors:
            raise ValueError(("{0} is not a valid contrast type for  "
                              "contrast_type{1} please choose from the "
                              "following contrasts provided by the chosen "
                              "atlas: ").format(ctype, idx+1),
                             ", ".join(mgdm_intensity_priors))

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_image1)

        seg_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdm-seg', ))

        lbl_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdm-lbls'))

        mems_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdm-mems'))

        dist_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdm-dist'))
        if overwrite is False \
            and os.path.isfile(seg_file) \
            and os.path.isfile(lbl_file) \
            and os.path.isfile(mems_file) \
            and os.path.isfile(dist_file) :

            print("skip computation (use existing results)")
            output = {
                'segmentation': seg_file,
                'labels': lbl_file,
                'memberships': mems_file,
                'distance': dist_file
            }
            return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create mgdm instance
    mgdm = nighresjava.BrainMgdmMultiSegmentation2()

    # set mgdm parameters
    mgdm.setAtlasFile(atlas_file)
    mgdm.setTopologyLUTdirectory(topology_lut_dir)
    #mgdm.setOutputImages('label_memberships')
    mgdm.setOutputImages('segmentation')
    mgdm.setAdjustIntensityPriors(adjust_intensity_priors)
    mgdm.setComputePosterior(compute_posterior)
    mgdm.setPosteriorScale_mm(posterior_scale)
    mgdm.setDiffuseProbabilities(diffuse_probabilities)
    mgdm.setSteps(n_steps)
    mgdm.setMaxIterations(max_iterations)
    mgdm.setTopology(topology)
    mgdm.setNormalizeQuantitativeMaps(normalize_qmaps)
    # set to False for "quantitative" brain prior atlases
    # (version quant-3.0.5 and above)

    # load contrast image 1 and use it to set dimensions and resolution
    img = load_volume(contrast_image1)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    mgdm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    mgdm.setResolutions(resolution[0], resolution[1], resolution[2])

    # convert orientation information to mgdm slice and orientation info
    sliceorder, LR, AP, IS = _get_mgdm_orientation(affine, mgdm)
    mgdm.setOrientations(sliceorder, LR, AP, IS)

    # input image 1
    mgdm.setContrastImage1(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    mgdm.setContrastType1(contrast_type1)

    # if further contrast are specified, input them
    if contrast_image2 is not None:
        data = load_volume(contrast_image2).get_fdata()
        mgdm.setContrastImage2(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
        mgdm.setContrastType2(contrast_type2)

        if contrast_image3 is not None:
            data = load_volume(contrast_image3).get_fdata()
            mgdm.setContrastImage3(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
            mgdm.setContrastType3(contrast_type3)

            if contrast_image4 is not None:
                data = load_volume(contrast_image4).get_fdata()
                mgdm.setContrastImage4(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
                mgdm.setContrastType4(contrast_type4)

    # execute MGDM
    try:
        mgdm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    seg_data = np.reshape(np.array(mgdm.getSegmentedBrainImage(),
                                   dtype=np.int32), dimensions, 'F')

    dist_data = np.reshape(np.array(mgdm.getLevelsetBoundaryImage(),
                                    dtype=np.float32), dimensions, 'F')

    # if using label_memberships output,
    # membership and labels output has a 4th dimension, set to 6
    dimensions4d = [dimensions[0], dimensions[1], dimensions[2], 6]

    lbl_data = np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),
                                   dtype=np.int32), dimensions, 'F')
    mems_data = np.reshape(np.array(mgdm.getPosteriorMaximumMemberships4D(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(seg_data)
    seg = nb.Nifti1Image(seg_data, affine, header)

    header['cal_max'] = np.nanmax(dist_data)
    dist = nb.Nifti1Image(dist_data, affine, header)

    header['cal_max'] = np.nanmax(lbl_data)
    lbls = nb.Nifti1Image(lbl_data, affine, header)

    header['cal_max'] = np.nanmax(mems_data)
    mems = nb.Nifti1Image(mems_data, affine, header)

    if save_data:
        save_volume(seg_file, seg)
        save_volume(dist_file, dist)
        save_volume(lbl_file, lbls)
        save_volume(mems_file, mems)
        output = {
            'segmentation': seg_file,
            'labels': lbl_file,
            'memberships': mems_file,
            'distance': dist_file
        }
    else:
        output = {
            'segmentation': seg,
            'labels': lbls,
            'memberships': mems,
            'distance': dist
        }

    return output





subject='sub-AHEAD122017'

in_dir = '/local_raid/data/pbautin/data/sub-AHEAD122017/'
out_dir = '/local_raid/data/pbautin/data/sub-AHEAD122017/cortex_segmentation/'

blockface = '/local_raid/data/pbautin/data/sub-AHEAD122017/anat/sub-AHEAD122017_blockface-image.nii.gz'

# 1. an image of bright voxels is computed as 1/(1 ((img-800)/100)^2)
print('cauchy estimate')
# cauchy = out_dir+subject+'_cauchy-08.nii.gz'
# if not os.path.isfile(cauchy):
#     img = nighres.io.load_volume(blockface)
#     data = img.get_fdata()
#     data = 1.0/(1.0 + (data-800)*(data-800)/(100*100))
#     nii = nibabel.Nifti1Image(data,img.affine,img.header)
#     nighres.io.save_volume(cauchy,nii)
#
#
# # 2. use the results to define csf priors
# dark = recursive_ridge_diffusion(input_image=blockface,ridge_filter='2D',ridge_intensities='dark',
#                                             max_iter=0,save_data=True,max_scale=0,output_dir=out_dir)
# light = recursive_ridge_diffusion(input_image=cauchy,ridge_filter='2D',ridge_intensities='light',
#                                             max_iter=0,save_data=True,max_scale=0,output_dir=out_dir)
#
# print('CSF partial voluming')
# pvcsf = out_dir + subject + '_pvcsf.nii.gz'
# if not os.path.isfile(pvcsf):
#     dark = nighres.io.load_volume(dark['propagation'])
#     light = nighres.io.load_volume(light['propagation'])
#     nii = nibabel.Nifti1Image(numpy.maximum(dark.get_fdata(), light.get_fdata()),
#                                dark.affine,dark.header)
#     nighres.io.save_volume(pvcsf,nii)

dark = None
light = None
nii = None

pvcsf='/local_raid/data/pbautin/data/sub-AHEAD122017/cortex_segmentation/sub-AHEAD122017_pvcsf.nii.gz'

# 3. MGDM
mgdm = mgdm_segmentation(contrast_image1=blockface, contrast_type1='bflidark03', contrast_image2=pvcsf, contrast_type2='PV',
                                       compute_posterior=True,n_steps=5,max_iterations=100,
                                       adjust_intensity_priors=False,
                                       normalize_qmaps=False,diffuse_probabilities=True,
                                       atlas_file='/local_raid/data/pbautin/data/sub-AHEAD122017/brain-atlas-insitu.txt',
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
