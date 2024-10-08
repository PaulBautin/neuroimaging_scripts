#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# Script to Run COMMIT on Diffusion MRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# Structural connectomes reconstructed from dMRI have poor accuracy and can be improved
# with filtering methods.
#
# References:
# - Original Paper: https://doi.org/10.1126%2Fsciadv.aba8245
# - GitHub Repository: https://github.com/daducci/COMMIT
#
# Before Running:
# - Ensure MATLAB version 2017b or newer is installed with the Image Processing
#   Toolbox and Signal Processing Toolbox.
# - Clone the NORDIC GitHub repository.
#
# Input Data:
# - MICA-PNC Pilot014 preprocessed dMRI images
#
# Previous Implementation:
# - The tractogram data was previously filtered with SIFT
#
# ------------------------------------------------------------------------------


# Import and configure commit package
import commit
from commit import trk2dictionary
import subprocess
import amico
import os
import numpy as np
commit.setup()
import glob
import dipy
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
import nibabel as nib


# test
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.viz import window, actor, colormap
from operator import itemgetter, attrgetter



# Function to convert neuroimaging files using mrconvert
def run_mrconvert(input_file, output_file, options=None):
    """
    Convert neuroimaging files using mrconvert.

    :param input_file: Path to the input file (e.g., .mif, .nii, .nii.gz).
    :param output_file: Path to the output file (desired format).
    :param options: List of additional command line options for mrconvert.
    """
    # Build the command as a list of strings
    command = ['mrconvert', input_file, output_file]

    # Add additional options if provided
    if options:
        command.extend(options)

    try:
        # Run the command and capture output and errors
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Output the command's response for logging
        print(result.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode('utf-8')}")


# Function to convert neuroimaging files using mrconvert
def run_sh2peaks(input_file, output_file, options=None):
    """
    Convert neuroimaging files using mrconvert.

    :param input_file: Path to the input file (e.g., .mif, .nii, .nii.gz).
    :param output_file: Path to the output file (desired format).
    :param options: List of additional command line options for mrconvert.
    """
    # Build the command as a list of strings
    command = ['sh2peaks', input_file, output_file]

    # Add additional options if provided
    if options:
        command.extend(options)

    try:
        # Run the command and capture output and errors
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Output the command's response for logging
        print(result.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode('utf-8')}")



output_fiber_assignment = '/local_raid/data/pbautin/results/COMMIT/fibers_assignment.txt'
connectome = '/local_raid/data/pbautin/results/COMMIT/connectome.csv'
filename_tractogram = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography.tck'
output_tractogram = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography_connecting.tck'
output_tractogram_cluster = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography_connecting_cluster.tck'
atlas = '/local_raid/data/pbautin/results/micapipe/tmp/25361_micapipe_post-dwi_HC007/HC007_schaefer-100-full_dwi.nii.gz'
os.system( 'dice_connectome_build ' + output_fiber_assignment + ' ' + connectome + ' -t '+ filename_tractogram +' -a ' + atlas +' -f')
#os.system('dice_tractogram_cluster ' + filename_tractogram + ' 10 ' + output_tractogram_cluster + ' --atlas ' + atlas + ' --tmp_folder ' + '/local_raid/data/pbautin/results/COMMIT/tmp ' + '-k -f')
#os.system( 'dice_tractogram_split ' + filename_tractogram + ' ' + output_fiber_assignment + ' --output_folder /local_raid/data/pbautin/results/COMMIT/tmp/bundles_split -f')
os.system( 'dice_tractogram_sort ' + filename_tractogram + ' ' + atlas + ' ' + output_tractogram_cluster + ' --tmp_folder ' + '/local_raid/data/pbautin/results/COMMIT/tmp_2 ' + '-k -f')

# First, we retrieve the number of streamlines present in each bundle from the connectome previously calculated, i.e.
C = np.loadtxt( connectome, delimiter=',' )
C = np.triu( C ) # be sure to get only the upper-triangular part of the matrix
group_size = C[C>0].astype(np.int32)

# With this information, we create an array with the indices of the streamlines composing each bundle:
tmp = np.insert( np.cumsum(group_size), 0, 0 )
group_idx = np.fromiter( [np.arange(tmp[i],tmp[i+1]) for i in range(len(tmp)-1)], dtype=np.object_ )
np.save('/local_raid/data/pbautin/results/COMMIT/group_idx_commit.npy', group_idx)
print(group_idx)


from operator import itemgetter, attrgetter
atlas_img = nib.load(atlas)
fnames = [(i, int(i.split("bundles/bundle_")[1].split("-")[0]), int(i.split("bundles/bundle_")[1].split("-")[1].split(".tck")[0])) for i in glob.glob('/local_raid/data/pbautin/results/COMMIT/tmp_2/bundles/bundle*.tck')]
tasks = [i[0] for i in sorted(fnames, key=itemgetter(1, 2))]
qb = QuickBundles(threshold=1000000000)

cl = []
n = 0
for t in tasks:
    tractogram = load_tractogram(t, atlas_img, bbox_valid_check=False)
    clusters = qb.cluster(tractogram.streamlines)
    for c in clusters:
        cl.append(np.array(c.indices) + n)
    cl.append(np.array(range(len(tractogram))) + n) # .append(np.array(range(len(tractogram))) + n)
    n = n + len(tractogram)
group_idx = np.array(cl, dtype=object)
np.save('/local_raid/data/pbautin/results/COMMIT/group_idx_commit_my.npy', group_idx)
print(group_idx)


    # interactive = False
    # scene = window.Scene()
    # colormap = colormap.create_colormap(np.arange(len(clusters)))
    # scene.clear()
    # scene.SetBackground(1, 1, 1)
    # scene.add(actor.streamtube(tractogram.streamlines, window.colors.white, opacity=0.05))
    # scene.add(actor.streamtube(clusters.centroids, colormap, linewidth=0.4))
    # window.record(scene, out_path='fornix_centroids.png', size=(600, 600))
    # if interactive:
    #     window.show(scene)





# #print([np.load(i).astype(np.int32) for i in glob.glob('/local_raid/data/pbautin/results/COMMIT/tmp/bundles/bundle*.npy')])
# #os.system( 'dice_tractogram_sort ' + filename_tractogram + ' ' + atlas + ' ' + output_tractogram + ' -f')
# print(np.array(streamlines, dtype=object))

# Example usage
input_file = '/local_raid/data/pbautin/data/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-preproc_dwi.mif'  # Input neuroimaging file
output_dwi = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-preproc_dwi.nii.gz'
output_bvec = '/local_raid/data/pbautin/results/COMMIT/dwi.bvec'  # Desired output file format
output_bval = '/local_raid/data/pbautin/results/COMMIT/dwi.bval'  # Desired output file format
options = ['-export_grad_fsl', output_bvec, output_bval]  # Optional arguments (e.g., setting strides)
run_mrconvert(input_file, output_dwi, options)
run_mrconvert(input_file, output_dwi)

input_sh = '/local_raid/data/pbautin/data/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_model-CSD_map-FOD_desc-wmNorm.nii.gz'
output_peak = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-preproc_peaks.nii.gz'
run_sh2peaks(input_sh, output_peak)

tissue_segmentation = '/local_raid/data/pbautin/data/sub-HC007/ses-01/dwi/sub-HC007_ses-01_space-dwi_desc-5tt.nii.gz'
output_wmmask = '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-preproc_wmmask.nii.gz'
options = ['-coord', '3', '2', '-axes', '0,1,2']
run_mrconvert(tissue_segmentation, output_wmmask, options)


# Import the tractogram and necessary files
# The output tractogram from MrTrix has been already converted to the format accepted by COMMIT
# (no need of     fiber_shift         = 0.5,)
    # blur_clust_thr      = 2,
    # blur_clust_groupby  = None,
    # blur_core_extent    = 2,
    # blur_gauss_extent   = 0.9,
trk2dictionary.run(
    filename_tractogram = output_tractogram_cluster,
    filename_mask       = output_wmmask,
    filename_peaks      = output_peak,
    path_out            = '/local_raid/data/pbautin/results/COMMIT',
    peaks_use_affine    = True,
    fiber_shift    = 0.5,
    ndirs = 1,
)

# Load the diffusion data
out_scheme = '/local_raid/data/pbautin/results/COMMIT/dwi.scheme'
output_msdki = '/local_raid/data/pbautin/results/model_results/msdki/axonal_volume_fraction.nii.gz'
amico.util.fsl2scheme(output_bval, output_bvec, out_scheme)
mit = commit.Evaluation()
mit.set_config('doNormalizeSignal', False)
mit.set_config('doMergeB0', False)
mit.set_config('doNormalizeKernels', True)
mit.set_verbose(4)
# mit.load_data(output_dwi, out_scheme, b0_thr=61)
mit.load_data(output_msdki)



# # Set the forward model
# mit.set_model( 'StickZeppelinBall' )
# d_par       = 1.7E-3             # Parallel diffusivity [mm^2/s]
# d_perps_zep = [ 0.51E-3 ]        # Perpendicular diffusivity(s) [mm^2/s] (derek jones study put to 0.61E-3)
# d_isos      = [ 1.7E-3, 3.0E-3 ] # Isotropic diffusivity(s) [mm^2/s]
# mit.model.set( d_par, d_perps_zep, d_isos )
# mit.generate_kernels( regenerate=True )
# mit.load_kernels()

# Set model and generate the kernel
mit.set_model( 'VolumeFractions' )
mit.generate_kernels( regenerate=True )
mit.load_kernels()

# Load the sparse data-structure
mit.load_dictionary('/local_raid/data/pbautin/results/COMMIT')

# Build the linear operator A
mit.set_threads()
mit.build_operator()

# Fit the model to the data
mit.fit( tol_fun=1e-3, max_iter=1000 )

# Save results
mit.save_results(path_suffix="_COMMIT1")

# If you need the estimated coefficients for all the compartments
x_ic, x_ec, x_iso = mit.get_coeffs()
print( x_ic.size ) # this shows there are 283522 streamline coefficients (one for each)



# First, we retrieve the number of streamlines present in each bundle from the connectome previously calculated, i.e.
# C = np.loadtxt( connectome, delimiter=',' )
# C = np.triu( C ) # be sure to get only the upper-triangular part of the matrix
# group_size = C[C>0].astype(np.int32)
#
# # With this information, we create an array with the indices of the streamlines composing each bundle:
# tmp = np.insert( np.cumsum(group_size), 0, 0 )
# group_idx = np.fromiter( [np.arange(tmp[i],tmp[i+1]) for i in range(len(tmp)-1)], dtype=np.object_ )

# Now, we initialize a dictionary of additional parameters for the group lasso regularisation.
# In this way the individual weight for each bundle is computed internally to avoid the introduction of
# a penalization bias due to different bundles' size (see Eq. 6).
params_IC = {}
params_IC['group_idx'] = group_idx
params_IC['group_weights_cardinality'] = True
params_IC['group_weights_adaptive'] = True

perc_lambda = 0.00025 # change to suit your needs

# set the regularisation
mit.set_regularisation(
    regularisers   = ('group_lasso', None, None),
    is_nonnegative = (True, True, True),
    lambdas        = (perc_lambda, None, None),
    params         = (params_IC, None, None)
)

mit.fit( tol_fun=1e-3, max_iter=1000 )
mit.save_results( path_suffix="_COMMIT2" )




############# post-Processing
# tckedit -minweight 0.000000000001 -tck_weights_in /local_raid/data/pbautin/results/COMMIT/Results_VolumeFractions_COMMIT1/streamline_weights.txt \
# -tck_weights_out /local_raid/data/pbautin/results/COMMIT/Results_VolumeFractions_COMMIT1/streamline_weights_filtered.txt \
# /local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-3M_tractography.tck \
# /local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-3M_tractography_filtered.tck


# tckmap /local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-3M_tractography.tck \
# /local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-3M_tractography_tdi_filtered.nii.gz \
# -template /local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-preproc_peaks.nii.gz \
# -tck_weights_in /local_raid/data/pbautin/results/COMMIT/Results_VolumeFractions_COMMIT1/streamline_weights_filtered.txt


#  tck2connectome '/local_raid/data/pbautin/results/COMMIT/sub-HC007_ses-01_space-dwi_desc-iFOD2-1M_tractography_connecting_cluster.tck' \
#  '/local_raid/data/pbautin/results/micapipe/tmp/12732_micapipe_post-dwi_HC007/HC007_schaefer-100-full_dwi.nii.gz' \
#  '/local_raid/data/pbautin/results/COMMIT/HC007_schaefer-100-full_dwi_connectome.txt' \
#  -tck_weights_in '/local_raid/data/pbautin/results/COMMIT/Results_VolumeFractions_COMMIT2/streamline_weights.txt'
#
# import matplotlib.pyplot as plt
# import numpy as np
# conn = np.loadtxt('/local_raid/data/pbautin/results/COMMIT/HC007_schaefer-100-full_dwi_connectome.txt')
# plt.imshow(conn)
# plt.show()
