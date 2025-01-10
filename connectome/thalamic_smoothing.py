#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Connectome Visualization and Smoothing Pipeline

This script provides a pipeline for the following tasks:
1. Mapping high-resolution structural connectivities using the Connectome Spatial Smoothing (CSS) package.
2. Applying a smoothing kernel to the connectome.
3. Filtering the connectome based on a specific brain structure.
4. Saving the modified data into a new CIFTI file.
5. Visualizing the CIFTI data on a surface plot.

Usage:
Run this script in a terminal, supplying the required command-line arguments.
>>> python CCS_2.py <in_tractogram> <surf_lh> <surf_rh> <cifti_file> <out_file>
"""

import argparse
import numpy as np
import scipy.sparse as sparse
import nibabel as nib
import matplotlib.pyplot as plt
from Connectome_Spatial_Smoothing import CSS as css
from dipy.io.stateful_tractogram import (Origin, Space,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram, load_tractogram, load_tck
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels

from nilearn import maskers, plotting, image


#from neuromaps import nulls
#from neuromaps import stats as stats_neuromaps
#from scipy import stats
#from neuromaps.images import dlabel_to_gifti
#from neuromaps.datasets import fetch_fslr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    p.add_argument("in_tractogram", help="Path of the input tractograms (.trk).")
    p.add_argument("surf_lh", help="Path to left hemisphere surface")
    p.add_argument("surf_rh", help="Path to right hemisphere surface")
    p.add_argument("cifti_file", help="Path to cifti file")
    #p.add_argument("out_file", help="Output file path (.trk).")
    #p.add_argument("--weight_file", help="weight file path (.txt).", default=None)
    #p.add_argument("--dps_name", help="Name of data per streamline to load.", default=None)
    return p


def plot_surface(cifti_file):
    """
    Plot a surface representation of a CIFTI file.

    Parameters
    ----------
    cifti_file : str or nib.Cifti2Image
        The CIFTI file or image object to plot.
    """
    surfaces = fetch_fslr()
    plt.style.use('dark_background')
    print(surfaces)
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial', 'ventral'], zoom=1.2)
    p.add_layer(cifti_file, cbar=True, cmap='autumn')
    fig = p.build()
    plt.show(block=True)


def load_streamline_dps(sft, dps_name):
    """
    load streamline data

    Parameters
    ----------
    sft : statefull tractogram
        The tractogram with dps data
    """
    new_data_per_streamline = [s for s in sft.data_per_streamline[dps_name]]
    return new_data_per_streamline


def save_cifti_file(streamline_incidence, cifti_file, cifti_output):
    """
    Save streamline incidence data into a new CIFTI file.

    Parameters
    ----------
    streamline_incidence : ndarray
        The streamline incidence data.
    cifti_file : str
        Path to the original CIFTI file.
    cifti_output : str
        Path to save the new CIFTI file.

    Returns
    -------
    new_img : nib.Cifti2Image
        The new CIFTI image object.
    """
    # Load original CIFTI file once to improve efficiency
    original_cifti = nib.load(cifti_file)
    original_data = original_cifti.get_fdata()

    # Apply the streamline incidence to the original data
    new_data = streamline_incidence[np.newaxis, ...] * np.ones_like(original_data)

    # Create a new CIFTI image object with the updated data
    new_img = nib.Cifti2Image(new_data, header=original_cifti.header, nifti_header=original_cifti.nifti_header)

    # Save the new CIFTI image to disk
    new_img.to_filename(cifti_output)

    return new_img


def filter_connectome(conn, cifti_file, structure='CIFTI_STRUCTURE_BRAIN_STEM'):
    """
    Filter the connectome matrix based on a specific brain structure.

    Parameters
    ----------
    conn : scipy.sparse.csr_matrix or np.ndarray
        The connectome matrix.
    cifti_file : str
        Path to the CIFTI file.
    structure : str, optional
        The brain structure to filter by. Default is 'CIFTI_STRUCTURE_BRAIN_STEM'.

    Returns
    -------
    filtered_conn : scipy.sparse.csr_matrix or np.ndarray
        The filtered connectome matrix.
    """
    import nibabel as nib
    import numpy as np

    # Load the CIFTI file and get the BrainModelAxis
    img_cifti = nib.load(cifti_file)
    bm_axis = img_cifti.header.get_axis(1)  # BrainModelAxis

    # # Check if the desired structure exists
    # if structure not in bm_axis.structures:
    #     raise KeyError(f"Structure '{structure}' not found in CIFTI file.")
    mask = np.zeros(len(img_cifti.get_fdata().ravel()), dtype=bool)
    for name, data_indices, model in bm_axis.iter_structures():  # Iterates over volumetric and surface structures
        if name in structure:
            mask[data_indices] = True

    # Filter the connectome matrix
    filtered_conn = conn[:, mask]

    return filtered_conn





def streamlines_from_hdf5(in_tractogram, output_sft):
    import h5py
    hdf5_file = h5py.File(in_tractogram, 'r')

    # Keep track of the order of connections/streamlines in relation to the
    # tractogram as well as the number of streamlines for each connection.
    bundle_groups_len = []
    hdf5_keys = list(hdf5_file.keys())
    streamlines = []
    for key in hdf5_keys:
        tmp_streamlines = reconstruct_streamlines_from_hdf5(hdf5_file, key)
        streamlines.extend(tmp_streamlines)
        bundle_groups_len.append(len(tmp_streamlines))

    offsets_list = np.cumsum([0]+bundle_groups_len)
    ref = '/home/pabaua/dev_tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-pl007_ses-v1/Register_T1/sub-pl007_ses-v1__t1_warped.nii.gz'
    sft = StatefulTractogram(streamlines, ref,Space.VOX, origin=Origin.TRACKVIS)
    save_tractogram(sft, output_sft)
    return sft





def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")




def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    # Build argument parser and parse command-line arguments and verify that all input files exist
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the input files.
    #print("Loading file {}".format(args.in_tractogram))
    #sft = load_tck(args.in_tractogram, reference='/local_raid/data/pbautin/data/MNI152Volumes/MNI152_T1_0.8mm.nii.gz', bbox_valid_check=False)

    # if args.dps_name:
    #     args.weights = load_streamline_dps(sft, args.dps_name)
    # else:
    #     args.weights = None

    # Map high-resolution structural connectivity
    #output_sft = '/home/pabaua/dev_hiball/css_test/sft.trk'
    #sft_map = streamlines_from_hdf5('/home/pabaua/Desktop/sub-pl007_ses-v1__decompose.h5', output_sft)
    #args.in_tractogram = output_sft
    import itertools
    cifti = nib.load(args.cifti_file)
    axes = [cifti.header.get_axis(i) for i in range(cifti.ndim)]
    _glasser_cifti = '/local_raid/data/pbautin/downloads/Glasser360.32k_fs_LR.dlabel.nii'
    cifti_lh = surf_data_from_cifti(nib.load(_glasser_cifti).get_fdata(dtype=np.float32), axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT').ravel()
    cifti_rh = surf_data_from_cifti(nib.load(_glasser_cifti).get_fdata(dtype=np.float32), axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT').ravel()
    labeling = np.pad(np.concatenate((cifti_lh, cifti_rh), axis=0), (0,91282-64984), mode='constant', constant_values=0)

    fsLR_lh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    fsLR_rh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    plot_hemispheres(fsLR_lh, fsLR_rh, array_name=labeling[:64984], size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
                             nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid')

    img = cifti
    brain_models = [x for x in img.header.get_index_map(1).brain_models]
    names = np.array([[x.brain_structure, x.index_offset, x.index_count] for x in brain_models[:]])
    print(names)
    idx_0 = int(names[names[..., 0] == 'CIFTI_STRUCTURE_THALAMUS_LEFT'][0][1]) - int(names[names[..., 0] == 'CIFTI_STRUCTURE_ACCUMBENS_LEFT'][0][1])
    idx_f = int(names[names[..., 0] == 'CIFTI_STRUCTURE_THALAMUS_LEFT'][0][2]) +idx_0

    # subcortical regions
    subijk = np.array(list(itertools.chain.from_iterable([(x.voxel_indices_ijk) for x in brain_models[2:]])))
    subxyz = nib.affines.apply_affine(img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix, subijk)

    plt.style.use('dark_background')

    # Generate a colormap range for the subset (this is an example, adjust based on your data)
    subset_values = np.linspace(0, 1, len(subxyz[idx_0:idx_f]))  # Adjust if actual values exist
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # Optimized figure size

    # Left plot: Scatter in X-Y plane
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        subxyz[idx_0:idx_f][:, 0],
        subxyz[idx_0:idx_f][:, 1],
        c=subset_values,
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax1.scatter(
        subxyz[:, 0],
        subxyz[:, 1],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax1.axis('equal')
    ax1.set_title('X-Y Plane')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    fig.colorbar(scatter1, ax=ax1, orientation='vertical', label='Subset Intensity')

    # Right plot: Scatter in Y-Z plane
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        subxyz[idx_0:idx_f][:, 1],
        subxyz[idx_0:idx_f][:, 2],
        c=subset_values,
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax2.scatter(
        subxyz[:, 1],
        subxyz[:, 2],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax2.axis('equal')
    ax2.set_title('Y-Z Plane')
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('Z Coordinate')
    fig.colorbar(scatter2, ax=ax2, orientation='vertical', label='Subset Intensity')

    plt.tight_layout()
    plt.show()

    connectome = css.map_high_resolution_structural_connectivity(args.in_tractogram, args.surf_lh, args.surf_rh, threshold=5, cifti_file=args.cifti_file, subcortex=True)#, atlas_file=_glasser_cifti)#, weights=args.weights)

    # # Compute and apply the smoothing kernel to the connectome
    # smoothing_kernel = css.compute_smoothing_kernel(
    #     args.surf_lh, args.surf_rh, fwhm=4, epsilon=0.1, cifti_file=args.cifti_file, subcortex=True)
    # connectome = css.smooth_high_resolution_connectome(connectome, smoothing_kernel)

    # Save and load smoothed connectome to disk
    # sparse.save_npz('/home/pabaua/dev_hiball/css_test/smoothed_high_resolution_connectome.npz', connectome)
    # print("Loading connectome")
    # connectome = sparse.load_npz('/home/pabaua/dev_hiball/css_test/smoothed_high_resolution_connectome.npz')
    # print("Loaded connectome")

    mask = labeling[:64984] != 0
    connectome = filter_connectome(connectome, args.cifti_file, structure=['CIFTI_STRUCTURE_THALAMUS_LEFT']).toarray().T
    print(connectome.shape)
    conn_labeled = np.log(reduce_by_labels(connectome, labeling, red_op='sum')+1)
    corr_plot = plotting.plot_matrix(conn_labeled.T, figure=(10, 10), labels=None, cmap='Purples', vmax=1) # , vmin=-10, vmax=2
    plotting.show()
    #connectome = connectome.toarray()
    gm = GradientMaps(n_components=2, random_state=None, approach='dm', kernel='normalized_angle')
    gm.fit(conn_labeled)
    print(gm.gradients_[:, 0])
    print(gm.gradients_[:, 0].shape)

    # subcortical regions
    subijk = np.array(list(itertools.chain.from_iterable([(x.voxel_indices_ijk) for x in brain_models[2:]])))
    subxyz = nib.affines.apply_affine(img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix, subijk)

    plt.style.use('dark_background')
    # Generate a colormap range for the subset (this is an example, adjust based on your data)
    subset_values = np.linspace(0, 1, len(subxyz[idx_0:idx_f]))  # Adjust if actual values exist
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # Optimized figure size

    # Left plot: Scatter in X-Y plane
    ax1 = axes[0,0]
    scatter1 = ax1.scatter(
        subxyz[idx_0:idx_f][:, 0],
        subxyz[idx_0:idx_f][:, 1],
        c=gm.gradients_[:, 0],
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax1.scatter(
        subxyz[:, 0],
        subxyz[:, 1],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax1.axis('equal')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    #fig.colorbar(scatter1, ax=ax1, orientation='vertical', label='Subset Intensity')

    # Right plot: Scatter in Y-Z plane
    ax2 = axes[0,1]
    scatter2 = ax2.scatter(
        subxyz[idx_0:idx_f][:, 1],
        subxyz[idx_0:idx_f][:, 2],
        c=gm.gradients_[:, 0],
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax2.scatter(
        subxyz[:, 1],
        subxyz[:, 2],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax2.axis('equal')
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('Z Coordinate')
    fig.colorbar(scatter2, ax=ax2, orientation='vertical')

    # Left plot: Scatter in X-Y plane
    ax3 = axes[1,0]
    scatter3 = ax3.scatter(
        subxyz[idx_0:idx_f][:, 0],
        subxyz[idx_0:idx_f][:, 1],
        c=gm.gradients_[:, 1],
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax3.scatter(
        subxyz[:, 0],
        subxyz[:, 1],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax3.axis('equal')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    #fig.colorbar(scatter1, ax=ax1, orientation='vertical', label='Subset Intensity')

    # Right plot: Scatter in Y-Z plane
    ax4 = axes[1,1]
    scatter4 = ax4.scatter(
        subxyz[idx_0:idx_f][:, 1],
        subxyz[idx_0:idx_f][:, 2],
        c=gm.gradients_[:, 1],
        cmap='RdYlBu',
        s=20,
        alpha=0.9,
        edgecolors='none'
    )
    ax4.scatter(
        subxyz[:, 1],
        subxyz[:, 2],
        c='gray',
        alpha=0.5,
        s=10,
        edgecolors='none'
    )
    ax4.axis('equal')
    ax4.set_xlabel('Y Coordinate')
    ax4.set_ylabel('Z Coordinate')
    fig.colorbar(scatter4, ax=ax4, orientation='vertical')

    plt.tight_layout()
    plt.show()

    # print(gm.gradients_.shape)
    grad = [None] * 2
    for i in range(2):
        # map the gradient to the parcels
        grad[i] = map_to_labels(gm.gradients_[:, i], labeling[:64984], mask=mask, fill=np.nan)
    plot_hemispheres(fsLR_lh, fsLR_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.55)
    # connectome.data[np.isnan(connectome.data)] = 0.0
    # connectome.data[connectome.data < 0.01] = 0.0

    fsLR_lh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    fsLR_rh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    plot_hemispheres(fsLR_lh, fsLR_rh, array_name=gradients[0], size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
                             nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid')


    # Compute streamline incidence, normalize and save to a new CIFTI file
    # streamline_incidence = np.log1p(np.array(connectome.sum(axis=1))[..., 0]) / np.max(np.log1p(np.array(connectome.sum(axis=1))[..., 0]))
    cifti = nib.Cifti2Image(np.array(np.log1p(np.array(connectome.sum(axis=1))[..., 0])), header=cifti.header, nifti_header=cifti.nifti_header)
    cifti_lh = surf_data_from_cifti(cifti.get_fdata(dtype=np.float32), axes[1], 'CIFTI_STRUCTURE_CORTEX_LEFT').ravel()
    cifti_rh = surf_data_from_cifti(cifti.get_fdata(dtype=np.float32), axes[1], 'CIFTI_STRUCTURE_CORTEX_RIGHT').ravel()
    streamline_incidence = np.concatenate((cifti_lh, cifti_rh), axis=0)


    fsLR_lh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    fsLR_rh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    plot_hemispheres(fsLR_lh, fsLR_rh, array_name=streamline_incidence, size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
                             nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid')
    #print(streamline_incidence.shape)
    #cifti_si = save_cifti_file(streamline_incidence, args.cifti_file, args.out_file)
    # cifti_si = nib.load(args.out_file)
    # Plot surface using the new CIFTI data
    # plot_surface(cifti_file=cifti_si)


    # curvature = nib.load('/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Fs_ciftify/sub-pl007_ses-v1/MNINonLinear/fsaverage_LR32k/sub-pl007_ses-v1.curvature.32k_fs_LR_resampled.dscalar.nii').get_fdata()
    # print(np.min(np.abs(curvature)))
    # print(np.max(np.abs(curvature)))
    # sulc = nib.load('/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Fs_ciftify/sub-pl007_ses-v1/MNINonLinear/fsaverage_LR32k/sub-pl007_ses-v1.curvature.32k_fs_LR_resampled_abs.gii')
    # cifti_si = nib.load('/home/pabaua/dev_tpil/results/results_map_projection/test_css.dscalar.gii')
    # print(np.min(cifti_si.agg_data()))
    # print(np.max(cifti_si.agg_data()))
    # nulls_t = nulls.alexander_bloch(cifti_si, atlas='fsLR', density='32k', n_perm=500, seed=1234)
    # corr, p = stats_neuromaps.compare_images(cifti_si, sulc, nulls=nulls_t)
    # print(corr)
    # print(p)
if __name__ == "__main__":
    main()

#/local_raid/data/pbautin/software/neuroimaging_scripts/connectome/thalamic_smoothing.py /local_raid/data/pbautin/data/dTOR_full_tractogram_full.tck  /local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-white.surf.gii /local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/surf/sub-mni_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-white.surf.gii /local_raid/data/pbautin/downloads/ones.dscalar.nii
