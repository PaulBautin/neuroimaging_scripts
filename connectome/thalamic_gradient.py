
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Network control theory control vs. chronic pain patients
#
# example: python nct_main.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from nctpy.energies import get_control_inputs, integrate_u, minimum_energy_fast
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
    expand_states
)
import time
#
from nilearn import maskers, plotting, image
import seaborn as sns
import scipy as sp
# import seaborn as sns
# import matplotlib.cm as cm
# import matplotlib as mpl
# import plotly.express as px
#
#
# from functions.tpil_stats_freesurfer import cortical_measures, cortical_measures_diff, cortical_measures_z_score
# from functions.tpil_sc_load import load_connectivity, find_files_with_common_name
# from functions.tpil_meta_analysis import fetch_neurosynth_data, get_studies_by_terms, apply_meta_analysis, apply_atlas_meta
#
#
# #TODO: add cognitive topographies from neurosynth or brainmap
# #TODO: test different meta-analysis techniques
# #TODO: test different time horizons
# #TODO: look up the activation values (could be binary)

#from nilearn.image import reorder_img, check_niimg_3d, get_data
from nibabel.affines import apply_affine
from scipy.ndimage import center_of_mass, label

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.gradient import GradientMaps


def find_parcellation_cut_coords(
    labels_img,
    background_label=0,
    return_label_names=False,
):
    """Return coordinates of center of mass of 3D parcellation atlas.

    Parameters
    ----------
    labels_img : 3D Nifti1Image
        A brain parcellation atlas with specific mask labels for each
        parcellated region.

    background_label : int, default=0
        Label value used in labels_img to represent background.

    return_label_names : bool, default=False
        Returns list of labels.

    label_hemisphere : 'left' or 'right', default='left'
        Choice of hemisphere to compute label center coords for.
        Applies only in cases where atlas labels are lateralized.

    Returns
    -------
    coords : numpy.ndarray of shape (n_labels, 3)
        Label regions cut coordinates in image space (mm).

    labels_list : list, optional
        Label regions. Returned only when return_label_names is True.

    """
    # Prepare image data
    #labels_img = reorder_img(check_niimg_3d(labels_img), copy_header=True)
    labels_data = labels_img.get_fdata()
    labels_affine = labels_img.affine

    # Get unique labels excluding the background
    unique_labels = np.unique(labels_data)
    unique_labels = unique_labels[unique_labels != background_label]

    # Compute centers of mass for all labels in the selected hemisphere
    centers = center_of_mass(labels_data, labels=labels_data, index=unique_labels)
    centers = np.array(centers)


    # Filter out labels with no voxels in the selected hemisphere
    valid = ~np.isnan(centers[:, 0])
    valid_labels = unique_labels[valid]
    valid_centers = centers[valid]

    # Transform voxel coordinates to world coordinates
    world_coords = apply_affine(labels_affine, valid_centers)

    if return_label_names:
        return world_coords, valid_labels
    else:
        return world_coords
#
#
# def plot_points(values, atlas, lut, metric='z_score'):
#     df_z_score = pd.concat([values['lh'].stack(), values['rh'].stack()]).to_frame().reset_index().rename(columns={'level_1':'regions',0:metric}).set_index('regions')
#     coords, labels_list = plotting.find_parcellation_cut_coords(atlas, return_label_names=True)
#     print(labels_list)
#     df_coords = pd.DataFrame({'coords':coords.tolist(), 'labels_list':labels_list}).set_index('labels_list')
#     df_coords = df_coords.join(lut.set_index('label')).reset_index()
#     df_coords['region_fs'] = df_coords['region'].str.split('/').str[-1]
#     print(df_coords.set_index('region_fs'))
#     df = df_coords.set_index('region_fs').join(df_z_score).dropna(axis=0)
#     print(df)
#     # plotting.plot_markers(values['v1'], coords, title="Volume z-score per node (CLBP - control) v1", node_vmin=-2, node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
#     # plt.show()
#     # plotting.plot_markers(values['v2'], coords, title="Volume z-score per node (CLBP - control) v2", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
#     # plt.show()
#     # plotting.plot_markers(values['v3'], coords, title="Volume z-score per node (CLBP - control) v3", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
#     # plt.show()
#
#
# def plot_heatmap(df_con, df_clbp, ses='v1'):
#     df_con = df_con.groupby('session').get_group(ses).drop(['session','subject'], axis=1)
#     df_clbp = df_clbp.groupby('session').get_group(ses).drop(['session','subject'], axis=1)
#     # figure
#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 3), sharex=True, sharey=True)
#     print(df_con.groupby('x0').mean())
#     print(np.max(df_con.groupby('x0').mean().values))
#     df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
#     print(df_g1)
#     g1 = sns.heatmap(df_g1,annot=True, ax=ax[0], cmap='Blues', square=True, vmin=0, vmax=1)
#     g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=12)
#     g1.set_xticklabels(g1.get_xticklabels(), rotation=45, fontsize=12, ha='right')
#     ax[0].set(ylabel='', xlabel='', title='transition energy for controls')
#     df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
#     g2 = sns.heatmap(df_g2, annot=True, ax=ax[1], cmap='Blues', square=True, vmin=0, vmax=1)
#     #g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=12)
#     g2.set_xticklabels(g2.get_xticklabels(), rotation=45, fontsize=12, ha='right')
#     ax[1].set(ylabel='', xlabel='', title='transition energy for CLBPs')
#     df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
#     g3 = sns.heatmap(df_zscore, annot=True, ax=ax[2], vmin=-2, vmax=2, cmap='RdBu', square=True)
#     #g3.set_yticklabels(g3.get_yticklabels(), rotation=0, fontsize=12)
#     g3.set_xticklabels(g3.get_xticklabels(), rotation=45, fontsize=12, ha='right')
#     ax[2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control)')
#     plt.show()
#
#
# def plot_heatmap_v1v2v3(df_con_all, df_clbp_all):
#     df_con = df_con_all.groupby('session').get_group('v1').drop(['session','subject'], axis=1)
#     df_clbp = df_clbp_all.groupby('session').get_group('v1').drop(['session','subject'], axis=1)
#     # figure
#     fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(28, 21), sharex=True, sharey=True)
#     df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
#     print(df_g1)
#     g11 = sns.heatmap(df_g1,annot=True, ax=ax[0,0], cmap='Blues', vmin=0, vmax=1)
#     g11.set_yticklabels(g11.get_yticklabels(), rotation=0, fontsize=12)
#     g11.set_xticklabels(g11.get_xticklabels(), rotation=45, fontsize=12)
#     ax[0,0].set(ylabel='', xlabel='', title='transition energy for controls v1')
#     df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
#     g12 = sns.heatmap(df_g2, annot=True, ax=ax[0,1], cmap='Blues', vmin=0, vmax=1)
#     g12.set_yticklabels(g12.get_yticklabels(), rotation=0, fontsize=12)
#     g12.set_xticklabels(g12.get_xticklabels(), rotation=45, fontsize=12)
#     ax[0,1].set(ylabel='', xlabel='', title='transition energy for CLBPs v1')
#     df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
#     g13 = sns.heatmap(df_zscore, annot=True, ax=ax[0,2], vmin=-2, vmax=2, cmap='RdBu')
#     g13.set_yticklabels(g13.get_yticklabels(), rotation=0, fontsize=12)
#     g13.set_xticklabels(g13.get_xticklabels(), rotation=45, fontsize=12)
#     ax[0,2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v1')
#
#     # figure
#     df_con = df_con_all.groupby('session').get_group('v2').drop(['session','subject'], axis=1)
#     df_clbp = df_clbp_all.groupby('session').get_group('v2').drop(['session','subject'], axis=1)
#     df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
#     print(df_g1)
#     g21 = sns.heatmap(df_g1,annot=True, ax=ax[1,0], cmap='Blues', vmin=0, vmax=1)
#     g21.set_yticklabels(g21.get_yticklabels(), rotation=0, fontsize=12)
#     g21.set_xticklabels(g21.get_xticklabels(), rotation=45, fontsize=12)
#     ax[1,0].set(ylabel='', xlabel='', title='transition energy for controls v2')
#     df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
#     g22 = sns.heatmap(df_g2, annot=True, ax=ax[1,1], cmap='Blues', vmin=0, vmax=1)
#     #g22.set_yticklabels(g22.get_yticklabels(), rotation=0, fontsize=12)
#     g22.set_xticklabels(g22.get_xticklabels(), rotation=45, fontsize=12)
#     ax[1,1].set(ylabel='', xlabel='', title='transition energy for CLBPs v2')
#     df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
#     g23 = sns.heatmap(df_zscore, annot=True, ax=ax[1,2], vmin=-2, vmax=2, cmap='RdBu')
#     #g23.set_yticklabels(g23.get_yticklabels(), rotation=0, fontsize=12)
#     g23.set_xticklabels(g23.get_xticklabels(), rotation=45, fontsize=12)
#     ax[1,2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v2')
#
#     # figure
#     df_con = df_con_all.groupby('session').get_group('v3').drop(['session','subject'], axis=1)
#     df_clbp = df_clbp_all.groupby('session').get_group('v3').drop(['session','subject'], axis=1)
#     df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
#     print(df_g1)
#     g31 = sns.heatmap(df_g1,annot=True, ax=ax[2, 0], cmap='Blues', vmin=0, vmax=1)
#     g31.set_yticklabels(g31.get_yticklabels(), rotation=0, fontsize=12)
#     g31.set_xticklabels(g31.get_xticklabels(), rotation=30, fontsize=12, ha='right')
#     ax[2, 0].set(ylabel='', xlabel='', title='transition energy for controls v3')
#     df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
#     g32 = sns.heatmap(df_g2, annot=True, ax=ax[2, 1], cmap='Blues', vmin=0, vmax=1)
#     #g32.set_yticklabels(g32.get_yticklabels(), rotation=0, fontsize=12)
#     g32.set_xticklabels(g32.get_xticklabels(), rotation=30, fontsize=12, ha='right')
#     ax[2, 1].set(ylabel='', xlabel='', title='transition energy for CLBPs v3')
#     df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
#     g33 = sns.heatmap(df_zscore, annot=True, ax=ax[2, 2], vmin=-2, vmax=2, cmap='RdBu')
#     #g33.set_yticklabels(g33.get_yticklabels(), rotation=0, fontsize=12)
#     g33.set_xticklabels(g33.get_xticklabels(), rotation=30, fontsize=12, ha='right')
#     ax[2, 2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v3')
#     plt.show()
#
#
# def plot_average_control(df_avg_con, df_avg_clbp, atlas, ses='v1'):
#     coords, labels_list = plotting.find_parcellation_cut_coords(atlas, return_label_names=True)
#     print(np.array(labels_list).shape)
#     df_avg_con = df_avg_con[df_avg_con['session'] == ses]
#     df_avg_clbp = df_avg_clbp[df_avg_clbp['session'] == ses]
#     print(df_avg_clbp)
#     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True, sharey=True)
#     plotting.plot_markers(df_avg_con.drop('subject', axis=1).groupby('session').mean(), coords, title="average control per node control", node_cmap='RdYlBu', alpha=0.9, axes=ax[0])
#     plotting.plot_markers(df_avg_clbp.drop('subject', axis=1).groupby('session').mean(), coords, title="average control per node clbp", node_cmap='RdYlBu', alpha=0.9, axes=ax[1])
#     df_zscore = (df_avg_clbp.drop('subject', axis=1).groupby('session').mean() - df_avg_con.drop('subject', axis=1).groupby('session').mean()) / df_avg_con.drop('subject', axis=1).groupby('session').std()
#     plotting.plot_markers(df_zscore, coords, title="average control per node zscore", node_cmap='RdYlBu', alpha=0.9, axes=ax[2])
#     plt.show()
#
#
# def normalize_connectomes(df_A):
#     system = 'continuous'  # option 'discrete'
#     df_A = df_A.drop('roi', axis=1)
#     #df_A_norm = df_A.groupby(['session', 'subject']).apply(lambda x: matrix_normalization(A=x.drop(['session','subject'], axis=1).iloc[-200:,-200:], c=1, system=system))
#     df_A_norm = df_A.groupby(['session', 'subject']).apply(lambda x: matrix_normalization(A=x.drop(['session','subject'], axis=1), c=1, system=system))
#     return df_A_norm.reset_index().drop(labels='level_2', axis=1)
#
#
# def get_yeo_parcellation(atlas_img):
#     def label_extractor(img_yeo, data_yeo, i):
#         data_yeo_copy = data_yeo.copy()
#         data_yeo_copy[data_yeo_copy != i] = 0
#         data_yeo_copy[data_yeo_copy == i] = 1
#         img_yeo_1 = nib.Nifti1Image(data_yeo_copy, img_yeo.affine, img_yeo.header)
#         return img_yeo_1
#
#
#     # strategy is the name of a valid function to reduce the region with
#     roi = maskers.NiftiLabelsMasker(atlas_img, strategy='mean', background_label=0)
#     networks = {1:'visual', 2:'somatomotor', 3:'dorsal attention', 4:'ventral attention', 5:'limbic', 6:'frontoparietal', 7:'default'}
#     yeo = datasets.fetch_atlas_yeo_2011()
#     img_yeo = nib.load(yeo.thick_7)
#     data_yeo = img_yeo.get_fdata()
#     img_dict = {i: label_extractor(img_yeo, data_yeo, i) for i in np.delete(np.unique(data_yeo), 0)}
#     dict_signal = {networks[k]: roi.fit_transform(v) for k, v in img_dict.items()}
#     img_signal = {k: roi.inverse_transform(v) for k, v in dict_signal.items()}
#     plotting.plot_stat_map(img_signal['limbic'], title='limbic', display_mode='x', cut_coords=[-40, -20, 0, 20, 40], colorbar=True)
#     plt.show()
#     return dict_signal
#
#
#
# def get_state_traj(df_A_norm, dict_states, out_dir, fname):
#     # set parameters
#     system = 'continuous'  # option 'discrete'
#     T = 3  # time horizon
#     rho = 1  # mixing parameter for state trajectory constraint
#     S = np.eye(215)  # nodes in state trajectory to be constrained
#     B = np.eye(215)  # which nodes receive input, uniform full control set
#     if os.path.isfile(os.path.join(out_dir, fname)):
#         df = pd.read_pickle(os.path.join(out_dir, fname))
#     else:
#         # df_A_norm = df_A_norm.groupby(['session']).get_group(('v1'))
#         # df_A_norm = df_A_norm.groupby(['session','subject']).get_group(('v1','sub-pl002'))
#         df = df_A_norm.groupby(['session', 'subject']).apply(lambda x: pd.Series({(x_f,x_i): np.sum(integrate_u(get_control_inputs(A_norm=x.drop(['session','subject', 'B'], axis=1), T=T, B=x['B'].mean(), x0=dict_states[x_i], xf=dict_states[x_f], system=system, rho=rho, S=S)[1])) for x_i in dict_states.keys() for x_f in dict_states.keys()}))
#         df = df.stack().reset_index().rename(columns={'level_2':'x0'})
#         df.to_pickle(os.path.join(out_dir, fname))
#     return df
#
#
# def get_states_from_meta_analysis(schaefer_atlas, out_dir):
#     # get states from meta-analysis
#     term_list = ["pain", "reward", "stress", "anxiety", "depression", "learning", "memory"]
#     neurosynth_dset = fetch_neurosynth_data(out_dir)
#     neurosynth_dset_by_term = get_studies_by_terms(neurosynth_dset, term_list)
#     meta_dict = apply_meta_analysis(neurosynth_dset_by_term, out_dir)
#     dict_states = apply_atlas_meta(meta_dict, schaefer_atlas)
#     return dict_states
#
#
# def get_states_from_yeo_atlas(out_dir):
#     # load node-to-system mapping
#     data_dir = '/home/pabaua/dev_tpil/data/Schaefer'
#     system_labels = np.array(np.loadtxt(os.path.join(data_dir, "pnc_schaefer200_system_labels.txt"), dtype=str))
#     # use list of system names to create states
#     states, state_labels = convert_states_str2int(system_labels)
#     state_dict = {k: np.append(np.zeros(15), normalize_state((system_labels == k).astype(int))) for k in state_labels}
#     return state_dict
#
#
# def apply_atlas(meta_dict, atlas):
#     """
#     Apply Nilearn NiftiLabelsMasker to the dictionnary of NiMARE results.MetaResults
#
#     Parameters
#     ----------
#     meta_dict : dict
#         Dictionary that contains terms of interest as keys and state map for each term as values
#     atlas : nibabel.nifti1.Nifti1Image
#         label atlas to use for the NiftiLabelsMasker
#
#     Returns
#     -------
#     state_dict : dict
#         Where keys are terms of interest and values are the corresponding normalized states numpy array (N_atlas_regions,)
#     """
#     # strategy is the name of a valid function to reduce the region with
#     roi = maskers.NiftiLabelsMasker(atlas, strategy='mean')
#     state_dict = {k: normalize_state(roi.fit_transform(v)).T[...,0] for k, v in meta_dict.items()}
#     img_signal = {k: roi.inverse_transform(v) for k, v in state_dict.items()}
#     plotting.plot_stat_map(img_signal['cPDM'], title='cPDM', display_mode='ortho', colorbar=True)
#     plt.show()
#     return state_dict
#
#
# def compute_avg_control(df_A_norm ,labels_list, out_dir, fname):
#     """
#     Apply Nilearn NiftiLabelsMasker to the dictionnary of NiMARE results.MetaResults
#
#     Parameters
#     ----------
#     meta_dict : dict
#         Dictionary that contains terms of interest as keys and state map for each term as values
#     atlas : nibabel.nifti1.Nifti1Image
#         label atlas to use for the NiftiLabelsMasker
#
#     Returns
#     -------
#     state_dict : dict
#         Where keys are terms of interest and values are the corresponding normalized states numpy array (N_atlas_regions,)
#     """
#     if os.path.isfile(os.path.join(out_dir, fname)):
#         df_avg_c = pd.read_pickle(os.path.join(out_dir, fname))
#     else:
#         df_avg_c = df_A_norm.groupby(['session', 'subject']).apply(lambda x: pd.Series(ave_control(A_norm=x.drop(['session','subject', 'B'], axis=1), system='continuous'), index=labels_list)).reset_index()
#         df_avg_c.to_pickle(os.path.join(out_dir, fname))
#     return df_avg_c
def get_p_val_string(p_val):
    if p_val == 0.0:
        p_str = "-log10($\mathit{:}$)>25".format('{p}')
    elif p_val < 0.05:
        p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
    # elif p_val < 0.001:
    #     p_str = '$\mathit{:}$ < 0.001'.format('{p}')
    # elif p_val >= 0.001 and p_val < 0.05:
    #     p_str = '$\mathit{:}$ < 0.05'.format('{p}')
    else:
        p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

    return p_str


def my_reg_plot(x, y, xlabel, ylabel, ax, c='gray', annotate='pearson', regr_line=True, kde=True, fontsize=20):
    if len(x.shape) > 1 and len(y.shape) > 1:
        if x.shape[0] == x.shape[1] and y.shape[0] == y.shape[1]:
            mask_x = ~np.eye(x.shape[0], dtype=bool) * ~np.isnan(x)
            mask_y = ~np.eye(y.shape[0], dtype=bool) * ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
        else:
            mask_x = ~np.isnan(x)
            mask_y = ~np.isnan(y)
            mask = mask_x * mask_y
            indices = np.where(mask)
    elif len(x.shape) == 1 and len(y.shape) == 1:
        mask_x = ~np.isnan(x)
        mask_y = ~np.isnan(y)
        mask = mask_x * mask_y
        indices = np.where(mask)
    else:
        print('error: input array dimension mismatch.')

    try:
        x = x[indices]
        y = y[indices]
    except:
        pass

    try:
        c = c[indices]
    except:
        pass

    # kde plot
    if kde == True:
        try:
            sns.kdeplot(x=x, y=y, ax=ax, color='gray', thresh=0.05, alpha=0.5)
        except:
            pass

    # regression line
    if regr_line == True:
        color_blue = sns.color_palette("Set1")[1]
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color_blue)

    # scatter plot
    if type(c) == str:
        ax.scatter(x=x, y=y, c=c, s=10, alpha=0.8)
    else:
        ax.scatter(x=x, y=y, c=c, cmap='viridis', s=5, alpha=0.8)

    # axis options
    ax.set_xlabel(xlabel, labelpad=-0.5)
    ax.set_ylabel(ylabel, labelpad=-0.5)
    ax.tick_params(pad=-2.5)
    ax.grid(False)
    sns.despine(right=True, top=True, ax=ax)

    # annotation
    r, r_p = sp.stats.pearsonr(x, y)
    rho, rho_p = sp.stats.spearmanr(x, y)
    if type(annotate) == str:
        if annotate == 'pearson':
            textstr = '$\mathit{:}$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'spearman':
            textstr = '$\\rho$ = {:.2f}, {:}'.format(rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
        elif annotate == 'both':
            textstr = '$\mathit{:}$ = {:.2f}, {:}\n$\\rho$ = {:.2f}, {:}'.format('{r}', r, get_p_val_string(r_p),
                                                                                 rho, get_p_val_string(rho_p))
            ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize,
                    verticalalignment='top')
    elif type(annotate) == tuple:
        coef = annotate[0]
        p = annotate[1]
        textstr = 'coef = {:.2f}, {:}'.format(coef, get_p_val_string(p))
        ax.text(0.05, 0.975, textstr, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    else:
        pass

def convert_states_str2int(states_str):
    """This function takes a list of strings that designate a distinct set of binary brain states and returns
    a numpy array of integers encoding those states alongside a list of keys for those integers.

    Args:
        states_str (N, list): a list of strings that designate which regions belong to which states.
            For example, states = ['Vis', 'Vis', 'Vis', 'SomMot', 'SomMot', 'SomMot']

    Returns:
        states (N, numpy array): array of integers denoting which node belongs to which state.
        state_labels (n_states, list): list of keys corresponding to integers.
            For example, if state_labels[1] = 'SomMot' then the integer 1 in `states` corresponds to 'SomMot'.
            Together, a binary state can be extracted like so: x0 = states == state_labels.index('SomMot')

    """
    n_states = len(states_str)
    state_labels = np.unique(states_str)

    states = np.zeros(n_states)
    for i, state in enumerate(state_labels):
        for j in np.arange(n_states):
            if state == states_str[j]:
                states[j] = i

    return states.astype(int), state_labels

def plot_figs():
    import matplotlib.animation as animation
    import matplotlib.colors as colors
    import matplotlib.cm as cm

    # Example placeholder color array for initialization
    initial_color = x[0, :]  # Start with the first time frame

    # Set up the figure and axis
    fig, ax = plt.subplots()
    sc = ax.scatter(df_conversion['coor.x'], df_conversion['coor.y'], c=initial_color, alpha=0.9, cmap='Purples', s=30)
    ax.axis('equal')
    # Define color normalization based on the range of x
    norm = colors.Normalize(vmin=np.min(x), vmax=np.max(x))

    # Add colorbar using ScalarMappable with the same colormap and normalization
    sm = cm.ScalarMappable(cmap='Purples', norm=norm)
    sm.set_array([])  # Dummy array for colorbar
    cbar = plt.colorbar(sm, ax=ax)

    # Add a text annotation for displaying the time
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top')

    # Update function for each frame
    def update(frame):
        # Update color with values from x at the given time frame
        sc.set_array(x[frame, :])
        # Update the time text
        time_text.set_text(f'Time: {frame}')
        return sc, time_text

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(x.shape[0]), interval=10, blit=True)

    plt.show()

    e_fast = minimum_energy_fast(A_norm=A_norm, T=T, B=B, x0=x0_mat, xf=xf_mat)
    e_fast = e_fast.transpose().reshape(n_states, n_states, n_nodes)
    e_fast = np.sum(e_fast, axis=2)  # sum over nodes

    # Define figure dimensions and layout settings
    fig, axes = plt.subplots(len(state_labels), 2, figsize=(3.2, 16.8))  # Optimized figure size
    # Define fixed axis limits based on data range
    x_min, x_max = df_conversion['coor.x'].min(), df_conversion['coor.x'].max()
    y_min, y_max = df_conversion['coor.y'].min(), df_conversion['coor.y'].max()
    z_min, z_max = df_conversion['coor.z'].min(), df_conversion['coor.z'].max()

    # Loop over each state label for plotting
    for i, label in enumerate(state_labels):
        # First subplot: x-y plane scatter plot
        print(label)
        ax1 = axes[i, 0]
        ax1.scatter(coords[:,0], coords[:,1], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
        ax1.scatter(coords[:,0], coords[:,1], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.axis('equal')
        ax1.axis('off')

        # Second subplot: y-z plane scatter plot
        ax2 = axes[i, 1]
        ax2.scatter(coords[:,1], coords[:,2], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
        ax2.scatter(coords[:,1], coords[:,2], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
        ax2.set_xlim(y_min, y_max)
        ax2.set_ylim(z_min, z_max)
        ax2.axis('equal')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()
    plt.rcParams.update({'font.size': 22})
    plot = plotting.plot_matrix(e_fast, figure=(10, 10), labels=state_labels, cmap='Purples')
    plt.show()

    # Define figure dimensions and layout settings
    fig, axes = plt.subplots(len(state_labels), 2, figsize=(3.2, 16.8))  # Optimized figure size
    # Define fixed axis limits based on data range
    x_min, x_max = df_conversion['coor.x'].min(), df_conversion['coor.x'].max()
    y_min, y_max = df_conversion['coor.y'].min(), df_conversion['coor.y'].max()
    z_min, z_max = df_conversion['coor.z'].min(), df_conversion['coor.z'].max()

    # Loop over each state label for plotting
    for i, label in enumerate(state_labels):
        # First subplot: x-y plane scatter plot
        print(label)
        ax1 = axes[i, 0]
        ax1.scatter(df_conversion['coor.x'], df_conversion['coor.y'], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
        ax1.scatter(df_conversion['coor.x'], df_conversion['coor.y'], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.axis('equal')
        ax1.axis('off')

        # Second subplot: y-z plane scatter plot
        ax2 = axes[i, 1]
        ax2.scatter(df_conversion['coor.y'], df_conversion['coor.z'], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
        ax2.scatter(df_conversion['coor.y'], df_conversion['coor.z'], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
        ax2.set_xlim(y_min, y_max)
        ax2.set_ylim(z_min, z_max)
        ax2.axis('equal')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()
    plt.rcParams.update({'font.size': 22})
    plot = plotting.plot_matrix(e_fast, figure=(10, 10), labels=state_labels, cmap='Purples')
    plt.show()

    plot = plotting.plot_matrix(diff_arr, figure=(10, 10), labels=state_labels, cmap='Purples')
    plt.show()


 # Gradients aka eigen vector of the diffusion map embedding

def fslr5k_dm_lr(mtx, mask_5k, Ngrad=3, log=True, S=0):
    """
    Create the gradients from the SC or GD matrices.
    Use log=False for GD gradients
    """
    if log != True:
     mtx_log = mtx
    else:
     # log transform the connectome
     mtx_log = np.log(mtx)
    # Replace NaN with 0
    mtx_log[np.isnan(mtx_log)] = 0
    # Replace negative infinite with 0
    mtx_log[np.isneginf(mtx_log)] = 0
    # Replace infinite with 0
    mtx_log[~np.isfinite(mtx_log)] = 0
    # replace 0 values with almost 0
    mtx_log[mtx_log==0] = np.finfo(float).eps
    # Left and right mask
    indx_L = np.where(mask_5k[0:4842]==1)[0]
    indx_R = np.where(mask_5k[4842:9684]==1)[0]
    # Left and right SC
    mtx_L = mtx_log[0:4842, 0:4842]
    mtx_R = mtx_log[4842:9684, 4842:9684]

    # Slice the matrix
    mtx_L_masked = mtx_L[indx_L, :]
    mtx_L_masked = mtx_L_masked[:, indx_L]
    mtx_R_masked = mtx_R[indx_R, :]
    mtx_R_masked = mtx_R_masked[:, indx_R]

    # mtx Left hemi
    mtx_L = GradientMaps(n_components=Ngrad, random_state=None, approach='dm', kernel='normalized_angle')
    mtx_L.fit(mtx_L_masked, sparsity=S)

    # mtx Right hemi
    mtx_R = GradientMaps(n_components=Ngrad, alignment='procrustes', kernel='normalized_angle'); # align right hemi to left hemi
    mtx_R.fit(mtx_R_masked, sparsity=S, reference=mtx_L.gradients_)

    # Left and right gradients concatenated
    mtx_gradients = np.concatenate((mtx_L.gradients_, mtx_R.aligned_), axis=0)

    # Boolean mask
    mask_surf = mask_5k != 0

    # Get the index of the non medial wall regions
    indx = np.where(mask_5k==1)[0]

    # Map gradients to surface
    grad = [None] * Ngrad
    for i, g in enumerate(mtx_gradients.T[0:Ngrad,:]):
        # create a new array filled with NaN values
        g_nan = np.full(mask_surf.shape, np.nan)
        g_nan[indx] = g
        grad[i] = g_nan
    return(mtx_gradients, grad)



def main():
    """
    main function, gather stats and call plots
    """
    params = {"ytick.color" : "w",
              "xtick.color" : "w",
              "axes.labelcolor" : "w",
              "axes.edgecolor" : "w",
              'font.size': 22}
    plt.rcParams.update(params)
    plt.style.use('dark_background')
    out_dir = os.path.abspath("/local_raid/data/pbautin/results/nct/")
    # os.makedirs(out_dir, exist_ok=True)

    # Number of vertices of the fsLR-5k matrices (per hemisphere)
    N5k = 9684
    # Number of gradients to calculate
    Ngrad=10
    # Number of gradients to plot
    Nplot=3
    # Labels for plotting based on Nplot
    labels=['G'+str(x) for x in list(range(1,Nplot+1))]

    # Load fsLR-5k inflated surface
    micapipe='/local_raid/data/pbautin/software/micapipe'
    f5k_lhi = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    f5k_rhi = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')

    # fsLR-5k mask
    mask_lh = nib.load(micapipe + '/surfaces/fsLR-5k.L.mask.shape.gii').darrays[0].data
    mask_rh = nib.load(micapipe + '/surfaces/fsLR-5k.R.mask.shape.gii').darrays[0].data
    mask_5k = np.concatenate((mask_lh, mask_rh), axis=0)


    ########## Plot structural connectome
    lut_sc='/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_subcortical-cerebellum_mics.csv'
    lut='/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-200_mics.csv'
    conn='/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-01/dwi/connectomes/sub-Pilot014_ses-01_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii'
    sc_5k = nib.load(conn).darrays[0].data
    sc_5k = np.triu(sc_5k,1)+sc_5k.T
    print(sc_5k.shape)
    # Calculate the gradients
    sc_dm, grad = fslr5k_dm_lr(sc_5k, mask_5k, Ngrad=Ngrad, log=False, S=0.9)
    # PLot the gradients (G2-G4)
    plot_hemispheres(f5k_lhi, f5k_rhi, array_name=grad[1:Nplot+1], cmap='RdBu_r', nan_color=(0, 0, 0, 1),
        zoom=1.3, size=(900, 750), embed_nb=False, color_range='sym',
        color_bar='right', label_text={'left': labels})


    # lut1 = pd.read_csv(lut_sc)
    # lut2 = pd.read_csv(lut)
    # indx = sorted(list(set(lut1['mics']).union(set(lut2['mics']))))
    # # set index for python
    # indx = [i - 1 for i in indx]
    # mtx_sc = M[np.ix_(indx, indx)][48:, 48:]
    # A_norm = matrix_normalization(np.triu(mtx_sc,1)+mtx_sc.T, system='continuous', c=1)
    # #A_norm = np.delete(np.delete(A_norm, Ndim, axis=0), Ndim, axis=1)
    # corr_plot = plotting.plot_matrix(np.log((np.triu(mtx_sc,1)+mtx_sc.T)+1), figure=(10, 10), labels=None, cmap='Purples') # , vmin=-10, vmax=2
    # plotting.show()
    #
    #
    # ############### get BigBrainWarp G2 microstructure gradient and plot on surf
    # fsLR_lh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    # fsLR_rh = read_surface('/local_raid/data/pbautin/software/micapipe/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    # profile_gii_l = '/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii'
    # profile_gii_r = '/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii'
    # profile_all = np.hstack(np.concatenate((nib.load(profile_gii_l).darrays[0].data,
    #                                nib.load(profile_gii_r).darrays[0].data), axis=0)).astype(float)
    # print("profile shape {}".format(profile_all.shape))
    #
    # ############### get Schaerfer parcellation and plot on surf
    # annot_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-200_conte69_lh.label.gii').darrays[0].data + 1000
    # annot_lh[annot_lh == 1000] = 1000
    # annot_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-200_conte69_rh.label.gii').darrays[0].data + 1900
    # annot_rh[annot_rh == 1900] = 2000
    # annot_all = np.hstack(np.concatenate((annot_lh, annot_rh), axis=0)).astype(float)
    # print('annot all shape {}'.format(annot_all.shape))
    #
    # ############ Concatenate micro gradient and shaeffer parecellation and plot on surf
    # df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-200_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
    # df_label.fillna('other', inplace=True)
    # d = {'mics': annot_all, 'profile': profile_all}
    # df_hist = pd.DataFrame(data=d)
    # df_mean_p = df_hist.groupby('mics').mean().reset_index().merge(df_label, on='mics')
    # df_mean = df_hist.merge(df_mean_p, on='mics', suffixes=('', '_mean'))
    #
    # df_state_p = df_mean.groupby('label').mean().reset_index()[['profile', 'label']].sort_values('profile')
    # df_state_p['label_int'] = np.arange(8)
    # df_state = df_mean.merge(df_state_p, on='label', suffixes=('', '_state_mean'))
    #
    # profile_all[(annot_all == 1000) + (annot_all == 2000)] = np.nan
    # profile_all_mean = df_mean.profile_mean.values
    # profile_all_mean[(annot_all == 1000) + (annot_all == 2000)] = np.nan
    # profile_all_mean_state = df_state.profile_state_mean.values
    # profile_all_mean_state[(annot_all == 1000) + (annot_all == 2000)] = np.nan
    #
    # #annot_all[(annot_all == 1000) + (annot_all == 2000)] = np.nan
    #
    # plot_hemispheres(fsLR_lh, fsLR_rh, array_name=annot_all, size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
    #                          nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid', screenshot=True, filename='/home/bic/pbautin/Downloads/Figure_surf_1.png', scale=4)
    #
    # plot_hemispheres(fsLR_lh, fsLR_rh, array_name=profile_all, size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
    #                          nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=False, layout_style='grid', screenshot=True, filename='/home/bic/pbautin/Downloads/Figure_surf_0.png', scale=4)
    # plot_hemispheres(fsLR_lh, fsLR_rh, array_name=profile_all_mean, size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
    #                          nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid', screenshot=True, filename='/home/bic/pbautin/Downloads/Figure_surf_2.png', scale=4)
    # plot_hemispheres(fsLR_lh, fsLR_rh, array_name=profile_all_mean_state, size=(400, 350), color_bar='bottom', zoom=1.25, embed_nb=False, interactive=False, share='both', background=(0,0,0),
    #                          nan_color=(0, 0, 0, 1), color_range=None, cmap="Purples", transparent_bg=True, layout_style='grid', screenshot=True, filename='/home/bic/pbautin/Downloads/Figure_surf_3.png', scale=4)
    #
    #
    # ############ Minimum State-to-state energy
    # states_int = df_state.groupby('mics')['label_int'].mean().values
    # state_labels = df_state_p.label.values
    # x0_mat, xf_mat = expand_states(states_int)
    # # x0_mat = np.array([normalize_state(x0_mat[:,i]) for i in np.arange(x0_mat.shape[1])]).T
    # # xf_mat = np.array([normalize_state(xf_mat[:,i]) for i in np.arange(x0_mat.shape[1])]).T
    # T = 1   # time horizon
    # n_nodes = A_norm.shape[0]
    # n_states = len(np.unique(states_int))
    #
    # B = np.eye(n_nodes) # set all nodes to control nodes
    #
    # x, u, error = get_control_inputs(A_norm, T, B, x0_mat[:,5], xf_mat[:,6], system='continuous', rho=1, S='identity', xr='zero', expm_version='scipy')
    # e_fast = minimum_energy_fast(A_norm=A_norm, T=T, B=B, x0=x0_mat, xf=xf_mat)
    # e_fast = e_fast.transpose().reshape(n_states, n_states, n_nodes)
    # e_fast = np.sum(e_fast, axis=2)  # sum over nodes
    # e_fast = np.delete(np.delete(e_fast, 1, axis=0), 1, axis=1)
    #
    #
    # labels = np.delete(df_state_p.label.values, 1, axis=0)
    # corr_plot = plotting.plot_matrix(e_fast, figure=(10, 10), labels=labels, cmap='Purples') # , vmin=-10, vmax=2
    # plotting.show()
    #
    # labels = np.delete(df_state_p.label.values, 1, axis=0)
    # asym = e_fast - e_fast.T
    # corr_plot = plotting.plot_matrix(asym, figure=(10, 10), labels=labels, cmap='Purples') # , vmin=-10, vmax=2
    # plotting.show()
    #
    # diff_arr = df_state_p['profile'].values - df_state_p['profile'].values[:, None]
    # diff_arr = np.delete(np.delete(diff_arr, 1, axis=0), 1, axis=1)
    # corr_plot = plotting.plot_matrix(diff_arr, figure=(10, 10), labels=labels, cmap='Purples') # , vmin=-10, vmax=2
    # plotting.show()
    #
    # f, ax = plt.subplots(1, 1, figsize=(10, 10))
    # upper_indx = np.triu_indices(7, k=1)
    # print(asym)
    # my_reg_plot(diff_arr[upper_indx], asym[upper_indx], 'hierarchy distance', 'energy asymmetry', ax, annotate='spearman')
    # plt.show()
    #
    # f, ax = plt.subplots(1, 1, figsize=(10, 10))
    # my_reg_plot(diff_arr, e_fast, 'hierarchy distance', 'energy', ax, annotate='spearman')
    # plt.show()
    #
    # plt.figure(figsize=(10,10))
    # plt.boxplot(e_fast[::-1].T, tick_labels=labels[::-1], vert=False)
    # plt.show()
    #
    # # Initialize the color palette
    # pal = sns.color_palette(palette='Purples', n_colors=7)
    # data = pd.DataFrame(e_fast).melt(var_name="label", value_name="value", ignore_index=False)
    # data = data.merge(data.groupby('label').mean(), on='label', suffixes=('', '_mean'))
    # # Initialize the FacetGrid object
    # g = sns.FacetGrid(data, row="label", hue="value_mean", aspect=30, height=1.5, palette=pal)
    # g.fig.set_size_inches(10, 10)  # Set the figure size here
    # # Draw the density plots
    # g.map(sns.kdeplot, "value",
    #       bw_adjust=1, clip_on=False,
    #       fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=1, bw_adjust=1)
    # g.map(plt.axhline, y=0, lw=1, clip_on=False, color='w')
    # g.set_titles("")
    # g.set(yticks=[], ylabel="", xlabel="energy (a.u.)")
    # g.despine(bottom=True, left=True)
    # plt.show()
    #
    #
    # atlas = nib.load('/local_raid/data/pbautin/data/MNI152Volumes/micapipe/micapipe_v0.2.0/sub-mni/ses-01/parc/sub-mni_ses-01_space-nativepro_T1w_atlas-schaefer-200_all.nii.gz')
    # coords, labels_list = find_parcellation_cut_coords(atlas, return_label_names=True, background_label=0)
    # coords = coords[49:, :]
    # coords = np.delete(coords, 100, axis=0)
    # states_int = states_int[1:]
    # states_int = np.delete(states_int, 100, axis=0)
    # states_int[states_int > 1] = states_int[states_int > 1] - 1
    # print(states_int.shape)
    #
    # # Define figure dimensions and layout settings
    # fig, axes = plt.subplots(len(labels), 2, figsize=(3.2, 16.8))  # Optimized figure size
    # # Define fixed axis limits based on data range
    # x_min, x_max = coords[:,0].min(), coords[:,0].max()
    # y_min, y_max = coords[:,1].min(), coords[:,1].max()
    # z_min, z_max = coords[:,2].min(), coords[:,2].max()
    #
    #
    # # Loop over each state label for plotting
    # for i, label in enumerate(labels):
    #     print(i)
    #     # First subplot: x-y plane scatter plot
    #     ax1 = axes[i, 0]
    #     ax1.scatter(coords[:,0], coords[:,1], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
    #     ax1.scatter(coords[:,0], coords[:,1], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
    #     ax1.set_xlim(x_min, x_max)
    #     ax1.set_ylim(y_min, y_max)
    #     ax1.axis('equal')
    #     ax1.axis('off')
    #
    #     # Second subplot: y-z plane scatter plot
    #     ax2 = axes[i, 1]
    #     ax2.scatter(coords[:,1], coords[:,2], alpha=0.9, c=(states_int != i), cmap='Greys', s=30)
    #     ax2.scatter(coords[:,1], coords[:,2], alpha=0.5, c=(states_int == i), cmap='Purples', s=30)
    #     ax2.set_xlim(y_min, y_max)
    #     ax2.set_ylim(z_min, z_max)
    #     ax2.axis('equal')
    #     ax2.axis('off')
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
