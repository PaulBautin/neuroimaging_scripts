from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Connectivity and communication of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh

from nctpy.energies import integrate_u, get_control_inputs
from nctpy.pipelines import ComputeControlEnergy, ComputeOptimizedControlEnergy
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    convert_states_str2int,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
)
from nctpy.plotting import roi_to_vtx, null_plot, surface_plot, add_module_lines, set_plotting_params

from scipy.integrate import simpson as simps
import scipy as sp


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

    return states.astype(float), state_labels


def main():
    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    # load .csv associated with schaefer 400
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label = df_label[df_label.network != 'medial_wall']
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    states, state_labels = convert_states_str2int(df_label['network'].values)

    #### Load connectivity matrix
    C = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    C = np.average(np.array([nib.load(f).darrays[0].data for f in C[:2]]), axis=0)
    C = np.log(np.triu(C,1)+C.T + 1)
    C = C[49:, 49:]
    C = np.delete(np.delete(C, 200, axis=0), 200, axis=1)
    N = C.shape[0] # Number of nodes
    C[np.eye(N, dtype=bool)] = 0
    C = C / np.mean(C[~np.eye(N, dtype=bool)])

    ##### Network control theory
    n_nodes = C.shape[0]
    control_tasks = []
    control_set = np.eye(n_nodes)
    control_set[df_label.network == 'SalVentAttn', df_label.network == 'SalVentAttn'] = 0
    trajectory_constraints = np.eye(n_nodes)
    trajectory_constraints[df_label.network == 'SalVentAttn', df_label.network == 'SalVentAttn'] = 0
    rho = 1
    n_states = len(state_labels)
    for initial_idx in np.arange(n_states):
        initial_state = normalize_state(states == initial_idx)  # initial state
        for target_idx in np.arange(n_states):
            target_state = normalize_state(states == target_idx)  # target state
            control_task = dict()  # initialize dict
            control_task["x0"] = initial_state  # store initial state
            control_task["xf"] = target_state  # store target state
            control_task["B"] = control_set  # store control set
            control_task["S"] = trajectory_constraints  # store state trajectory constraints
            control_task["rho"] = rho  # store rho
            control_tasks.append(control_task)
    compute_control_energy = ComputeControlEnergy(A=C, control_tasks=control_tasks, system="continuous", c=1, T=1)
    compute_control_energy.run()
    energy_matrix = np.reshape(compute_control_energy.E, (n_states, n_states))
    energy_matrix_delta = energy_matrix.transpose() - energy_matrix


    # ##### Network control theory
    # n_nodes = C.shape[0]
    # control_tasks = []
    # control_set = np.eye(n_nodes)
    # trajectory_constraints = np.eye(n_nodes)
    # rho = 1
    # n_states = len(state_labels)
    # for initial_idx in np.arange(n_states):
    #     initial_state = normalize_state(states == initial_idx)  # initial state
    #     for target_idx in np.arange(n_states):
    #         target_state = normalize_state(states == target_idx)  # target state
    #         control_task = dict()  # initialize dict
    #         control_task["x0"] = initial_state  # store initial state
    #         control_task["xf"] = target_state  # store target state
    #         control_task["B"] = control_set  # store control set
    #         control_task["S"] = trajectory_constraints  # store state trajectory constraints
    #         control_task["rho"] = rho  # store rho
    #         control_tasks.append(control_task)
    # compute_control_energy = ComputeControlEnergy(A=C, control_tasks=control_tasks, system="continuous", c=1, T=1)
    # compute_control_energy.run()
    # energy_matrix_with_sal = np.reshape(compute_control_energy.E, (n_states, n_states))
    # energy_matrix_delta_with_sal = energy_matrix.transpose() - energy_matrix

    f, ax = plt.subplots(1, 3, figsize=(14, 8))
    sns.heatmap(
        energy_matrix,
        ax=ax[0],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy", "shrink": 0.25},
    )
    # plot without self-transitions
    mask = np.zeros_like(energy_matrix)
    mask[np.eye(n_states) == 1] = True
    sns.heatmap(
        energy_matrix,
        ax=ax[1],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy", "shrink": 0.25},
        mask=mask,
    )
    # plot energy asymmetries
    mask = np.triu(np.ones_like(energy_matrix, dtype=bool))
    sns.heatmap(
        energy_matrix_delta,
        ax=ax[2],
        square=True,
        linewidth=0.5,
        cbar_kws={"label": "energy (delta)", "shrink": 0.25},
        mask=mask,
        cmap="RdBu_r",
        center=0,
    )
    for cax in ax:
        cax.set_ylabel("initial state (x0)")
        cax.set_xlabel("target state (xf)")
        cax.set_yticklabels(state_labels, rotation=0, size=6)
        cax.set_xticklabels(state_labels, rotation=90, size=6)
    f.tight_layout()
    plt.show()





    


if __name__ == "__main__":
    main()