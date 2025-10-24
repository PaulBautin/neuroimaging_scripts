import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation

from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from scipy.stats import zscore


import sys
sys.path.insert(0, '/local_raid/data/pbautin/software/elapy') # Add the path to the other repo
import elapy as ela

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
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-32k inflated surface
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.midthickness.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.midthickness.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    #### load yeo atlas 7 network
    atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    yeo_surf = np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0).astype(float)
    df_yeo_surf = pd.DataFrame(data={'mics': yeo_surf})
    df_yeo_surf['index'] = df_yeo_surf.index
    df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    df_yeo_surf.loc[df_yeo_surf['network'] == 'medial_wall', 'network_int'] = np.nan
    salience_border = array_operations.get_labeling_border(surf_32k, np.asarray(df_yeo_surf['network'] == 'SalVentAttn'))
    df_yeo_surf.loc[salience_border == 1, 'network_int'] = 7

    func = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-01/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
    func = np.vstack([nib.load(f).darrays[0].data for f in func[:2]])
    print(func.shape)

    # Add timepoints as separate columns
    df_func = df_yeo_surf[['network']].copy()
    timepoint_cols = [f"{i}" for i in range(func.shape[0])]
    df_func = pd.DataFrame(func.T, columns=timepoint_cols)
    df_func['network'] = df_yeo_surf['network'].values
    df_func = df_func[df_func['network'] != 'medial_wall']
    df_func_mean = df_func.groupby('network').mean()
    df_func_mean = df_func_mean.T.apply(zscore).T
    print(df_func_mean)
    df_func_mean[df_func_mean >= 0] = 1
    df_func_mean[df_func_mean < 0] = 0
    data = df_func_mean.astype(int)
    
    print(data)
    h, W = ela.fit_exact(data)  # exact fitting based on likelihood function
    #h, W = ela.fit_approx(data) # approximated fitting based on pseudo-likelihood function

    acc1, acc2 = ela.calc_accuracy(h, W, data)
    print(acc1, acc2)

    graph = ela.calc_basin_graph(h, W, data)

    D = ela.calc_discon_graph(h, W, data, graph)
    freq, trans, trans2 = ela.calc_trans(data, graph)
    P = ela.calc_trans_bm(h, W, data)

    ela.plot_local_min(data, graph)
    plt.show()
    ela.plot_basin_graph(graph)
    plt.show()
    ela.plot_discon_graph(D)
    plt.show()
    ela.plot_landscape(D)
    plt.show()
    # ela.plot_landscape3d(D)
    # plt.show()
    ela.plot_trans(freq, trans, trans2)
    plt.show()


if __name__ == "__main__":
    main()