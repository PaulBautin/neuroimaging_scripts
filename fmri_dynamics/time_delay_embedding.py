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

from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, r2_score
from sklearn import linear_model

from numpy.linalg import lstsq
from scipy import stats

from scipy.sparse.linalg import spsolve, lsqr


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

def time_embed(x: np.ndarray, n_embeddings: int) -> np.ndarray:
    """
    Perform time-delay embedding using sliding windows.

    Parameters
    ----------
    x : np.ndarray
        Time series data of shape (n_samples, n_channels).
    n_embeddings : int
        Number of consecutive time steps in each embedding vector.
        Must be odd.

    Returns
    -------
    X : np.ndarray
        Time-embedded data of shape
        (n_samples - n_embeddings + 1, n_channels * n_embeddings).
        Each row is a flattened window:
            [x[t], x[t+1], ..., x[t+n_embeddings-1]] across all channels.
    """
    if x.ndim != 2:
        raise ValueError("Input must be 2D: (n_samples, n_channels).")
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    n_samples, n_channels = x.shape
    if n_embeddings > n_samples:
        raise ValueError("n_embeddings cannot exceed number of samples.")

    # Extract sliding windows of shape (n_windows, n_embeddings, n_channels)
    X = np.lib.stride_tricks.sliding_window_view(x, (n_embeddings, n_channels))
    # Drop the redundant channel dimension introduced by the window spec
    X = X.reshape(-1, n_embeddings, n_channels)

    # Flatten embedding: (n_windows, n_embeddings * n_channels)
    X = X.reshape(X.shape[0], -1)

    return X

def normalize_to_range(data, target_min, target_max):
    """
    Normalizes a NumPy array or list of numerical data to a specified target range.

    Args:
        data (np.array or list): The input numerical data.
        target_min (float): The desired minimum value of the normalized range.
        target_max (float): The desired maximum value of the normalized range.

    Returns:
        np.array: The normalized data within the target range.
    """
    data = np.array(data) # Ensure data is a NumPy array for min/max operations
    
    original_min = np.nanmin(data)
    original_max = np.nanmax(data)

    if original_min == original_max: # Handle cases where all values are the same
        return np.full_like(data, (target_min + target_max) / 2)

    # Normalize to 0-1 range first
    normalized_0_1 = (data - original_min) / (original_max - original_min)

    # Scale to the target range
    scaled_data = target_min + (normalized_0_1 * (target_max - target_min))
    return scaled_data


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

    func_files = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-01/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
    func = np.vstack([zscore(nib.load(f).darrays[0].data, axis=0) for f in func_files[:7]])

    df_func = pd.DataFrame(func.T)
    df_func['network'] = df_yeo_surf['network'].values
    df_func_net = df_func.groupby('network').mean().transform(lambda x : zscore(x))

    func_sal = df_func_net[df_func_net.index == 'SalVentAttn'].to_numpy().T
    func[:, df_yeo_surf.network == 'medial_wall'] = np.nan
    print(func.shape)

    delay = 10
    reg = linear_model.Ridge()
    func_sal = np.concatenate((func_sal[:-delay,:], func_sal[2:-delay+2,:], func_sal[4:-delay+4,:], func_sal[6:-delay+6,:], func_sal[8:-delay+8,:], func_sal[10:,:]), axis=1)
    func = func[delay:,np.any(~np.isnan(func), axis=0)]
    reg.fit(func_sal, func)
    df_yeo_surf.loc[df_yeo_surf.network != 'medial_wall', 'linear_reg'] = reg.coef_[:,1]
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['linear_reg'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    func = np.vstack([zscore(nib.load(f).darrays[0].data, axis=0) for f in func_files[10:11]])

    df_func = pd.DataFrame(func.T)
    df_func['network'] = df_yeo_surf['network'].values
    df_func_net = df_func.groupby('network').mean().transform(lambda x : zscore(x))

    func_sal = df_func_net[df_func_net.index == 'SalVentAttn'].to_numpy().T
    func_sal = np.concatenate((func_sal[:-delay,:], func_sal[2:-delay+2,:], func_sal[4:-delay+4,:], func_sal[6:-delay+6,:], func_sal[8:-delay+8,:], func_sal[10:,:]), axis=1)
    func[:, df_yeo_surf.network == 'medial_wall'] = np.nan
    print(func.shape)

    df_yeo_surf.loc[df_yeo_surf.network != 'medial_wall', 'linear_reg_pred'] = reg.predict(func_sal)[0,:]
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['linear_reg_pred'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    print(r2_score(func[delay:,np.any(~np.isnan(func), axis=0)], reg.predict(func_sal)))




if __name__ == "__main__":
    main()