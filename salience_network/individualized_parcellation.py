import scipy.io as sio
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels

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
    state_labels = np.sort(np.unique(states_str))

    states = np.zeros(n_states)
    for i, state in enumerate(state_labels):
        for j in np.arange(n_states):
            if state == states_str[j]:
                states[j] = i

    return states.astype(float), state_labels

#### set custom plotting parameters
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')
yeo7_colors = mp.colors.ListedColormap(np.array([
                        [0, 118, 14, 255],
                        [230, 148, 34, 255],
                        [205, 62, 78, 255],
                        [120, 18, 134, 255],
                        [220, 248, 164, 255],
                        [70, 130, 180, 255],
                        [196, 58, 250, 255]]) / 255)
mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)
cmap_types = mp.colors.ListedColormap(np.array([
                        [127, 140, 172, 255],
                        [139, 167, 176, 255],
                        [171, 186, 162, 255],
                        [218, 198, 153, 255],
                        [253, 211, 200, 255],
                        [252, 229, 252, 255],
                        [252, 229, 252, 255]])/255)
mp.colormaps.register(name='CustomCmap_type', cmap=cmap_types)

surf_lh, surf_rh = load_conte69()

# Load the .mat file
parcels_struct = sio.loadmat('/local_raid/data/pbautin/results/parcellation/ind_parcellation_gMSHBM/test_set/2_sess/beta50/Ind_parcellation_MSHBM_sub1_w50_MRF10_beta50.mat')  # Adjust the filename as needed
lh_labels = parcels_struct['lh_labels'][:,0].astype(np.float32) + 1000
rh_labels = parcels_struct['rh_labels'][:,0].astype(np.float32) + 1800
rh_labels[rh_labels == 1800] = 2000
labels = np.concatenate((lh_labels, rh_labels), axis=0).astype(float)
df_yeo_surf = pd.DataFrame(data={'mics': labels})

df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv')
df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
df_yeo_surf = df_yeo_surf.merge(df_label, on='mics', validate="many_to_one", how='left')
state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
state[state == np.where(state_name == 'medial_wall')[0]] = np.nan
print(np.unique(state))
print(state_name)
#salience = state.copy()
#salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
#state_stack = np.vstack([state, salience])
plot_hemispheres(surf_lh, surf_rh, array_name=state, size=(1200, 600), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                         nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)
