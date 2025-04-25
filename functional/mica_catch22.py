import pycatch22 as catch22
import os
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import glob
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore


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

def RAD(x, centre=False, tau=1):
    """
    Harris, Gollo, & Fulcher's rescaled auto-density (RAD) noise-insensitive
    metric for inferring the distance to criticality.

    Parameters
    ----------
    x: array
        A time-series input vector

    centre : boolean
        Whether to centre the time series and take absolute values

    tau: integer
        The embedding and differencing delay in units of the timestep

    Returns
    -------
    The RAD feature value.
    """

    # ensure that x is in the form of a numpy array
    x = np.array(x)
    
    # if specified: centre the time series and take the absolute value
    if centre:
        x = x - np.median(x)
        x = np.abs(x)

    # Delay embed at interval tau
    y = x[tau:]
    x = x[:-tau]

    # Median split
    subMedians = x < np.median(x)
    superMedianSD = np.std(x[~subMedians], ddof=1)
    subMedianSD = np.std(x[subMedians], ddof=1)

    # Properties of the auto-density
    sigma_dx = np.std(y - x)
    densityDifference = (1/superMedianSD) - (1/subMedianSD)

    # return RAD
    return sigma_dx * densityDifference


params = {"ytick.color" : "w",
            "xtick.color" : "w",
            "axes.labelcolor" : "w",
            "axes.edgecolor" : "w",
            'font.size': 22}
plt.rcParams.update(params)
plt.style.use('dark_background')


# Load the data
surf_lh, surf_rh = load_conte69()
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
#atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
# Load data for multiple subjects
func_32k = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
# Load the data for each subject
func_32k = [nib.load(f).darrays[0].data for f in func_32k]
# Apply the Yeo atlas to the data
func_400 = np.array([reduce_by_labels(f, yeo_surf, red_op='mean', axis=0) for f in func_32k])
print(len(func_400))
print(func_400[1].shape[0])
# cut the data to 275 time points
func_400 = np.array([f[:275, :] for f in func_400])

df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
df_label.fillna('other', inplace=True)
df = pd.DataFrame(data={'mics': yeo_surf})
df = df.merge(df_label, on='mics',validate="many_to_one", how='left')
print(df)
df.fillna('other', inplace=True)
state, state_name = convert_states_str2int(df['label'].values)
state = state.astype(float)
# salience
salience = state == np.where(state_name == 'SalVentAttn')[0][0]
print(salience)
# apply salience mask on func_32k
salience_32k = np.array([s[:, salience] for s in func_32k])
print(salience_32k[0].shape)

#func_400 = salience_32k

def compute_features(x):
    res = catch22.catch22_all(x, catch24=True)
    return res['values']

print(f"Number of cores available: {os.cpu_count()}")
threads_to_use = max(1, os.cpu_count() // 2)  # Use half the cores, ensuring at least 1 thread

# Compute the features for each time series
# using parallel processing
results_list = Parallel(n_jobs=threads_to_use)(
    delayed(compute_features)(func_400[sub][:, i]) 
    for sub in range(len(func_400)) 
    for i in range(func_400[0].shape[1])
)
# Reshape the results into a 3D matrix: Subjects x Regions x Features
results = np.asarray(results_list).reshape(len(func_400), func_400[0].shape[1], -1)
print(f"Results shape: {results.shape} (Subjects x Regions x Features)")

# plot imshow of the results matrix
plt.figure(figsize=(10, 10))
plt.imshow(zscore(np.mean(results, axis=0)), aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('Catch22 features')
plt.ylabel('Cortical regions')
plt.show()

# compute and plot the region per region similarity
# compute the correlation matrix
print(np.mean(results, axis=0).shape)
corr_matrix = np.corrcoef(zscore(np.mean(results, axis=0)))
plt.figure(figsize=(10, 10))
plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Cortical regions')
plt.ylabel('Cortical regions')
plt.show()

# plot the first two PCA components on surfaces
pca = PCA(n_components=10)
pca.fit_transform(zscore(np.mean(results, axis=0)).T)
# scatter plot of variance explained
plt.figure(figsize=(10, 10))
plt.scatter(np.arange(0,10), pca.explained_variance_ratio_)
plt.xlabel('PCA component')
plt.ylabel('Variance explained')
plt.show()

plot_values = pca.components_[0]
mask = (yeo_surf != 1000) & (yeo_surf != 2000)
plot_values = map_to_labels(plot_values, yeo_surf, mask=mask, fill=np.nan)
# mask = salience
# When salience true multiply by plot_values
# plot_values = np.zeros(salience.shape)
# plot_values[salience] = pca.components_[0]
# plot_values[plot_values == 0] = np.nan
# print(plot_values)



plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

plot_values = pca.components_[1]
mask = (yeo_surf != 1000) & (yeo_surf != 2000)
plot_values = map_to_labels(plot_values, yeo_surf, mask=mask, fill=np.nan)
# plot_values = np.zeros(salience.shape)
# plot_values[salience] = pca.components_[1]
# plot_values[plot_values == 0] = np.nan
# print(plot_values)
plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

# print the results for the first time series
# feature_index = np.argwhere(np.asarray(catch22.catch22_all(func_400[0], catch24 = True)['names']) == 'SP_Summaries_welch_rect_centroid')[0][0]
# plot_values = map_to_labels(results[:,feature_index], yeo_surf)
# plot_values = map_to_labels(feature_rad, yeo_surf)
# plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                              nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


# threads_to_use = os.cpu_count()
# feature_rad = np.asarray(Parallel(n_jobs=threads_to_use)(
#     delayed(RAD)(func_400[i]) for i in range(len(func_400))))