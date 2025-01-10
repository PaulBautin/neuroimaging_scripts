from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Cytoarchitecture of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

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
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels
from scipy import stats

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


plt.style.use('dark_background')

# load the conte69 hemisphere surfaces and spheres
surf_lh, surf_rh = load_conte69()
sphere_lh, sphere_rh = load_conte69(as_sphere=True)


# Load the data from the specified text file
data = np.loadtxt('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_den-32k_desc-profiles.txt', delimiter=',')
base = np.ones((50, 64984))
data_entropy = stats.entropy(data, base, axis=0)



#################################
# load econo atlas
econo_surf_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_lh.label.gii').darrays[0].data
econo_surf_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/economo_conte69_rh.label.gii').darrays[0].data
econo_surf = np.hstack(np.concatenate((econo_surf_lh, econo_surf_rh), axis=0)).astype(float)
# Hardcoded based on table data in Garcia-Cabezas (2021)
econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])
econ_ctb = econ_ctb[[0] + list(range(2, 45))]
#mask = econo_surf != 0
surf_type = map_to_labels(econ_ctb.astype(float), econo_surf)#, mask=mask, fill=np.nan
surf_type[surf_type == 0] = np.nan
print(np.unique(surf_type))
surf_type_lh = surf_type[:32492]
surf_type_rh = surf_type[32492:]
cmap_types = mp.colors.ListedColormap(np.array([
    [127, 140, 172, 255],
    [139, 167, 176, 255],
    [171, 186, 162, 255],
    [218, 198, 153, 255],
    [253, 211, 200, 255],
    [252, 229, 252, 255],
    [252, 229, 252, 255]])/255)
mp.colormaps.register(name='CustomCmap', cmap=cmap_types)

# surf_type_arr = [surf_type[surf_type != i] == 0 for i in np.unique(surf_type)]
# surf_type_arr = np.array(surf_type_arr[0])
# print(surf_type_arr.shape)
def surf_type_isolation(surf_type_test, i):
    # Work on a copy of the input array to avoid modifying the original
    surf_type_copy = surf_type_test.copy()
    surf_type_copy[surf_type_copy != i] = np.nan
    return surf_type_copy

# Generate a new array with isolated surf types
print(np.unique(surf_type))
surf_type_arr = np.array([surf_type_isolation(surf_type, i) for i in np.unique(surf_type)])
print(np.unique(surf_type_arr))

# plot_hemispheres(surf_lh, surf_rh, array_name=surf_type_arr[:6], size=(800, 1400), share='both', background=(0,0,0),
#                          nan_color=(250, 250, 250, 1), cmap='CustomCmap', transparent_bg=True)
# plot_hemispheres(surf_lh, surf_rh, array_name=surf_type, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                          nan_color=(250, 250, 250, 1), cmap='CustomCmap', transparent_bg=True)

#####################################33
# load yeo atlas 7 network
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 2000
yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
print(np.unique(yeo_surf))
# Concatenate micro gradient and shaeffer parecellation
df_label = pd.read_csv('/local_raid/data/pbautin/software/micapipe/parcellations/lut/lut_schaefer-400_mics.csv').set_index('mics').label.str.split('_').str[2].reset_index()
df_label.fillna('other', inplace=True)
print(df_label)
df = pd.DataFrame(data={'mics': yeo_surf})
df = df.merge(df_label, on='mics',validate="many_to_one", how='left')
state, state_name = convert_states_str2int(df['label'].values)
print(np.unique(state_name))
state = state.astype(float)

# network composition
surf_type[np.isnan(surf_type)] = 7
unique_vals = np.unique(surf_type)
salience_composition = {val: [] for val in unique_vals}
dict_state_composition = {}
dict_state = {}
for s_name in state_name:
    dict_value ={}
    # isolate state
    val = state == np.where(state_name == s_name)[0][0]
    val = val.astype(float)
    val[val == 0] = np.nan
    dict_state.update({s_name:val})
    # find composition
    composition = surf_type[~np.isnan(val)] * val[~np.isnan(val)]
    unique, counts = np.unique(composition, return_counts=True)
    counts = (counts / len(composition)) * 100
    counts_dict = dict(zip(unique, counts))
    for value in unique_vals:
        dict_value.update({value:counts_dict.get(value, 0)})
        dict_state_composition.update({s_name:dict_value})

print("hererr")
print("dict {}".format(dict_state_composition))

# composition = surf_type[~np.isnan(salience)] * salience[~np.isnan(salience)]
# unique, counts = np.unique(composition, return_counts=True)
# counts = (counts / len(composition)) * 100
# counts_dict_real = dict(zip(unique, counts))


salience = state == np.where(state_name == 'SalVentAttn')[0][0]
salience = salience.astype(float)
salience[salience == 0] = np.nan
salience_lh = salience.astype(float)[:32492]
salience_rh = salience.astype(float)[32492:]
# salience = salience.astype(float) *  np.where(state_name == 'SalVentAttn')[0][0]
# salience[salience != np.where(state_name == 'SalVentAttn')[0][0]] = np.nan
print(salience)
# state[state == 7] = np.nan
# print(np.unique(state))
# #print(np.where(state_name == 'SalVentAttn'))
# #salience = df['label'].values == 'SalVentAttn'
# #salience = salience.astype(float)
#
# # print(salience.shape)

composition = data[:,~np.isnan(salience)] * salience[~np.isnan(salience)]
print(composition.shape)
gm = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle')
gm.fit(composition.T)
print(gm.gradients_[:, 0].shape)
colors = plt.cm.coolwarm((gm.gradients_[:, 0] + np.abs(np.min(gm.gradients_[:, 0]))) / (np.max(gm.gradients_[:, 0]) + np.abs(np.min(gm.gradients_[:, 0]))))
for i in range(len(composition[0,:])):
    plt.plot(range(0,50), composition[:,i], color=colors[i])
plt.show()


arr = np.zeros(64984)
arr[arr == 0] = np.nan
arr[~np.isnan(salience)] = gm.gradients_[:, 0] / 0.00187
plot_hemispheres(surf_lh, surf_rh, array_name=arr, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                         nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')

#
# yeo7_colors = mp.colors.ListedColormap(np.array([
#                         [0, 118, 14, 255],
#                         [230, 148, 34, 255],
#                         [205, 62, 78, 255],
#                         [120, 18, 134, 255],
#                         [220, 248, 164, 255],
#                         [70, 130, 180, 255],
#                         [196, 58, 250, 255]]) / 255)
# mp.colormaps.register(name='CustomCmap_yeo', cmap=yeo7_colors)
# state = np.vstack([state, salience])
# plot_hemispheres(surf_lh, surf_rh, array_name=state, size=(1200, 600), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                          nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)
# plot_hemispheres(surf_lh, surf_rh, array_name=salience, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
#                          nan_color=(250, 250, 250, 1), cmap='CustomCmap_yeo', transparent_bg=True)


from brainspace.null_models import SpinPermutations
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

# Let's create some rotations
n_rand = 10

sp = SpinPermutations(n_rep=n_rand, random_state=0)
sp.fit(sphere_lh, points_rh=sphere_rh)

surf_type[np.isnan(surf_type)] = 7
salience_rotated = np.hstack(sp.randomize(salience_lh, salience_rh))


composition = surf_type[~np.isnan(salience)] * salience[~np.isnan(salience)]
unique, counts = np.unique(composition, return_counts=True)
counts = (counts / len(composition)) * 100
counts_dict_real = dict(zip(unique, counts))


unique_vals = np.unique(surf_type)
#unique_vals = unique_vals[~np.isnan(unique_vals)]  # Remove NaN values from unique keys
salience_composition = {val: [] for val in unique_vals}

for n in range(n_rand):
    composition = surf_type[~np.isnan(salience_rotated[n])] * salience_rotated[n][~np.isnan(salience_rotated[n])]
    unique, counts = np.unique(composition, return_counts=True)
    counts = (counts / len(composition)) * 100

    counts_dict = dict(zip(unique, counts))
    for val in unique_vals:
        salience_composition[val].append(counts_dict.get(val, 0))


# Convert dictionary to DataFrame for plotting
data = pd.DataFrame(salience_composition)
# Melt the DataFrame to long format
data_melted = data.melt(var_name='Key', value_name='Count')
print(data_melted)
#x_arr = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag']

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.barplot(x='Key', y='Count', data=data_melted, color='white',fill=False)#,vert=False
sns.scatterplot(x=range(0,7), y=counts_dict_real.values(), color=cmap_types.colors)
plt.title('Salience Network Composition')
plt.xlabel('Cortical Types')
plt.ylabel('Percentage')
plt.show()
