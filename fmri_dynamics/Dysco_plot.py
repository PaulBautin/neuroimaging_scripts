import os
import numpy as np
import nibabel as nb
from scipy.io import loadmat
import random
import nibabel as nib

import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import combinations

from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.datasets import load_gradient, load_marker, load_conte69
from brainspace.gradient import GradientMaps, kernels
from scipy import stats




current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
project_dir = '/home/pabaua/dev_mni/DySCo'
classy_functions_path = os.path.join(project_dir, 'Python', 'core_functions', 'classy')
sys.path.append(classy_functions_path)
core_functions_path = os.path.join(project_dir, 'Python', 'core_functions')
sys.path.append(core_functions_path)
print(project_dir)

from CLS_compute_eigs_all import Compute_Eigs
from CLS_dysco_distance import Dysco_distance
from CLS_dysco_reconf import Dysco_reconf_distance
from CLS_dysco_norm import Dysco_Norm

from compute_eigenvectors_sliding_cov import compute_eigs_cov
from dysco_distance import dysco_distance
from dysco_mode_alignment import dysco_mode_alignment
from dysco_norm import dysco_norm
from fMRI_Processing.surf_cifti_data import surf_data_from_cifti


#### load the conte69 hemisphere surfaces and spheres
surf_lh, surf_rh = load_conte69()


# RUN this cell to load the saved .npy file
brain = np.load('/home/pabaua/dev_mni/DySCo/Python/Pipelines/Test_brain/Test_Brain_1.npy')
brain = nib.load('/home/pabaua/Downloads/sub-PNC001_ses-01_03/sub-PNC001/ses-01/func/desc-me_task-rest_bold/surf/sub-PNC001_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
plot_hemispheres(surf_lh, surf_rh, array_name=brain[2,:], size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                nan_color=(250, 250, 250, 1), cmap='inferno', transparent_bg=True)


half_window_size = 10
n_eigen = 10
eigen_path = '/home/pabaua/dev_mni/DySCo/Python/Pipelines/Test_brain/eigen_data_mica.npz'

if os.path.exists(eigen_path):
    data = np.load(eigen_path)
    eigenvectors, eigenvalues = data['eigenvectors'], data['eigenvalues']
    print("Loaded saved eigenvectors and eigenvalues.")
else:
    print("Computing eigenvectors and eigenvalues...")
    eigenvectors, eigenvalues = compute_eigs_cov(brain, n_eigen, half_window_size)
    np.savez(eigen_path, eigenvectors=eigenvectors, eigenvalues=eigenvalues)
    print("Saved computed eigenvectors and eigenvalues.")

#### recurrence matrix EVD
plot_hemispheres(surf_lh, surf_rh, array_name=eigenvectors[2,:,0], size=(1200, 900), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                nan_color=(250, 250, 250, 1), cmap='inferno', transparent_bg=True)
plot_hemispheres(surf_lh, surf_rh, array_name=eigenvectors[2,:,1], size=(1200, 900), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                nan_color=(250, 250, 250, 1), cmap='inferno', transparent_bg=True)
print(eigenvectors.shape)
plt.imshow(eigenvalues)
plt.show()



#### Norm
norm2 = dysco_norm(eigenvalues, 2)
metastability = np.std(norm2)
print(metastability)
plt.plot(norm2)
plt.show()

#### Entropy
entropy = eigenvalues / np.tile(np.sum(eigenvalues, axis=0), (n_eigen, 1))
entropy = -np.sum(np.log(entropy) * entropy, axis=0)
plt.plot(entropy)
plt.show()


#### Distance
T = eigenvectors.shape[0]
fcd = np.zeros((T, T))

for i in range(T):
    for j in range(i + 1, T):
        fcd[i, j] = dysco_distance(eigenvectors[i, :, :], eigenvectors[j, :, :], 2)
        fcd[j, i] = fcd[i, j]

plt.imshow(fcd)
plt.show()

#### Reconfiguration speed
lag = 20
speed = np.zeros(T - lag)
for i in range(T - lag):
    speed[i] = fcd[i, i + lag]
    
plt.plot(speed)
plt.show()










