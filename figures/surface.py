# Set the environment
import os
import glob
import numpy as np
import nibabel as nib
import seaborn as sns
import vtk
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from IPython.display import Image, display
import matplotlib.pyplot as plt


def load_qmri(qmri='', surf='fsLR-32k'):
    '''
    This function loads the qMRI intensity maps from midthickness surface
    '''
    # List the files
    files_lh = sorted(glob.glob(f"{dir_maps}/*_hemi-L_surf-{surf}_label-midthickness_{qmri}.func.gii"))
    files_rh = sorted(glob.glob(f"{dir_maps}/*_hemi-R_surf-{surf}_label-midthickness_{qmri}.func.gii"))

    # Load map data
    surf_map=np.concatenate((nib.load(files_lh[0]).darrays[0].data, nib.load(files_rh[0]).darrays[0].data), axis=0)

    return(surf_map)

def plot_qmri(qmri='',  surf='fsLR-32k', label='pial', cmap='rocket', rq=(0.15, 0.95)):
    '''
    This function plots the qMRI intensity maps on the pial surface
    '''
    # Load the data
    map_surf = load_qmri(qmri, surf)
    print('Number of vertices: ' + str(map_surf.shape[0]))

    # Load the surfaces
    surf_lh=read_surface(f'{dir_surf}/{subjectID}_hemi-L_space-nativepro_surf-{surf}_label-{label}.surf.gii', itype='gii')
    surf_rh=read_surface(f'{dir_surf}/{subjectID}_hemi-R_space-nativepro_surf-{surf}_label-{label}.surf.gii', itype='gii')

    # Color range based in the quantiles
    crange=(np.quantile(map_surf, rq[0]), np.quantile(map_surf, rq[1]))

    # Plot the group T1map intensitites
    plot_hemispheres(surf_lh, surf_rh, array_name=map_surf, size=(900, 250), color_bar='bottom', zoom=1.25, share='both',
                     nan_color=(0, 0, 0, 1), cmap=cmap, color_range=crange, transparent_bg=False, screenshot = False)



# This variable will be different for each subject
subjectID='sub-Pilot014_ses-02'
subjectDir='/local_raid/data/pbautin/results/micapipe/micapipe_v0.2.0/sub-Pilot014/ses-02'

# Set paths and variables
dir_FS = 'freesurfer/' + subjectID
dir_surf = subjectDir + '/surf/'
dir_maps = subjectDir + '/maps/'

# # Load native pial surface
# pial_lh = read_surface(dir_FS+'/surf/lh.pial', itype='fs')
# pial_rh = read_surface(dir_FS+'/surf/rh.pial', itype='fs')
#
# # Load native white matter surface
# wm_lh = read_surface(dir_FS+'/surf/lh.white', itype='fs')
# wm_rh = read_surface(dir_FS+'/surf/rh.white', itype='fs')
#
# # Load native inflated surface
# inf_lh = read_surface(dir_FS+'/surf/lh.inflated', itype='fs')
# inf_rh = read_surface(dir_FS+'/surf/rh.inflated', itype='fs')
#
# # Load fsaverage5
# fs5_lh = read_surface('freesurfer/fsaverage5/surf/lh.pial', itype='fs')
# fs5_rh = read_surface('freesurfer/fsaverage5/surf/rh.pial', itype='fs')
#
# # Load fsaverage5 inflated
# fs5_inf_lh = read_surface('freesurfer/fsaverage5/surf/lh.inflated', itype='fs')
# fs5_inf_rh = read_surface('freesurfer/fsaverage5/surf/rh.inflated', itype='fs')

# Load fsLR 32k
print("ready to load data")
micapipe='/local_raid/data/pbautin/software/micapipe'
f32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.surf.gii', itype='gii')
f32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.surf.gii', itype='gii')
print("data was loaded")

# # Load fsLR 32k inflated
# f32k_inf_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
# f32k_inf_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
#
# # Load Load fsLR 5k
# f5k_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.surf.gii', itype='gii')
# f5k_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.surf.gii', itype='gii')
#
# # Load fsLR 5k inflated
# f5k_inf_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
# f5k_inf_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')

# Plot of T1map on fsLR-32k

plot_qmri('T1map', 'fsLR-32k')
