# Load required packages
import os
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib as plt

# Set the working directory to the 'out' directory
# os.chdir("/local_raid/data/pbautin/results/micapipe/tmp") # <<<<<<<<<<<< CHANGE THIS PATH TO YOUR OUT DIRECTORY

# This variable will be different for each subject
sub='HC007'           # <<<<<<<<<<<< CHANGE THIS SUBJECT's ID
ses='01'              # <<<<<<<<<<<< CHANGE THIS SESSION
subjectID=f'sub-{sub}_ses-{ses}'
subjectDir=f'/local_raid/data/pbautin/data/3t_data/micapipe_v0.2.0/sub-{sub}/ses-{ses}'
tracts = '5M'

# Here we define the atlas
atlas='schaefer-100'

plt.style.use('dark_background')

# Set the path to the the structural cortical connectome
filter='COMMIT2'
cnt_sc_cor_commit_two = f'{subjectDir}/dwi/connectomes/{subjectID}_space-dwi_atlas-{atlas}_desc-iFOD2-{tracts}-{filter}_full-connectome.shape.gii'
mtx_sc_commit_two = nib.load(cnt_sc_cor_commit_two).darrays[0].data
mtx_scSym_commit_two = np.triu(mtx_sc_commit_two,1)+mtx_sc_commit_two.T > 0
corr_plot = plotting.plot_matrix(np.log(mtx_scSym_commit_two + 1), figure=(10, 10), labels=None, cmap='Purples') # , vmin=-10, vmax=2
plotting.show()

# Set the path to the the structural cortical connectome
filter='COMMIT1'
cnt_sc_cor_commit_one = f'{subjectDir}/dwi/connectomes/{subjectID}_space-dwi_atlas-{atlas}_desc-iFOD2-{tracts}-{filter}_full-connectome.shape.gii'
mtx_sc_commit_one = nib.load(cnt_sc_cor_commit_one).darrays[0].data
mtx_scSym_commit_one = np.triu(mtx_sc_commit_one,1)+mtx_sc_commit_one.T > 0
corr_plot = plotting.plot_matrix(np.log(mtx_scSym_commit_one + 1), figure=(10, 10), labels=None, cmap='Purples') # , vmin=-10, vmax=2
plotting.show()

# Set the path to the the structural cortical connectome
corr_plot = plotting.plot_matrix(np.log(mtx_scSym_commit_two + 1)-np.log(mtx_scSym_commit_one + 1), figure=(10, 10), labels=None, cmap='Purples') # , vmin=-10, vmax=2
plotting.show()

# Set the path to the the structural cortical connectome
filter='SIFT2'
cnt_sc_cor = f'{subjectDir}/dwi/connectomes/{subjectID}_space-dwi_atlas-{atlas}_desc-iFOD2-{tracts}-{filter}_full-connectome.shape.gii'
mtx_sc = nib.load(cnt_sc_cor).darrays[0].data
mtx_scSym = np.triu(mtx_sc,1)+mtx_sc.T
corr_plot = plotting.plot_matrix(np.log(mtx_scSym + 1), figure=(10, 10), labels=None, cmap='Purples') # , vmin=-10, vmax=2
plotting.show()

# Set the path to the the structural cortical connectome
cnt_sc_EL = cnt_sc_cor= f'{subjectDir}/dwi/connectomes/{subjectID}_space-dwi_atlas-{atlas}_desc-iFOD2-{tracts}-{filter}_full-edgeLengths.shape.gii'
mtx_scEL = nib.load(cnt_sc_EL).darrays[0].data
mtx_scELSym = np.triu(mtx_scEL,1)+mtx_scEL.T
corr_plot = plotting.plot_matrix(mtx_scELSym, figure=(10, 10), labels=None, cmap='Purples', vmin=0, vmax=200)
plotting.show()
