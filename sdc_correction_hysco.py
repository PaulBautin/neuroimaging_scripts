from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the image and domain information
# change this function call to be the filepath for your data
data = DataObject('/home/pabaua/Downloads/raw_HC001-HC005/sub-HC002/ses-01/dwi/sub-HC002_ses-01_acq-b700-41_dir-AP_firstvol.nii.gz' \
                  , '/home/pabaua/Downloads/raw_HC001-HC005/sub-HC002/ses-01/dwi/sub-HC002_ses-01_dir-PA_firstvol.nii.gz' \
                    , 1, device=device,dtype=torch.float32)

loss_func = EPIMRIDistortionCorrection(data, 300, 1e-4, regularizer=myLaplacian3D, PC = JacobiCG)
# initialize the field map
B0 = loss_func.initialize(blur_result=True)
# set-up the optimizer
# change path to be where you want logfile and corrected images to be stored
opt = GaussNewton(loss_func, max_iter=500, verbose=True, path='results/gnpcg-Jac/')
# optimize!
opt.run_correction(B0)
# save field map and corrected images
opt.apply_correction()
# see plot of corrected images
opt.visualize()