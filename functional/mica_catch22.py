import pycatch22 as catch22
import os
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69

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

# generate list where each entry is a length 1000 time series
#dataset = [np.random.randn(1000) for _ in range(50000)]
# Generate sample time series
surf_lh, surf_rh = load_conte69()
atlas_yeo_lh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
atlas_yeo_rh = nib.load('/local_raid/data/pbautin/software/micapipe/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 2000
atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
func_32k = nib.load('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC032/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC032_ses-a1_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
func_400 = reduce_by_labels(func_32k, yeo_surf, red_op='mean', axis=0).T


def compute_features(x):
    res = catch22.catch22_all(x, catch24 = True)
    return res['values'] # just return the values 

print(f"Number of cores available: {os.cpu_count()}")

threads_to_use = os.cpu_count()
feature_rad = np.asarray(Parallel(n_jobs=threads_to_use)(
    delayed(RAD)(func_400[i]) for i in range(len(func_400))))



print(feature_rad.shape)
results_list = Parallel(n_jobs=threads_to_use)(
    delayed(compute_features)(func_400[i]) for i in range(len(func_400))
)
results = np.asarray(results_list)

# print the results for the first time series
feature_index = np.argwhere(np.asarray(catch22.catch22_all(func_400[0], catch24 = True)['names']) == 'SP_Summaries_welch_rect_centroid')[0][0]
plot_values = map_to_labels(results[:,feature_index], yeo_surf)
plot_values = map_to_labels(feature_rad, yeo_surf)
plot_hemispheres(surf_lh, surf_rh, array_name=plot_values, size=(1200, 300), zoom=1.25, color_bar='bottom', share='both', background=(0,0,0),
                             nan_color=(250, 250, 250, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')