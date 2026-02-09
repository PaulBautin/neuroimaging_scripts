import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc
import nibabel as nib
import glob

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation


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
    print(df_label)

    func = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-01/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii')
    data = np.vstack([nib.load(f).darrays[0].data for f in func[:3]])
    data_schaefer = reduce_by_labels(data, yeo_surf, red_op='mean', axis=0)
    data_cortex = data_schaefer[:,df_label['network'] != 'medial_wall']
    data_salience = data_schaefer[:,df_label['network'] == 'SalVentAttn']
    print(data_schaefer.shape)

    T_t = glhmm.auxiliary.make_indices_from_T(np.array([data.shape[0]/3, data.shape[0]/3, data.shape[0]/3]))
    data_preprocessed, indices_preprocessed, log = preproc.preprocess_data(data_cortex, indices=T_t, pca=10)
    data_preprocessed_sal, indices_preprocessed_sal, log_sal = preproc.preprocess_data(data_salience, indices=T_t, pca=5)

    hmm = glhmm.glhmm(K=4, covtype='full', model_mean='state', model_beta='state', preproclogX=log_sal, preproclogY=log)
    hmm.train(X=data_preprocessed_sal, Y=data_preprocessed, indices=indices_preprocessed)
    print(hmm)


    K = hmm.hyperparameters["K"] # the number of states
    q = data.shape[1] # the number of parcels/channels
    state_means = np.zeros(shape=(q, K))
    state_means = hmm.get_means() # the state means in the shape (no. features, no. states)
    
    df_label.loc[df_label['network'] != 'medial_wall', 'state_means_0'] = state_means[:,0]
    df_label.loc[df_label['network'] != 'medial_wall', 'state_means_1'] = state_means[:,1]
    df_label.loc[df_label['network'] != 'medial_wall', 'state_means_2'] = state_means[:,2]
    df_label.loc[df_label['network'] != 'medial_wall', 'state_means_3'] = state_means[:,3]
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'state_means_0', 'state_means_1', 'state_means_2', 'state_means_3']], on='mics', validate="many_to_one", how='left')
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['state_means_0'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['state_means_1'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['state_means_2'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['state_means_3'].values, size=(1200, 300), zoom=1.3, color_bar='bottom', share='both',
                        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)



    cmap = "coolwarm"
    plt.imshow(state_means, cmap=cmap,interpolation="none")
    plt.colorbar(label='Activation Level') # Label for color bar
    plt.title("State mean activation")
    plt.xticks(np.arange(K), np.arange(1,K+1))
    plt.gca().set_xlabel('State')
    plt.gca().set_ylabel('Brain region')
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()





if __name__ == "__main__":
    main()