import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import distance
import nibabel as nib


# Notations
# W: weight matrix
# h: external field
# n: number of variables
# k: number of observations
# m: number of all states, 2**n

def load_testdata(n=1):
  data_file_name = f'test_data/testdata_{n}.dat'
  roi_file_name = 'test_data/roiname.dat'
  X = pd.read_table(data_file_name, header=None)
  X.index = pd.read_csv(roi_file_name, header=None).squeeze()
  return (X==1).astype(int)

def binarize(X):
  return ((X.T - X.T.mean()).T >=0).astype(int)

def calc_state_no(X):
  return X.astype(str).sum().apply(lambda x:int(x,base=2))

def gen_all_state(X_in):
    n = len(X_in)
    X =  np.array([list(bin(i)[2:].rjust(n,'0'))
                for i in range(2**n)]).astype(int).T
    return pd.DataFrame(X, index=X_in.index)

def calc_energy(h, W, X_in):
  X = 2*X_in-1
  return -0.5 * (X * W.dot(X)).sum() - h.dot(X)

def calc_prob(h, W, X):
  energy  = calc_energy(h, W, X)
  energy -= energy.min()  # avoid overflow
  prob    = np.exp(-energy)
  return prob / prob.sum()

# pseudo-likelihood
def fit_approx(X_in, max_iter=10**3, alpha=0.9):
  X      = 2*X_in-1
  n, k   = X.shape
  h      = np.zeros(n)
  W      = np.zeros((n,n))
  X_mean = X.mean(axis=1)
  X_corr = X.dot(X.T) / k
  np.fill_diagonal(X_corr.values, 0)
  for i in range(max_iter):
    Y  = np.tanh(W.dot(X).T + h) # k * n
    h += alpha * (X_mean - Y.mean(axis=0))
    Z  = X.dot(Y) / k
    Z.columns = Z.index
    Z  = (Z + Z.T) / 2
    np.fill_diagonal(Z.values, 0)
    W += alpha * (X_corr - Z)
    if np.allclose(X_mean, Y.mean(axis=0)) and np.allclose(X_corr, Z):
      break
  return h, W

# likelihood
def fit_exact(X_in, max_iter=10**4, alpha=0.5):
  X      = 2*X_in-1
  X_all  = gen_all_state(X)
  X2_all = 2*X_all-1
  n, k   = X.shape
  m      = 2**n
  h      = np.zeros(n)
  W      = np.zeros((n,n))
  X_mean = X.mean(axis=1)
  X_corr = X.dot(X.T) / k
  np.fill_diagonal(X_corr.values, 0)
  for i in range(max_iter):
    p      = calc_prob(h, W, X_all)
    Y_mean = X2_all.dot(p)
    Y_corr = X2_all.dot(np.diag(p)).dot(X2_all.T)
    np.fill_diagonal(Y_corr.values, 0)
    h += alpha * (X_mean - Y_mean)
    W += alpha * (X_corr - Y_corr)
    if np.allclose(X_mean, Y_mean) and np.allclose(X_corr, Y_corr):
      break
  return h, W

def calc_accuracy(h, W, X):
  freq  = calc_state_no(X).value_counts()
  p_n   = freq / freq.sum()
  q     = X.mean(axis=1)
  X_all = gen_all_state(X)
  p_1   = (X_all.T * q + (1-X_all).T * (1-q)).T.prod()
  p_2   = calc_prob(h, W, X_all)
  def entropy(p):
    return (-p * np.log2(p)).sum()
  acc1 = (entropy(p_1) - entropy(p_2)) / (entropy(p_1) - entropy(p_n))
  d1   = (p_n * np.log2(p_n / p_1.iloc[p_n.index])).sum()
  d2   = (p_n * np.log2(p_n / p_2.iloc[p_n.index])).sum()
  acc2  = (d1-d2)/d1
  return acc1, acc2

def calc_adjacent(X):
  X_all = gen_all_state(X)
  out_list = [calc_state_no(X_all)]
  for i in X_all.index:
    Y = X_all.copy()
    Y.loc[i] = 1 - Y.loc[i]
    out_list.append(calc_state_no(Y))
  return pd.concat(out_list, axis=1)

def calc_basin_graph(h, W, X):
  X_all = gen_all_state(X)
  A = calc_adjacent(X)
  energy = calc_energy(h, W, X_all)
  min_idx = energy.values[A].argmin(axis=1)
  graph = pd.DataFrame()
  graph['source'] = A.index.values
  graph['target'] = A.values[A.index, min_idx]
  graph['energy'] = energy
  G = nx.from_pandas_edgelist(graph, create_using=nx.DiGraph)
  graph['state_no'] = None
  conn = sorted(nx.weakly_connected_components(G), key=len)[::-1]
  for i, node_set in enumerate(conn):
    graph.loc[list(node_set),'state_no'] = i + 1
  return graph

def calc_trans(X, graph):
  sr = graph.loc[calc_state_no(X)].state_no
  freq = sr.value_counts().sort_index()
  freq.name = 'freq'
  sr = sr[sr.diff()!=0]  # change points only
  trans = pd.crosstab(sr.values[:-1], sr.values[1:])
  trans.index.name ='src'
  trans.columns.name ='dst'
  out_list = []
  for i in freq.index:
    for j in freq.index:
      sr2 = sr[sr.isin([i,j])]
      sr2 = sr2[sr2.diff()!=0]
      count = int(sr2.size / 2)
      out_list.append(dict(src=i, dst=j, count=count))
  trans2 = pd.DataFrame(out_list)  # includes indirect transitions
  trans2 = trans2.set_index(['src','dst'])['count'].unstack()
  np.fill_diagonal(trans2.values, 0)
  return freq, trans, trans2

def calc_discon_graph_sub(i_input, H, A):
  m, n = A.shape
  C = np.inf * np.ones(m)
  C[i_input] = H[i_input]
  I = set(range(m))
  while I:
    I_list = list(I)
    i = I_list[np.argmin(C[I_list])]
    I.remove(i)
    for j in A[i]:
      if j in I:
        if C[i] <= H[j]:
          C[j] = H[j]
        else:
          C[j] = min(C[i], C[j])
  return C

def calc_discon_graph(h, W, X, graph):
  X_all = gen_all_state(X)
  A = calc_adjacent(X).values[:, 1:]  # remove self-loop
  H = calc_energy(h, W, X_all).values
  df = graph[graph.source==graph.target]
  local_idx = df.index
  out_list = []
  for i_input in local_idx:
    C = calc_discon_graph_sub(i_input, H, A)
    out_list.append(C[local_idx])
  D = pd.DataFrame(np.array(out_list), index=df.state_no,
                   columns=df.state_no)
  D = D.sort_index().sort_index(axis=1)
  return D

def uniform_layout(G, alpha=0.1, n_iter=None, seed=None, **kwargs):
  pos = nx.spring_layout(G, seed=seed, **kwargs)
  X = np.array(list(pos.values()))
  if n_iter == None:
    n_iter = 10 * len(G)
  for _ in range(n_iter):
    D = distance.squareform(distance.pdist(X))
    np.fill_diagonal(D, None)
    X += alpha * (X - X[np.nanargmin(D, axis=0)])
    X = X.clip(-1,1)
  return dict(zip(pos.keys(), X))

def calc_trans_bm(h, W, X):
  X_all  = gen_all_state(X)
  X2_all = 2 * X_all - 1
  Y      = W.dot(X2_all).T + h
  Q      = 1/(1+np.exp(-Y))
  out_list = []
  for _, q in Q.iterrows():
    p = (X_all.T * q + (1-X_all).T * (1-q)).T.prod()
    out_list.append(p)
  P = pd.concat(out_list, axis=1)  # index: dst, col: src
  return P

def calc_depth_threshold(n, k, n_repeat=100, method='exact', alpha=0.05):
  np.random.seed(12345)
  out_list = []
  for _ in range(n_repeat):
    X = pd.DataFrame(np.random.randint(2, size=(n,k)))
    if method == 'exact':
      h, W = fit_exact(X)
    else:
      h, W = fit_approx(X)
    graph = calc_basin_graph(h, W, X)
    D = calc_discon_graph(h, W, X, graph)
    D -= D.values.diagonal()
    np.fill_diagonal(D.values, None)
    out_list.append(D.min().values)
  return pd.Series(np.concatenate(out_list)).quantile(1-alpha)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

def plot_local_min(data, graph):
  df = graph[graph.source == graph.target]
  n = len(data)
  X =  np.array([list(bin(i)[2:].rjust(n,'0'))
                 for i in df.index]).astype(int).T
  df = pd.DataFrame(X, index=data.index, columns=df.state_no)
  df = df.sort_index(axis=1)

  fig, ax = plt.subplots(figsize=(4,4))
  sns.heatmap(data=df, ax=ax, linecolor='w', lw=2, square=True,
              cmap=sns.color_palette('Paired', 2),
              cbar_kws=dict(ticks=[0.25,0.75], shrink=0.25, aspect=2))
  ax.tick_params(length=0)
  ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
  ax.set_title('Local minimum patterns', fontsize=16, pad=10)
  ax.set_xlabel('State number')
  ax.set_ylabel(None)
  cax = ax.collections[0].colorbar.ax
  cax.set_yticklabels(['0','1'])
  cax.tick_params(length=0)
  fig.tight_layout()
  fig.show()

def main():
    from brainspace.utils.parcellation import map_to_labels, reduce_by_labels
    from brainspace.plotting import plot_hemispheres
    from brainspace.datasets import load_conte69
    # Load the data
    surf_lh, surf_rh = load_conte69()
    atlas_yeo_lh = nib.load('/home/pabaua/dev_mni/micapipe/parcellations/schaefer-100_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load('/home/pabaua/dev_mni/micapipe/parcellations/schaefer-100_conte69_rh.label.gii').darrays[0].data + 1950
    atlas_yeo_rh[atlas_yeo_rh == 1950] = 2000
    yeo_surf = np.hstack(np.concatenate((atlas_yeo_lh, atlas_yeo_rh), axis=0)).astype(float)
    print(np.unique(yeo_surf))
    df_func = nib.load('/home/pabaua/Downloads/sub-PNC001_ses-01_03/sub-PNC001/ses-01/func/desc-me_task-rest_bold/surf/sub-PNC001_ses-01_surf-fsLR-32k_desc-timeseries_clean.shape.gii').darrays[0].data
    func_400 = pd.DataFrame(reduce_by_labels(df_func, yeo_surf, red_op='mean', axis=0).T)
    df_label = pd.read_csv('/home/pabaua/dev_mni/micapipe/parcellations/lut/lut_schaefer-100_mics.csv')
    df_label = pd.concat([df_label.label, func_400], axis=1)
    df_label = df_label[df_label.label != 'medial_wall'].set_index('label')
    data = binarize(df_label)
    # Fit an Ising model to the data.
    h, W = fit_approx(data)
    print(h)
    # Check fitting accuracy scores (1: best, 0: worst).
    #acc1, acc2 = calc_accuracy(h, W, data)
    #print('accuracy:', acc1, acc2)
    # Calculate a basin graph.
    graph = calc_basin_graph(h, W, data)
    plot_local_min(data, graph)
    # Calculate a disconnectivity graph.
    D = calc_discon_graph(h, W, data, graph)
    # Calculate each state's frequency and transitions between states.
    freq, trans, trans2 = calc_trans(data, graph)


if __name__ == "__main__":
    main()