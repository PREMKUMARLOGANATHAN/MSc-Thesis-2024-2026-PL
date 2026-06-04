import numpy as np
import pandas as pd

def build_shadow(X, E, tau):
    ''' This function is used to build the shadow manifold of a time series data.
  Use the created shadow manifold for visulisaition purposes or further processing.
  Input Parameters:

  X - 1D time series data
  tau - delay
  E - Embedding dimension

  '''
    X = np.asarray(X)
    T = len(X)

    N = T - (E - 1) * tau
    M = np.zeros((N,E))

    for i in range(E):
      M[:, i] = X[((E - 1) - i) * tau : (E - 1 - i) * tau + N]

    return M

def shadow_train_maker(X, E, tau, Tp):

   X = np.asarray(X)
   T = len(X)
   N = T - (E - 1) * tau

   i = np.asarray(np.arange(0, N))[:, None]
   j = np.asarray(np.arange(E))[None, :]

   idx = i + (E - 1) * tau -j * tau

   shadow_M = X[idx]

   pred_M = np.full(idx.shape, np.nan)
   pred_mask = (idx + Tp) < T
   pred_M[pred_mask] = X[(idx + Tp)[pred_mask]]
   pred_M = pd.DataFrame(pred_M).dropna()
   pred_M = np.asarray(pred_M)

   return shadow_M, pred_M

def sampler(M, sample):
   
   idx = np.random.randint(0, M.shape[0], size = sample)
   new_M = M[idx, :]

   return new_M, idx