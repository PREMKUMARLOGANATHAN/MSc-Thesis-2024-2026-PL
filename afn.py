import manifold_visualisation as mv
import numpy as np
from scipy.spatial import KDTree

def nbrs(x, metric, wind, max_neigh):

    tree = KDTree(x)
    n = len(x)

    d = np.zeros(n)
    ix = np.zeros(n, dtype=int)

    k_neighbors = min(n, 2 * wind + 2)
    distances, indices = tree.query(x, k = k_neighbors, p = 2, workers = 3)

    for i in range(n):
        valid = (np.abs(indices[i] - i) > wind) & (distances[i] > 0)
        
        if np.any(valid):
            d[i] = distances[i][valid][0]
            ix[i] = indices[i][valid][0]
        else:
            d[i] = np.inf 
            ix[i] = i 

    return d, ix

def euclidean_dist(x, y):
    
    result = np.sqrt(np.array(list(map(np.sum, (x - y) ** 2))))
    return result

def afn(x, e, tau, metric, wind, max_neigh = None):
   
    if max_neigh is None:
        max_neigh = e + 1

    M1 = mv.build_shadow(x[:-tau], e, tau)
    M2 = mv.build_shadow(x, e + 1, tau)

    dist, indx = nbrs(M1, metric, wind, max_neigh)
    # valid = indx >= 0

    E = euclidean_dist(M2, M2[indx]) / dist
    Es = np.abs(M2[:, -1] - M2[indx, -1])

    return np.mean(E), np.mean(Es)