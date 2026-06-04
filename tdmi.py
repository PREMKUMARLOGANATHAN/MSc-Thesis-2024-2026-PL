import numpy as np

def mi(x, y, bins):

    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y

def tdmi(x, max_tau, bins):

    N = len(x)
    maxtau = min(N, max_tau)

    ii = np.empty(maxtau)
    ii[0] = mi(x, x, bins)

    for tau in range(1, maxtau):
        ii[tau] = mi(x[:-tau], x[tau:], bins)

    return ii