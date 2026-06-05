"""
This file is written by Prem Kumar Loganathan for the MSc Thesis.

"""

import CCM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def ccm_result(func, X, Y, L_range, tau_y, tau_x, E_y, E_x, L1 = 'X causes Y', L2 = 'Y causes X', N = 100):
    
    '''
    
    Function Usage: 
        This function is used to do bootstrapped convergent cross mapping as per 'Detecting Causality in Complex Ecosystems'
        Here the results will not be truncated to [0, 1], but the idea is to make sense why it is negative.
        
        By default this function sees both direction. So please give labels mannually to avoid the predefined labels by author.
        
        Happy Learning!
        
    Parameters:
        
        func: CCM.CCM => Convergent cross mapping function created, Refer github
        X: The causal variable
        Y: The effect variable 
        L_range: The range of library size to check upon
        tau_y, E_y : The state space reconstruction parameters for Y
        tau_x, E_x : The state space reconstruction parameters for X
        L1 : Label for 1st direction of causality testing
        L2 : Label for 2nd direction of causality testing
        N : No of bootstapping sample
        
    '''
    
    Xhat_My_all = np.zeros((len(L_range), N))
    Yhat_Mx_all = np.zeros((len(L_range), N))
    
    n = len(X)
    
    for idx, L in tqdm(enumerate(L_range)):
        for b in range(N):
    
            start_ind = np.random.randint(0, n - L)
            idxs = np.arange(start_ind, start_ind + L)
    
            X_boot = X[idxs]
            Y_boot = Y[idxs]
    
            ccm_XY = func(X_boot, Y_boot, E_y, tau_y, L) # see if X causes Y
            ccm_YX = func(Y_boot, X_boot, E_x, tau_x, L) # see if Y causes X
    
            r_xy = ccm_XY.causality()[0][1]
            r_yx = ccm_YX.causality()[0][1]
    
            Xhat_My_all[idx, b] = r_xy
            Yhat_Mx_all[idx, b] = r_yx
            
    fig, ax = plt.subplots(1, 1, figsize = (7, 6))

    bp1 = ax.boxplot(Xhat_My_all.T, positions = L_range, showmeans = True, showfliers = False, 
                     meanline = False, meanprops = dict(marker = None),
                     boxprops = dict(color = 'blue'), whiskerprops = dict(color = 'blue'),
                     capprops = dict(color = 'blue'), medianprops = dict(lw = 0))
    bp2 = ax.boxplot(Yhat_Mx_all.T, positions = L_range, showmeans = True, showfliers = False, 
                     meanline = False, meanprops = dict(marker = None),
                     boxprops = dict(color = 'red'), whiskerprops = dict(color = 'red'),
                     capprops = dict(color = 'red'), medianprops = dict(lw = 0))
    
    ax.plot(L_range, Xhat_My_all.mean(axis = 1), color = 'blue', lw = 1.2, label = L1)
    ax.plot(L_range, Yhat_Mx_all.mean(axis = 1), color = 'red', lw = 1.2, label = L2)
    
    ax.set_xlabel('Library Size [L]')
    ax.set_ylabel(r'$CCM\ Skill\ (\rho)$')            
    ax.legend(loc = 'upper right')
    ax.grid('--', color = 'grey', alpha = 0.4)
    ax.set_title('Convergent Cross Mapping Skill as \na function of Library Size', loc = 'left')
            
    return ax
