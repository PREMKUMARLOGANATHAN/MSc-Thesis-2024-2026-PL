import numpy as np
import matplotlib.pyplot as plt

def extended_ccm(func, X, Y, L, tau_y, tau_x, E_y, E_x, tp, variable):
    
    ccm_XY = func(X, Y, tau_y, E_y, L) # Checking if X -> Y
    ccm_YX = func(Y, X, tau_x, E_x, L) # Checking if Y -> X
    rho_xy, rho_yx = [], []
    for i in range(-tp, tp+1):
        rho_xy.append(ccm_XY.causality(tp = i)[0][1])
        rho_yx.append(ccm_YX.causality(tp = i)[0][1])

    fig, ax = plt.subplots(1, 1, figsize = (6, 4))
    ax.plot(np.arange(-tp, tp+1), rho_xy, 'o-', color = 'blue', label = f'AMR causes {variable}', lw = 1.2)
    ax.plot(np.arange(-tp, tp+1), rho_yx, '^-', color = 'red', label = f'{variable} causes AMR', lw = 1.2)
    ax.set_xlabel('Time Horizon (Tp)')
    ax.set_ylabel(r'$CCM\ Skill\ (\rho)$')
    ax.legend(loc = 'upper right')  
    ax.grid(color = 'grey', alpha = 0.3)
    ax.set_title('Convergent Cross Mapping Skill as \na Function of Library Size', loc = 'left')
    return ax