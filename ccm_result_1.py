import matplotlib.pyplot as plt
from tqdm import tqdm

def ccm_result_1(func, X, Y, L_range, tau_y, tau_x, E_y, E_x):
    Xhat_My, Yhat_Mx = [], []
    for L in tqdm(L_range): 
        ccm_XY = func(X, Y, tau_y, E_y, L) # define new ccm object # Testing for X -> Y
        ccm_YX = func(Y, X, tau_x, E_x, L) # define new ccm object # Testing for Y -> X    

        rho_xy = ccm_XY.causality()[0][1]
        rho_yx = ccm_YX.causality()[0][1]

        Xhat_My.append(rho_xy) 
        Yhat_Mx.append(rho_yx) 

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.plot(L_range, Xhat_My, label='AMU causes Temp', linewidth = 1.5)
    ax.plot(L_range, Yhat_Mx, c='r', label='Temp causes AMU', linewidth = 1.5)
    ax.set_xlabel('Time Series Length [L]',)
    ax.set_ylabel(r'$CCM\ Skill\ (\rho)$')
    ax.legend(loc = 'upper right')  
    ax.grid(color = 'grey', alpha = 0.3)
    ax.set_title('Convergent Cross Mapping Skill as \na Function of Library Size', loc = 'left')
    return ax