# Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import manifold_visualisation as mv
from scipy.spatial import KDTree

# Inspired by Prince Xavier CCM github, pyEDM and telusko youtube channel,
# Written by Prem Kumar Loganathan, for the thesis  
# The class always is in the format of class_func(target, columns, params)
# 
# Here and in pyEDM target - causal and column - effect 

# So, here the class_func(X, Y, params): goes by to test if X causes Y

class CCM:

    def __init__(self, X, Y, E, tau, L):

        self.X = X
        self.Y = Y
        self.E = E
        self.tau = tau
        self.L = L
        self.My = mv.build_shadow(self.Y[:self.L], self.E, self.tau)
        self.steps, self.tree = self.distance(self.My)

    def distance(self, M):
        
        # KDTree only returns a class variable (tree) - 
        # this will be used to calculate the distance using query.

        steps = np.arange((self.E - 1) * self.tau, self.L)
        tree = KDTree(M)

        return steps, tree

    def nearby_distance(self, t, steps, tree):

        time_ind = t - (self.E - 1) * self.tau
        m_ind = tree.data[time_ind]
        dist, indx = tree.query(m_ind, k = self.E + 1)
        mask = steps[indx] != t
        nearby_index = indx[mask][:self.E + 1]
        nearby_dist = dist[mask][:self.E + 1]
        original_steps = steps[nearby_index]

        return nearby_dist, original_steps
    
    def prediction(self, t, tp = 0):

        nearby_dist, original_steps = self.nearby_distance(t, self.steps, self.tree)
        u = np.exp(-nearby_dist / np.max([1e-6, nearby_dist[0]]))
        w = u /np.sum(u)

        target_time = t + tp
        neighbor_time = original_steps + tp

        valid_mask = (neighbor_time >= 0) & (neighbor_time < len(self.X))
        neighbor_time = neighbor_time[valid_mask]
        w = w[valid_mask]

        if len(neighbor_time) == 0:
            return None, None

        X_true = self.X[target_time]
        X_cor = np.array(self.X)[neighbor_time]
        X_hat = (w * X_cor).sum()

        return X_true, X_hat
    
    def causality(self, tp = 0):

        X_true_list = []
        X_hat_list = []

        for t in self.steps:
            
            if t + tp < 0 or t + tp >= len(self.X):
                continue

            X_true, X_hat = self.prediction(t, tp = tp)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)

        r, p = np.corrcoef(X_true_list, X_hat_list)

        return r, p
    
    def plot_ccm_correlation(self):

        X_true_list = []
        X_hat_list = []

        for t in self.steps:
            X_true, X_hat = self.prediction(t, tp = 0)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)

        r, p = np.corrcoef(X_true_list, X_hat_list)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(X_true_list, X_hat_list, s = 7.5)
        ax.set_xlabel('X(t) [Observed]')
        ax.set_ylabel(r'\hat{X}(t) [Predicted]')
        ax.set_title(f'Correlation Coefficient = {r[1]}') 
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(X_true_list, c = 'k', lw = 1.5, alpha = 0.3, label = 'Observed')
        ax.plot(X_hat_list, c = 'blue', lw = 1.5, alpha = 0.4, label = 'Predicted')
        ax.legend(loc = 'upper right')
        ax.grid('--', c = 'grey', alpha = 0.3)
        ax.set_xlabel('Time [Months]')
        ax.set_ylabel('Original and Predicted Values')
        ax.set_xmargin(0)
        plt.show()

    def cross_map_visualiser(self):

        Mx = mv.build_shadow(self.X[:self.L], self.E, self.tau)
        My = mv.build_shadow(self.Y[:self.L], self.E, self.tau)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

        ax1.scatter(Mx[:, 0], Mx[:, 1], s = 7, c = "#0D39EC6D")
        ax2.scatter(My[:, 0], My[:, 1], s = 7, c = "#FE0000")

        steps, tree = self.distance(Mx)
        t = np.random.randint(len(Mx)//2)
        
        nearby_dist, original_steps = self.nearby_distance(t, steps, tree)
        base_row = t - (self.E - 1) * self.tau
        
        ax1.scatter(Mx[base_row, 0], Mx[base_row, 1], c = 'k', s = 50, label = 'Centroid')
        ax2.scatter(My[base_row, 0], My[base_row, 1], c = 'k', s = 50, label = 'Cross Mapping')

        for j in range(min(3, len(original_steps))):
            row = original_steps[j] - (self.E - 1) * self.tau

            A_t, A_lag = Mx[row, :2]
            B_t, B_lag = My[row, :2]

            ax1.scatter(A_t, A_lag, c = "#01FF1B", s = 15, label = f'Neighbours' if j == 0 else None)
            ax2.scatter(B_t, B_lag, c = '#01FF1B', s = 15, label = f'Mapped Points' if j == 0 else None)

        ax1.set_title(f'Neighbours used for Prediction')
        ax1.set_xlabel('$X_t$')
        ax1.set_ylabel('$X_{t-\\tau}$')
        ax1.legend(loc = 'upper right')

        ax2.set_title(f'Target Manifold Mapping')
        ax2.set_xlabel('$Y_t$')
        ax2.set_ylabel('$Y_{t-\\tau}$')
        ax2.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()