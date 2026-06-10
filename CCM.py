# Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import manifold_visualisation as mv
from scipy.spatial import KDTree

# Inspired by Prince Xavier CCM github, pyEDM and telusko youtube channel,
# Written by Prem Kumar Loganathan, for the thesis on 'Impact of Climate Change on AntiMicrobial Resistane'
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
        # this 'tree' will be used to calculate the distance using query.

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
