#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:01:05 2018

Fit a Gaussian mixture model to a capacity trace and generate traces from the trained model

@author: Matteo Gadaleta
"""

import pandas as pd
import numpy as np
import deepdish as dd
from sklearn.mixture import BayesianGaussianMixture 
from scipy.stats import multivariate_normal


class GMM_Channel():
    
    def __init__(self, n_features, n_components):
        """
        n_features: Number of consecutive samples to consider in the joint probability distribution
        n_components: Number of gaussians of the GMM model
        """
        self.n_features = n_features
        self.n_components = n_components
        
        
    def _get_covariances(self, gmm):
        
        # Return all the estimated covariance matrices
        n_components = gmm.get_params()['n_components']
        cv_type      = gmm.get_params()['covariance_type']
        
        if cv_type == 'full':
            return gmm.covariances_
        if cv_type == 'tied':
            cov = gmm.covariances_
            return np.tile(cov,(n_components,1,1))
        if cv_type == 'diag':
            cov_diag = gmm.covariances_
            cov = []
            for diag in cov_diag:            
                cov.append(np.diag(diag))
            return np.array(cov)
        if cv_type == 'spherical':
            cov_spher = gmm.covariances_
            cov = []
            for spher in cov_spher:
                dim = gmm.degrees_of_freedom_prior_
                id_mat = np.identity(dim)
                cov.append(id_mat * spher)
        return np.array(cov)
    
    
    def _create_training_matrix(self, seq, n_feat):
        matrix = [seq[idx-n_feat:idx] for idx in np.arange(n_feat, len(seq))]
        labels = [seq[idx] for idx in np.arange(n_feat, len(seq))]
        return np.array(matrix), np.array(labels)
    
    
    def _create_custom_gmm(self, means, covs, weigths):
    
        n_comps = len(means)
        
        gmm = BayesianGaussianMixture(n_comps)
        gmm.means_ = np.array(means).reshape(-1,1)
        gmm.covariances_ = np.array(covs).reshape(-1,1,1)
        gmm.weights_ = np.array(weigths)
        gmm._check_is_fitted = lambda: True
        
        return gmm
    
    
    def fit(self, sequence):
        
#        # Normalize sequence
#        self.norm_std = np.std(sequence)
#        sequence = sequence / self.norm_std
        
        # Create training matrix
        train_matrix, labels = self._create_training_matrix(sequence, self.n_features)
        
        # Include labels in feature matrix
        train_matrix_extended = np.column_stack((train_matrix, labels))
        N, dims_TOT = np.shape(train_matrix_extended)
    
        # GMM fit
        gmm = BayesianGaussianMixture(n_components=self.n_components, covariance_type='full', max_iter=1000)
        gmm.fit(train_matrix_extended)
    
        # Get gmm_parameters
        alpha = gmm.weights_
        mu = gmm.means_
        sigma = self._get_covariances(gmm)
        n_components = gmm.n_components
            
        self.gmm_parameters = {'n_components': n_components, 'n_features': self.n_features, 'alpha': alpha, 'mu': mu, 'sigma': sigma}
        
    
    def _create_cond_model(self, x):
        """
        See 2006 - Sun - A Bayesian Network Approach  to Traffic Flow Forecasting
        """
        
        gmm_parameters = self.gmm_parameters
        
        gmm_sigma = gmm_parameters['sigma']
        gmm_mu = gmm_parameters['mu']
        gmm_alpha = gmm_parameters['alpha']
        n_components = gmm_parameters['n_components']
        # Analyze each component separately
        cond_means = []
        cond_vars = []
        cond_betas = []
        for comp_num in range(n_components):
            # Get current component
            sigma = gmm_sigma[comp_num]
            mu = gmm_mu[comp_num]
            alpha = gmm_alpha[comp_num]
            # Separate dependent variables (E) and target (F)
            sigma_EE = sigma[:-1, :-1]
            sigma_FE = sigma[-1, :-1].reshape([-1,1])
            sigma_EF = sigma[:-1, -1].reshape([1,-1])
            sigma_FF = sigma[-1, -1]
            mu_E = mu[:-1]
            mu_F = mu[-1]
            # Conditional mean
            sigma_EE_inv = np.linalg.pinv(sigma_EE)
            mu_F_given_E = mu_F - np.matmul(np.matmul(sigma_EF, sigma_EE_inv), (mu_E - x).reshape(-1, 1)).squeeze()
            sigma_F_given_E = sigma_FF - np.matmul(np.matmul(sigma_EF, sigma_EE_inv), sigma_FE).squeeze()
            beta = alpha * np.max([multivariate_normal(mu_E, sigma_EE).pdf(x), 1e-100])
            # Save
            cond_means.append(mu_F_given_E)
            cond_vars.append(sigma_F_given_E)
            cond_betas.append(beta)
        # Normalize betas
        norm_value = np.linalg.norm(cond_betas, ord=1)
        cond_betas = np.exp(np.log(cond_betas) - np.log(norm_value))
        
        # Create generative GMM
        cond_model = self._create_custom_gmm(cond_means, cond_vars, cond_betas)
            
        return cond_model
    
    
    def sample(self, n_samples, random_seed):
                
        # Set random seed
        np.random.seed(random_seed)
        
        # Generate signal
        signal = list(np.random.rand(self.n_features) * 1e7)
        for idx in range(n_samples + self.n_features * 20):
            x = signal[-self.n_features:]
            cond_model = self._create_cond_model(x)
            new_sample = cond_model.sample()[0].squeeze()
            signal.append(np.max([0, new_sample]))
        # Crop the initial part
        signal = signal[self.n_features * 21:]
        
#        # Denormalize sequence
#        signal = signal * self.norm_std
        
        return np.array(signal)
    
    
    def save_model(self, path):
        
        dd.io.save(path, self.gmm_parameters)
        
        
    def load_model(self, path):
        
        gmm_parameters = dd.io.load(path)
        self.gmm_parameters = gmm_parameters
        self.n_components = gmm_parameters['n_components']
        self.n_features = gmm_parameters['n_features']
    
    
    
if __name__ == '__main__':
            
    import matplotlib.pyplot as plt
    from pathlib import Path

    dataset_path = Path('4g_gent_dataset')
    scenarios = ['bicycle', 'bus', 'car', 'foot', 'train', 'tram']
    n_feat = 5
    n_components = 3
    
    for scenario in scenarios:
        print('#############')
        print('## SCENARIO: %s' % scenario)
        # Load data
        filepath_list = [d for d in dataset_path.iterdir() if scenario in d.name]
        # Concatenate data
        capacity_conc = []
        for filepath in filepath_list:
            print(filepath.name)
            # Extract capacity
            single_df = pd.read_csv(filepath, sep=' ', names=['timestamp', 'time', 'lat', 'long', 'bytes', 'period'])
            capacity = np.array(single_df['bytes'].values * 8, dtype=np.float)
            # Concatenate capacity
            capacity_conc.extend(list(capacity))
            
        # Train model
        model = GMM_Channel(n_feat, n_components)
        model.fit(capacity_conc)

        # Save model
        out_folder = Path('gmm_models')
        out_folder.mkdir(parents=True, exist_ok=True)
        out_filename = '%s.h5' % scenario
        model.save_model(out_folder / out_filename)
        
        # Generate example
        gen_signal = model.sample(len(capacity_conc), random_seed=123)
        fig, axs = plt.subplots(2,1, figsize=(12, 8))
        axs[0].set_title('GENERATED SIGNAL (%s) - n_components: %d, n_feat: %d' % (scenario, model.n_components, model.n_features))
        axs[0].plot(gen_signal)
        axs[1].set_title('ORIGINAL SIGNAL (%s)' % scenario)
        axs[1].plot(capacity_conc)
        plt.tight_layout()
        plt.savefig(out_folder / ('%s.png' % scenario))
        
        
