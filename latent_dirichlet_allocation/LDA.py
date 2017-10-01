from __future__ import absolute_import, division, print_function

__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu

def random_initialization(K_, D_, V_):
    theta_ = nu.log(np.random.dirichlet(np.ones(K_), size = D_))
    beta_ = nu.log(np.random.dirichlet(np.ones(V_), size = K_))
    return (theta_, beta_)

def update_theta(phi_, K_, D_):
    count_ = np.zeros((D_, K_))
    for d in range(D_):
        count_[d,:] = np.sum(phi_[d], axis = 0)
    theta_ = nu.log(nu.normalize(count_))
    return (theta_)

def update_beta(Y_, phi_, K_, D_, V_):
    count_ = np.zeros((K_, V_))
    for k in range(K_):
        for d in range(D_):
            for i in range(len(Y_[d])):
                count_[k,Y_[d][i]] += phi_[d][i,k]
    beta_ = nu.log(nu.normalize(count_))
    return (beta_)

def E_step(Y_, theta_, beta_, K_, D_):
    lower_bound_ = 0
    phi_ = []
    for d in range(D_):
        logphid = np.zeros((Y_[d].size, K_))
        logphid += theta_[d,:]
        for k in range(K_):
            logphid[:,k] += beta_[k,:][Y_[d]]
        phid_ = nu.normalize_log_across_row(logphid)
        lower_bound_ += np.sum(phid_ * (logphid - nu.log(phid_)))
        #lower_bound_ += np.sum(np.logaddexp.reduce(logphid, axis = 1))
        phi_.append(phid_)
    return (phi_, lower_bound_)

def M_step(Y_, phi_, K_, D_, V_):
    theta_ = update_theta(phi_, K_, D_)
    beta_ = update_beta(Y_, phi_, K_, D_, V_)
    return (theta_, beta_)

def LDA(Y_, K_, D_, V_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    print('Start Inference...')
    theta_, beta_ = initializer(K_, D_, V_)
    lower_bound = np.array([])
    continue_ = True
    while (continue_):
        print('*', end = '')
        phi_, lower_bound_ = E_step(Y_, theta_, beta_, K_, D_)
        lower_bound = np.append(lower_bound, lower_bound_)
        theta_, beta_ =M_step(Y_, phi_, K_, D_, V_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                print('  done!')
    print('theta')
    print(np.exp(theta_))
    print('beta')
    print(np.exp(beta_))
    
