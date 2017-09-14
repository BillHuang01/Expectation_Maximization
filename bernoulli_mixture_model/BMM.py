__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu

def random_initialization(Y_, K_):
    N_, D_ = Y_.shape
    pi_ = np.random.dirichlet(np.ones(K_))
    mu_ = np.random.beta(1, 1, size = (K_, D_))
    return (pi_, mu_)

def E_step(Y_, pi_, mu_):
    N_, D_ = Y_.shape
    K_ = mu_.shape[0]
    logZ_ = np.zeros((N_, K_))
    logZ_ += nu.log(pi_)
    for d in range(D_):
        y_ = Y_[:,d]
        for k in range(K_):
            logZ_[(y_==1),k] += nu.log(mu_[k,d])
            logZ_[(y_==0),k] += nu.log(1 - mu_[k,d])
    Z_ = nu.normalize_log_across_row(logZ_)
    lower_bound_ = np.sum(Z_ * (logZ_ - nu.log(Z_)))
    return (Z_, lower_bound_)

def M_step(Y_, Z_):
    N_, K_ = Z_.shape
    D_ = Y_.shape[1]
    Nk_ = np.sum(Z_, axis = 0)
    pi_ = Nk_ / N_
    mu_ = np.zeros((K_, D_))
    for k in range(K_):
        mu_[k,:] = np.sum(Y_.T * Z_[:,k], axis = 1) / Nk_[k]
    return (pi_, mu_)

def BMM(Y_, K_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    pi_, mu_ = initializer(Y_, K_)
    lower_bound = np.array([])
    continue_ = True
    while (continue_):
        Z_, lower_bound_ = E_step(Y_, pi_, mu_)
        lower_bound = np.append(lower_bound, lower_bound_)
        pi_, mu_ = M_step(Y_, Z_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
    group_ = np.argmax(Z_, axis = 1)
    return (group_)
