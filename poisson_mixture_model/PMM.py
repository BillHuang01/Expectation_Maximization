__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu
from scipy import stats
import sys

def random_initialization(Y_, K_):
    mean_ = np.mean(Y_)
    pi_ = np.random.dirichlet(np.ones(K_))
    lambda_ = np.random.gamma(1.0, mean_, size = K_)
    return (pi_, lambda_)

def E_step(Y_, pi_, lambda_):
    N_ = Y_.size
    K_ = lambda_.size
    logZ_ = np.zeros((N_, K_))
    logZ_ += nu.log(pi_)
    for k in range(K_):
        logZ_[:,k] += stats.poisson.logpmf(Y_, lambda_[k])
    logZ_ = np.nan_to_num(logZ_) # to avoid overflow in addtion
    Z_ = nu.normalize_log_across_row(logZ_)
    lower_bound_ = np.sum(Z_ * (logZ_ - nu.log(Z_)))
    return (Z_, lower_bound_)

def M_step(Y_, Z_):
    N_, K_ = Z_.shape
    Nk_ = np.sum(Z_, axis = 0)
    pi_ = Nk_ / N_
    lambda_ = np.dot(Z_.T, Y_) / Nk_
    return (pi_, lambda_)

def PMM(Y_, K_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    print('Start Inference...')
    pi_, lambda_ = initializer(Y_, K_)
    lower_bound = np.array([])
    continue_ = True
    while (continue_):
        sys.stdout.write('*')
        Z_, lower_bound_ = E_step(Y_, pi_, lambda_)
        lower_bound = np.append(lower_bound, lower_bound_)
        pi_, lambda_ = M_step(Y_, Z_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                sys.stdout.write('  done!\n')
    print('pi')
    print(pi_)
    print('lambda')
    print(lambda_)
