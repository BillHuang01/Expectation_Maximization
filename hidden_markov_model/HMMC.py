from __future__ import absolute_import, division, print_function

# Hidden Markov Model with Categorical Distribution

__author__ = 'billhuang'

import numpy as np
import kmean
import numerical_utils as nu

def random_initialization(Y_, J_):
    C_ = np.unique(Y_).size
    pi0_ = nu.log(np.random.dirichlet(np.ones(J_)))
    A_ = nu.log(np.random.dirichlet(np.ones(J_), size = J_))
    Bstar_ = nu.log(np.random.dirichlet(np.ones(J_), size = C_))
    return (pi0_, A_, Bstar_)

def sync_B(Y_, Bstar_):
    B_ = Bstar_[Y_, :]
    return (B_)

def pass_message_forward(pi0_, A_, B_):
    T_, K_ = B_.shape
    M_ = np.zeros((T_, K_))
    M_[0,:] = pi0_ + B_[0,:]
    for t in range(1, T_):
        M_[t,:] = nu.log_matrix_multiply_vector(A_.T, M_[(t-1),:]) + B_[t,:]
    return (M_)

def pass_message_backward(A_, B_):
    R_ = np.zeros(B_.shape)
    R_[-1,:] = 0
    for t in range(B_.shape[0] - 2, -1, -1):
        R_[t,:] = nu.log_matrix_multiply_vector(A_, (B_[(t+1),:] + R_[(t+1),:]))
    return (R_)

def sync_Q(M_, R_):
    Qu_ = M_ + R_
    logQ_ = (Qu_.T - np.logaddexp.reduce(Qu_, axis = 1)).T
    Q_ = np.exp(logQ_)
    return (Q_)

def sync_N(M_, R_, A_, B_):
    T_, K_ = B_.shape
    xi_ = np.zeros((T_ - 1, K_, K_))
    for t in range(0, T_ - 1):
        xi_[t,:] = (A_.T + M_[t,:]).T + B_[(t+1),:] + R_[(t+1),:]
    xi_ = xi_ - np.logaddexp.reduce(M_[-1,:])
    N_ = np.sum(np.exp(xi_), axis = 0)
    return (N_)

def sync_A(N_):
    A_ = nu.normalize_across_row(N_)
    return (nu.log(A_))

def update_params(Y_, Q_):
    C_ = np.unique(Y_).size
    K_ = Q_.shape[1]
    Bstar_ = np.zeros((C_, K_))
    for c in range(C_):
        Bstar_[c,:] = np.sum(Q_[(Y_==c),:], axis = 0)
    Qsum_ = np.sum(Q_, axis = 0)
    Bstar_ = Bstar_ / Qsum_
    return (nu.log(Bstar_))

def compute_lower_bound(pi0_, A_, B_, Q_, N_):
    lower_bound_ = np.sum(N_ * A_) + np.sum(Q_[0,:] * pi0_) + np.sum(Q_ * B_)
    return (lower_bound_)

'''
def compute_lower_bound(M_):
    lower_bound_ = np.sum(np.logaddexp.reduce(M_, axis = 1))
    return (lower_bound_)
'''

def E_step(Y_, pi0_, A_, Bstar_):
    B_ = sync_B(Y_, Bstar_)
    M_ = pass_message_forward(pi0_, A_, B_)
    R_ = pass_message_backward(A_, B_)
    Q_ = sync_Q(M_, R_)
    N_ = sync_N(M_, R_, A_, B_)
    #lower_bound_ = compute_lower_bound(M_)
    lower_bound_ = compute_lower_bound(pi0_, A_, B_, Q_, N_)
    return (Q_, N_, lower_bound_)

def M_step(Y_, Q_, N_):
    pi0_ = nu.log(Q_[0,:])
    A_ = sync_A(N_)
    Bstar_ = update_params(Y_, Q_)
    return (pi0_, A_, Bstar_)

def HMM(Y_, K_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    print('Start Inference...')
    pi0_, A_, Bstar_ = initializer(Y_, K_)
    lower_bound = np.array([])
    continue_ = True
    i = 0;
    while (continue_):
        print('*', end = '')
        Q_, N_, lower_bound_ = E_step(Y_, pi0_, A_, Bstar_)
        lower_bound = np.append(lower_bound, lower_bound_)
        pi0_, A_, Bstar_ = M_step(Y_, Q_, N_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
                print('  done!')
    print('A')
    print(np.exp(A_))
    print('B')
    print(np.exp(Bstar_))

