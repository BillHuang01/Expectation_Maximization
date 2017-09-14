# Hidden Markov Model with Multivariate Gaussian Emission Distribution
# each dimension are independent (covariance of different dimensions are 0)

__author__ = 'billhuang'

import numpy as np
import kmean
import numerical_utils as nu
from scipy import stats

def random_initialization(Y_, K_):
    pi0_, A_ = init_transition(K_)
    D_ = Y_.shape[1]
    mu_ = np.random.normal(0, 1, size = (K_, D_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        s_[k,:] = np.eye(D_)
    B_ = sync_B(Y_, mu_, s_)
    return (pi0_, A_, B_)

def kmean_initialization(Y_, K_):
    pi0_, A_ = init_transition(K_)
    D_ = Y_.shape[1]
    label_ = kmean.kmean(Y_, K_)
    mu_ = np.zeros((K_, D_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        Yk_ = Y_[label_ == k]
        mu_[k,:] = np.mean(Yk_, axis = 0)
        s_[k,:] = np.eye(D_)
    B_ = sync_B(Y_, mu_, s_)
    return (pi0_, A_, B_)

def init_transition(K_):
    pi0_ = nu.log(np.random.dirichlet(np.ones(K_)))
    A_ = nu.log(np.random.dirichlet(np.ones(K_), size = K_))
    return (pi0_, A_)

def sync_B(Y_, mu_, s_):
    T_ = Y_.shape[0]
    K_ = mu_.shape[0]
    B_ = np.zeros((T_, K_))
    for k in range(K_):
        B_[:,k] = stats.multivariate_normal.logpdf(Y_, mu_[k,:], s_[k,:])
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
    logQ_ = (Qu_.T - nu.log_sum_across_row(Qu_)).T
    Q_ = np.exp(logQ_)
    return (Q_)

def sync_N(M_, R_, A_, B_):
    T_, K_ = B_.shape
    xi_ = np.zeros((T_ - 1, K_, K_))
    for t in range(0, T_ - 1):
        xi_[t,:] = (A_.T + M_[t,:]).T + B_[(t+1),:] + R_[(t+1),:]
    xi_ = xi_ - nu.log_sum_vector(M_[-1,:])
    N_ = np.sum(np.exp(xi_), axis = 0)
    return (N_)

def sync_A(N_):
    A_ = nu.normalize_across_row(N_)
    return (nu.log(A_))

def update_params(Y_, Q_, N_):
    T_, D_ = Y_.shape
    K_ = Q_.shape[1]
    q_ = np.sum(Q_, axis = 0)
    mu_ = np.dot(np.diag(1/q_), np.dot(Q_.T, Y_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        ym_ = Y_ - mu_[k,:]
        s_[k,:] = np.dot(ym_.T, np.dot(np.diag(Q_[:,k]), ym_)) / q_[k]
    return (mu_, s_)

def compute_lower_bound(pi0_, A_, B_, Q_, N_):
    lower_bound_ = np.sum(N_ * A_) + np.sum(Q_[0,:] * pi0_) + np.sum(Q_ * B_) 
    return (lower_bound_)

def E_step(pi0_, A_, B_):
    M_ = pass_message_forward(pi0_, A_, B_)
    R_ = pass_message_backward(A_, B_)
    Q_ = sync_Q(M_, R_)
    N_ = sync_N(M_, R_, A_, B_)
    lower_bound_ = compute_lower_bound(pi0_, A_, B_, Q_, N_)
    return (Q_, N_, lower_bound_)

def M_step(Y_, Q_, N_):
    pi0_ = nu.log(Q_[0,:])
    A_ = sync_A(N_)
    mu_, s_ = update_params(Y_, Q_, N_)
    B_ = sync_B(Y_, mu_, s_)
    return (pi0_, A_, B_)

def HMM(Y_, K_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    pi0_, A_, B_ = initializer(Y_, K_)
    lower_bound = np.array([])
    continue_ = True
    while (continue_):
        Q_, N_, lower_bound_ = E_step(pi0_, A_, B_)
        lower_bound = np.append(lower_bound, lower_bound_)
        pi0_, A_, B_ = M_step(Y_, Q_, N_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
    hidden_state_ = np.argmax(Q_, axis = 1)
    return (hidden_state_)
    
    
