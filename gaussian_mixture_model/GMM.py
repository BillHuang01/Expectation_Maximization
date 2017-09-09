import numpy as np
import kmean
import numerical_utils as nu
from scipy import stats

def random_initialization(Y_, K_):
    D_ = Y_.shape[1]
    pi_ = np.random.dirichlet(np.ones(K_))
    mu_ = np.random.normal(0, 1, size = (K_, D_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        s_[k,:] = np.eye(D_)
    return (pi_, mu_, s_)

def kmean_initialization(Y_, K_):
    N_, D_ = Y_.shape
    label_ = kmean.kmean(Y_, K_)
    pi_ = np.bincount(label_) / N_
    mu_ = np.zeros((K_, D_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        Yk_ = Y_[label_ == k]
        mu_[k,:] = np.mean(Yk_, axis = 0)
        s_[k,:] = np.eye(D_)
    return (pi_, mu_, s_)

def E_step(Y_, pi_, mu_, s_):
    N_ = Y_.shape[0]
    K_ = pi_.size
    logZ_ = np.zeros((N_, K_))
    for k in range(K_):
        logZ_[:,k] = stats.multivariate_normal.logpdf(Y_, mu_[k,:], s_[k,:])
        logZ_[:,k] += nu.log(pi_[k])
    Z_ = nu.normalize_log_across_row(logZ_)
    lower_bound_ = np.sum(Z_ * (logZ_ - nu.log(Z_)))
    return (Z_, lower_bound_)

def M_step(Y_, Z_):
    N_, D_ = Y_.shape
    K_ = Z_.shape[1]
    Nk_ = np.sum(Z_, axis = 0)
    pi_ = Nk_ / N_
    mu_ = np.dot(np.diag(1 / Nk_), np.dot(Z_.T, Y_))
    s_ = np.zeros((K_, D_, D_))
    for k in range(K_):
        ym_ = Y_ - mu_[k,:]
        s_[k,:] = np.dot(ym_.T, np.dot(np.diag(Z_[:,k]), ym_)) / Nk_[k]
    return (pi_, mu_, s_)

def GMM(Y_, K_, eps = np.power(0.1, 3),
        initializer = random_initialization):
    pi_, mu_, s_ = initializer(Y_, K_)
    lower_bound = np.array([])
    continue_ = True
    while (continue_):
        Z_, lower_bound_ = E_step(Y_, pi_, mu_, s_)
        lower_bound = np.append(lower_bound, lower_bound_)
        pi_, mu_, s_ = M_step(Y_, Z_)
        if (lower_bound.size > 1):
            if ((np.exp(lower_bound[-1] - lower_bound[-2]) - 1) < eps):
                continue_ = False
    group_ = np.argmax(Z_, axis = 1)
    return (group_)
        
