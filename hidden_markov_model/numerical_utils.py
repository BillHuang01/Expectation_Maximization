__author__ = 'billhuang'

import numpy as np

def log(x):
    '''
    safe log for handling the case with zero count
    '''
    return (np.nan_to_num(np.log(x)))
'''
def normalize_log_across_row(UN):
    # ULP: unnormalized_log_probability
    # subtract each row by its max
    MLP = np.transpose(np.transpose(UN) - np.max(UN, axis=1))
    UP = np.exp(MLP)
    N = np.transpose(np.transpose(UP)/np.sum(UP, axis=1))
    return (N)
'''

def normalize_log_across_row2(logM):
    '''
    safer way to convert log probability to normalized
    probability across row in a matrix than simply exponential
    each term and then normalize
    '''
    logP = np.logaddexp.reduce(logM, axis = 1)
    NlogM = (logM.T - logP).T
    M = np.exp(NlogM)
    return (M)

'''
def log_sum_vector(logv):
    m = np.max(logv)
    nlogv = logv - m
    log_sum = np.log(np.sum(np.exp(nlogv))) + m
    return (log_sum)
'''
# got replaced by np.logaddexp.reduce(logv)

'''
def log_sum_across_row(LM):
    m = np.max(LM, axis = 1)
    # minus the max for each row
    NM = np.transpose(np.transpose(LM) - m)
    row_sum_log = np.log(np.sum(np.exp(NM), axis=1)) + m
    return (row_sum_log)
'''
# got replaced by np.logaddexp.reduce(LM, axis = 1)

def log_matrix_multiply_vector(logM, logv):
    logNM = logM + logv
    logP = np.logaddexp.reduce(logNM, axis = 1)
    return (logP)

def normalize_across_row(M):
    sum_ = np.sum(M, axis = 1)
    nM = (M.T / sum_).T
    return (nM)
