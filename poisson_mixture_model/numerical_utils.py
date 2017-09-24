__author__ = 'billhuang'

import numpy as np

def log(x):
    '''
    safe log for handling the case with zero count
    '''
    return (np.nan_to_num(np.log(x)))

def normalize_log_across_row(UN):
    '''
    safer way to convert log probability to normalized
    probability across row in a matrix than simply exponential
    each term and then normalize
    '''
    # ULP: unnormalized_log_probability
    # subtract each row by its max
    MLP = np.transpose(np.transpose(UN) - np.max(UN, axis=1))
    UP = np.exp(MLP)
    N = np.transpose(np.transpose(UP)/np.sum(UP, axis=1))
    return (N)
