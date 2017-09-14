__author__ = 'billhuang'

import numpy as np
import sys

def log(x):
    '''
    safe log for handling the case with zero count
    '''
    if (np.sum(x==0) > 0):
        x += np.power(0.1, 320)
    return (np.log(x))

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
