__author__ = 'billhuang'

import numpy as np

def get_center(label, feature, K):
    '''
    calculate the center for each cluster
    label: data point label for being in which class
    feature: data point feature
    K: number of clusters
    simply calculate the mean for the data in each label cluster to
    get the cluster center
    return: K*D matrix with each row represent the center for each class
    '''
    D = feature.shape[1]
    center = np.zeros((K,D))
    for k in range(K):
        center[k,:] = np.mean(feature[label==k,:], axis=0)
    return center

def dist_to_center(center, feature, K):
    '''
    calculate the distance of each point to all the K centers
    return N*K distance matrix with entry n,k being distance of
    data point n to cluster center k
    '''
    N = feature.shape[0]
    distance = np.zeros((N,K))
    for k in range(K):
        distance[:,k] = np.sum(np.square(feature - center[k,:]), axis = 1)
    return distance

def initial_center(K, N):
    '''
    initial center by picking each label for only one point
    the rest data points does not have any label (which use -1)
    those points' coordinate will be the center
    '''
    label = np.repeat(-1, N)
    # pick one point to be the center
    label[np.random.choice(N, K, replace=False)] = np.arange(K, dtype=int)
    return label

def initial_label(K, N):
    '''
    initial label by randomly picking a label for each point
    while loop to ensure that each label must have one data point
    '''
    label = np.random.randint(K, size = N)
    while (np.unique(label).size != K):
        label = np.random.randint(K, size = N)
    return label

def kmean(feature, K, initialization=initial_center):
    '''
    feature: N*D matrix (N data points each with D features)
    K: number of clusters
    initialization: initialization method (initial label or initial center)
    '''
    N, D = feature.shape
    previous_label = np.zeros(N, dtype=int)
    updated_label = initialization(K, N)
    while (np.sum(previous_label != updated_label) != 0):
        previous_label = updated_label
        center = get_center(previous_label, feature, K)
        distance = dist_to_center(center, feature, K)
        updated_label = np.argmin(distance, axis = 1)
    return updated_label
    
