__author__ = 'billhuang'

import numpy as np
import GMM

# SET RANDOM SEED
np.random.seed(1234)

# GENERATE DATA
print('Generate Data...')

N = np.array([10,20,30])

mu = np.array([[0.5, 1.2],
               [-1.2, 0.3],
               [3, -3]])

cov = np.zeros((3, 2, 2))
cov[0,:] = 0.5 * np.eye(2)
cov[1,:] = 0.3 * np.eye(2)
cov[2,:] = np.eye(2)

print('pi')
print(N/np.sum(N))
print('mu')
print(mu)
print('s')
print(cov)

Y = np.zeros((np.sum(N), 2))

for i in range(N[0]):
    Y[i,:] = np.random.multivariate_normal(mu[0,:], cov[0,:])
for i in range(N[0], (N[0] + N[1])):
    Y[i,:] = np.random.multivariate_normal(mu[1,:], cov[1,:])
for i in range((N[0] + N[1]), np.sum(N)):
    Y[i,:] = np.random.multivariate_normal(mu[2,:], cov[2,:])

GMM.GMM(Y, 3, initializer = GMM.kmean_initialization)
    
