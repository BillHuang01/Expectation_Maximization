__author__ = 'billhuang'

import numpy as np
import HMMG
import HMMC

# SET RANDOM SEED
np.random.seed(1234)

# HMM WITH GAUSSIAN EMISSION DISTRIBUTION

# GENERATE DATA

print('Generate Data...')

A = np.array([[0.8,0.1,0.1],
              [0.2,0.6,0.2],
              [0.1,0.15,0.75]])

mu = np.array([[0,0],
               [2,2],
               [-2,-2]])

cov = np.eye(2)

print('A')
print(A)

print('mu')
print(mu)

print('s')
print(cov)

print()

N = 500

Y = np.zeros((N,2))
Z = np.zeros(N, dtype = int)
Z[0] = np.random.choice(3)
Y[0,:] = np.random.multivariate_normal(mu[Z[0],:], cov)

for i in range(1, N):
    Z[i] = np.random.choice(3, p = A[Z[i-1],:])
    Y[i,:] = np.random.multivariate_normal(mu[Z[i],:], cov)

HMMG.HMM(Y, 3, initializer = HMMG.kmean_initialization)


# HMM WITH CATEGORICAL EMISSION DISTRIBUTION

print('\n\n\n')

print('Generate Data...')

A = np.array([[0.8,0.1,0.1],
              [0.2,0.6,0.2],
              [0.1,0.15,0.75]])

B = np.array([[0.9, 0.05, 0.05],
             [0.1, 0.9, 0.0],
             [0.05, 0.15, 0.8]])

print('A')
print(A)
print('B')
print(B)
print()

N = 500

Y = np.zeros(N, dtype = int)
Z = np.zeros(N, dtype = int)
Z[0] = np.random.choice(3)
Y[0] = np.random.choice(3, p = B[Z[0],:])

for i in range(1, N):
    Z[i] = np.random.choice(3, p = A[Z[i-1],:])
    Y[i] = np.random.choice(3, p = B[Z[i],:])

HMMC.HMM(Y, 3)
