__author__ = 'billhuang'

import numpy as np
import PMM

# SET RANDOM SEED
np.random.seed(1234)

# CREATE DATA
print('Generate Data...')
N = np.array([20, 40, 60])

lambda_ = np.array([5, 20, 35])

print('pi')
print(N/np.sum(N))

print('lambda')
print(lambda_)

Y = np.zeros(np.sum(N), dtype = int)
for i in range(N[0]):
    Y[i] = np.random.poisson(lambda_[0])
for i in range(N[0], (N[0]+N[1])):
    Y[i] = np.random.poisson(lambda_[1])
for i in range((N[0]+N[1]), np.sum(N)):
    Y[i] = np.random.poisson(lambda_[2])

# INFERENCE BY EM
PMM.PMM(Y, 3)

