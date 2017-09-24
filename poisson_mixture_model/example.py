__author__ = 'billhuang'

import numpy as np
import PMM

# SET RANDOM SEED
np.random.seed(1234)

# CREATE DATA
N = [20, 40, 60]

lambda_ = np.array([5, 25, 45])

Y = np.zeros(np.sum(N), dtype = int)
for i in range(N[0]):
    Y[i] = np.random.poisson(lambda_[0])
for i in range(N[0], (N[0]+N[1])):
    Y[i] = np.random.poisson(lambda_[1])
for i in range((N[0]+N[1]), np.sum(N)):
    Y[i] = np.random.poisson(lambda_[2])

# INFERENCE BY EM
label = PMM.PMM(Y, 3)
print(label)

