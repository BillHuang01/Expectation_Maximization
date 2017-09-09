import numpy as np
import BMM

# SET RANDOM SEED
np.random.seed(1234)

# CREATE DATA
N = [10, 20, 30]

mu = np.array([[1.0, 0.1, 0.1, 0.1, 0.2],
               [0.2, 1.0, 1.0, 0.1, 0.3],
               [0.1, 0.2, 0.1, 1.0, 1.0]])

mu = np.array([[0.2, 0.8, 0.1, 0.2, 0.7],
               [0.3, 0.2, 0.7, 0.3, 0.1],
               [0.9, 1.0, 0.5, 0.8, 0.2]])

Y = np.zeros((np.sum(N), 5), dtype = int)
for i in range(N[0]):
    for d in range(5):
        Y[i,d] = np.random.binomial(1, mu[0,d])
for i in range(N[0], (N[0]+N[1])):
    for d in range(5):
        Y[i,d] = np.random.binomial(1, mu[1,d])
for i in range((N[0]+N[1]), np.sum(N)):
    for d in range(5):
        Y[i,d] = np.random.binomial(1, mu[2,d])

# INFERENCE BY EM
label = BMM.BMM(Y, 3)
print(label)

