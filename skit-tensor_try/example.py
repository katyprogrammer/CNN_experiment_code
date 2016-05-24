import logging
import numpy as np
from scipy.io.matlab import loadmat
from sktensor import dtensor, cp_als

# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)

# Load Matlab data and convert it to dense tensor format
mat = loadmat('Sensory_Bread/brod.mat')
T = dtensor(mat['X'])

# choose low-rank
R = 8

# Decompose tensor using CP-ALS
P, fit, itr, exectimes = cp_als(T, R, init='random')

Y = None
# approximate
for i in range(R):
    y = P.lmbda[i] * np.outer(P.U[0][:,i], P.U[1][:,i])
    Y = y if Y is None else Y + y

print('norm of origin: {0}'.format(np.linalg.norm(T)))
print('norm of approx with {1} rank-1 tensor: {0}'.format(np.linalg.norm(Y), R))
print('norm of difference: {0}'.format(np.linalg.norm(T-Y)))
