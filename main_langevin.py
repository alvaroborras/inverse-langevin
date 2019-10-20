import sys
import os.path
import numpy as np
from numpy import tanh
import inverse_langevin
import time

file_coefs = 'inverse_langevin_coefficients.npy'

if not os.path.isfile(file_coefs):
  inverse_langevin.compute_coefficients()

coefs = np.load(file_coefs, mmap_mode='r')

nsp = coefs[0,0]
dx  = coefs[1,0]
xir = coefs[2,0]
a   = coefs[3,0]
b   = coefs[4,0]
nsp1 = nsp + 1

ntimes = 10000000
y_max = 1000
y_exact = 1e-2 + np.random.rand(ntimes) * y_max
x = 1/tanh(y_exact) - 1/y_exact

ySpline = np.zeros( (ntimes, 1) )


s = [int(x[i]/dx) for i in range(ntimes)]

start = time.time()
for i in range(ntimes):
  if s[i] < nsp1:
    xinc = x[i] - coefs[4,s[i]]
    xinc2 = np.power(xinc, 2)
    xinc3 = np.power(xinc, 3)
    ySpline[i] = coefs[0,s[i]]*xinc3 + coefs[1,s[i]]*xinc2 +  coefs[2,s[i]]*xinc + coefs[3,s[i]]
  else:
    ySpline[i] = ( a*x[i] + b ) / (1 - x[i]**2)
  


print(f'{time.time() - start} s elapsed.')