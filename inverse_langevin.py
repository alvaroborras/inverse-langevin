import numpy as np 
from numpy import tanh
import scipy.optimize, scipy.interpolate


def compute_coefficients(npoints = 100001, xir = 0.980):
    
    f = lambda y : xir - 1/tanh(y) + 1/y
    
    yir = scipy.optimize.broyden1(f, 1/xir, f_tol=1e-16)
    dyir = yir ** 2 / (1 + yir ** 2 - (yir / tanh(yir))**2)

    y1 = np.linspace(0, 2 * yir, 2 * npoints - 1)
    x1 = np.insert(1/tanh(y1[1:]) - 1/y1[1:],0,0)

    sp1 = scipy.interpolate.CubicSpline(x1, y1, bc_type='natural')

    x2 = np.linspace(0, xir, npoints)
    dx = x2[1] - x2[0]

    y2 = sp1(x2)
    sp2 = scipy.interpolate.CubicSpline(x2, y2, bc_type=((1,1), (1,1)))


    coef = np.zeros( (5, npoints) )
    btemp = yir*( 1 - xir**2 );

    
    coef[0,0] = npoints - 1
    coef[1,0] = dx
    coef[2,0] = xir
    coef[3,0] = -2*btemp*xir/(1 - xir**2) + dyir*(1 - xir**2)
    coef[4,0] = btemp - coef[3,0]*xir

    coef[0:3, 1:npoints-1] = sp2.c[:3, 1:npoints-1]
    coef[4, 1:npoints-1] = x2[:npoints-2]
    
    np.save('inverse_langevin_coefficients', coef)

if __name__ == "__main__":

  print('Computing coefficients...')
  compute_coefficients()