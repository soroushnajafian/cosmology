from __future__ import division
import math

try:
    import numpy as np
except ImportError:
    print('Numpy is required, please make sure it is installed properly')

try:
    import scipy as sp
except ImportError:
    print('Scipy is required, please make sure it is installed properly')

from scipy.interpolate import InterpolatedUnivariateSpline
import scalar_sol

class Quintessence:
    def __init__(self, xi, yi, zi, Gamma, wm, h):
        self.xi = float(xi)
        self.yi = float(yi)
        self.zi = float(zi)
        self.Gamma = float(Gamma)
        self.wm = float(wm)
        self.h = float(h)
        
        self.Qui = scalar_sol.Quin(self.xi, self.yi, self.zi, self.Gamma, self.wm)
        self.aas, self.Zs, self.xs, self.ys, self.zs = self.Qui.solu(0) # solve the equation only once, otherwise, the initial conditions will be changed

    def E(self, z):
        # the present day value of x, y, z is the final value of xi, yi, zi in the solution part
        self.x0 = self.xs[len(self.xs)-1]
        self.y0 = self.ys[len(self.ys)-1]
        self.z0 = self.zs[len(self.zs)-1]
        self.Om0 = 1-(self.x0**2+self.y0**2)

        self.order = np.argsort(self.Zs)  # the InterpolatedUnivariateSpline requires the first argument must increase
        self.EE = (self.Om0*(1+self.Zs)**3/(1-self.xs**2-self.ys**2))**0.5
        self.EEc = InterpolatedUnivariateSpline(self.Zs[self.order], self.EE[self.order], k=3)
        return self.EEc(z)

    def Ode(self, z):
        self.order = np.argsort(self.Zs)
        self.Ophi = self.xs**2+self.ys**2
        self.Ophi_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.Ophi[self.order], k=3)
        return self.Ophi_in(z)
            
    def w_de(self, z):
        self.order = np.argsort(self.Zs)
        self.w_de1 = (self.xs**2-self.ys**2)/(self.xs**2+self.ys**2)
        self.w_de_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.w_de1[self.order], k=3)
        return self.w_de_in(z)

    def weff(self, z):
        self.order = np.argsort(self.Zs)
        self.weff1 = self.wm+(1-self.wm)*self.xs**2-(1+self.wm)*self.yx**2
        self.weff_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.weff1[self.order], k=3)
        return self.weff_in(z)














