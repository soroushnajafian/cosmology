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
from scipy.misc import derivative
from scipy import interpolate
from scipy.interpolate import splrep
from scipy.interpolate import splev
import scalar_sol

class Quintessence:
    def __init__(self, xi, yi, zi, Gamma, wm, h):
        self.Ok0 = float(0) # assume the flat universe
        self.xi = float(xi)
        self.yi = float(yi)
        self.zi = float(zi)
        self.Gamma = float(Gamma)
        self.wm = float(wm)
        self.h = float(h)
        
        self.Qui = scalar_sol.Quin(self.xi, self.yi, self.zi, self.Gamma, self.wm)
        self.aas, self.Zs, self.xs, self.ys, self.zs = self.Qui.solu(0) # solve the equation only once, otherwise, the initial conditions will be changed
        
        # the present day value of x, y, z is the final value of xi, yi, zi in the solution part
        self.x0 = self.xs[len(self.xs)-1]
        self.y0 = self.ys[len(self.ys)-1]
        self.z0 = self.zs[len(self.zs)-1]
        self.Om0 = 1-(self.x0**2+self.y0**2)
        print(self.Om0)

    def E(self, z):

        self.order = np.argsort(self.Zs)  # the InterpolatedUnivariateSpline requires the first argument must increase
        self.EE = (self.Om0*(1+self.Zs)**3/(1-self.xs**2-self.ys**2))**0.5
        self.EEc = InterpolatedUnivariateSpline(self.Zs[self.order], self.EE[self.order], k=3)
        return self.EEc(z)

    def Ed(self, z):
        '''
        Calculate the H(z)/(1+z)
        '''
        return self.E(z)/(1+z)

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
        self.weff1 = self.wm+(1-self.wm)*self.xs**2-(1+self.wm)*self.ys**2
        self.weff_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.weff1[self.order], k=3)
        return self.weff_in(z)

    def Ep(self, z):
        '''
        The derivative of the expansion factor with respect to redshift z
        '''
        return derivative(self.E, z, dx = 1e-6)
    
    def q(self, z):
        '''
        The deceleration factor as a function of redshift z
        '''
        return -1.0+(1.0+z)*self.Ep(z)/self.E(z)

    def D_H(self):
        '''
        The Hubble distance from: David Hogg, arxiv: astro-ph/9905116v4
        '''
        return 3000/self.h
    
    def D_Hz(self, z):
        return self.D_H()/self.E(z)

    def chi(self, z):
        '''
        It is not a observable, but needed to calculate luminosity distance and others
        '''
        self.zz = np.linspace(0, z, 100)
        self.Ez = 1.0/np.array((map(self.E, self.zz)))
        self.cc = interpolate.splrep(self.zz, self.Ez, s=0)
        return interpolate.splint(0, z, self.cc)

    def D_L(self, z):
        '''
        Luminosity distance
        '''
        if self.Ok0 > 0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*math.sinh(np.sqrt(self.Ok0)*self.chi(z))
        elif self.Ok0 == 0:
            return self.D_H()*(1+z)*self.chi(z)
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*math.sin(np.sqrt(-self.Ok0)*self.chi(z))

    def D_A(self, z):
        '''
        Angular diameter distance
        '''
        return self.D_L(z)/(1+z)**2

    def D_C(self, z):
        '''
        Line of sight comoving distance
        '''
        return self.D_H()*self.chi(z)

    def D_M(self, z):
        '''
        Transverse comoving distance
        '''
        return self.D_L(z)/(1+z)
    
    def D_V(self, z):    #   zzx!!! for BAO use
        '''
        volume averaged distance
        '''
        return (z*self.D_Hz(z)*self.D_M(z)**2)**(1.0/3.0)

    def rd(self, Onu, Oba):
        '''
        Use the numerically calibrated approximation to calculate the sound horizon at the drag epoch. Added two parameters Onu and Oba have prior from CMB or other experiments. This function is only for BAO use. For some models with different neutrino theories, it should be changed.
            '''
        self.h = abs(self.h)
        self.Onu = float(Onu)
        self.Oba = float(Oba)
        
        return 55.154*np.exp(-72.3*(self.Onu*self.h**2.0+0.0006)**2.0)/(self.Om0*self.h**2)**0.25351/(self.Oba*self.h**2)**0.12807
    
    def rd_nu(self, Onu, Oba, Neff):
        
        self.h = abs(self.h)
        self.Onu = abs(float(Onu))
        self.Oba = abs(float(Oba))
        self.Neff = float(Neff)
        return 56.067*np.exp(-49.7*(self.Onu*self.h**2.0+0.002)**2.0)/(abs(self.Om0)*self.h**2)**0.25351/(self.Oba*self.h**2)**0.12807/(1+(self.Neff-3.406)/30.60)

    def mu(self, z):
        '''
        Distance module
        '''
        return np.log10(self.D_L(z))+25

    def Dp(self, z, D, D1):
        '''
        This function and the next function are decomposed from the second-order differential equation of the linear growth rate D(z)
        '''
        self.D = float(D)
        self.D1 = float(D1)
        return D1

    def D1p(self, z, D, D1):
        self.D = float(D)
        self.D1 = float(D1)
        return 3/2*(1/self.E(z))**2*self.Om0*(1+z)*self.D-self.D1*self.Ep(z)/self.E(z)+self.D1/(1+z)
    
    def D_solu(self, z):
        '''
        Solve the differential equation of D(z). The outputs are redshift, D, dD/dz, and f = dlnD/dlna
        '''
        self.z00 = 0
        self.z = float(z)
        self.dz = (self.z-0.0)/1000 # step size: the number of steps between 0 and z is 1000
        self.D0 = 1
        self.D10 = -self.Om0**0.6
        self.Ds = self.D0
        self.D1s = self.D10
        self.zs = self.z00
        self.fs = self.D10
        while self.z00<z:
            self.k1 = self.dz*self.Dp(self.z00, self.D0, self.D10)
            self.l1 = self.dz*self.D1p(self.z00, self.D0, self.D10)
            
            self.k2 = self.dz*self.Dp(self.z00+self.dz/2, self.D0+self.k1/2, self.D10+self.l1/2)
            self.l2 = self.dz*self.D1p(self.z00+self.dz/2, self.D0+self.k1/2, self.D10+self.l1/2)
            
            self.k3 = self.dz*self.Dp(self.z00+self.dz/2, self.D0+self.k2/2, self.D10+self.l2/2)
            self.l3 = self.dz*self.D1p(self.z00+self.dz/2, self.D0+self.k2/2, self.D10+self.l2/2)
            
            self.k4 = self.dz*self.Dp(self.z00+self.dz/2, self.D0+self.k3, self.D10+self.l3)
            self.l4 = self.dz*self.D1p(self.z00+self.dz/2, self.D0+self.k3, self.D10+self.l3)
            
            self.D0 = self.D0+1/6*(self.k1+2*self.k2+2*self.k3+self.k4)
            self.D10 = self.D10+1/6*(self.l1+2*self.l2+2*self.l3+self.l4)
            
            self.z00 = self.z00+self.dz
            self.fss = -(1+self.z00)/self.D0*self.D10
            
            self.zs = np.append(self.zs, self.z00)
            self.Ds = np.append(self.Ds, self.D0)
            self.D1s = np.append(self.D1s, self.D10)
            self.fs = np.append(self.fs, self.fss)
        
        return self.zs, self.Ds, self.D1s, self.fs


class Phantom(Quintessence):
    def __init__(self, xi, yi, zi, Gamma, wm, h):
        self.Ok0 = float(0) # assume the flat universe
        self.xi = float(xi)
        self.yi = float(yi)
        self.zi = float(zi)
        self.Gamma = float(Gamma)
        self.wm = float(wm)
        self.h = float(h)
        
        self.Pha = scalar_sol.Phan(self.xi, self.yi, self.zi, self.Gamma, self.wm)
        self.aas, self.Zs, self.xs, self.ys, self.zs = self.Pha.solu(0) # solve the equation only once, otherwise, the initial conditions will be changed
        
        # the present day value of x, y, z is the final value of xi, yi, zi in the solution part
        self.x0 = self.xs[len(self.xs)-1]
        self.y0 = self.ys[len(self.ys)-1]
        self.z0 = self.zs[len(self.zs)-1]
        self.Om0 = 1-(-self.x0**2+self.y0**2)
        print(self.Om0)

    def E(self, z):
    
        self.order = np.argsort(self.Zs)  # the InterpolatedUnivariateSpline requires the first argument must increase
        self.EE = (self.Om0*(1+self.Zs)**3/(1+self.xs**2-self.ys**2))**0.5
        self.EEc = InterpolatedUnivariateSpline(self.Zs[self.order], self.EE[self.order], k=3)
        return self.EEc(z)

    def Ode(self, z):
        self.order = np.argsort(self.Zs)
        self.Ophi = -self.xs**2+self.ys**2
        self.Ophi_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.Ophi[self.order], k=3)
        return self.Ophi_in(z)

    def w_de(self, z):
        self.order = np.argsort(self.Zs)
        self.w_de1 = (-self.xs**2-self.ys**2)/(-self.xs**2+self.ys**2)
        self.w_de_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.w_de1[self.order], k=3)
        return self.w_de_in(z)
    
    def weff(self, z):
        self.order = np.argsort(self.Zs)
        self.weff1 = self.wm-(1-self.wm)*self.xs**2-(1+self.wm)*self.ys**2
        self.weff_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.weff1[self.order], k=3)
        return self.weff_in(z)


class Tachyon(Quintessence):
    def __init__(self, xi, yi, zi, Gamma, wm, h):
        self.Ok0 = float(0) # assume the flat universe
        self.xi = float(xi)
        self.yi = float(yi)
        self.zi = float(zi)
        self.Gamma = float(Gamma)
        self.wm = float(wm)
        self.h = float(h)
        
        self.Tac = scalar_sol.Tach(self.xi, self.yi, self.zi, self.Gamma, self.wm)
        self.aas, self.Zs, self.xs, self.ys, self.zs = self.Tac.solu(0) # solve the equation only once, otherwise, the initial conditions will be changed
        
        # the present day value of x, y, z is the final value of xi, yi, zi in the solution part
        self.x0 = self.xs[len(self.xs)-1]
        self.y0 = self.ys[len(self.ys)-1]
        self.z0 = self.zs[len(self.zs)-1]
        self.Om0 = 1-self.y0**2/np.sqrt(1-self.x0**2)
        print(self.Om0)
    
    def E(self, z):
        self.order = np.argsort(self.Zs)  # the InterpolatedUnivariateSpline requires the first argument must increase
        self.EE = (self.Om0*(1+self.Zs)**3/(1-self.ys**2/np.sqrt(1-self.xs**2)))**0.5
        self.EEc = InterpolatedUnivariateSpline(self.Zs[self.order], self.EE[self.order], k=3)
        return self.EEc(z)
    
    def Ode(self, z):
        self.order = np.argsort(self.Zs)
        self.Ophi = self.ys**2/np.sqrt(1-self.xs**2)
        self.Ophi_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.Ophi[self.order], k=3)
        return self.Ophi_in(z)
    
    def w_de(self, z):
        self.order = np.argsort(self.Zs)
        self.w_de1 = self.xs**2-1
        self.w_de_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.w_de1[self.order], k=3)
        return self.w_de_in(z)

    def Ep(self, z):
        return derivative(self.E, z, dx = 1e-6)

    def weff(self, z):
        return -1.0+2.0/3.0*(1.0+z)*self.Ep(z)/self.E(z)

class DiGhCondensate(Quintessence):
    def __init__(self, xi, yi, c, wm, ll, h):
        self.Ok0 = float(0) # assume the flat universe
        self.xi = float(xi)
        self.yi = float(yi)
        self.c = float(c)
        self.wm = float(wm)
        self.ll = float(ll)
        self.h = float(h)
        
        self.DGC = scalar_sol.DiGhCon(self.xi, self.yi, self.c, self.wm, self.ll)
        self.aas, self.Zs, self.xs, self.ys = self.DGC.solu(0) # solve the equation only once, otherwise, the initial conditions will be changed
        
        # the present day value of x, y, z is the final value of xi, yi, zi in the solution part
        self.x0 = self.xs[len(self.xs)-1]
        self.y0 = self.ys[len(self.ys)-1]
        self.Om0 = 1+self.x0**2-3*self.c*self.x0**4/self.y0**2
        print(self.Om0)

    def E(self, z):
        self.order = np.argsort(self.Zs)  # the InterpolatedUnivariateSpline requires the first argument must increase
        self.EE = (self.Om0*(1+self.Zs)**3/(1+self.xs**2-3*self.xs**4/self.ys**2))**0.5
        self.EEc = InterpolatedUnivariateSpline(self.Zs[self.order], self.EE[self.order], k=3)
        return self.EEc(z)
    
    def Ode(self, z):
        self.order = np.argsort(self.Zs)
        self.Ophi = self.ys**2/np.sqrt(1-self.xs**2)
        self.Ophi_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.Ophi[self.order], k=3)
        return self.Ophi_in(z)
    
    def w_de(self, z):
        self.order = np.argsort(self.Zs)
        self.w_de1 = (1-self.c*self.xs**2/self.ys**2)/(1-3*self.c*self.xs**2/self.ys**2)
        self.w_de_in = InterpolatedUnivariateSpline(self.Zs[self.order], self.w_de1[self.order], k=3)
        return self.w_de_in(z)
    
    def Ep(self, z):
        return derivative(self.E, z, dx = 1e-6)
    
    def weff(self, z):
        return -1.0+2.0/3.0*(1.0+z)*self.Ep(z)/self.E(z)

