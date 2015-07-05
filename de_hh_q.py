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

from scipy.misc import derivative
from scipy import interpolate
from scipy.interpolate import splrep
from scipy.interpolate import splev


class q_Linear:
    '''
    This is the class for the parameterization of deceleration factor: q=q0+q1*z.
    The input parameters: q0: constant
                              q1: constant
                              h: dimensionless Hubble constant
    '''
    def __init__(self, q0, q1, h):
        self.Ok0 = float(0)
        self.q0 = float(q0)
        self.q1 = float(q1)
        self.h = float(h)
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    
    def E(self, z):
        return math.e**(self.q1*z)*(1+z)**(1+self.q0-self.q1)
    
    def w_de(self, z):
        print("Geometric description has no meanings of matter and dark energy")
        return
    
    '''
    The following part is model independent, so they are also used by the other parameterizations.
    '''
    
    def Ep(self, z):
        '''
        The derivative of the expansion factor with respect to redshift z
        '''
        return derivative(self.E, z, dx = 1e-6)
    
    def q(self, z):
        '''
        The deceleration factor as a function of redshift z
        '''
        return -1.0+(1.0+z)*self.Ep(z)/self.E(z) # it has a direct expression: q = q0 + q1*z
    
    def weff(self, z):
        '''
        The effective equation of state of the universe, it can be used to read out different era of the universe: radiation-dominated, matter-dominated, and dark energy dominated.
        '''
        return -1.0+2.0/3.0*(1.0+z)*self.Ep(z)/self.E(z)
    
    def D_H(self):
        '''
        The Hubble distance from: David Hogg, arxiv: astro-ph/9905116v4
        '''
        return 3000/self.h
    
    def chi(self, z):
        '''
        It is not a observable, but needed to calculate luminosity distance and others
        '''
        self.zz = np.linspace(0, z, 1000)
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

    def mu(self, z):
        '''
        Distance module
        '''
        return np.log10(self.D_L(z))+25

    def D_solu(self, z):
        print("No solution to growth rate and growth factor")
        return

class q_CPL(q_Linear):
    '''
    This is the class for the parameterization of deceleration factor: q=q0+q1*z/(1+z).
    The input parameters: q0: constant
                          q1: constant
                          h: dimensionless Hubble constant
    '''
    def __init__(self, q0, q1, h):
        self.Ok0 = float(0)
        self.q0 = float(q0)
        self.q1 = float(q1)
        self.h = float(h)
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    
    def E(self, z):
        return math.e**(-self.q1*z/(1+z))*(1+z)**(1+self.q0+self.q1)

