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
from scipy.integrate import odeint
from scipy import interpolate
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.integrate import quad
from Numericaltools import NuIntegral
from ctypes import *
from Numericaltools import WCSF1_tab, WCSF2_tab, WCSF3_tab, Invdisttree
import sys


class LCDM:
    '''
    This is the class for standard \LambdaCDM model.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, h, Growth = False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Or0-self.Ok0)
        self.h = float(h)
        self.modelN = 4
        if Growth  == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Ok0*(1+z)**2+self.Ode0)**0.5
    
    def Ed(self, z):
        '''
        Calculate the H(z)/(1+z)
        '''
        return self.E(z)/(1+z)
    
    def w_de(self, z):
        return -1
    
    '''
    The following part is model independent, so they are also used by the other dark energy models.
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
        return -1.0+(1.0+z)*self.Ep(z)/self.E(z)
    
    def weff(self, z):
        '''
        The effective equation of state of the universe, it can be used to read out different era of the universe: radiation-dominated, matter-dominated, and dark energy dominated.
        '''
        return -1.0+2.0/3.0*(1.0+z)*self.Ep(z)/self.E(z)

    def D_H(self):
        '''
        The Hubble distance from: David Hogg, arxiv: astro-ph/9905116v4
        '''
        return 2997.9245/self.h
    
    def D_Hz(self, z):#  zzx!!! for BAO use
        '''
        redshift dependent hubble radius
        '''
        return self.D_H()/self.E(z)
    
    def chi_inte(self, z):
        if self.E(z) == 0.0:  # very rarely happens: some strange cosmology
            return 1.0/(self.E(z)+1e-15)
        else:
            return 1.0/self.E(z)
    
    def D_L(self, z):
        '''
        Luminosity distance
        '''
        r = quad(self.chi_inte, 0.0, z)[0]
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)

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
    
    def D_M(self, z):    #  zzx!!! for BAO use
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
        self.Onu = float(Onu)
        self.Oba = float(Oba)
        return 55.154*np.exp(-72.3*(self.Onu*self.h**2.0+0.0006)**2.0)/(abs(self.Om0)*self.h**2)**0.25351/(self.Oba*self.h**2)**0.12807
    
    def rd_nu(self, Onu, Oba, Neff):
        
        self.Onu = float(Onu)
        self.Oba = float(Oba)
        self.Neff = float(Neff)
        return 56.067*np.exp(-49.7*(self.Onu*self.h**2.0+0.002)**2.0)/(abs(self.Om0)*self.h**2)**0.25351/(self.Oba*self.h**2)**0.12807/(1+(self.Neff-3.406)/30.60)


    def mu(self, z):
        '''
        Distance module
        '''
        #return 42.39+5*np.log10(self.D_L(z)/self.D_H()/self.h)
        return 5.0*np.log10(self.D_L(z)/self.D_H())
    
    def t_H(self):
        return 9.78/self.h  # the unit is Gyr
    
    def chit(self, z):
        '''
        It is not a observable, but needed to calculate the lookback time
        '''
        if z <=3.0:
            self.zz = np.linspace(0, z, 1000)
        elif z>3.0:
            self.zz = np.hstack((np.linspace(0, 3.0, 800), np.linspace(3.0+3./800, z, 200)))
        self.ETz = 1.0/np.array((1+self.zz)*(map(self.E, self.zz)))
        self.ccT = interpolate.splrep(self.zz, self.ETz, s=0)
        return interpolate.splint(0, z, self.ccT)
    
    def LB_T(self, z):
        return self.t_H()*self.chit(z)  #the unit is Gyr
    

    def Om_z(self, z):
        '''
        The defination of the Om(z) parameter as a function of redshift z.
        '''
        if self.E(z) == 0:
            return self.Om0*(1+z)**3.0/(self.E(z)+1.0e-15)**2.0
        else:
            return self.Om0*(1+z)**3.0/self.E(z)**2.0
    
    def Geff(self, z):
        '''
        The effective newton constant. It is 1 in LCDM model but in other modified gravity models,
        it is a function of redshift z. It appears in the purterbation equation, not the real newton constant
        in the particular modified gravity theory.
        '''
        return 1

    def growth_equation(self, gr, z):
        '''
        The linear growth equation of purterbation as a function of redshift z.
        '''
        self.gr = gr
        return [self.gr[1], -(self.Ep(z)/self.E(z)-1.0/(1+z))*self.gr[1]+self.Geff(z)*3.0/2.0*self.Om_z(z)/(1+z)**2*self.gr[0]]

    def growth_sol1(self):
        '''
        Solve the growth equation. The initial condition is chosen at around z=30. Solve the equation
        to z=0
        '''
        self.gr0 = [1./(30.+1.), -1.0/(1.0+30.0)**2.0]
        return odeint(self.growth_equation, self.gr0, self.zs_gr)
    
    def D_z_1(self, z):
        '''
        Represent the solution of D as a function of z by the use of interpolation.
        '''
        self.Dz1 = self.solution[:,0]
        self.orderD = np.argsort(self.zs_gr)
        self.Dz11 = UnivariateSpline(self.zs_gr[self.orderD], self.Dz1[self.orderD], k=3, s=0)
        return self.Dz11(z)

    def D_z(self, z):
        '''
        Normalize the solution of D(z) to be D(0)=1
        '''
        return self.D_z_1(z)/self.D_z_1(0)

    def Dp_z(self, z):
        '''
        Represent the solution of D' as a function of z by the use of interpolation
        '''
        self.Dpz1 = self.solution[:,1]
        self.orderDp = np.argsort(self.zs_gr)
        self.Dpz11 = UnivariateSpline(self.zs_gr[self.orderDp], self.Dpz1[self.orderDp], k=3, s=0)
        return self.Dpz11(z)

    def f_z(self, z):
        '''
        Change the function of D' to f=dlnD/dlna (the growth rate)
        '''
        return -(1+z)/self.D_z_1(z)*self.Dp_z(z)

    def Dfsig8(self, z, sigma8):
        '''
        The observable of growth rate data: f(z)D(z)*sigma8
        '''
        self.sigma8=sigma8
        return self.D_z(z)*self.f_z(z)*self.sigma8

    def GrowthIntegrad_a(self, a):
        z = 1./a-1.0
        return 1./(self.E(z)**3.0*a*a*a)

    def growth_D(self, z):  # only correct for universe with matter and CC. When the curvature or radiation is included, it's not applicable anymore.
        self.af = 1./(1.+z)
        self.r = quad(self.GrowthIntegrad_a, 1e-7, self.af)
        self.gr_D = self.E(z)*self.r[0]
        self.gr_Nor = quad(self.GrowthIntegrad_a, 1e-7, 1.0)[0]
        return self.gr_D/self.gr_Nor


class Flat_LCDM(LCDM):
    '''
    This is the flat LCDM model. It has only two input parameters: Om0 and h.
    '''
    def __init__(self, Om0, Or0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Or0 = float(Or0)
        self.h = float(h)
        self.Ok0 = float(0)
        self.Ode0 = float(1-self.Om0-self.Or0)
        self.modelN = 3
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Or0*(1.+z)**4.+self.Ode0)**0.5


class Topo_defc_2D(LCDM):
    '''
    This is the class for the FRW cosmology with 2D topological defects. The equation of state is w_x = -2/3.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ori = float(Or0)
        self.Otopo = float(1-self.Om0-self.Ok0-self.Or0)
        self.modelN = 4
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Otopo*(1.+z))**0.5
    
    def w_de(self, z):
        return -2/3

class Phan_DE(LCDM):
    '''
    This is the class for the FRW cosmology with Phantom dark energy. The equation of state is w_x = -4/3.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          h: dimensionless Hubble constant
        '''
    def __init__(self, Om0, Ok0, Or0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Oph = float(1-self.Om0-self.Ok0-self.Or0)
        self.modelN = 4
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    
    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Oph*(1.+z)**(-1.))**0.5

    def w_de(self, z):
        return -4/3

class XCDM(LCDM):
    '''
    This is the class for the XCDM cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          w: the equation of state of dark energy
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, w, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1.-self.Om0-self.Ok0-self.Or0)
        self.w = float(w)
        self.h = float(h)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*np.power((1.0+z), 3.0*(1.0+self.w)))**0.5
    
    def w_de(self, z):
        return self.w

class CG(LCDM):
    '''
    This is the class for the Chaplygin gas model (CG), the equation of state of dark energy is p = -A/\rho.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          As: constant related to A   0<A<1
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, As, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.As = float(As)
        self.h = float(h)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
    def E(self,z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*abs(self.As+(1.-self.As)*np.power((1.0+z), 6.0))**0.5)**0.5

    def ODE(self, z):
        '''
        The energy component of dark energy
        '''
        return self.E(z)**2.-self.Om0*(1.+z)**3.-self.Ok0*(1.+z)**2.-self.Or0*(1.+z)**4.
    
    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)
    
    def w_de(self, z):
        '''
        The "equivalent" equation of state of dark energy
        '''
        return self.ODEp(z)/3./self.ODE(z)*(1.+z)-1.

class GCG(CG):
    '''
    http://arxiv.org/pdf/1105.1870.pdf
    This is the class for the Generalized Chaplygin gas model (GCG), the equation of state of dark energy is p = -A/\rho^\alpha.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          As: constant related to A
                          alpha: a constant in the range 0 < \alpha < 1
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, As, alpha, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.As = float(As)
        self.alpha = float(alpha)
        self.h  = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*abs(self.As+(1.-self.As)*(np.power((1.+z), 3.*(1.+self.alpha))))**(1./(1.+self.alpha)))**0.5

class MCG(CG):
    '''
    This is the class for the Modified Cahplygin gas model (MCG)  http://arxiv.org/pdf/0905.2281v2.pdf
    '''

    def __init__(self, Om0, Ok0, Or0, As, alpha, B, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.As = float(As)
        self.alpha = float(alpha)
        self.B = float(B)
        self.h  = float(h)
        self.modelN = 7
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
               
    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*abs(self.As+(1.-self.As)*(np.power((1.+z), 3.*(1.+self.B)*(1.+self.alpha))))**(1./(1.+self.alpha)))**0.5


class H_LOG(CG):
    '''
        This is the class for the XCDM cosmology.
        The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
        Ok0: the current energy density fraction of curvature
        w: the equation of state of dark energy
        h: dimensionless Hubble constant
        '''
    def __init__(self, Om0, Ok0, Or0, beta, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1.-self.Om0-self.Ok0-self.Or0)
        self.alpha = np.exp(self.Ode0)
        self.beta = float(beta)
        self.h = float(h)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+np.log(self.alpha+self.beta*z))**0.5

class W_Linear(LCDM):
    '''
    This is the class for the dark energy model with a linear parameterization of equation of state of the dark energy: w=w0+w1*z
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          w0 and w1: constants related to the EoS parameterization
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, w0 ,w1, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*(1.+z)**(3.*(self.w0-self.w1+1.))*np.power(math.e, 3.*self.w1*z))**0.5

    def w_de(self, z):
        return self.w0+self.w1*z

class W_CPL(LCDM):
    '''
    This is the class for the CPL parameterization of the equation of state of dark energy: w=w0+w1*z/(1+z)
    The input parameters have the same meanings as W_linear class
    '''
    def __init__(self, Om0, Ok0, Or0, w0 ,w1, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*(1.+z)**(3.*(self.w0+self.w1+1.))*np.power(math.e, -3.*self.w1*z/(1.+z)))**0.5
    
    def w_de(self, z):
        return self.w0+self.w1*z/(1+z)

class W_JBP(LCDM):
    '''
    http://arxiv.org/pdf/astro-ph/0409161v2.pdf
    This is the class for the JBP parameterization of the equation of state of dark energy.
    The input parameters have the same meanings as W_linear class
    '''
    def __init__(self, Om0, Ok0, Or0, w0 ,w1, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    
    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0*(1.+z)**(3.*(self.w0+1.))*np.power(math.e, 3.*self.w1*z**2./2.0/(1.+z)**2))**0.5
    def w_de(self, z):
        return self.w0+self.w1*z/(1+z)**2

class W_Hann(LCDM):
    '''
    http://arxiv.org/pdf/astro-ph/0409161v2.pdf
    This is the class for the Hannestad parameterization of the equation of state of dark energy.
    The input parameters have the same meanings as W_linear class
    '''
    def __init__(self, Om0, Ok0, Or0, w0 ,w1, zt, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Or0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.zt = float(zt)
        self.n = float(n)
        self.at = 1/(self.zt+1)
        self.h = float(h)
        self.modelN = 8
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    
    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Ode0*(1+z)**(3*(self.w1+1))*(((self.w1+self.w0*self.at**self.n)*(1+z)**self.n)/(self.w1+self.w0*self.at**self.n*(1+z)**self.n))**(3*(self.w1-self.w0)/self.n))**0.5
    
    def w_de(self, z):
        return (1+((1+z)/(1+self.zt))**self.n)/(1.0/self.w0+1.0/self.w1*((1+z)/(1+self.zt))**self.n)


class W_Interp(LCDM):
    '''
    This is the class for the interpolation of the dark energy EoS
    '''

    def __init__(self, Om0, Ok0, Or0, w0, w1, w2, w3, w4, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode = 1.0-self.Om0-self.Ok0-self.Or0
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.w_p = np.array([self.w0, self.w1, self.w2, self.w3, self.w4])
        self.h = float(h)
        self.modelN = 9
        self.zt = [0.0, 0.2, 0.4, 0.6, 1.8]
        self.a = np.ones(len(self.zt))
        self.b = np.ones(len(self.zt))
        for i in range(len(self.zt)-1):
            self.a[i] = (self.w_p[i+1]-self.w_p[i])/(self.zt[i+1]-self.zt[i])
            self.b[i] = -self.zt[i]*(self.w_p[i+1]-self.w_p[i])/(self.zt[i+1]-self.zt[i])+self.w_p[i]
        self.a[-1]=0.0
        self.b[-1]=self.w_p[-1]
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def w_de(self, z):
        if (self.zt[0] <=z) & (z <= self.zt[1]):
            return self.a[0]*z+self.b[0]
        elif (self.zt[1] < z) & (z <= self.zt[2]):
            return self.a[1]*z+self.b[1]
        elif (self.zt[2] < z) & (z <= self.zt[3]):
            return self.a[2]*z+self.b[2]
        elif (self.zt[3] < z) & (z <= self.zt[4]):
            return self.a[3]*z+self.b[3]
        elif self.zt[4] < z:
            return self.a[4]*z+self.b[4]
    '''
    def func1(self, z):
        return (1+self.w_de(z))/(1+z)
    
    def fz_func1(self, z):
        return math.exp(3*quad(self.func1, 0, z)[0])
    '''

    def fz_func(self, z):
        if (self.zt[0] <=z) & (z <= self.zt[1]):
            self.deint = self.a[0]*z+(self.b[0]-self.a[0]+1.0)*np.log(1.0+z)
        
        elif (self.zt[1] < z) & (z <= self.zt[2]):
            self.deint = self.a[0]*self.zt[1]+(self.b[0]-self.a[0]+1.0)*np.log(1.0+self.zt[1]) + self.a[1]*(z-self.zt[1])+(self.b[1]-self.a[1]+1.0)*np.log((1.0+z)/(1.0+self.zt[1]))

        elif (self.zt[2] < z) & (z <= self.zt[3]):
            self.deint = self.a[0]*self.zt[1]+(self.b[0]-self.a[0]+1.0)*np.log(1.0+self.zt[1]) + self.a[1]*(self.zt[2]-self.zt[1])+(self.b[1]-self.a[1]+1.0)*np.log((1.0+self.zt[2])/(1.0+self.zt[1]))+ self.a[2]*(z-self.zt[2])+(self.b[2]-self.a[2]+1.0)*np.log((1.0+z)/(1.0+self.zt[2]))

        elif (self.zt[3] < z) & (z <= self.zt[4]):
            self.deint = self.a[0]*self.zt[1]+(self.b[0]-self.a[0]+1.0)*np.log(1.0+self.zt[1]) + self.a[1]*(self.zt[2]-self.zt[1])+(self.b[1]-self.a[1]+1.0)*np.log((1.0+self.zt[2])/(1.0+self.zt[1]))+ self.a[2]*(self.zt[3]-self.zt[2])+(self.b[2]-self.a[2]+1.0)*np.log((1.0+self.zt[3])/(1.0+self.zt[2]))+self.a[3]*(z-self.zt[3])+(self.b[3]-self.a[3]+1.0)*np.log((1.0+z)/(1.0+self.zt[3]))
        else:
            self.deint = self.a[0]*self.zt[1]+(self.b[0]-self.a[0]+1.0)*np.log(1.0+self.zt[1]) + self.a[1]*(self.zt[2]-self.zt[1])+(self.b[1]-self.a[1]+1.0)*np.log((1.0+self.zt[2])/(1.0+self.zt[1]))+ self.a[2]*(self.zt[3]-self.zt[2])+(self.b[2]-self.a[2]+1.0)*np.log((1.0+self.zt[3])/(1.0+self.zt[2]))+self.a[3]*(self.zt[4]-self.zt[3])+(self.b[3]-self.a[3]+1.0)*np.log((1.0+self.zt[4])/(1.0+self.zt[3]))+(1+self.w_p[-1])*np.log((1+z)/(1+self.zt[4]))

        return math.exp(3*self.deint)

    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Ode*self.fz_func(z))**0.5

    def Geff(self, z):
        return 1.0

class W_Piece(LCDM):
    '''
    This is the class for the interpolation of the dark energy EoS
    '''
    def __init__(self, Om0, Ok0, Or0, w0, w1, w2, w3, w4, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode = 1.0-self.Om0-self.Ok0-self.Or0
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.w_p = np.array([self.w0, self.w1, self.w2, self.w3, self.w4])
        self.h = float(h)
        self.modelN = 9
        self.zt = [0.0, 0.2, 0.4, 0.6, 1.8]
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def w_de(self, z):
        if (self.zt[0] <=z) & (z <= self.zt[1]):
            return self.w0
        elif (self.zt[1] < z) & (z <= self.zt[2]):
            return self.w1
        elif (self.zt[2] < z) & (z <= self.zt[3]):
            return self.w2
        elif (self.zt[3] < z) & (z <= self.zt[4]):
            return self.w3
        elif self.zt[4] < z:
            return self.w4

    def fz_func(self, z):
        
        if (self.zt[0] <=z) & (z <= self.zt[1]):
            return (1.0+z)**(3*(1+self.w_p[0]))
        
        elif (self.zt[1] < z) & (z <= self.zt[2]):
            return ((1.0+self.zt[1])/(1.0+self.zt[0]))**(3.0*(1.0+self.w_p[0]))*((1.0+z)/(1.0+self.zt[1]))**(3.0*(1+self.w_p[1]))
        
        elif (self.zt[2] < z) & (z <= self.zt[3]):
            return ((1.0+self.zt[1])/(1.0+self.zt[0]))**(3*(1+self.w_p[0]))*((1.0+self.zt[2])/(1.0+self.zt[1]))**(3*(1+self.w_p[1]))*((1.0+z)/(1.0+self.zt[2]))**(3*(1.0+self.w_p[2]))
        
        elif (self.zt[3] < z) & (z <= self.zt[4]):
            return ((1.0+self.zt[1])/(1.0+self.zt[0]))**(3*(1+self.w_p[0]))*((1.0+self.zt[2])/(1.0+self.zt[1]))**(3*(1+self.w_p[1]))*((1.0+self.zt[3])/(1.0+self.zt[2]))**(3*(1.0+self.w_p[2]))*((1.0+z)/(1.0+self.zt[3]))**(3*(1+self.w_p[3]))
        
        else:
            return ((1.0+self.zt[1])/(1.0+self.zt[0]))**(3*(1+self.w_p[0]))*((1.0+self.zt[2])/(1.0+self.zt[1]))**(3*(1+self.w_p[1]))*((1.0+self.zt[3])/(1.0+self.zt[2]))**(3*(1.0+self.w_p[2]))*((1.0+self.zt[4])/(1.0+self.zt[3]))**(3*(1+self.w_p[3]))*((1.0+z)/(1.0+self.zt[4]))**(3*(1+self.w_p[4]))

    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Ode*self.fz_func(z))**0.5

    def Geff(self, z):
        return 1.0


class Casimir(CG):
    '''
    http://arxiv.org/pdf/astro-ph/0606731.pdf
    This is the class for the FRW cosmology with Casimir effect. Negative (1+z)^4 type contribution in the expansion factor.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          Ocass0:
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Ocas0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ocas0 = float(Ocas0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Ocas0-self.Or0)
        self.h = float(h)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode0+self.Ocas0*(1.+z)**4.)**0.5
'''
not sure, need to check the paper
class CGB:
    def __init__(self, Om0, )
'''

class Card(LCDM):
    # http://arxiv.org/pdf/astro-ph/0302064.pdf
    '''
    http://arxiv.org/pdf/astro-ph/0509415.pdf
    This is the class for the Cardissian expansion cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          The curvature is zero
                          Or0: the current energy density fraction of radiation
                          n: constant related to the Cardassian term in the Friedmann equation
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, q, n, h, Growth=False):
        
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.q = float(q)
        self.n = float(n)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return np.sqrt(abs((self.Om0*(1.+z)**3.+self.Or0*(1.+z)**4.)*(1.+np.power( self.Om0+self.Or0, self.q*(1.-self.n) )*(((1.-self.Ok0)/(self.Om0+self.Or0))**self.q-1.0)/np.power( self.Om0*(1.+z)**3.+self.Or0*(1.+z)**4., self.q*(1.-self.n) ))**(1./self.q)+self.Ok0*(1.+z)**2.))
    
    def ODE(self, z):
        '''
        The energy component of dark energy
        '''
        return self.E(z)**2.-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-self.Ok0*(1.+z)**2.
    
    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)
    
    def w_de(self, z):
        '''
        The "equivalent" equation of state of dark energy
        '''
        return self.ODEp(z)/3/self.ODE(z)*(1+z)-1

class sDGP(CG):
    '''
    http://arxiv.org/pdf/0905.1112.pdf
    http://arxiv.org/pdf/0905.1735v2.pdf    ==> Geff
    This is the class for the DGP cosmology: positive branch
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Occ: Cosmological Constant.
                          h: dimensionless Hubble constant    
    '''
    def __init__(self, Om0, Ok0, Or0, Occ, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Occ = float(Occ)
        self.h = float(h)
        self.ODGP = float(1.0-self.Om0-self.Or0-self.Ok0-self.Occ)
        self.Orc = self.ODGP**2.0/4.0/(1-self.Ok0)
        self.alpha = 2*np.sqrt(1-self.Ok0)*(3-4*self.Ok0+2*self.Om0*self.Ok0+self.Ok0**2)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        #if 2*self.Orc-self.ODGP > 0:
            #print('Warning: sDGP: The parameters are not appropriate, please choose model: nDGP')


    def E(self, z):
        return abs((np.sqrt(abs(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Occ+self.Orc))+np.sqrt(self.Orc))**2+self.Ok0*(1+z)**2)**0.5
    '''
    The equation of state of dark energy is calculated by the same method in CG class: a effective EoS
    '''

    def Geff(self,z):
        return (4*self.Om_z(z)**2-4*(1-self.Ok0)**2+self.alpha)/(3*self.Om_z(z)**2-3*(1-self.Ok0)**2+self.alpha)



class nDGP(CG):
    '''
    http://arxiv.org/pdf/0905.1112.pdf
    This is the class for the DGP cosmology: normal branch
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Occ: Cosmological Constant.
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Occ, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Occ = float(Occ)
        self.h = float(h)
        self.ODGP = float(1.0-self.Om0-self.Or0-self.Ok0-self.Occ)
        self.Orc = self.ODGP**2.0/4.0/(1-self.Ok0)
        self.alpha = 2*np.sqrt(1-self.Ok0)*(3-4*self.Ok0+2*self.Om0*self.Ok0+self.Ok0**2)

        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        #if 2*self.Orc-self.ODGP < 0:
            #print('Warning: nDGP: The parameters are not appropriate, please choose model: sDGP')


    def E(self, z):
        return abs((np.sqrt(abs(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Occ+self.Orc))-np.sqrt(self.Orc))**2+self.Ok0*(1+z)**2)**0.5

    def Geff(self,z):
        return (4*self.Om_z(z)**2-4*(1-self.Ok0)**2+self.alpha)/(3*self.Om_z(z)**2-3*(1-self.Ok0)**2+self.alpha)

'''
DDG should be the same as DGP model: DDG is the cosmological solution of DGP gravity?
class DDG(CG): # according to the paper, redefine r0h0 = r0*H0 to get the constraint condition

    #This is the class for the DDG cosmology
    #The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                         # r0h0: constant defined as r0*H0.
                         # h: dimensionless Hubble constant
    
    def __init__(self, Om0, r0h0, h):
        self.Ok0 = float(0)  # assume the curvature is zero
        self.Om0 = float(Om0)
        self.r0h0 = float(r0h0)
        self.Ode0 = float(1+1/self.r0h0-self.Om0)
        self.h =float(h)
        self.modelN = 3

    def E(self, z):
        return (-0.5/self.r0h0+np.sqrt(self.Om0*(1+z)**3+self.Ode0+1/4/self.r0h0**2))**0.5
'''
class RS(LCDM):
    # http://arxiv.org/pdf/hep-th/0101060.pdf
    '''
    This is the class for the RS braneworld cosmology without cosmological constant
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Odr0: the current energy density fraction from dark radiation
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Odr0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Odr0 = float(Odr0)
        self.Oll0 = float((1-self.Om0-self.Ok0-self.Or0-self.Odr0)/(self.Om0+self.Or0)**2)
        self.h = float(h)
        self.modelN = 5
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Odr0*(1+z)**4+self.Oll0*(self.Om0*(1+z)**3+self.Or0*(1+z)**4)**2)**0.5
    
    def ODE(self, z):
        '''
            The energy component of dark energy
            '''
        return self.E(z)**2-self.Om0*(1+z)**3+self.Odr0*(1+z)**4-self.Oro*(1+z)**4  # Note: different from GC, GCG and DE_Card
    
    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)
    
    def w_de(self, z):
        '''
            The "equivalent" equation of state of dark energy
            '''
        return self.ODEp(z)/3/self.ODE(z)*(1+z)-1

class RSL(RS):
    '''
    This is the class for the RS braneworld cosmology with cosmological constant
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Odr0: the current energy density fraction from dark radiation
                          Oll0: the current energy density fraction contributed from brane tension
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Odr0, Ode0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Odr0 = float(Odr0)
        self.Ode0 = float(Ode0)
        self.Oll0 = float((1-self.Om0-self.Ok0-self.Or0-self.Odr0-self.Ode0)/(self.Om0+self.Or0)**2)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Odr0*(1+z)**4+self.Oll0*(self.Om0*(1+z)**3+self.Or0*(1+z)**4)**2+self.Ode0)**0.5
    # the calculation of effective equation of state of dark energy is the same as RS model

class S_Brane1(CG):
    #http://arxiv.org/pdf/astro-ph/0202346.pdf
    '''
    This is the class for the Shtanov Brane1 cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Osig0: the current energy density fraction from brane tension
                          Oll0: the current energy density fraction contributed from Planck mass scales in different dimensions
                          The dark radiation is neglected.
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Osig0, Oll0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Osig0 = float(Osig0)
        self.Oll0 = float(Oll0)
        self.Ode0 = float((self.Om0+self.Ok0+self.Or0+self.Osig0-1)**2.0/4.0/self.Oll0-1+self.Ok0)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

        if self.Om0+self.Ok0+self.Osig0+self.Or0 - 1 < 0:
            print('Warning: S_Brane1: The parameters are not appropriate, please choose model: S_Brane2')

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Osig0+2*self.Oll0-2*np.sqrt(self.Oll0)*np.sqrt(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Osig0+self.Oll0+self.Ode0))**0.5
    # Since the dark radiation is ignored, the dark energy component and its effective equation of state is calculated the same way as CG model.

class S_Brane2(CG):
    '''
    This is the class for the Shtanov Brane2 cosmology. 
    The meanings of the input parameters are the same as Shtanov Brane1 model.
    '''
    def __init__(self, Om0, Ok0, Or0, Osig0, Oll0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Osig0 = float(Osig0)
        self.Oll0 = float(Oll0)
        self.Ode0 = float((self.Om0+self.Ok0+self.Or0+self.Osig0-1)**2.0/4.0/self.Oll0-1+self.Ok0)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

        if self.Om0+self.Ok0+self.Osig0+self.Or0 - 1 > 0:
            print('Warning: S_Brane2: The parameters are not appropriate, please choose model: S_Brane1')
    
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Osig0+2*self.Oll0+2*np.sqrt(self.Oll0)*np.sqrt(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Osig0+self.Oll0+self.Ode0))**0.5


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
        self.modelN = 3
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    
    def E(self, z):
        return math.e**(self.q1*z)*(1+z)**(1+self.q0-self.q1)
    
    def Ed(self, z):
        '''
        Calculate the H(z)/(1+z)
        '''
        return self.E(z)/(1+z)
    
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
        
    def D_Hz(self, z):#  zzx!!! for BAO use
        '''
        redshift dependent hubble radius
        '''
        return self.D_H()/self.E(z)
    
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
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*self.chi(z))
        elif self.Ok0 == 0:
            return self.D_H()*(1+z)*self.chi(z)
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*self.chi(z))

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
        self.Onu = abs(float(Onu))
        self.Oba = abs(float(Oba))
        return 55.154*np.exp(-72.3*(self.Onu*self.h**2.0+0.0006)**2.0)/(abs(self.Om0)*self.h**2)**0.25351/(self.Oba*self.h**2)**0.12807
    
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
        self.modelN = 3
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    
    def E(self, z):
        return math.e**(-self.q1*z/(1+z))*(1+z)**(1+self.q0+self.q1)


'''
class MAG:
    # find the related papers
    def __init__(self, Om0, Ok0, Ophi0):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ophi0 = float(Ophi0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Ode0)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ophi0*(1+z)**6+self.Ode0)**0.5
'''
# the 8,9,10th models in the paper should be considered more carefully.

class EDE(CG):
    #http://arxiv.org/pdf/1503.01235.pdf
    '''
    This is the class for the early dark energy model.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ode: the energy density fraction of early dark energy
                          w0: equation of state
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, Ode, w0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode = float(Ode)
        self.w0 = float(w0)
        self.Od0 = float(1.0-self.Om0-self.Or0-self.Ok0)
        self.h = float(h)
        self.modelN = 6
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    '''
        The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
        '''
    
    def E(self, z):
        if (self.Od0+(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.)*(1.+z)**(-3.-3.*self.w0)) == 0.0:
            print(self.Om0, self.Ok0, self.Ode, self.w0)
        self.Od = (self.Od0-self.Ode*(1.-np.power(1.+z, (3.*self.w0)) ))/(self.Od0+(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.)*(1.+z)**(-3.-3.*self.w0))+self.Ode*(1.-np.power(1.+z, (3.*self.w0)))
        if self.Od == 1.0:
            self.Od = self.Od + 1e-15
        return abs((self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.)/(1.-self.Od))**0.5


class OA(LCDM):
    '''
    This is the Oscillating ansatz: comparison of cosmological models using recent supernova data
    '''
    def __init__(self, Om0, Ok0, Or0, a1, a2, a3, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a3 = float(a3)
        self.h = float(h)
        self.modelN = 7
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.0+z)**3.0+self.Ok0*(1.0+z)**2.0+self.Or0*(1.0+z)**4.0+self.a1*np.cos(self.a2*z**2.0+self.a3)+(1.0-self.a1*np.cos(self.a3)-self.Om0))**0.5

class fT_PL(LCDM):
    '''
    This is the class for the power law f(T) gravity model
    '''
    def __init__(self, Om0, Or0, b, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.h = float(h)
        self.b = float(b)
        self.modelN = 4  #  5 when Ok0 is added
        self.OF0 = 1.-self.Om0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Expo(self, Ee, z):
        self.Ee = Ee
        return self.Ee**2.-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-(1.-self.Om0-self.Or0)*self.Ee**(2.*self.b)

    def E(self, z):

        return fsolve(self.Expo, 1.0, args=(z))
    
    def chi_inte(self, z):
        return 1.0/self.E(z)
    
    def D_L(self, z):
        r = quad(self.chi_inte, 0.0, z)[0]
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)

    def Geff(self, z):
        if self.b == 0.5:
            self.b = self.b + 1e-15
        return 1.0/(1.+self.b*self.OF0/(1.-2.*self.b)/self.E(z)**(2.*(1.-self.b)))


class fT_EXP(fT_PL):
    '''
    This is the class for the exponential f(T) gravity model
    '''
    def __init__(self, Om0, Or0, b, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(0)
        self.Or0 = float(Or0)
        self.h = float(h)
        if b==0.0:
            print("parameter b is 0")
            b = b+1e-10
        self.p = float(1.0/b)
        self.modelN = 4  
        self.OF0 = 1.0-self.Om0-self.Or0
        self.alpha = self.OF0/(1.-(1.+2.*self.p)*np.exp(-self.p))
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Expo(self, Ee, z):
        self.Ee = Ee
        return self.Ee**2.-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-self.alpha*(1.-(1.+2.*self.p*self.Ee**2.)*np.exp(-self.p*self.Ee**2.))

    def E(self, z):
        
        return fsolve(self.Expo, 1, args=(z))

    def Geff(self, z):
        if (2.*self.p+1.)*np.exp(-self.p) == 1:
            self.p = self.p+1e-15
        return 1.0/(1.+self.OF0*self.p*np.exp(-self.p*self.E(z)**2.)/(1.-(1.+2.*self.p)*np.exp(-self.p)))


class fT_EXP_Linder(fT_PL):
    '''
        This is the class for the exponential f(T) gravity model proposed by Linder
        '''
    def __init__(self, Om0, Or0, b, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.h = float(h)
        if b==0.0:
            print("parameter b is 0")
            b = b+1e-10
        self.p = float(1.0/b)
        self.modelN = 4  #  5 when Ok0 is added
        self.OF0 = 1.-self.Om0-self.Or0
        self.alpha = self.OF0/(1.-(1.+self.p)*np.exp(-self.p))
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Expo(self, Ee, z):
        self.Ee = Ee
        return self.Ee**2.0-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-self.alpha*(1.-(1.+self.p*self.Ee)*np.exp(-self.p*self.Ee))

    def E(self, z):
        
        return fsolve(self.Expo, 1.0, args=(z))

    def Geff(self, z):
        if np.power(self.p+1., self.p) == 1.0:
            self.p = self.p+1e-15
        return 1.0/(1.+self.OF0*self.p*np.exp(-self.p*self.E(z))/(2.*self.E(z)*(1.-(1.+self.p)*np.exp(-self.p))))

class fT_LOG(fT_PL):
    '''
    This is the class for the logarithmic f(T) gravity model
    '''
    def __init__(self, Om0, Or0, q, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.h = float(h)
        self.q = float(q)
        self.modelN = 4
        self.OF0 = 1.-self.Om0-self.Or0
        self.alpha = self.OF0*np.sqrt(self.q)/2.0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(0.5*np.sqrt(self.OF0**2.+4.*(self.Om0*(1.+z)**3.+self.Or0*(1.+z)**4.))+self.OF0/2.0)

    def Geff(self, z):
        if np.sqrt(self.q)/self.E(z) == 1:
            self.q = self.q+1e-15
        return 1.0/(1.+self.OF0/2./self.E(z)*(np.log(np.sqrt(self.q)/self.E(z))-1.))

class fT_tanh(fT_PL):
    '''
    This is the class for the hyperbolic-tangent f(T) gravity model
    http://arxiv.org/pdf/1008.3669.pdf
    '''
    def __init__(self, Om0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
        self.modelN = 4
        self.OF0 = 1.-self.Om0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Expo(self, Ee, z):
        self.Ee = Ee
        return self.Ee**2.-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-self.OF0*self.Ee**(2.*(self.n-1.))*(2.*np.cosh(1./self.Ee**2.)**(-2.)+(1.-2.*self.n)*self.Ee**2.*np.tanh(1./self.Ee**2.))/(2.*np.cosh(1.)**(-2.)+(1.-2.*self.n)*np.tanh(1.))
    
    def E(self, z):
        
        return fsolve(self.Expo, 1, args=(z))
    
    def chi(self, z):
        return quad(self.chi_inte, 0.0, z)[0]
    
    def chi_inte(self, z):
        return 1.0/self.E(z)

    def Geff(self, z):
        if 2.*np.cosh(1.)**(-2.)+(1.-2.*self.n)*np.tanh(1.) == 0:
            self.n = self.n+1e-15
        return 1.0/(1.+self.OF0*self.E(z)**(2.*(self.n-2.))*(self.n*self.E(z)**2.*np.tanh(1./self.E(z)**2.)-np.cosh(1./self.E(z)**2.)**(-2.))/(2.*np.cosh(1.)**(-2.)+(1.-2.*self.n)*np.tanh(1.)))

class SL_DE(CG):
    '''
    This is the class for the Slow Roll dark energy model. The EoS of dark energy in this model is modeled by -1 plus a perturbation term which is related to the expansion factor.
    '''
    def __init__(self, Om0, Ok0, Or0, dW0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.dW0 = float(dW0)
        self.h = float(h)
        self.modelN = 5
        self.Ode = 1.0-self.Om0-self.Or0-self.Ok0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode*abs((1.+z)**3./(self.Om0*(1.+z)**3.+self.Ok0*(1.+z)**2.+self.Or0*(1.+z)**4.+self.Ode))**(self.dW0/self.Ode))**0.5

class QCD_Ghost(CG):
    '''
    This is the class for the QCD ghost dark energy model.
    '''

    def __init__(self, Om0, Ok0, Or0, gamma, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)  # add Ok0 as a free parameter later
        self.Or0 = float(Or0)
        self.gamma = float(gamma)
        if self.gamma == 0.0:
            self.gamma = self.gamma+1e-15
        self.h = float(h)
        self.modelN = 5
        self.kappa = (1.0-(self.Om0+self.Or0+self.Ok0)/self.gamma)/2.0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):

        return self.kappa+abs(self.kappa**2.+(self.Om0*(1+z)**3.+self.Or0*(1.+z)**4.+self.Ok0*(1+z)**2.)/self.gamma)**0.5


class PNGB(LCDM):
    '''
    This is the class for the PNGB model: Pseudo-Nambu Goldstone boson model.
    The prior of the parameters: -2 < w0 < -0.4, 0 < F < 8
    information: http://arxiv.org/pdf/0704.2064.pdf
    '''

    def __init__(self, Om0, Ok0, Or0, w0, F, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.w0 = float(w0)
        self.F = float(F)
        if self.F == 0.0:
            self.F = self.F+1e-15
        self.h = float(h)
        self.modelN = 6
        self.Ode0 = 1.0-self.Om0-self.Or0-self.Ok0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        self.rhox = np.exp(3.0*(1+self.w0)/self.F*(1.0-1.0/np.power(1.0+z, self.F)))
           
        return min(10.0**50, abs(self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Or0*(1+z)**4+self.Ode0*self.rhox)**0.5)

    def w_de(self, z):
        return -1.0+(1.0+self.w0)*(1.0+z)**(-self.F)

class HDE(LCDM):
    '''
    This is the holographic dark energy model: the IR cut-off is the future event horizon
    '''

    def __init__(self, Om0, Ok0, Or0, c, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = -float(Ok0)   #  note: the paper has a different definition of Ok, so there is a negative sign.
        self.Or0 = float(Or0)
        self.c = float(c)
        self.h = float(h)
        self.Ode0 = 1.0-self.Om0-self.Or0+self.Ok0
        self.Ode_solution = self.Ode_sol()
        if Growth == True:
            self.zs_gr = np.hstack((np.linspace(30, 2, 400), np.linspace(1.99, 0, 400)))
            self.solution = self.growth_sol1()

    def Odep(self, Odee, z):
        self.Odee = Odee
        self.gamma = self.Ok0/self.Om0
        self.beta = self.Or0/self.Om0
        
        self.fac1 = 1-self.c**2*self.gamma*(1.-self.Odee)/self.Odee/(self.beta*(1.+z)**2.0+1.+z-self.gamma)
        if self.fac1 < 0.0:
            #print("Parameters are not physical.")
            self.fac1 = abs(self.fac1)
        
        self.fac2 = self.Odee*(1.-self.Odee)*(1.+2.*self.beta*(1.+z))/(self.beta*(1.0+z)**2 + 1. + z - self.gamma)
        
        return -2.0*self.Odee**1.5*(1.-self.Odee)/(1.0+z)/self.c*np.sqrt(self.fac1) - self.fac2

    #zs_Ode = np.hstack((np.linspace(0.0, 1.0, 500), np.linspace(1.0+1./500, 3.0, 300), np.linspace(3.0+3./300, 10, 300), np.linspace(10.0+10./300, 1100, 300)))
    zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))


    def Ode_sol(self):
        return odeint(self.Odep, self.Ode0, self.zs_Ode)

    def Ode_z(self, z):
        self.orderODE = np.argsort(self.zs_Ode)
        self.Ode_intp = UnivariateSpline(self.zs_Ode[self.orderODE], self.Ode_solution[self.orderODE], k=3, s=0)
        return self.Ode_intp(z)

    def E(self, z):
        return abs((self.Om0*(1.0+z)**3.0-self.Ok0*(1.0+z)**2.0+self.Or0*(1.0+z)**4.0)/(1-self.Ode_z(z)))**0.5
    
    def chi(self, z):
        return quad(self.chi_inte, 0.0, z)[0]
    
    def chi_inte(self, z):
        return 1.0/self.E(z)

    def w_de(self, z):
        self.w_de_fac = self.Ode_z(z)-self.c**2*self.Ok0
        if self.w_de_fac < 0.0:
            #print("bad parameters")
            self.w_de_fac = -self.w_de_fac
        return -1.0/3.0*(1.0+2.0/self.c*np.sqrt(self.w_de_fac))


class ADE(LCDM):
    '''
    This is the holographic dark energy model: the IR cut-off is the conformal age of the universe
    '''
    
    def __init__(self, Om0, Ok0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
        self.Ode0 = 1.0-self.Om0-self.Or0-self.Ok0
        self.Ode_solution = self.Ode_sol()
        if Growth == True:
            self.zs_gr = np.hstack((np.linspace(30, 2, 400), np.linspace(1.99, 0, 400)))
            self.solution = self.growth_sol1()

    def Odep(self, Odee, z):
        self.Odee = Odee
        self.Gz = (self.Or0*(1.0+z)**2-self.Ok0)/(self.Om0*(1.0+z)+self.Or0*(1.0+z)**2+self.Ok0)

        return -self.Odee*(1.0-self.Odee)/(1.0+z)*(3.0+self.Gz-2.0*(1.0+z)/self.n*np.sqrt(self.Odee))

    zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))
    
    def Ode_sol(self):
        return odeint(self.Odep, self.Ode0, self.zs_Ode)
    
    def Ode_z(self, z):
        self.orderODE = np.argsort(self.zs_Ode)
        self.Ode_intp = UnivariateSpline(self.zs_Ode[self.orderODE], self.Ode_solution[self.orderODE], k=3, s=0)
        return self.Ode_intp(z)
    
    def E(self, z):
        return abs((self.Om0*(1.0+z)**3.0+self.Ok0*(1.0+z)**2.0+self.Or0*(1.0+z)**4.0)/(1-self.Ode_z(z)))**0.5
    
    def chi(self, z):
        return quad(self.chi_inte, 0.0, z)[0]
    
    def chi_inte(self, z):
        return 1.0/self.E(z)
    
    def w_de(self, z):
        return -1.0+2.0*(1.0+z)/3.0/self.n*np.sqrt(self.Ode_z(z))


class RDE(LCDM):
    '''
    This is the holographic dark energy model with the Ricci scalar as the scale
    '''

    def __init__(self, Om0, Ok0, Or0, alpha, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.alpha = float(alpha)
        if self.alpha == 0.0:
            self.alpha = self.alpha + 1e-15
        self.h = float(h)
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()


    def E(self, z):
        return abs(2.0*self.Om0/(2.0-self.alpha)*(1.0+z)**3.0+self.Ok0*(1.0+z)**2.0+self.Or0*(1.0+z)**4.0+(1-self.Or0-self.Ok0-2.0*self.Om0/(2.0-self.alpha))*np.power(1.0+z, 4.0-2.0/self.alpha) )**0.5

    def ODE(self, z):
        '''
        The energy component of dark energy
        '''
        return self.E(z)**2-self.Om0*(1+z)**3-self.Ok0*(1+z)**2-self.Or0*(1+z)**4

    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)

    def w_de(self, z):
        '''
        The "equivalent" equation of state of dark energy
        '''
        return self.ODEp(z)/3/self.ODE(z)*(1+z)-1

class PolyCDM(CG):
    '''
    This is class for the Poly CDM cosmology, the parameters include the spatial curvature: a prior needed: 0.0+-0.1
    '''

    def __init__(self, Om0, Ok0, Or0, Om1, Om2, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Om1 = float(Om1)
        self.Om2 = float(Om2)
        self.h = float(h)
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1.0+z)**3.0+self.Ok0*(1.0+z)**2.0+self.Or0*(1.0+z)**4.0+self.Om1*(1.0+z)**2.0+self.Om2*(1.0+z)+(1-self.Om0-self.Om1-self.Om2-self.Ok0-self.Or0))**0.5

class NeutrinoCos(LCDM):
    I = NuIntegral()

    def __init__(self, Om0, Ok0, Or0, mnu, Neff, h, Degenerate=False, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.mnu = float(mnu)
        self.Neff = float(Neff)
        self.h = float(h)
        self.Degenerate = Degenerate

        if self.Degenerate == False:
            self.mnuone = self.mnu
        else:
            self.mnuone = self.mnu/self.Neff

        self.Tnu = (4./11.)**(1./3)*2.7255*(3.046/3.0)**0.25
        self.prefix0 = 4.48130979e-7*2.7255**4*((4./11.)**(4./3.0))
        self.prefix = self.prefix0*(3.046/3.0)
        self.omnuh2today=self.Onuh2(0.0)
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Onuh2(self, z):
        if self.mnuone == 0.0:
            return self.Neff*7./8.*self.prefix0*(1.0+z)**4.0
        mnuOT = self.mnuone/(self.Tnu*(1.0+z))*11604.5193
        if self.Degenerate:
            return 3.0*self.I.SevenEights(mnuOT)*self.prefix*(1.0+z)**4.0 + (self.Neff-3.014)*7.0/8.0*self.prefix0*(1.0+z)**4.0
        else:
            return ((self.I.SevenEights(mnuOT)*self.prefix+(self.Neff-1.015)*7.0/8.0*self.prefix0))*(1.0+z)**4.0

    def E(self, z):
        return abs(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Onuh2(z)/self.h**2.0+self.Ok0*(1.0+z)**2.0+(1.0-self.Om0-self.Ok0-self.Or0-self.Onuh2(0.0)/self.h**2.0))**0.5

    def growth_equation(self, gr, z):
        '''
        The linear growth equation of purterbation as a function of redshift z.
        '''
        self.gr = gr
        return [self.gr[1], -(self.Ep(z)/self.E(z)-1.0/(1+z))*self.gr[1]+self.Geff(z)*3.0/2.0*self.Om_z(z)*(1-self.Onuh2(0)/self.h**2.0/(self.Om0+self.Onuh2(0)/self.h**2))/(1+z)**2*self.gr[0]]


class Quintessence_PL2(LCDM):
    '''
    This is the class for the quintessence scalar field cosmology with a power low potential.
    '''
    def __init__(self, Om0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
        
        sc = cdll.LoadLibrary("./Cpro/Quint_PL.dylib")
        sc.E.argtypes = [c_double, c_double, POINTER(POINTER(c_double)), POINTER(POINTER(c_double))]
        a1 = POINTER(c_double)()
        b1 = POINTER(c_double)()
        self.jj = sc.E(self.n, self.Om0, byref(a1), byref(b1))

        self.za1 = a1[0:self.jj]
        self.Ea1 = b1[0:self.jj]

        
        self.z_max = max(self.za1)
        self.EE_max = max(self.Ea1)
        self.Om_max = self.Om0*(1.+self.z_max)**3./self.E(self.z_max)**2
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        if z <=self.z_max:
            self.Ess = UnivariateSpline(self.za1, self.Ea1, k=3, s=0)
            return self.Ess(z)
        else:
            Emax = self.Ess(self.z_max)
            if self.Om_max>=1.0:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0)**0.5 * Emax/Emax_norm
            else:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0+1.0-self.Om0-self.Or0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+1.0-self.Om0-self.Or0)**0.5 * Emax/Emax_norm


    def chi(self, z):   # use the new chi to replace the one in LCDM module. This is much faster.
        if z<=self.z_max:
            self.id = np.argmin(abs(np.array(z)-self.za1))
            self.xx = np.hstack((0.0, self.za1[0:self.id-1], z))   # the solution from the C code may not start from z=0 and end at z=z(input), we need to add z=0 and z=z and the corresponding E(z).   Otherwise, this may introduce a 1% error.
            self.yy = np.hstack((1.0, self.Ea1[0:self.id-1], self.E(z)))
            self.cc = np.trapz(np.array(1.0)/self.yy, self.xx)
            return self.cc
        else:
            z_in = np.linspace(self.z_max, z, 200)
            E_in = map(self.E, z_in)
            chi_sec = np.trapz(np.array(1.0)/np.array(E_in), z_in)
            return self.chi(self.z_max)+chi_sec
                
                
    def chi_inte(self, z):
        if self.E(z) == 0.0:  # very rarely happens: some strange cosmology
            return 1.0/(self.E(z)+1e-15)
        else:
            return 1.0/self.E(z)
    
    def D_L(self, z):
        #r = quad(self.chi_inte, 0.0, z)[0]
        r = self.chi(z)
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)

class Quintessence_PL(LCDM):
    '''
    This is the class for the quintessence scalar field cosmology with a power low potential.
    '''
    def __init__(self, Om0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
        
        sc = cdll.LoadLibrary("./Cpro/Quint_PL2.dylib")
        sc.E1.argtypes = [c_double, c_double]
        self.jj = sc.E1(self.n, self.Om0)
        self.jj = self.jj
        
        sc.E2.argtypes = [c_double, c_double, c_int]
        sc.E2.restype = POINTER(c_double)
        c = sc.E2(self.n, self.Om0, self.jj)
        cc = np.array(np.fromiter(c, dtype=np.float64, count=self.jj*2))
        sc.free_mem.argtypes = [POINTER(c_double)]
        sc.free_mem.restype = None
        sc.free_mem(c)
        
        self.za1 = np.array(cc[0::2])
        self.Ea1 = np.array(cc[1::2])
        
        
        w1 = np.where(np.isnan(self.za1)==False)
        self.za1 = self.za1[w1]
        self.Ea1 = self.Ea1[w1]
        
        w2 = np.where(np.isnan(self.Ea1)==False)
        self.Ea1 = self.Ea1[w2]
        self.za1 = self.za1[w2]
        
        r = max(int(len(self.za1)/1000), 1)   #  too many elements slow down the interpolation, so just select 1000 of them.
        self.za1 = self.za1[0::r]
        self.Ea1 = self.Ea1[0::r]

        self.z_max = max(self.za1)
        #print("zmax", self.z_max)
        self.EE_max = max(self.Ea1)
        #print("Emax", self.EE_max)
        self.Om_max = self.Om0*(1.+self.z_max)**3./self.E(self.z_max)**2
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        if z <=self.z_max:
            self.Ess = UnivariateSpline(self.za1, self.Ea1, k=3, s=0)
            return self.Ess(z)
        else:
            Emax = self.Ess(self.z_max)
            if self.Om_max>=1.0:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0)**0.5 * Emax/Emax_norm
            else:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0+1.0-self.Om0-self.Or0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+1.0-self.Om0-self.Or0)**0.5 * Emax/Emax_norm

    def chi(self, z):   # use the new chi to replace the one in LCDM module. This is much faster.
        if z<=self.z_max:
            self.id = np.argmin(abs(np.array(z)-self.za1))
            self.xx = np.hstack((0.0, self.za1[0:self.id-1], z))   # the solution from the C code may not start from z=0 and end at z=z(input), we need to add z=0 and z=z and the corresponding E(z).   Otherwise, this may introduce a 1% error.
            self.yy = np.hstack((1.0, self.Ea1[0:self.id-1], self.E(z)))
            self.cc = np.trapz(np.array(1.0)/self.yy, self.xx)
            return self.cc
        else:
            z_in = np.linspace(self.z_max, z, 200)
            E_in = map(self.E, z_in)
            chi_sec = np.trapz(np.array(1.0)/np.array(E_in), z_in)
            return self.chi(self.z_max)+chi_sec
                
                
    def chi_inte(self, z):
        if self.E(z) == 0.0:  # very rarely happens: some strange cosmology
            return 1.0/(self.E(z)+1e-15)
        else:
            return 1.0/self.E(z)
    
    def D_L(self, z):
        #r = quad(self.chi_inte, 0.0, z)[0]
        r = self.chi(z)
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)



class Quintessence_EXP2(LCDM):
    '''
    This is the class for the quintessence scalar field cosmology with a power low potential.
    '''
    def __init__(self, Om0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
        
        sc = cdll.LoadLibrary("./Cpro/Quint_EXP.dylib")
        sc.E.argtypes = [c_double, c_double, POINTER(POINTER(c_double)), POINTER(POINTER(c_double))]
        a1 = POINTER(c_double)()
        b1 = POINTER(c_double)()
        self.jj = sc.E(self.n, self.Om0, byref(a1), byref(b1))
        
        self.za1 = a1[0:self.jj]
        self.Ea1 = b1[0:self.jj]
        
        if self.jj<4:
            self.za1 = np.zeros(5)
            self.Ea1 = np.ones(5)
 
        self.z_max = max(self.za1)
        self.EE_max = max(self.Ea1)
        self.Om_max = self.Om0*(1.+self.z_max)**3./self.E(self.z_max)**2
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        if z <=self.z_max:
            self.Ess = UnivariateSpline(self.za1, self.Ea1, k=3, s=0)
            return self.Ess(z)
        else:
            Emax = self.Ess(self.z_max)
            if self.Om_max>=1.0:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0)**0.5 * Emax/Emax_norm
            else:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0+1.0-self.Om0-self.Or0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+1.0-self.Om0-self.Or0)**0.5 * Emax/Emax_norm
                    
                    
    def chi(self, z):   # use the new chi to replace the one in LCDM module. This is much faster.
        if z<=self.z_max:
            self.id = np.argmin(abs(np.array(z)-self.za1))
            self.xx = np.hstack((0.0, self.za1[0:self.id-1], z))   # the solution from the C code may not start from z=0 and end at z=z(input), we need to add z=0 and z=z and the corresponding E(z).   Otherwise, this may introduce a 1% error.
            self.yy = np.hstack((1.0, self.Ea1[0:self.id-1], self.E(z)))
            self.cc = np.trapz(np.array(1.0)/self.yy, self.xx)
            return self.cc
        else:
            z_in = np.linspace(self.z_max, z, 200)
            E_in = map(self.E, z_in)
            chi_sec = np.trapz(np.array(1.0)/np.array(E_in), z_in)
            return self.chi(self.z_max)+chi_sec


    def chi_inte(self, z):
        if self.E(z) == 0.0:  # very rarely happens: some strange cosmology
            return 1.0/(self.E(z)+1e-15)
        else:
            return 1.0/self.E(z)
                
    def D_L(self, z):
        #r = quad(self.chi_inte, 0.0, z)[0]
        r = self.chi(z)
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)

class Quintessence_EXP(LCDM):
    '''
    This is the class for the quintessence scalar field cosmology with a power low potential.
    '''
    def __init__(self, Om0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)
       
        sc = cdll.LoadLibrary("./Cpro/Quint_EXP2.dylib")
        sc.E1.argtypes = [c_double, c_double]
        self.jj = sc.E1(self.n, self.Om0)
        self.jj = self.jj
        #print(self.jj, self.Om0, self.n, self.h)
        
        sc.E2.argtypes = [c_double, c_double, c_int]
        sc.E2.restype = POINTER(c_double)
        c = sc.E2(self.n, self.Om0, self.jj)
        cc = np.array(np.fromiter(c, dtype=np.float64, count=self.jj*2))
        sc.free_mem.argtypes = [POINTER(c_double)]
        sc.free_mem.restype = None
        sc.free_mem(c)
        
        self.za1 = np.array(cc[0::2])
        self.Ea1 = np.array(cc[1::2])
        
        if self.jj<4:
            self.za1 = np.zeros(5)
            self.Ea1 = np.ones(5)
        
        
        w1 = np.where(np.isnan(self.za1)==False)
        self.za1 = self.za1[w1]
        self.Ea1 = self.Ea1[w1]
        
        w2 = np.where(np.isnan(self.Ea1)==False)
        self.Ea1 = self.Ea1[w2]
        self.za1 = self.za1[w2]
        
        r = max(int(len(self.za1)/1000), 1)   #  too many elements slow down the interpolation, so just select 1000 of them.
        self.za1 = self.za1[0::r]
        self.Ea1 = self.Ea1[0::r]
        
        self.z_max = max(self.za1)
        #print("zmax", self.z_max)
        self.EE_max = max(self.Ea1)
        #print("Emax", self.EE_max)
        self.Om_max = self.Om0*(1.+self.z_max)**3./self.E(self.z_max)**2
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        if z <=self.z_max:
            self.Ess = UnivariateSpline(self.za1, self.Ea1, k=3, s=0)
            return self.Ess(z)
        else:
            Emax = self.Ess(self.z_max)
            if self.Om_max>=1.0:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0)**0.5 * Emax/Emax_norm
            else:
                Emax_norm = (self.Om0*(1.0+self.z_max)**3.0+self.Or0*(1.0+self.z_max)**4.0+1.0-self.Om0-self.Or0)**0.5
                return (self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+1.0-self.Om0-self.Or0)**0.5 * Emax/Emax_norm
                    
    def chi(self, z):   # use the new chi to replace the one in LCDM module. This is much faster.
        if z<=self.z_max:
            self.id = np.argmin(abs(np.array(z)-self.za1))
            self.xx = np.hstack((0.0, self.za1[0:self.id-1], z))   # the solution from the C code may not start from z=0 and end at z=z(input), we need to add z=0 and z=z and the corresponding E(z).   Otherwise, this may introduce a 1% error.
            self.yy = np.hstack((1.0, self.Ea1[0:self.id-1], self.E(z)))
            self.cc = np.trapz(np.array(1.0)/self.yy, self.xx)
            return self.cc
        else:
            z_in = np.linspace(self.z_max, z, 200)
            E_in = map(self.E, z_in)
            chi_sec = np.trapz(np.array(1.0)/np.array(E_in), z_in)
            return self.chi(self.z_max)+chi_sec


    def chi_inte(self, z):
        if self.E(z) == 0.0:  # very rarely happens: some strange cosmology
            return 1.0/(self.E(z)+1e-15)
        else:
            return 1.0/self.E(z)
                
    def D_L(self, z):
        #r = quad(self.chi_inte, 0.0, z)[0]
        r = self.chi(z)
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)



class WCSFa(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, Epinf, Zts, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = abs(float(Eps))
        self.sign_Eps = np.sign(Eps)
        self.Epinf = abs(float(Epinf))
        self.Zts = float(Zts)
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)

    def F2(self, x):
        return 2.0**0.5*(1.0-np.log(1.0+np.power(x, 3.0))/np.power(x, 3.0))-self.F(x)

    def w_de(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return -1.0+self.sign_Eps*(2.0/3.0*(self.Epinf**0.5+(self.Eps**0.5-(2.0*self.Epinf)**0.5)*self.F(xx)+self.Zts*self.F2(xx))**2.0)

    def Ode_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.sign_Eps*(2.0/3.0*(self.Epinf**0.5+(self.Eps**0.5-(2.0*self.Epinf)**0.5)*self.F(xx)+self.Zts*self.F2(xx))**2.0)/(1.0+z)
    
    def E(self, z):
        self.fde = np.exp(3.0*quad(self.Ode_int, 0, z)[0])
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)

class WCSFb(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, Epinf, Zts, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = abs(float(Eps))
        self.sign_Eps = np.sign(Eps)
        self.Epinf = abs(float(Epinf))
        self.Zts = float(Zts)
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
        self.zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))
        self.Es = map(self.EE, self.zs_Ode)
    
    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)
    
    def F2(self, x):
        return 2.0**0.5*(1.0-np.log(1.0+np.power(x, 3.0))/np.power(x, 3.0))-self.F(x)
    
    def w_de(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return -1.0+self.sign_Eps*(2.0/3.0*(self.Epinf**0.5+(self.Eps**0.5-(2.0*self.Epinf)**0.5)*self.F(xx)+self.Zts*self.F2(xx))**2.0)
    
    def Ode_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.sign_Eps*(2.0/3.0*(self.Epinf**0.5+(self.Eps**0.5-(2.0*self.Epinf)**0.5)*self.F(xx)+self.Zts*self.F2(xx))**2.0)/(1.0+z)
    
    def EE(self, z):
        self.fde = np.exp(3.0*quad(self.Ode_int, 0, z)[0])
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)
    
    def E(self, z):
        self.OE_intp = UnivariateSpline(self.zs_Ode, self.Es, k=3, s=0)
        return self.OE_intp(z)

class WCSF1(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = float(Eps)
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
        self.zs_Ode = WCSF1_tab(self.aeq).zs_Ode
        self.FFs = WCSF1_tab(self.aeq).FFs
    
    def FFsInt(self, z):
        self.FF = UnivariateSpline(self.zs_Ode, self.FFs, k=3, s=0)
        return self.FF(z)
    
    def E(self, z):
        self.fde = np.exp(2.0*self.Eps*self.FFsInt(z))
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)

class WCSF2(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, Epinf, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = float(Eps)
        self.Epinf = np.sign(self.Eps)*abs(Epinf)
        if self.Epinf == 0.0:
            self.Epinf = 1e-10
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
        self.zs_Ode = WCSF2_tab(self.aeq).zs_Ode
        self.FFs = WCSF2_tab(self.aeq).FFs
        self.Fs = WCSF2_tab(self.aeq).Fs
    
    def FFsInt(self, z):
        self.FF = UnivariateSpline(self.zs_Ode, self.FFs, k=3, s=0)
        return self.FF(z)

    def FsInt(self, z):
        self.F = UnivariateSpline(self.zs_Ode, self.Fs, k=3, s=0)
        return self.F(z)
    
    def E(self, z):
        self.fde = np.exp(2.0*self.Epinf*(np.log(1.0+z)+2.0*(np.sqrt(self.Eps/self.Epinf)-np.sqrt(2.0))*self.FsInt(z)+(np.sqrt(self.Eps/self.Epinf)-np.sqrt(2))**2.0*self.FFsInt(z)))
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)

class WCSF3(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, Epinf, Zts, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = float(Eps)
        self.Epinf = np.sign(self.Eps)*abs(Epinf)
        if self.Epinf == 0.0:
            self.Epinf = 1e-10
        self.Zts = float(Zts)
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
        
        self.zs_Ode = WCSF3_tab(self.aeq).zs_Ode
        self.FFs = WCSF3_tab(self.aeq).FFs
        self.Fs = WCSF3_tab(self.aeq).Fs
    
    def FFsInt(self, z):
        self.FF = UnivariateSpline(self.zs_Ode, self.FFs, k=3, s=0)
        return self.FF(z)
    
    def FsInt(self, z):
        self.F = UnivariateSpline(self.zs_Ode, self.Fs, k=3, s=0)
        return self.F(z)
    
    def E(self, z):
        self.fde = np.exp(2.0*self.Epinf*(np.log(1.0+z)+2.0*(np.sqrt(self.Eps/self.Epinf)-np.sqrt(2.0))*self.FsInt(z)+(np.sqrt(self.Eps/self.Epinf)-np.sqrt(2))**2.0*self.FFsInt(z)))
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)


class WCSFc(LCDM):
    '''
    This is the class for the weakly-coupled canonical scalar field model. The equation of state of dark energy is parameterized by 1-3 parameters.
    '''
    def __init__(self, Om0, Ok0, Or0, Eps, Epinf, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Eps = abs(float(Eps))
        self.sign_Eps = np.sign(float(Eps))
        if self.Eps == 0.0:
            self.sign_Eps = np.sign(1.0)
        self.Epinf = abs(float(Epinf))
        self.h = float(h)
        self.aeq = np.power(self.Om0/(1.0-self.Om0), 1.0/(3.0-1.08*(1.0-self.Om0)*self.Eps))
        self.Ode0 = 1.0-self.Om0-self.Ok0-self.Or0
        self.Zts = 0.0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
    
    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)
    
    def F2(self, x):
        return 2.0**0.5*(1.0-np.log(1.0+np.power(x, 3.0))/np.power(x, 3.0))-self.F(x)
    
    def w_de(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return -1.0+self.sign_Eps*(2.0/3.0*(self.Epinf**0.5+(self.Eps**0.5-(2.0*self.Epinf)**0.5)*self.F(xx)+self.Zts*self.F2(xx))**2.0)
    
    def Fi_1(self, x):
        return -2.0/3.0*np.sqrt(1.0/x**3.0+1.0)+np.log(np.sqrt(1./x**3.0+1.0)+1.0)/3.0-np.log(abs(np.sqrt(1./x**3.0+1.0)-1.0))/3.0
    
    def Fi_2(self, x):
        return -(np.arcsinh(x**1.5)+x**1.5*np.sqrt(x**3.0+1.0))/3.0/x**3.0
    
    def FFi_1(self, x):
        return np.log(abs(x))-1.0/3.0/x**3.0
    
    def FFi_2(self, x):
        return (3.0*np.log(abs(x))-2.0*(x**3.0+1.0)**1.5*np.arcsinh(x**1.5)/x**4.5-1.0/x**3.0)/9.0
    
    def FFi_3(self, x):
        ash = np.arcsinh(x**1.5)
        S1 = 2.0/9.0*(np.log(abs(np.exp(ash)-1.0))+np.log(np.exp(ash)+1.0))
        S2f = 2.0*np.exp(2*ash) * ( np.exp(2*ash) * (2*ash * (6.0*ash - (np.exp(ash)-1.0)*(np.exp(ash)+1.0)*(np.exp(2.0*ash)-3.0) )+np.exp(2.0*ash)-2.0)+1.0)
        S2d = 9.0*(np.exp(ash)-1.0)**4.0*(np.exp(ash)+1.0)**4.0
        return -S1-S2f/S2d
    
    def Fin(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return (self.Fi_1(xx)-self.Fi_2(xx))-(self.Fi_1(1.0/self.aeq)-self.Fi_2(1.0/self.aeq))
    
    def FFin(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return (self.FFi_1(xx)-2.0*self.FFi_2(xx)+self.FFi_3(xx))-(self.FFi_1(1.0/self.aeq)-2.0*self.FFi_2(1.0/self.aeq)+self.FFi_3(1.0/self.aeq))
    
    def Fctin(self,z):
        xx = 1.0/(1.0+z)/self.aeq
        return np.log(xx)-np.log(1.0/self.aeq)
    
    def Ode_int(self, z):
        return self.sign_Eps*2.0*(self.Epinf*self.Fctin(z)+2.0*np.sqrt(self.Epinf)*(np.sqrt(self.Eps)-np.sqrt(2.0*self.Epinf))*self.Fin(z)+(np.sqrt(self.Eps)-np.sqrt(2.0*self.Epinf))**2.0*self.FFin(z))
    
    def E(self, z):
        self.fde = np.exp(-self.Ode_int(z))
        return np.power(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0+self.Ode0*self.fde, 0.5)



class KGBM(LCDM):
    '''
    This is the class for the Kinetic gravity braiding model
    '''
    def __init__(self, Om0, Ok0, Or0, n, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.h = float(h)
        self.n = float(n)
        self.modelN = 4
        self.OF0 = 1.-self.Om0-self.Or0-self.Ok0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def Expo(self, Ee, z):
        self.Ee = Ee
        return self.Ee**2.-self.Om0*(1.+z)**3.-self.Or0*(1.+z)**4.-self.Ok0*(1.+z)**2.-self.OF0*self.Ee**(-2.0/(2.0*self.n-1.0))
    
    def E(self, z):
        
        return fsolve(self.Expo, 1.0, args=(z))
    
    def chi_inte(self, z):
        return 1.0/self.E(z)
    
    def D_L(self, z):
        r = quad(self.chi_inte, 0.0, z)[0]
        if self.Ok0 > 0.0:
            return self.D_H()*(1+z)/np.sqrt(self.Ok0)*np.sinh(np.sqrt(self.Ok0)*r)
        elif self.Ok0 == 0.0:
            return self.D_H()*(1+z)*r
        else:
            return self.D_H()*(1+z)/np.sqrt(-self.Ok0)*np.sin(np.sqrt(-self.Ok0)*r)

    def Geff(self, z):
        if 5.0*self.n-self.Om_z(z) == 0.0:
            self.n = self.n+1e-10
        return (2.0*self.n+3.0*self.n*self.Om_z(z)-self.Om_z(z))/(self.Om_z(z)*(5.0*self.n-self.Om_z(z)))

class KGBM_n1(LCDM):
    '''
    This is the class for the Kinetic gravity braiding model with n=1
    '''
    def __init__(self, Om0, Ok0, Or0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.h = float(h)
        self.modelN = 4
        self.OF0 = 1.-self.Om0-self.Or0-self.Ok0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        
        return abs(0.5*self.Ok0*(1.0+z)**2.0 + 0.5*self.Om0*(1.0+z)**3.0 + 0.5*self.Or0*(1.0+z)**4.0 + np.sqrt(self.OF0+(1.0+z)**4.0 / 4.0 * (self.Ok0+self.Om0*(1.0+z)+self.Or0*(1.0+z)**2.0)**2.0))**0.5

    def Geff(self, z):
        n=1.0
        if 5.0*n-self.Om_z(z) == 0.0:
            n =n+1e-10
        return (2.0*n+3.0*n*self.Om_z(z)-self.Om_z(z))/(self.Om_z(z)*(5.0*n-self.Om_z(z)))

class Gal_tracker(LCDM):
    '''
    This is the class for the tracker solution of Galileon gravity model
    '''
    def __init__(self, Om0, Ok0, Or0, h, Growth=False):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.h = float(h)
        self.modelN = 4
        self.OF0 = 1.-self.Om0-self.Or0-self.Ok0
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()
                
    def E(self, z):
        return abs(0.5*self.Ok0*(1.0+z)**2.0 + 0.5*self.Om0*(1.0+z)**3.0 + 0.5*self.Or0*(1.0+z)**4.0 + np.sqrt(self.OF0+(1.0+z)**4.0 / 4.0 * (self.Ok0+self.Om0*(1.0+z)+self.Or0*(1.0+z)**2.0)**2.0))**0.5
    
    def Geff(self, z):
        ##  Consider a very special case of Galileon cosmology: alpha=beta=0, so the effective newton constant can be approximated as Geff = 1+OF0/4
        self.OF0_z = 1.0-(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.Ok0*(1.0+z)**2.0)/self.E(z)**2.0
        return 1+self.OF0_z/4.0

class LCDM_coup1(LCDM):
    '''
    This is the class for the coupled dark energy model mimicking a LCDM expansion: Type 1: constant w
    '''
    def __init__(self, Omc0, Ok0, Or0, wc, h, Growth=False):
        self.Omc0 = float(Omc0)   #  the input matter fraction is the coupled one
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.wc = float(wc)
        self.h = float(h)
        self.modelN = 5
        self.Odec0 = 1.0-self.Omc0-self.Or0-self.Ok0
        
        self.Om0 = 1.0+self.wc-self.wc*self.Omc0
        self.Ode0 = 1.0-self.Om0-self.Or0-self.Ok0
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Ok0*(1+z)**2+self.Ode0)**0.5

    def rhomc(self, z):
        return self.h**2.0*(1.0+self.wc)*self.Ode0/self.wc+self.h**2.0*self.Om0*(1.0+z)**3.0

    def rhomcp(self, z):
        return derivative(self.rhomc, z, dx=1e-6)

    def Qc(self, z):
        return 3.0*self.h**2*self.Ode0*(1.0+self.wc)*self.h*self.E(z)/self.wc

    def Qcp(self, z):
        return derivative(self.Qc, z, dx=1e-6)

    def test_function(self, z):  #  just test the terms in the perturbation equation is h-independent
        #return 1./self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z)
        #return self.Qc(z)/self.rhomc(z)/self.h/self.E(z)/(1.0+z)**2.0
        #return 1.0/self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z)**2.0*self.rhomcp(z)
        return 1.0/self.h/self.E(z)/(1.0+z)/self.rhomc(z)*self.Qcp(z)

    def growth_equation(self, gr, z):
        '''
        The linear growth equation of purterbation in the interacting dark energy and dark matter model.
        '''
        self.gr = gr
        return [self.gr[1], -(self.Ep(z)/self.E(z)-1.0/(1.0+z)-1./self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z))*self.gr[1]-(-1.5*self.rhomc(z)/(self.h*self.E(z)*(1.0+z))**2.0+self.Qc(z)/self.rhomc(z)/self.h/self.E(z)/(1.0+z)**2.0+1.0/self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z)**2.0*self.rhomcp(z)-1.0/self.h/self.E(z)/(1.0+z)/self.rhomc(z)*self.Qcp(z))*self.gr[0]]


class LCDM_coup2(LCDM):
    '''
        This is the class for the coupled dark energy model mimicking a LCDM expansion: Type 1: constant w
        '''
    def __init__(self, Omc0, Ok0, Or0, wa, wb, h, Growth=False):
        self.Omc0 = float(Omc0)   #  the input matter fraction is the coupled one
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.wa = float(wa)
        self.wb = float(wb)
        self.h = float(h)
        self.modelN = 5
        self.Odec0 = 1.0-self.Omc0-self.Or0-self.Ok0
        
        self.Om0 = 1.0+self.wa-self.wa*self.Omc0
        self.Ode0 = 1.0-self.Om0-self.Or0-self.Ok0
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return abs(self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Ok0*(1+z)**2+self.Ode0)**0.5
    
    def wc(self, z):
        return self.wa+self.wb*(-np.log(1.0+z))
    
    def rhomc(self, z):
        return self.h**2.0*(1.0+self.wc(z))*self.Ode0/self.wc(z)+self.h**2.0*self.Om0*(1.0+z)**3.0
    
    def rhomcp(self, z):
        return derivative(self.rhomc, z, dx=1e-6)
    
    def Qc(self, z):
        return self.h**2*(-self.wb+3.0*self.wc(z)+3.0*self.wc(z)**2.0)*self.Ode0*self.h*self.E(z)/self.wc(z)**2.0
    
    def Qcp(self, z):
        return derivative(self.Qc, z, dx=1e-6)
    
    def growth_equation(self, gr, z):
        '''
        The linear growth equation of purterbation in the interacting dark energy and dark matter model.
        '''
        self.gr = gr
        return [self.gr[1], -(self.Ep(z)/self.E(z)-1.0/(1.0+z)-1./self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z))*self.gr[1]-(-1.5*self.rhomc(z)/(self.h*self.E(z)*(1.0+z))**2.0+self.Qc(z)/self.rhomc(z)/self.h/self.E(z)/(1.0+z)**2.0+1.0/self.h/self.E(z)/(1.0+z)*self.Qc(z)/self.rhomc(z)**2.0*self.rhomcp(z)-1.0/self.h/self.E(z)/(1.0+z)/self.rhomc(z)*self.Qcp(z))*self.gr[0]]


class fR1(LCDM):
    '''
    This is class of the f(R) gravity which mimicks the LCDM expansion: test Om and A
    '''
    def __init__(self, Om0, Or0, A, h, Growth=False, n=0.1):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.A = float(A)
        self.h = float(h)
        self.OL = float(1.0-self.Om0-self.Or0)
        self.n = n
        self.xx_fr = np.linspace(-2, 0, 1000)
        self.fr_solution = self.fieldeq_sol1()
        #print(self.Om0, self.A)

        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return np.sqrt(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.OL)

    def EE(self, x):  # x = lna
        return self.Om0*np.exp(-3.0*x)+self.Or0*np.exp(-4.0*x)+self.OL

    def EEp(self, x):
        return -3.0*self.Om0*np.exp(-3.0*x)-4.0*self.Or0*np.exp(-4.0*x)

    def EEpp(self, x):
        return 9.0*self.Om0*np.exp(-3.0*x)+16.0*self.Or0*np.exp(-4.0*x)

    def EEppp(self, x):
        return -27.0*self.Om0*np.exp(-3.0*x)-64.0*self.Or0*np.exp(-4.0*x)

    def fieldeq(self, fr_y, x):
        self.fr_y = fr_y
        return [self.fr_y[1], (1.0+0.5*self.EEp(x)/self.EE(x)+(4*self.EEpp(x)+self.EEppp(x))/(4*self.EEp(x)+self.EEpp(x)))*self.fr_y[1]-0.5*(4*self.EEp(x)+self.EEpp(x))/self.EE(x)*self.fr_y[0]-3.0*self.OL*(4*self.EEp(x)+self.EEpp(x))/self.EE(x)]

    def fieldeq_sol1(self):
        p = (-7.0+np.sqrt(73))/4.0
        ai = 0.01
        yi = self.A*ai**p-6.0*self.OL
        ypi = self.A*p*ai**p
        self.fr_y0 = [yi, ypi]
        return odeint(self.fieldeq, self.fr_y0, self.xx_fr)
    
    def Rc(self, x):
        return 3.0*(4.0*self.EE(x)+self.EEp(x))
    
    def Rcp(self, x):
        return 3.0*(4.0*self.EEp(x)+self.EEpp(x))
    
    def Rcpp(self, x):
        return 3.0*(4.0*self.EEpp(x)+self.EEppp(x))

    def fR_y(self, x):
        fR_yx = UnivariateSpline(self.xx_fr, self.fr_solution[:,0], k=3, s=0)
        return fR_yx(x)

    def fR_yp(self, x):
        fR_ypx = UnivariateSpline(self.xx_fr, self.fr_solution[:,1], k=3, s=0)
        return fR_ypx(x)

    def fR_ypp(self, x):
        return (1.0+0.5*self.EEp(x)/self.EE(x)+(4.0*self.EEpp(x)+self.EEppp(x))/(4.0*self.EEp(x)+self.EEpp(x)))*self.fR_yp(x) - 0.5*(4.0*self.EEp(x)+self.EEpp(x))/self.EE(x)*self.fR_y(x)-3.0*self.OL*(4.0*self.EEp(x)+self.EEpp(x))/self.EE(x)

    def fR_fR(self, x):
        return self.fR_yp(x)/self.Rcp(x)

    def fR_B(self, x):
        return 2.0/3.0/(1.0+self.fR_fR(x))/(4.0*self.EEp(x)+self.EEpp(x))*self.EE(x)/self.EEp(x)*(self.fR_ypp(x)-self.fR_yp(x)*(4.0*self.EEpp(x)+self.EEppp(x))/(4.0*self.EEp(x)+self.EEpp(x)))

    def fR_B0(self):
        return self.fR_B(0.0)

    def k2fRR(self, x):
        fR_Fac = (self.fR_ypp(x)*self.Rcp(x)-self.Rcpp(x)*self.fR_yp(x))/self.Rcp(x)**2.0
        return fR_Fac/self.Rcp(x)*self.n*self.n*(2997.9)**2.0

    def Geffx(self, x):
        return 1.0/(1.0+self.fR_fR(x))*(1.0+4.0*self.k2fRR(x)/np.exp(2.0*x)/(1.0+self.fR_fR(x)))/(1.0+3.0*self.k2fRR(x)/np.exp(2.0*x)/(1.0+self.fR_fR(x)))

    def Geff(self, z):
        xc = np.log(1.0/(1.0+z))
        return self.Geffx(xc)


class fR2(LCDM):
    '''
    This is class of the f(R) gravity which mimicks the LCDM expansion: designer model
    '''
    def __init__(self, Om0, Or0, B0, h, Growth=False, n=0.1, Asearch=None):
        self.Om0 = float(Om0)
        self.Ok0 = 0.0
        self.Or0 = float(Or0)
        self.B0 = float(B0)
        self.h = float(h)
        self.OL = float(1.0-self.Om0-self.Or0)
        self.n = 0.1
        invdisttree = Invdisttree(zip(Asearch[:,0], Asearch[:,2]), Asearch[:,1], leafsize=10, stat=1)
        self.A = invdisttree(np.array([self.Om0, self.B0]), nnear=10, eps=0, p=2)
        self.xx_fr = np.linspace(-2, 0, 1000)
        self.fr_solution = self.fieldeq_sol1()
        #print(self.Om0, self.B0, self.A)
        
        
        if Growth == True:
            self.zs_gr = np.linspace(30,0,500)
            self.solution = self.growth_sol1()

    def E(self, z):
        return np.sqrt(self.Om0*(1.0+z)**3.0+self.Or0*(1.0+z)**4.0+self.OL)
    
    def EE(self, x):  # x = lna
        return self.Om0*np.exp(-3.0*x)+self.Or0*np.exp(-4.0*x)+self.OL
    
    def EEp(self, x):
        return -3.0*self.Om0*np.exp(-3.0*x)-4.0*self.Or0*np.exp(-4.0*x)
    
    def EEpp(self, x):
        return 9.0*self.Om0*np.exp(-3.0*x)+16.0*self.Or0*np.exp(-4.0*x)
    
    def EEppp(self, x):
        return -27.0*self.Om0*np.exp(-3.0*x)-64.0*self.Or0*np.exp(-4.0*x)
    
    def fieldeq(self, fr_y, x):
        self.fr_y = fr_y
        return [self.fr_y[1], (1.0+0.5*self.EEp(x)/self.EE(x)+(4*self.EEpp(x)+self.EEppp(x))/(4*self.EEp(x)+self.EEpp(x)))*self.fr_y[1]-0.5*(4*self.EEp(x)+self.EEpp(x))/self.EE(x)*self.fr_y[0]-3.0*self.OL*(4*self.EEp(x)+self.EEpp(x))/self.EE(x)]
    
    def fieldeq_sol1(self):
        p = (-7.0+np.sqrt(73))/4.0
        ai = 0.01
        yi = self.A*ai**p-6.0*self.OL
        ypi = self.A*p*ai**p
        self.fr_y0 = [yi, ypi]
        return odeint(self.fieldeq, self.fr_y0, self.xx_fr)
    
    def Rc(self, x):
        return 3.0*(4.0*self.EE(x)+self.EEp(x))
    
    def Rcp(self, x):
        return 3.0*(4.0*self.EEp(x)+self.EEpp(x))
    
    def Rcpp(self, x):
        return 3.0*(4.0*self.EEpp(x)+self.EEppp(x))
    
    def fR_y(self, x):
        fR_yx = UnivariateSpline(self.xx_fr, self.fr_solution[:,0], k=3, s=0)
        return fR_yx(x)
    
    def fR_yp(self, x):
        fR_ypx = UnivariateSpline(self.xx_fr, self.fr_solution[:,1], k=3, s=0)
        return fR_ypx(x)
    
    def fR_ypp(self, x):
        return (1.0+0.5*self.EEp(x)/self.EE(x)+(4.0*self.EEpp(x)+self.EEppp(x))/(4.0*self.EEp(x)+self.EEpp(x)))*self.fR_yp(x) - 0.5*(4.0*self.EEp(x)+self.EEpp(x))/self.EE(x)*self.fR_y(x)-3.0*self.OL*(4.0*self.EEp(x)+self.EEpp(x))/self.EE(x)
    
    def fR_fR(self, x):
        return self.fR_yp(x)/self.Rcp(x)
    
    def fR_B(self, x):
        return 2.0/3.0/(1.0+self.fR_fR(x))/(4.0*self.EEp(x)+self.EEpp(x))*self.EE(x)/self.EEp(x)*(self.fR_ypp(x)-self.fR_yp(x)*(4.0*self.EEpp(x)+self.EEppp(x))/(4.0*self.EEp(x)+self.EEpp(x)))
    
    def fR_B0(self):
        return self.fR_B(0.0)
    
    def k2fRR(self, x):
        fR_Fac = (self.fR_ypp(x)*self.Rcp(x)-self.Rcpp(x)*self.fR_yp(x))/self.Rcp(x)**2.0
        return fR_Fac/self.Rcp(x)*self.n*self.n*(2997.9)**2.0
    
    def Geffx(self, x):
        return 1.0/(1.0+self.fR_fR(x))*(1.0+4.0*self.k2fRR(x)/np.exp(2.0*x)/(1.0+self.fR_fR(x)))/(1.0+3.0*self.k2fRR(x)/np.exp(2.0*x)/(1.0+self.fR_fR(x)))
    
    def Geff(self, z):
        xc = np.log(1.0/(1.0+z))
        return self.Geffx(xc)






