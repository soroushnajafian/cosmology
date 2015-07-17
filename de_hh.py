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


class LCDM:
    '''
    This is the class for standard \LambdaCDM model.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Or0: the current energy density of radiation
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Or0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Or0 = float(Or0)
        self.Ode0 = float(1-self.Om0-self.Or0-self.Ok0)
        self.h = float(h)
    '''
    The following part is model dependent: the expansion factor E(z) and the equation of state of the dark energy
    '''
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Or0*(1+z)**4+self.Ok0*(1+z)**2+self.Ode0)**0.5
    
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
    
    def t_H(self):
        return 9.78/self.h  # the unit is Gyr
    
    def chit(self, z):
        '''
        It is not a observable, but needed to calculate the lookback time
        '''
        self.zz = np.linspace(0, z, 1000)
        self.ETz = 1.0/np.array((1+self.zz)*(map(self.E, self.zz)))
        self.ccT = interpolate.splrep(self.zz, self.ETz, s=0)
        return interpolate.splint(0, z, self.ccT)
    
    def LB_T(self, z):
        return self.t_H()*self.chit(z)  #the unit is Gyr
    

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
        self.z0 = 0
        self.dz = 10**(-4)
        self.D0 = 1
        self.D10 = -self.Om0**0.6
        self.Ds = self.D0
        self.D1s = self.D10
        self.zs = self.z0
        self.fs = self.D10
        while self.z0<z:
            self.k1 = self.dz*self.Dp(self.z0, self.D0, self.D10)
            self.l1 = self.dz*self.D1p(self.z0, self.D0, self.D10)
            
            self.k2 = self.dz*self.Dp(self.z0+self.dz/2, self.D0+self.k1/2, self.D10+self.l1/2)
            self.l2 = self.dz*self.D1p(self.z0+self.dz/2, self.D0+self.k1/2, self.D10+self.l1/2)
        
            self.k3 = self.dz*self.Dp(self.z0+self.dz/2, self.D0+self.k2/2, self.D10+self.l2/2)
            self.l3 = self.dz*self.D1p(self.z0+self.dz/2, self.D0+self.k2/2, self.D10+self.l2/2)
        
            self.k4 = self.dz*self.Dp(self.z0+self.dz/2, self.D0+self.k3, self.D10+self.l3)
            self.l4 = self.dz*self.D1p(self.z0+self.dz/2, self.D0+self.k3, self.D10+self.l3)
            
            self.D0 = self.D0+1/6*(self.k1+2*self.k2+2*self.k3+self.k4)
            self.D10 = self.D10+1/6*(self.l1+2*self.l2+2*self.l3+self.l4)

            self.z0 = self.z0+self.dz
            self.fss = -(1+self.z0)/self.D0*self.D10
            
            self.zs = np.append(self.zs, self.z0)
            self.Ds = np.append(self.Ds, self.D0)
            self.D1s = np.append(self.D1s, self.D10)
            self.fs = np.append(self.fs, self.fss)
                
        return self.zs, self.Ds, self.D1s, self.fs




class Topo_defc_2D(LCDM):
    '''
    This is the class for the FRW cosmology with 2D topological defects. The equation of state is w_x = -2/3.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Otopo = float(1-self.Om0-self.Ok0)
        
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Otopo*(1+z))**0.5
    
    def w_de(self, z):
        return -2/3

class Phan_DE(LCDM):
    '''
    This is the class for the FRW cosmology with Phantom dark energy. The equation of state is w_x = -4/3.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          h: dimensionless Hubble constant
        '''
    def __init__(self, Om0, Ok0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Oph = float(1-self.Om0-self.Ok0)
    
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Oph*(1+z)**(-1))**0.5

    def w_de(self, z):
        return -4/3

class XCDM(LCDM):
    '''
    This is the class for the XCDM cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          w: the equation of state of dark energy
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, w, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ode0 = float(1-self.Om0-self.Ok0)
        self.w = float(w)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0*(1+z)**(3*(1+self.w)))**0.5
    
    def w_de(self, z):
        return self.w

class CG(LCDM):
    '''
    This is the class for the Chaplygin gas model (CG), the equation of state of dark energy is p = -A/\rho.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          As: constant related to A
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, As, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ode0 = float(1-self.Om0-self.Ok0)
        self.As = float(As)
        self.h = float(h)
        
    def E(self,z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0*(self.As+(1-self.As)*(1+z)**6)**0.5)**0.5

    def ODE(self, z):
        '''
        The energy component of dark energy
        '''
        return self.E(z)**2-self.Om0*(1+z)**3+self.Ok0*(1+z)**2
    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)
    def w_de(self, z):
        '''
        The "equivalent" equation of state of dark energy
        '''
        return self.ODEp(z)/3/self.ODE(z)*(1+z)-1

class GCG(CG):
    '''
    This is the class for the Generalized Chaplygin gas model (GCG), the equation of state of dark energy is p = -A/\rho^\alpha.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          As: constant related to A
                          alpha: a constant in the range 0 < \alpha < 1
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, As, alpha, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ode0 = float(1-self.Om0-self.Ok0)
        self.As = float(As)
        self.alpha = float(alpha)
        self.h  = float(h)
        
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0*(self.As+(1-self.As)*(1+z)**(3*(1+self.alpha)))**(1/(1+self.alpha)))**0.5

class W_Linear(LCDM):
    '''
    This is the class for the dark energy model with a linear parameterization of equation of state of the dark energy: w=w0+w1*z
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          w0 and w1: constants related to the EoS parameterization
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, w0 ,w1, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ode0 = float(1-self.Om0-self.Ok0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0*(1+z)**(3*(self.w0-self.w1+1))*math.e**(3*self.w1*z))**0.5

    def w_de(self, z):
        return self.w0+self.w1*z

class W_CPL(LCDM):
    '''
    This is the class for the CPL parameterization of the equation of state of dark energy: w=w0+w1*z/(1+z)
    The input parameters have the same meanings as W_linear class
    '''
    def __init__(self, Om0, Ok0, w0 ,w1, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ode0 = float(1-self.Om0-self.Ok0)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0*(1+z)**(3*(self.w0+self.w1+1))*math.e**(-3*self.w1*z/(1+z)))**0.5
    def w_de(self, z):
        return self.w0+self.w1*z/(1+z)

class DE_Casimir(CG):
    '''
    This is the class for the FRW cosmology with Casimir effect.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          radiation neglected
                          Ocass0:
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Ocass0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Ocass0 = float(Ocass0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Ocass0)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Ode0-self.Ocass0*(1+z)**4)**0.5
'''
not sure, need to check the paper
class CGB:
    def __int__(self, Om0, )
'''

class DE_Card(LCDM):
    ## not from 0604327 (it might be wrong), we use:http://iopscience.iop.org/0004-637X/588/1/1/fulltext/57352.text.html
    ## This is a special model that has nonzero Or0 (radiation is included)
    '''
    This is the class for the Cardissian expansion cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          The curvature is zero
                          Or0: the current energy density fraction of radiation
                          n: constant related to the Cardassian term in the Friedmann equation
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Or0, n, h):
        self.Ok0 = float(0)  # assume the curvature is zero
        self.Om0 = float(Om0)
        self.Or0 = float(Or0)
        self.n = float(n)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**4*(1/(1+z)+self.Or0/self.Om0+(1+z)**(4*self.n-4)*(1-self.Or0-self.Om0)/self.Om0*((1/(1+z)+self.Or0/self.Om0)/(1+self.Or0/self.Om0))**self.n))**0.5
    
    def ODE(self, z):
        '''
        The energy component of dark energy
        '''
        return self.E(z)**2-self.Om0*(1+z)**3+self.Or0*(1+z)**4  # Note: different from GC and GCG
    def ODEp(self, z):
        return derivative(self.ODE, z, dx = 1e-6)
    def w_de(self, z):
        '''
        The "equivalent" equation of state of dark energy
        '''
        return self.ODEp(z)/3/self.ODE(z)*(1+z)-1

class DGP(CG):
    '''
    This is the class for the DGP cosmology
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Orc0: bulk-induced term related to a crossover radius.
                          h: dimensionless Hubble constant    
    '''
    def __init__(self, Om0, Orc0, h):
        self.Om0 = float(Om0)
        self.Orc0 = float(Orc0)
        self.Ok0 = float(1-(np.sqrt(self.Om0+self.Orc0)+np.sqrt(self.Orc0))**2)
        self.h = float(h)
    
    def E(self, z):
        return ((np.sqrt(self.Om0*(1+z)**3+self.Orc0)+np.sqrt(self.Orc0))**2+self.Ok0*(1+z)**2)**0.5
    '''
    The equation of state of dark energy is calculated by the same method in CG class: a effective EoS
    '''
class DDG(CG): # according to the paper, redefine r0h0 = r0*H0 to get the constraint condition
    '''
    This is the class for the DDG cosmology
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          r0h0: constant defined as r0*H0.
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, r0h0, h):
        self.Ok0 = float(0)  # assume the curvature is zero
        self.Om0 = float(Om0)
        self.r0h0 = float(r0h0)
        self.Ode0 = float(1+1/self.r0h0-self.Om0)
        self.h =float(h)

    def E(self, z):
        return -0.5/self.r0h0+np.sqrt(self.Om0*(1+z)**3+self.Ode0+1/4/self.r0h0**2)

class RS(LCDM):
    '''
    This is the class for the RS braneworld cosmology without cosmological constant
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Odr0: the current energy density fraction from dark radiation
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Odr0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Odr0 = float(Odr0)
        self.Oll0 = float(1-self.Om0-self.Ok0-self.Odr0)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Odr0*(1+z)**4+self.Oll0*(1+z)**6)**0.5
    
    def ODE(self, z):
        '''
            The energy component of dark energy
            '''
        return self.E(z)**2-self.Om0*(1+z)**3+self.Odr0*(1+z)**4  # Note: different from GC, GCG and DE_Card
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
    def __init__(self, Om0, Ok0, Odr0, Oll0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Odr0 = float(Odr0)
        self.Oll0 = float(Oll0)
        self.Ode0 = float(1-self.Om0-self.Ok0-self.Odr0-self.Oll0)
        self.h = float(h)

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Odr0*(1+z)**4+self.Oll0*(1+z)**6+self.Ode0)**0.5
    # the calculation of effective equation of state of dark energy is the same as RS model

class S_Brane1(CG):
    '''
    This is the class for the Shtanov Brane1 cosmology.
    The input parameters: Om0: the current energy density fraction of matter including baryonic matter and dark matter
                          Ok0: the current energy density fraction of curvature
                          Osig0: the current energy density fraction from brane tension
                          Oll0: the current energy density fraction contributed from Planck mass scales in different dimensions
                          The dark radiation is neglected.
                          h: dimensionless Hubble constant
    '''
    def __init__(self, Om0, Ok0, Osig0, Oll0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Osig0 = float(Osig0)
        self.Oll0 = float(Oll0)
        self.Ode0 = float((self.Om0+self.Ok0+self.Osig0+2*self.Oll0-1)**2.0/4.0/self.Oll0-self.Om0-self.Osig0-self.Oll0)
        self.h = float(h)
        #print(self.Om0+self.Ok0+self.Osig0+2*self.Oll0)
        if self.Om0+self.Ok0+self.Osig0+2*self.Oll0 - 1 < 0:
            print('Warning: S_Brane1: The parameters are not appropriate, please choose model: S_Brane2')

    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Osig0+2*self.Oll0-2*np.sqrt(self.Oll0)*np.sqrt(self.Om0*(1+z)**3+self.Osig0+self.Oll0+self.Ode0))**0.5
    # Since the dark radiation is ignored, the dark energy component and its effective equation of state is calculated the same way as CG model.

class S_Brane2(CG):
    '''
    This is the class for the Shtanov Brane2 cosmology. 
    The meanings of the input parameters are the same as Shtanov Brane1 model.
    '''
    def __init__(self, Om0, Ok0, Osig0, Oll0, h):
        self.Om0 = float(Om0)
        self.Ok0 = float(Ok0)
        self.Osig0 = float(Osig0)
        self.Oll0 = float(Oll0)
        self.Ode0 = float((self.Om0+self.Ok0+self.Osig0+2.0*self.Oll0-1.0)**2.0/4.0/self.Oll0-self.Om0-self.Osig0-self.Oll0)
        self.h = float(h)
        #print(self.Om0+self.Ok0+self.Osig0+2*self.Oll0)
        if self.Om0+self.Ok0+self.Osig0+2*self.Oll0 - 1 > 0:
            print('Warning: S_Brane2: The parameters are not appropriate, please choose model: S_Brane1')
    
    def E(self, z):
        return (self.Om0*(1+z)**3+self.Ok0*(1+z)**2+self.Osig0+2*self.Oll0+2*np.sqrt(self.Oll0)*np.sqrt(self.Om0*(1+z)**3+self.Osig0+self.Oll0+self.Ode0))**0.5


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


















