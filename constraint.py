from __future__ import division

try:
    import numpy as np
except ImportError:
    print('Numpy is required, please make sure it is installed properly')

try:
    import scipy as sp
except ImportError:
    print('Scipy is required, please make sure it is installed properly')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
from de_plot import DE_vis


class BAO_DH:
    def __init__(self, DH):
        self.DH = [float(i) for i in DH]
    
    def chi2(self):
        self.ob, self.sig = np.loadtxt('BAO_Dh.dat', usecols=(1,2), unpack=True)
        self.chi2s = (self.DH-self.ob)**2.0/self.sig**2
        return sum(self.chi2s)

class BAO_DM:
    def __init__(self, DM):
        self.DM = [float(i) for i in DM]
    
    def chi2(self):
        self.ob, self.sig = np.loadtxt('BAO_Dm.dat', usecols=(1,2), unpack=True)
        self.chi2s = (self.DM-self.ob)**2.0/self.sig**2
        return sum(self.chi2s)

class BAO_DV:
    def __init__(self, DV):
        self.DV = [float(i) for i in DV]
    
    def chi2(self):
        self.ob, self.sig = np.loadtxt('BAO_Dv.dat', usecols=(1,2), unpack=True)
        self.chi2s = (self.DV-self.ob)**2.0/self.sig**2
        return sum(self.chi2s)


class SoundHorizon:
    def __init__(self, Om0, h):
        self.Om0 = float(Om0)
        self.h = float(h)
        self.sigmnu = 0.6
        self.Onv = 0.0107*self.sigmnu/self.h**2
        self.Ob = 0.022032/self.h**2
        #self.Onv = 0.0107*self.sigmnu/0.7**2
        #self.Ob = 0.022032/0.7**2

    def rd(self):
        return 55.154*math.e**(-72.3*(self.Onv*self.h**2+0.0006)**2)/(self.Om0*self.h**2)**0.25351/(self.Ob*self.h**2)**0.12807
