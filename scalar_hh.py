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

class Quintessence:
    def __init__(self, xi, yi, zi, Gamma, wm):
        self.xi = float(xi)
        self.yi = float(yi)
        self.zi = float(zi)
        self.Gamma = float(Gamma)
        self.wm = float(wm)

    def xp(self, x1, y1, z1):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.z1 = float(z1)
        return -3*self.x1+np.sqrt(6)/2*self.z1*self.y1**2+3/2*self.x1*((1-self.wm)*self.x1**2+(1+self.wm)*(1-self.y1**2))

    def yp(self, x2, y2, z2):
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.z2 = float(z2)
        return -np.sqrt(6)/2*self.z1*self.x1*self.y1+3/2*self.x1*((1-self.wm)*self.x1**2+(1+self.wm)*(1-self.y1**2))

    def zp(self, x3, y3, z3):
        self.x3 = float(x3)
        self.y3 = float(y3)
        self.z3 = float(z3)
        return -np.sqrt(6)*self.x3**2*(self.Gamma-1)*self.x3

    def solu(self, zz):
        self.zz = float(zz)
        self.ai = 0.001
        self.a = self.ai
        self.dN = 10**(-3)
        self.Zi = 1/self.ai - 1
        self.Z = self.Zi
        self.x = self.xi
        self.y = self.yi
        self.z = self.zi
        while self.ai < 1/(1+self.zz):
            self.k1 = self.dN*self.xp(self.xi, self.yi, self.zi)
            self.l1 = self.dN*self.yp(self.xi, self.yi, self.zi)
            self.m1 = self.dN*self.zp(self.xi, self.yi, self.zi)
            
            self.k2 = self.dN*self.xp(self.xi+self.k1/2, self.yi+self.l1/2, self.zi+self.m1/2)
            self.l2 = self.dN*self.yp(self.xi+self.k1/2, self.yi+self.l1/2, self.zi+self.m1/2)
            self.m2 = self.dN*self.zp(self.xi+self.k1/2, self.zi+self.l1/2, self.zi+self.m1/2)
            
            self.k3 = self.dN*self.xp(self.xi+self.k2/2, self.yi+self.l2/2, self.zi+self.m2/2)
            self.l3 = self.dN*self.yp(self.xi+self.k2/2, self.yi+self.l2/2, self.zi+self.m2/2)
            self.m3 = self.dN*self.zp(self.xi+self.k2/2, self.yi+self.l2/2, self.zi+self.m2/2)
            
            self.k4 = self.dN*self.xp(self.xi+self.k3, self.yi+self.l3, self.zi+self.m3)
            self.l4 = self.dN*self.yp(self.xi+self.k3, self.yi+self.l3, self.zi+self.m3)
            self.m4 = self.dN*self.zp(self.xi+self.k3, self.yi+self.l3, self.zi+self.m3)
            
            self.xi = self.xi+1/6*(self.k1+2*self.k2+2*self.k3+self.k4)
            self.yi = self.yi+1/6*(self.l1+2*self.l2+2*self.l3+self.l4)
            self.zi = self.zi+1/6*(self.m1+2*self.m2+2*self.m3+self.m4)
            
            self.ai = self.ai*(1+self.dN)
            self.Zi = 1/self.ai-1
            self.x = np.append(self.x, self.xi)
            self.y = np.append(self.y, self.yi)
            self.z = np.append(self.z, self.zi)
            self.Z = np.append(self.Z, self.Zi)
            self.a = np.append(self.a, self.ai)

        return self.a, self.Z, self.x, self.y, self.z










