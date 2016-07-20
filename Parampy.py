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

class Flat_LCDM(object):

    def __init__(self):
        self.Om0, self.h, self.Oba, self.sigma8 = 0.27, 0.7, 0.04, 0.8
        
    def param(self):
        return np.array([self.Om0, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.01, 0.5), (0.4, 1), (0.001, 0.2))
        
    def param_sigma(self):
        return np.hstack((self.param(), self.sigma8))
    
    def param_sigma_bnds(self):
        return list(np.vstack((self.param_bnds(), (0.001, 2.0))))

class LCDM(Flat_LCDM):

    def __init__(self):
        self.Om0, self.Ok0, self.h, self.Oba, self.sigma8 = 0.27, 0.00, 0.7, 0.04, 0.8
    
    def param(self):
        return np.array([self.Om0, self.Ok0, self.h, self.Oba])
        
    def param_bnds(sef):
        return ((0.01, 0.5), (-0.3, 0.3), (0.4, 1), (0.001, 0.2))

class XCDM(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.w, self.h, self.Oba, self.sigma8 = 0.298877, -0.003445, -1.0, 0.679585, 0.048743, 0.770251

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.w, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.w, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-2.0, -0.1), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-2.0, -0.1), (0.4, 1), (0.001, 0.2))

class WCPL(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba, self.sigma8 = 0.301812, -0.003029, -0.973540, 0.0, 0.675390, 0.049428, 0.772645
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.w0, self.w1, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3, -0.1), (-3, 3), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3, -0.1), (-3, 3), (0.4, 1), (0.001, 0.2))

class WJBP(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba, self.sigma8 = 0.34, 0.0, -1.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.w0, self.w1, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3, -0.1), (-5, 5), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3, -0.1), (-5, 5), (0.4, 1), (0.001, 0.2))

class WLinear(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba, self.sigma8 = 0.27, 0.0, -1.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.w0, self.w1, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.w0, self.w1, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3, -0.1), (-3, 3), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3, -0.1), (-3, 3), (0.4, 1), (0.001, 0.2))

class CG(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.As, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.8, 0.7, 0.04, 0.8

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.As, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.As, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0, 1), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0, 1), (0.4, 1), (0.001, 0.2))

class GCG(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.As, self.alpha, self.h, self.Oba, self.sigma8 = 0.30, 0.0, 0.8, 1.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.As, self.alpha, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.As, self.alpha, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0, 1), (0.0, 1.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0, 1), (0.0, 1.0), (0.4, 1), (0.001, 0.2))

class MCG(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.As, self.alpha, self.B, self.h, self.Oba, self.sigma8 = 0.30, 0.0, 0.8, 1.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.As, self.alpha, self.B, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.As, self.alpha, self.B, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0, 1), (0.0, 1.0), (0.0, 1), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0, 1), (0.0, 1.0), (0.0, 1), (0.4, 1), (0.001, 0.2))

class sDGP(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.h, self.Oba, self.sigma8 = 0.30, 0.0, 0.7, 0.04, 0.8

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.4, 1), (0.001, 0.2))


class sDGP_L(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Occ, self.h, self.Oba, self.sigma8 = 0.30, 0.0, 0.7, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Occ, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Occ, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 1.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 1.0), (0.4, 1), (0.001, 0.2))

class nDGP(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Occ, self.h, self.Oba, self.sigma8 = 0.30, 0.0, 0.7, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Occ, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Occ, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 1.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 1.0), (0.4, 1), (0.001, 0.2))

class EDE(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Ode, self.w0, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.05, -1.0, 0.7, 0.04, 0.8

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Ode, self.w0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Ode, self.w0, self.h, self.Oba])
                
    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 0.5), (-3.0, 0.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 0.5), (-3.0, 0.0), (0.4, 1), (0.001, 0.2))

class SL_DE(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.dW0, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.dW0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.dW0, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))

class Casimir(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Ocas0, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Ocas0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Ocas0, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-0.5, 0.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-0.5, 0.0), (0.4, 1), (0.001, 0.2))

class Card(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.q, self.n, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 1.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.q, self.n, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.q, self.n, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.001, 10.0), (-3.0, 2.0/3.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.001, 10.0), (-3.0, 2.0/3.0), (0.4, 1), (0.001, 0.2))

class PNGB(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.w0, self.F, self.h, self.Oba, self.sigma8 = 0.3, 0.0, -1.0, 6.0, 0.67, 0.04, 0.82

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.w0, self.F, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.w0, self.F, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3.0, 0.0), (0.0, 8.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3.0, 0.0), (0.0, 8.0), (0.4, 1), (0.001, 0.2))

class PolyCDM(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Om1, self.Om2, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Om1, self.Om2, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Om1, self.Om2, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-10.0, 10.0), (-10.0, 10.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-10.0, 10.0), (-10.0, 10.0), (0.4, 1), (0.001, 0.2))

class QCD_Ghost(Flat_LCDM):

    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.gamma, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 1.0, 0.7, 0.04, 0.8

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.gamma, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.gamma, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 4.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 4.0), (0.4, 1), (0.001, 0.2))

class OA(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.a1, self.a2, self.a3, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 1.0, 1.0, 0.1, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.a1, self.a2, self.a3, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.a1, self.a2, self.a3, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (0.4, 1), (0.001, 0.2))

class HDE(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.c, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.4, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.c, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.c, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.001, 2.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.001, 2.0), (0.4, 1), (0.001, 0.2))

class ADE(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.n, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 2.7, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.n, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.n, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.001, 1000.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.001, 1000.0), (0.4, 1), (0.001, 0.2))

class RDE(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.alpha, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 1.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.alpha, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.alpha, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 1.9), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 1.9), (0.4, 1), (0.001, 0.2))

class fT_PL(Flat_LCDM):

    def __init__(self):
        self.Om0, self.b, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.7, 0.04, 0.8

    def param(self):
        return np.array([self.Om0, self.b, self.h, self.Oba])

    def param_bnds(self):
        return ((0.01, 0.5), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))

class fT_EXP(Flat_LCDM):

    def __init__(self):
        self.Om0, self.b, self.h, self.Oba, self.sigma8 = 0.3, 1e-2, 0.7, 0.04, 0.8
    
    def param(self):
        return np.array([self.Om0, self.b, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.01, 0.5), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))

class fT_EXP_Linder(Flat_LCDM):
    
    def __init__(self):
        self.Om0, self.b, self.h, self.Oba, self.sigma8 = 0.3, 1e-2, 0.7, 0.04, 0.8
    
    def param(self):
        return np.array([self.Om0, self.b, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.01, 0.5), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))

class fT_tanh(Flat_LCDM):
    
    def __init__(self):
        self.Om0, self.n, self.h, self.Oba, self.sigma8 = 0.3, 1.65, 0.7, 0.04, 0.8
    
    def param(self):
        return np.array([self.Om0, self.n, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.01, 0.5), (1.5, 5.0), (0.4, 1), (0.001, 0.2))

class WCSF1(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Eps, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.7, 0.04, 0.8

    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Eps, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Eps, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-1.0, 1.0), (0.4, 1), (0.001, 0.2))

class WCSF2(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Eps, self.Epinf, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Eps, self.Epinf, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Eps, self.Epinf, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-1.0, 1.0), (0.0, 0.9), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-1.0, 1.0), (0.0, 0.9), (0.4, 1), (0.001, 0.2))

class WCSF3(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.Eps, self.Epinf, self.Zts, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.Eps, self.Epinf, self.Zts, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.Eps, self.Epinf, self.Zts, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.4, 1), (0.001, 0.2))

class Quintessence_PL(Flat_LCDM):
    
    def __init__(self):
        self.Om0, self.n, self.h, self.Oba, self.sigma8 = 0.3, 0.1, 0.7, 0.04, 0.8
    
    def param(self):
        return np.array([self.Om0, self.n, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.1, 0.5), (0.01, 5.0), (0.4, 1), (0.001, 0.2))

class Quintessence_EXP(Flat_LCDM):
    
    def __init__(self):
        self.Om0, self.n, self.h, self.Oba, self.sigma8 = 0.3, 0.1, 0.68, 0.05, 0.8
    
    def param(self):
        return np.array([self.Om0, self.n, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.1, 0.5), (0.01, 1.2), (0.4, 1), (0.001, 0.2))

class KGBM(Flat_LCDM):
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.n, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 10.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.n, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.n, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-1000.0, 1000.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-1000.0, 1000.0), (0.4, 1), (0.001, 0.2))

class KGBM_n1(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.h, self.Oba, self.sigma8 = 0.298877, 0.0, 0.679585, 0.048743, 0.770251
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.4, 1), (0.001, 0.2))

class Gal_tracker(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.7, 0.04, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.4, 1), (0.001, 0.2))

class LCDM_coup1(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Omc0, self.Ok0, self.wc, self.h, self.Oba, self.sigma8 = 0.3, 0.0, -1.0, 0.68, 0.048743, 0.770251
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Omc0, self.wc, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Omc0, self.Ok0, self.wc, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-2.0, -0.1), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-2.0, -0.1), (0.4, 1), (0.001, 0.2))

class LCDM_coup2(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Omc0, self.Ok0, self.wa, self.wb, self.h, self.Oba, self.sigma8 = 0.3, 0.0, -1.0, 0.0, 0.68, 0.048743, 0.770251
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Omc0, self.wa, self.wb, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Omc0, self.Ok0, self.wa, self.wb, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (-3.0, -0.1), (-3.0, 3.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (-3.0, -0.1), (-3.0, 3.0), (0.4, 1), (0.001, 0.2))

class H_LOG(Flat_LCDM):
    
    def __init__(self, Curvature):
        self.Curvature = Curvature
        self.Om0, self.Ok0, self.beta, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.0, 0.679585, 0.048743, 0.8
    
    def param(self):
        if self.Curvature == False:
            return np.array([self.Om0, self.beta, self.h, self.Oba])
        elif self.Curvature == True:
            return np.array([self.Om0, self.Ok0, self.beta, self.h, self.Oba])

    def param_bnds(self):
        if self.Curvature == False:
            return ((0.01, 0.5), (0.0, 2.0), (0.4, 1), (0.001, 0.2))
        elif self.Curvature == True:
            return ((0.01, 0.5), (-0.3, 0.3), (0.0, 2.0), (0.4, 1), (0.001, 0.2))

class fR(Flat_LCDM):

    def __init__(self):
        self.Om0, self.B0, self.h, self.Oba, self.sigma8 = 0.3, 0.0, 0.68, 0.05, 0.8
    
    def param(self):
        return np.array([self.Om0, self.B0, self.h, self.Oba])
    
    def param_bnds(self):
        return ((0.1, 0.5), (-0.1, 1.0), (0.4, 1), (0.001, 0.2))

