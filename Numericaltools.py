import numpy as np
from scipy import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree as KDTree

class NuIntegral:
    def __init__(self):
        rat=10**(arange(-4,5,0.1))
        intg=[]
        for r in rat:
            res=quad(lambda x:sqrt(x*x+r**2)/(exp(min(x,400))+1.0)*x**2, 0,1000)
            intg.append(res[0]/(1+r))
        intg=array(intg)
        intg*=7/8./intg[0]
        
        self.interpolator=interp1d(log(rat),intg)
        self.int_infty=0.2776566337
    
    def SevenEights(self,mnuOT):
        if (mnuOT<1e-4):
            return 7/8.
        elif (mnuOT>1e4):
            return self.int_infty*mnuOT
        else:
            return self.interpolator(log(mnuOT))*(1+mnuOT)

class Tools:
    def __init__(self):
        return
    
    def Inte1D(self, x, y, x0):
        self.ff = UnivariateSpline(x, y, k=1, s=0)
        if self.ff(x0)>=1.0:
            return 1.0
        elif self.ff(x0)<=0.0:
            return 0.0
        else:
            return self.ff(x0)

class Radiation:
    def __init__(self):
        self.Radiation_fac = 2.4728041043596488e-05


class Plot:
    def __init__(self):
        return

    def data(self, dataname, quantity):
        self.dataname=dataname
        self.quantity=quantity
        dir_d = '/Users/zhongxuzhai/Documents/work/research/DE/cosmology/'
        
        if self.dataname == 'CMASS' and self.quantity == 'DMrd':
            t=np.loadtxt(dir_d+'data/BAO_Dm.dat')
            return t[0]
        
        elif self.dataname == 'LyaF_Auto' and self.quantity == 'DMrd':
            t=np.loadtxt(dir_d+'data/BAO_Dm.dat')
            return t[1]

        elif self.dataname == 'LyaF_Cross' and self.quantity == 'DMrd':
            t=np.loadtxt(dir_d+'data/BAO_Dm.dat')
            return t[2]

        elif self.dataname == 'CMASS' and self.quantity == 'DHrd':
            t=np.loadtxt(dir_d+'data/BAO_Dh.dat')
            return t[0]
                
        elif self.dataname == 'LyaF_Auto' and self.quantity == 'DHrd':
            t=np.loadtxt(dir_d+'data/BAO_Dh.dat')
            return t[1]

        elif self.dataname == 'LyaF_Cross' and self.quantity == 'DHrd':
            t=np.loadtxt(dir_d+'data/BAO_Dh.dat')
            return t[2]

        elif self.dataname == '6dFGS' and self.quantity == 'DVrd':
            t=np.loadtxt(dir_d+'data/BAO_Dv.dat')
            return t[0]

        elif self.dataname == 'MGS' and self.quantity == 'DVrd':
            t=np.loadtxt(dir_d+'data/BAO_Dv.dat')
            return t[1]

        elif self.dataname == 'LOWZ' and self.quantity == 'DVrd':
            t=np.loadtxt(dir_d+'data/BAO_Dv.dat')
            return t[2]

        elif self.dataname == 'H0' and self.quantity == 'H0':
            return [0, 0.7302, 0.0179]

        elif self.dataname == 'SNe' and self.quantity == 'DL_Mod':
            t1=np.loadtxt(dir_d+'data/JLA31.dat')
            t2=np.loadtxt(dir_d+'data/JLA31_cov.dat')
            sig=np.sqrt(np.diag(t2))
            t=[[t1[i][0], t1[i][1], sig[i]] for i in range(len(sig))]
            return t

        elif self.dataname == 'Growth' and self.quantity == 'Dfsigma8':
            t=np.loadtxt(dir_d+'data/growthdata.dat')
            return t

    def err(self, Q, sigma=1):
        
        Nz = len(Q[0])
        N = len(Q)
        sigma_u = []
        sigma_l = []
        Q_c50 = []
        
        if sigma==1:
            frac1=0.16
            frac2=0.84
        
        elif sigma==2:
            frac1=0.025
            frac2=0.975
        
        for ii in range(Nz):
            Q_c = sorted(Q[:,ii])
            Q_c50.append(Q_c[int(N*0.5)])
            sigma_u.append(Q_c[int(N*frac1)])
            sigma_l.append(Q_c[int(N*frac2)])

        return np.array(sigma_u)-np.array(Q_c50), np.array(sigma_l)-np.array(Q_c50)

    def prediction(self, dataname, quantity):
        self.dataname=dataname
        self.quantity=quantity
        dir_d = '/Users/zhongxuzhai/Documents/work/research/DE/cosmology/'
        
        if self.dataname == 'eBOSS' and self.quantity == 'DMrd':
            t = np.loadtxt(dir_d+'data/prediction/eBOSS_DA.dat')
        
        elif self.dataname == 'eBOSS' and self.quantity == 'DHrd':
            t = np.loadtxt(dir_d+'data/prediction/eBOSS_DH.dat')
        
        elif self.dataname == 'eBOSS' and self.quantity == 'DVrd':
            t = np.loadtxt(dir_d+'data/prediction/eBOSS_DV.dat')

        elif self.dataname == 'eBOSS_comb' and self.quantity == 'Dfsigma8':
            t = np.loadtxt(dir_d+'data/prediction/eBOSS_comb_fsig.dat')

        elif self.dataname == 'eBOSS' and self.quantity == 'Dfsigma8':
            t = np.loadtxt(dir_d+'data/prediction/eBOSS_fsig.dat')

        elif self.dataname == 'DESI' and self.quantity == 'DMrd':
            t = np.loadtxt(dir_d+'data/prediction/DESI_DA.dat')
        
        elif self.dataname == 'DESI' and self.quantity == 'DHrd':
            t = np.loadtxt(dir_d+'data/prediction/DESI_DH.dat')

        elif self.dataname == 'DESI_LyaF' and self.quantity == 'DMrd':
            t = np.loadtxt(dir_d+'data/prediction/DESI_LyaF_DA.dat')
        
        elif self.dataname == 'DESI_LyaF' and self.quantity == 'DHrd':
            t = np.loadtxt(dir_d+'data/prediction/DESI_LyaF_DH.dat')

        elif self.dataname == 'DESI' and self.quantity == 'Dfsigma8':
            t = np.loadtxt(dir_d+'data/prediction/DESI_fsig.dat')

        return t

class WCSF1_tab:
    def __init__(self, aeq):
        self.aeq = aeq
        self.zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))
        self.FFs = map(self.FFint, self.zs_Ode)

    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)

    def FF_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F(xx)**2.0/(1.0+z)

    def FFint(self, z):
        return quad(self.FF_int, 0, z)[0]

class WCSF2_tab:
    def __init__(self, aeq):
        self.aeq = aeq
        self.zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))
        self.FFs = map(self.FFint, self.zs_Ode)
        self.Fs = map(self.Fint, self.zs_Ode)
    
    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)
    
    def FF_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F(xx)**2.0/(1.0+z)
    
    def FFint(self, z):
        return quad(self.FF_int, 0, z)[0]

    def F_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F(xx)/(1.0+z)

    def Fint(self, z):
        return quad(self.F_int, 0, z)[0]

class WCSF3_tab:
    def __init__(self, aeq):
        self.aeq = aeq
        self.zs_Ode = np.hstack((np.linspace(0.0, 2.99, 300), np.linspace(3.0, 1100, 200)))
        self.FFs = map(self.FFint, self.zs_Ode)
        self.Fs = map(self.Fint, self.zs_Ode)
        self.F2s = map(self.F2int, self.zs_Ode)
        self.F2F2s = map(self.F2F2int, self.zs_Ode)
        self.FF2s = map(self.FF2int, self.zs_Ode)
    
    def F(self, x):
        return np.sqrt(1.0+np.power(x, 3.0))/np.power(x, 1.5)-np.log(np.power(x, 1.5)+np.sqrt(1.0+np.power(x, 3.0)))/np.power(x, 3.0)
   
    def F2(self, x):
        return 2.0**0.5*(1.0-np.log(1.0+np.power(x, 3.0))/np.power(x, 3.0))-self.F(x)
            
    def FF_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F(xx)**2.0/(1.0+z)
    
    def FFint(self, z):
        return quad(self.FF_int, 0, z)[0]
    
    def F_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F(xx)/(1.0+z)
    
    def Fint(self, z):
        return quad(self.F_int, 0, z)[0]

    def F2_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F2(xx)/(1.0+z)
    
    def F2int(self, z):
        return quad(self.F2_int, 0, z)[0]

    def FF2_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F2(xx)**self.F(xx)/(1.0+z)
    
    def FF2int(self, z):
        return quad(self.FF2_int, 0, z)[0]

    def F2F2_int(self, z):
        xx = 1.0/(1.0+z)/self.aeq
        return self.F2(xx)**2.0/(1.0+z)
    
    def F2F2int(self, z):
        return quad(self.F2F2_int, 0, z)[0]


class Invdisttree:
    # this module is from stackoverflow
    """ inverse-distance-weighted interpolation using KDTree:
        invdisttree = Invdisttree( X, z )  -- data points, values
        interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3
        
        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights
        
        How many nearest neighbors should one take ?
        a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
        b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.
        
        p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.
        
        Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .
        
        A nice property of IDW is that it's scale-free around query points:
        if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
        the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
        In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
        is exceedingly sensitive to distance and to h.
        
        """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim
    
    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;
    
    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)
        
        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]






