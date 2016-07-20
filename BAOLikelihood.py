from scipy import *
import scipy.linalg as la
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import de_hh
import numpy as np
from scipy.interpolate import UnivariateSpline


class GaussBAODVLikelihood:
    
    def __init__(self, name, model, z, modelrd, maxchi2=1e30):
        
        self.name = name
        self.model = model
        self.z = float(z)
        self.DV_th = (self.z*self.model.D_Hz(self.z)*self.model.D_M(self.z)**2)**(1.0/3.0)
        self.rd_th = modelrd
        self.DVrd_th = self.DV_th/self.rd_th
        
        if self.name == 'SixdFGS':
            self.DVrd_obs = 3.047
            self.DVrderr = 0.137
        
        elif self.name == 'LOWZ':
            self.DVrd_obs = 8.467
            self.DVrderr = 0.167
        else:
            print("The data you entered can't be found.")
            exit()

        self.maxchi2=maxchi2
        self.DVrderr2=self.DVrderr**2
    
    
    def loglike(self):

        self.chi2=min(self.maxchi2, (self.DVrd_th-self.DVrd_obs)**2/(self.DVrderr2))

        return -self.chi2/2.0


class TabulatedBAODVLikelihood:

    def __init__(self, name, filename, model, z, modelrd):

        self.name = name
        self.model = model
        self.z = float(z)
        self.rd_th = modelrd
        self.DV_th = (self.z*self.model.D_Hz(self.z)*self.model.D_M(self.z)**2)**(1.0/3.0)
        self.DVrd_th = self.DV_th/self.rd_th
        
        self.filename = filename
        self.data=loadtxt(self.filename)
        self.chi2i=interp1d(self.data[:,0],self.data[:,1])
        
        self.obh2=0.021547; self.Om=0.31; self.h=0.67; self.Onu=0.0
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DV = (self.z*self.fid_th.D_Hz(self.z)*self.fid_th.D_M(self.z)**2.0)**(1.0/3.0)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DVrd = self.fid_DV/self.fid_rd

    def loglike(self):
        self.alpha=self.DVrd_th/self.fid_DVrd
        try:
            self.chi2=self.chi2i(self.alpha)
        except:
            #print "Note: alpha for ",self.name(),"out of lookup-table bounds"
            self.chi2=9
        return -self.chi2/2.0

class TabulatedBAOLikelihood:
    
    def __init__(self, name, Interalpha, model, z, modelrd):
        self.name = name
        self.z = float(z)
        self.Interalpha = Interalpha
        self.model = model
        self.rd_th = modelrd
        self.DM_th = self.model.D_M(self.z)
        self.DH_th = self.model.D_Hz(self.z)
        self.DMrd_th = self.DM_th/self.rd_th
        self.DHrd_th = self.DH_th/self.rd_th
        
        if self.name == 'CMASS':
            self.obh2=0.0224; self.Om=0.274; self.h=0.7; self.Onu=0
        elif self.name == 'LyaF_auto':
            self.obh2=0.0227; self.Om=0.27; self.h=0.7; self.Onu=0.0
        elif self.name == 'LyaF_cross':
            self.obh2=0.0227; self.Om=0.27; self.h=0.7; self.Onu=0.0
        else:
            print("The data you entered can't be found.")
            exit()
        
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DM = self.fid_th.D_M(self.z)
        self.fid_DH = self.fid_th.D_Hz(self.z)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DMrd = self.fid_DM/self.fid_rd
        self.fid_DHrd = self.fid_DH/self.fid_rd
    
    
    def loglike(self):
        self.alphaperp=self.DMrd_th/self.fid_DMrd
        self.alphapar=self.DHrd_th/self.fid_DHrd
        return self.Interalpha(self.alphaperp,self.alphapar)[0][0]


class GaussDALikelihood:

    def __init__(self, data, model, modelrd, maxchi2=1e30):

        self.model = model
        self.modelrd = modelrd
        self.data = data
        self.maxchi2 = maxchi2=1e30
        self.dataz = self.data[:,0]
        self.DA_derr2 = self.data[:,2]**2

    def DA_th(self):

        return map(self.model.D_A, self.dataz)/self.modelrd

    def loglike(self):
        self.chi2 = min(self.maxchi2, sum((self.DA_th()-self.data[:,1])**2/(self.DA_derr2)))
        return -self.chi2/2.0

class GaussDHLikelihood:


    def __init__(self, data, model, modelrd, maxchi2=1e30):
    
        self.model = model
        self.modelrd = modelrd
        self.data = data
        self.maxchi2 = maxchi2=1e30
        self.dataz = self.data[:,0]
        self.DH_derr2 = self.data[:,2]**2

    def DH_th(self):
        
        return map(self.model.D_Hz, self.dataz)/self.modelrd

    def loglike(self):
        self.chi2 = min(self.maxchi2, sum((self.DH_th()-self.data[:,1])**2/(self.DH_derr2)))
        return -self.chi2/2.0


class CMBLikelihood:

    def __init__(self, name, model, rd, z):
        
        self.name = name
        self.model = model
        self.rd = rd
        self.z = z
        self.Om = self.model.Om0
        self.Ob = self.model.Oba
        self.h = self.model.h
        
        if self.name == 'Planck':
            self.data_vec = array([   0.02245221,   0.1392552,   94.26486759])
            self.cov = array([[  1.28650185e-07,  -6.03648304e-07,   1.43024604e-05 ],
                              [ -6.03648304e-07,   7.55086874e-06,  -3.40923062e-05 ],
                              [ 1.43024604e-05,  -3.40923062e-05,   4.24432709e-03]])
            self.icov = la.inv(self.cov)

        else:
            print("The data you entered can't be found.")
            exit()

        self.DM_th = self.model.D_M(self.z)
        self.v3 = self.DM_th/self.rd
        self.v1 = self.Ob*self.h**2
        self.v2 = self.Om*self.h**2
        self.vec = array([self.v1, self.v2, self.v3])

    def loglike(self):
        self.delt = self.vec-self.data_vec
        return -dot(self.delt,dot(self.icov,self.delt))/2.0

class CMBSLikelihood:
    
    def __init__(self, name, model, rd, sigma8, z):
        
        self.name = name
        self.model = model
        self.rd = rd
        self.sigma8 = sigma8
        self.z = z
        self.Om = self.model.Om0
        self.Ob = self.model.Oba
        self.h = self.model.h
        
        if self.name == 'Planck':
            self.data_vec = array([   0.02245221,   0.1392552,   94.26486759,   0.03448874])
            self.cov = array([[  1.28650185e-07,  -6.03648304e-07,   1.43024604e-05,  3.44692891e-08 ],
                              [ -6.03648304e-07,   7.55086874e-06,  -3.40923062e-05,  -2.35840314e-07 ],
                              [  1.43024604e-05,  -3.40923062e-05,   4.24432709e-03,  -6.59458356e-07 ],
                              [  3.44692891e-08,  -2.35840314e-07,  -6.59458356e-07,  2.19859709e-07]])
            self.icov = la.inv(self.cov)
        
        else:
            print("The data you entered can't be found.")
            exit()
        
        self.DM_th = self.model.D_M(self.z)
        self.v3 = self.DM_th/self.rd
        self.v1 = self.Ob*self.h**2
        self.v2 = self.Om*self.h**2
        self.v4 = self.model.D_z(30.0)*self.sigma8
        self.vec = array([self.v1, self.v2, self.v3, self.v4])
    
    def loglike(self):
        self.delt = self.vec-self.data_vec
        return -dot(self.delt,dot(self.icov,self.delt))/2.0


class Growthdata:
    def __init__(self, model, modelsig8, maxchi2=1e30):
    
        self.model = model
        self.modelsig8 = modelsig8
        self.data = loadtxt('./data/growthdata.dat')
        self.maxchi2=maxchi2
        self.dataz = self.data[:,0]
        self.gr_derr2=self.data[:,2]**2

    def gr_th(self):

        return map(self.model.Dfsig8, self.dataz, np.ones(len(self.dataz))*self.modelsig8)


    def loglike(self):
    
        self.chi2=min(self.maxchi2, sum((self.gr_th()-self.data[:,1])**2/(self.gr_derr2)))
        
        return -self.chi2/2.0

class SNeLikelihood:
    def __init__(self, model, maxchi2=1e30):

        self.model = model
        self.data = loadtxt('./data/JLA31.dat')
        self.SNcov = loadtxt('./data/JLA31_cov.dat')
        self.SNcov += 3**2
        self.icov = la.inv(self.SNcov)
        self.maxchi2=maxchi2
        self.dataz = self.data[:,0]
            
    def mu_th(self):
        #return np.array(map(self.model.mu, self.dataz))
        return np.array([self.model.mu(zi)+43.0 for zi in self.dataz])

    def loglike(self):
        self.delt = self.mu_th()-self.data[:,1]
        return -dot(self.delt,dot(self.icov,self.delt))/2.0

class H0Likelihood:
    def __init__(self, model, maxchi2=1e30):
        
        self.model = model
        self.h_th = self.model.h
        self.h_err = 0.0179
        self.h_obs = 0.7302
        self.maxchi2=maxchi2

    def loglike(self):
        self.chi2=min(self.maxchi2, (self.h_th-self.h_obs)**2/(self.h_err**2))
        return -self.chi2/2.0


