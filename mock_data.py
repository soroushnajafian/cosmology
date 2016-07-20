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
from scipy import *
import scipy.linalg as la
import de_hh

class data_generate:

    def __init__(self, modelname, fitmodel, fitmodelrd, fitmodelsig8):

        self.modelname = modelname
        self.fitmodel = fitmodel
        self.fitmodelrd = fitmodelrd
        self.fitmodelsig8 = fitmodelsig8

        self.z = 0.106  # 6dFGS
        self.DV_mean = (self.z*self.fitmodel.D_Hz(self.z)*self.fitmodel.D_M(self.z)**2)**(1.0/3.0)
        self.DVrd_mean = self.DV_mean/self.fitmodelrd
        self.DVrd_obs = np.random.normal(self.DVrd_mean, 0.137)
        np.savetxt('./data/mock/mock_'+self.modelname+'_6dFGS.dat', np.array(self.DVrd_obs).reshape(1,))

        self.z = 0.32   #  LOWZ
        self.DV_mean = (self.z*self.fitmodel.D_Hz(self.z)*self.fitmodel.D_M(self.z)**2)**(1.0/3.0)
        self.DVrd_mean = self.DV_mean/self.fitmodelrd
        self.DVrd_obs = np.random.normal(self.DVrd_mean, 0.167)
        np.savetxt('./data/mock/mock_'+self.modelname+'_LOWZ.dat', np.array(self.DVrd_obs).reshape(1,))

        self.z = 0.15    # MGS
        self.DV_mean = (self.z*self.fitmodel.D_Hz(self.z)*self.fitmodel.D_M(self.z)**2)**(1.0/3.0)
        self.DVrd_mean = self.DV_mean/self.fitmodelrd
        self.obh2=0.021547; self.Om=0.31; self.h=0.67; self.Onu=0.0
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DV = (self.z*self.fid_th.D_Hz(self.z)*self.fid_th.D_M(self.z)**2.0)**(1.0/3.0)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DVrd = self.fid_DV/self.fid_rd
        self.fit_mean = self.DVrd_mean/self.fid_DVrd
        self.obs_shift = np.random.normal(self.fit_mean, 0.168/4.480*self.fit_mean)-self.fit_mean
        np.savetxt('./data/mock/mock_'+self.modelname+'_MGS.dat', zip((self.fit_mean-1.0395, self.obs_shift)))
        #   1.0395 is the value of alpha which has minimum of chi squared



        del self.obh2, self.Om, self.h, self.Onu
        self.z = 0.57   #  CMASS
        self.DM_fit = self.fitmodel.D_M(self.z)
        self.DH_fit = self.fitmodel.D_Hz(self.z)
        self.fit_mean = np.array([self.DM_fit/self.fitmodelrd, self.DH_fit/self.fitmodelrd])
        self.obh2=0.0224; self.Om=0.274; self.h=0.7; self.Onu=0.0
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DM = self.fid_th.D_M(self.z)
        self.fid_DH = self.fid_th.D_Hz(self.z)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DMrd = self.fid_DM/self.fid_rd
        self.fid_DHrd = self.fid_DH/self.fid_rd
        self.fit_alpha = self.fit_mean/[self.fid_DMrd, self.fid_DHrd]
        self.BAOcov = np.array([[(0.210/14.945*self.fit_alpha[0])**2, -0.52*0.210/14.945*self.fit_alpha[0]*0.73/20.75*self.fit_alpha[1]], [-0.52*0.210/14.945*self.fit_alpha[0]*0.73/20.75*self.fit_alpha[1], (0.73/20.75*self.fit_alpha[1])**2]])
        self.BAO_obs = np.random.multivariate_normal(self.fit_alpha, self.BAOcov)
        self.obs_shift = self.BAO_obs-self.fit_alpha
        
        self.bao_data = loadtxt('./data/CMASS.dat')
        self.alpha_min = self.bao_data[np.where(self.bao_data[:,2]==max(self.bao_data[:,2]))]
        self.alpha_min = self.alpha_min[0, 0:2]
        self.shift_chi2 = np.array(self.fit_alpha-self.alpha_min)
        np.savetxt('./data/mock/mock_'+self.modelname+'_CMASS_chi2_shift.dat', self.shift_chi2)
        np.savetxt('./data/mock/mock_'+self.modelname+'_CMASS.dat', self.obs_shift.T)


        del self.obh2, self.Om, self.h, self.Onu
        self.z = 2.34   #   LyaF_auto
        self.DM_fit = self.fitmodel.D_M(self.z)
        self.DH_fit = self.fitmodel.D_Hz(self.z)
        self.fit_mean = np.array([self.DM_fit/self.fitmodelrd, self.DH_fit/self.fitmodelrd])
        self.obh2=0.0227; self.Om=0.27; self.h=0.7; self.Onu=0.0
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DM = self.fid_th.D_M(self.z)
        self.fid_DH = self.fid_th.D_Hz(self.z)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DMrd = self.fid_DM/self.fid_rd
        self.fid_DHrd = self.fid_DH/self.fid_rd
        self.fit_alpha = self.fit_mean/[self.fid_DMrd, self.fid_DHrd]
        self.BAOcov = np.array([[(2.171/37.675*self.fit_alpha[0])**2, -0.43*2.171/37.675*self.fit_alpha[0]*0.28/9.18*self.fit_alpha[1]], [-0.43*2.171/37.675*self.fit_alpha[0]*0.28/9.18*self.fit_alpha[1], (0.28/9.18*self.fit_alpha[1])**2]])
        self.BAO_obs = np.random.multivariate_normal(self.fit_alpha, self.BAOcov)
        self.obs_shift = self.BAO_obs-self.fit_alpha
        
        self.bao_data = loadtxt('./data/lyabaoauto.txt')
        self.alpha_min = self.bao_data[np.where(self.bao_data[:,4]==min(self.bao_data[:,4]))]
        self.alpha_min = self.alpha_min[0, 0:2]
        self.shift_chi2 = np.array(self.fit_alpha-self.alpha_min)
        np.savetxt('./data/mock/mock_'+self.modelname+'_LyaF_auto_chi2_shift.dat', self.shift_chi2)
        np.savetxt('./data/mock/mock_'+self.modelname+'_LyaF_auto.dat', self.obs_shift.T)



        del self.obh2, self.Om, self.h, self.Onu
        self.z = 2.36  #   LyaF_cross
        self.DM_fit = self.fitmodel.D_M(self.z)
        self.DH_fit = self.fitmodel.D_Hz(self.z)
        self.fit_mean = np.array([self.DM_fit/self.fitmodelrd, self.DH_fit/self.fitmodelrd])
        self.obh2=0.0227; self.Om=0.27; self.h=0.7; self.Onu=0.0
        self.fid_th = de_hh.Flat_LCDM(self.Om, 0.0, self.h)
        self.fid_DM = self.fid_th.D_M(self.z)
        self.fid_DH = self.fid_th.D_Hz(self.z)
        self.fid_rd = self.fid_th.rd(self.Onu/self.h**2, self.obh2/self.h**2)
        self.fid_DMrd = self.fid_DM/self.fid_rd
        self.fid_DHrd = self.fid_DH/self.fid_rd
        self.fit_alpha = self.fit_mean/[self.fid_DMrd, self.fid_DHrd]
        self.BAOcov = np.array([[(1.344/36.288*self.fit_alpha[0])**2, -0.39*1.344/36.288*self.fit_alpha[0]*0.30/9.00*self.fit_alpha[1]], [-0.39*1.344/36.288*self.fit_alpha[0]*0.30/9.00*self.fit_alpha[1], (0.30/9.00*self.fit_alpha[1])**2]])
        self.BAO_obs = np.random.multivariate_normal(self.fit_alpha, self.BAOcov)
        self.obs_shift = self.BAO_obs-self.fit_alpha
        
        self.bao_data = loadtxt('./data/lyabaocross.scan')
        self.alpha_min = self.bao_data[np.where(self.bao_data[:,2]==min(self.bao_data[:,2]))]
        self.alpha_min = self.alpha_min[0, 0:2]
        self.shift_chi2 = np.array(self.fit_alpha-self.alpha_min)
        np.savetxt('./data/mock/mock_'+self.modelname+'_LyaF_cross_chi2_shift.dat', self.shift_chi2)
        np.savetxt('./data/mock/mock_'+self.modelname+'_LyaF_cross.dat', self.obs_shift.T)

        #  mock Growth data
        self.data = loadtxt('./data/growthdata.dat')
        self.dataz = self.data[:,0]
        self.fit_mean = map(self.fitmodel.Dfsig8, self.dataz, np.ones(len(self.dataz))*self.fitmodelsig8)
        self.obs = [np.random.normal(self.fit_mean[icc], self.data[icc,2]) for icc in range(len(self.dataz))]
        np.savetxt('./data/mock/mock_'+self.modelname+'_growthdata.dat', self.obs)

        # mock SNe data
        self.data = loadtxt('./data/JLA31.dat')
        self.SNcov = loadtxt('./data/JLA31_cov.dat')
        self.dataz = self.data[:,0]
        self.fit_mean = [self.fitmodel.mu(zi)+43.0 for zi in self.dataz]
        self.obs = np.random.multivariate_normal(self.fit_mean, self.SNcov).T
        np.savetxt('./data/mock/mock_'+self.modelname+'_SNe.dat', self.obs)

        # mock (compressed) CMB data, just for Planck data
        self.z = 1090.0
        self.fit_mean = np.array([self.fitmodel.Oba*self.fitmodel.h**2, self.fitmodel.Om0*self.fitmodel.h**2, self.fitmodel.D_M(self.z)/self.fitmodelrd])
        self.CMB_cov = array([[  1.28650185e-07,  -6.03648304e-07,   1.43024604e-05 ],
                              [ -6.03648304e-07,   7.55086874e-06,  -3.40923062e-05 ],
                              [ 1.43024604e-05,  -3.40923062e-05,   4.24432709e-03]])
        self.CMB_obs = np.random.multivariate_normal(self.fit_mean, self.CMB_cov).T
        np.savetxt('./data/mock/mock_'+self.modelname+'_CMB.dat', self.CMB_obs)
        
        #  mock (compressed) CMB data with sigma8 at z=30, just for Planck data
        self.z =1090.0
        self.fit_mean = np.array([self.fitmodel.Oba*self.fitmodel.h**2, self.fitmodel.Om0*self.fitmodel.h**2, self.fitmodel.D_M(self.z)/self.fitmodelrd, self.fitmodel.D_z(30.0)*self.fitmodelsig8])
        self.CMBS_cov = array([[  1.28650185e-07,  -6.03648304e-07,   1.43024604e-05,  3.44692891e-08 ],
                               [ -6.03648304e-07,   7.55086874e-06,  -3.40923062e-05,  -2.35840314e-07 ],
                               [  1.43024604e-05,  -3.40923062e-05,   4.24432709e-03,  -6.59458356e-07 ],
                               [  3.44692891e-08,  -2.35840314e-07,  -6.59458356e-07,  2.19859709e-07]])
        self.CMBS_obs = np.random.multivariate_normal(self.fit_mean, self.CMBS_cov).T
        np.savetxt('./data/mock/mock_'+self.modelname+'_CMBS.dat', self.CMBS_obs)


        # mock H0 data
        self.h_obs = np.random.normal(self.fitmodel.h, 0.0179)
        np.savetxt('./data/mock/mock_'+self.modelname+'_H0.dat', np.array(self.h_obs).reshape(1,))




















