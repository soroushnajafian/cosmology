from __future__ import division
import math
import numpy as np
import scipy as sp
from scipy import *
import de_hh
from BAOLikelihood import *
from BAOLikelihood_2 import *
from task import *
import scipy.optimize as op
import emcee
from Numericaltools import Radiation as NR

def lnprior(Om, Ok, h, Ob):
    if 0 < Om < 1.0 and -0.5 < Ok < 0.5 and 0 <= h <= 1.0 and 0 <= Ob <= Om:
        return 0.0
    else:
        return -np.inf

def lnprob(param, CMB=False, BAO=False, fsig8=False, SNe=False, H0=False):
    Om = param[0]
    Ok = param[1]
    h = param[2]
    Ob = param[3]
    sigma8 = []
    
    if fsig8 == True:
        sigma8 = param[4]
        if np.isinf(lnprior_sigma8(sigma8))==True:
            return -np.inf

    if np.isinf(lnprior(Om, Ok, h, Ob))==False:
        model = de_hh.LCDM(Om, Ok, NR().Radiation_fac/h**2*(1.0+0.2271*3.046), h, Growth=fsig8)
        rd = model.rd(0.0, Ob)
        return lnprob_joint(model, rd, sigma8=sigma8, CMB=CMB, BAO=BAO, fsig8=fsig8, SNe=SNe, H0=H0)
    else:
        return -np.inf

modelname = 'LCDM'
Nstep = 10000
Nburnin = 3000

dataname = ['BAO_CMB', 'BAO_CMB_H0', 'BAO_CMB_fsig8', 'BAO_CMB_fsig8_H0', 'BAO_CMB_SNe', 'BAO_CMB_SNe_H0', 'BAO_CMB_SNe_fsig8', 'BAO_CMB_SNe_fsig8_H0']
switch = [False, True]
ss=0
for i in [0,1]:
    SNe_key = switch[i]
    for j in [0,1]:
        fsig8_key = switch[j]
        for k in [0,1]:
            H0_key = switch[k]
            func = lambda *args: lnprob(*args, CMB=True, BAO=True, fsig8=fsig8_key, SNe=SNe_key, H0=H0_key)
            pos=np.loadtxt('output/bestfit/best_'+modelname+'_'+dataname[ss]+'.dat')[1:]
            nwalkers=2*len(pos)
            position = [pos+1e-3*np.random.randn(len(pos)) for jj in range(nwalkers)]
            task_MCMC(modelname=modelname, dataname=dataname[ss], position=position, nwalkers=nwalkers, Nstep=Nstep, Nburnin=Nburnin, logf=func)
            ss = ss+1





