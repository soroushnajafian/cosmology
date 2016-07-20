from __future__ import division
import math
import numpy as np
import scipy as sp
from scipy import *
import de_hh
from BAOLikelihood_mock import *
from BAOLikelihood_2_mock import *
import emcee
import scipy.optimize as op
from mock_data import *
from task import *
import Parampy
from Numericaltools import Radiation as NR

def lnprob(param, mockID, CMB=False, BAO=False, fsig8=False, SNe=False, H0=False):
    Om = param[0]
    Ok = param[1]
    h = param[2]
    Ob = param[3]
    sigma8 = []
    
    if fsig8 == True:
        sigma8 = param[4]

    model = de_hh.LCDM(Om, Ok, NR().Radiation_fac/h**2*(1.0+0.2271*3.046), h, Growth=fsig8)
    rd = model.rd(0.0, Ob)
    return lnprob_joint(modelname, model, rd, mockID=mockID, sigma8=sigma8, CMB=CMB, BAO=BAO, fsig8=fsig8, SNe=SNe, H0=H0)

modelname = 'LCDM'
model = Parampy.LCDM()
Ntest = 10
Nstep = 1
'''
dataname = ['BAO_CMB', 'BAO_CMB_H0', 'BAO_CMB_fsig8', 'BAO_CMB_fsig8_H0', 'BAO_CMB_SNe', 'BAO_CMB_SNe_H0', 'BAO_CMB_SNe_fsig8', 'BAO_CMB_SNe_fsig8_H0']
switch = [False, True]
ss=0
for i in [0,1]:
    SNe_key = switch[i]
    for j in [0,1]:
        fsig8_key = switch[j]
        for k in [0,1]:
            H0_key = switch[k]

            if fsig8_key == False:
                bnds = model.param_bnds()
                param = model.param()
            
            else:
                bnds = model.param_sigma_bnds()
                param = model.param_sigma()
        
            nll = lambda *args: -lnprob(*args, mockID=False, CMB=True, BAO=True, fsig8=fsig8_key, SNe=SNe_key, H0=H0_key)
            task_GoF(modelname=modelname, dataname=dataname[ss], N=Ntest, Func=nll, init_pos=param, bnds=bnds)
            
            ss=ss+1
'''
dataname='BBB'
bnds = model.param_sigma_bnds()
param = model.param_sigma()
nll = lambda *args: -lnprob(*args, mockID=False, CMB=True, BAO=True, fsig8=True, SNe=False, H0=False)
task_GoF(modelname=modelname, dataname=dataname, N=Ntest, Func=nll, init_pos=param, bnds=bnds)

