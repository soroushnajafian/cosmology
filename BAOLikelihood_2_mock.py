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
import de_hh
from BAOLikelihood_mock import *

#################################################

for i in range(3):
    if i == 0 :
        data = loadtxt("data/CMASS.dat")
        chi2col = -2
    if i == 1:
        data = loadtxt("data/lyabaoauto.txt")
        chi2col = 4
    if i == 2:
        data = loadtxt("data/lyabaocross.scan")
        chi2col = 2
    
    aperp = set()
    aparl = set()
    for line in data:
        aperp.add(line[0])
        aparl.add(line[1])
    aperp = sorted(list(aperp))
    aparl = sorted(list(aparl))

    logltab = zeros((len(aperp), len(aparl)))
    
    aperpp = array(aperp)
    aparll = array(aparl)
    
    for line in data:
        ii = aperp.index(line[0])
        jj = aparl.index(line[1])
        if chi2col>0:
            chi2 = line[chi2col]
            logltab[ii,jj]=-chi2/2.0
        else:
            logltab[ii,jj]=log(line[chi2col*-1]+1e-50)

    logltab = logltab-logltab.max()
    loglint = RectBivariateSpline(aperpp, aparll, logltab, kx=1, ky=1)
        
    if i == 0:
        loglint_CMASS = loglint
    if i == 1:
        loglint_LyaF_auto = loglint
    if i == 2:
        loglint_LyaF_cross = loglint
            
    del data, aperp, aparl, logltab, aperpp, aparll, loglint

########################################################################################

def lnlike_Gala(modelname, model, rd, mockID):
    
    z_6dFGS = 0.106
    SixdFGS_chi2 = GaussBAODVLikelihood('SixdFGS', modelname, model, z_6dFGS, rd, mock=mockID).loglike()

    z_LOWZ = 0.32
    LOWZ_chi2 = GaussBAODVLikelihood('LOWZ', modelname, model, z_LOWZ, rd, mock=mockID).loglike()

    z_MGS = 0.15
    MGS_chi2 = TabulatedBAODVLikelihood('MGS', "data/MGS.dat", modelname, model, z_MGS, rd, mock=mockID).loglike()
    
    z_CMASS = 0.57
    CMASS_chi2 = TabulatedBAOLikelihood('CMASS', loglint_CMASS, modelname, model, z_CMASS, rd, mock=mockID).loglike()
    
    return SixdFGS_chi2+LOWZ_chi2+MGS_chi2+CMASS_chi2

def lnlike_LyaF(modelname, model, rd, mockID):
    
    z_LyaFauto = 2.34
    LyaFauto_chi2 = TabulatedBAOLikelihood('LyaF_auto', loglint_LyaF_auto, modelname, model, z_LyaFauto, rd, mock=mockID).loglike()
    
    z_LyaFcross = 2.36
    LyaFcross_chi2 = TabulatedBAOLikelihood('LyaF_cross', loglint_LyaF_cross, modelname, model, z_LyaFcross, rd, mock=mockID).loglike()
    return LyaFauto_chi2+LyaFcross_chi2

def lnlike_CMB(modelname, model, rd, mockID):
    
    z_Plunck = 1090.0
    Planck_chi2 = CMBLikelihood('Planck', modelname, model, rd, z_Plunck, mock=mockID).loglike()
    return Planck_chi2

def lnlike_CMBS(modelname, model, sigma8, rd, mockID):
    
    z_Plunck = 1090.0
    Planck_chi2 = CMBSLikelihood('Planck', modelname, model, rd, sigma8, z_Plunck, mock=mockID).loglike()
    return Planck_chi2

def lnlike_fsig8(modelname, model, sigma8, mockID):
    
    fsig8_chi2 = Growthdata(modelname, model, sigma8, mock=mockID).loglike()
    return fsig8_chi2

def lnlike_SNe(modelname, model, mockID):
    
    return SNeLikelihood(modelname, model, mock=mockID).loglike()

def lnlike_H0(modelname, model, mockID):
    
    H0_chi2 = H0Likelihood(modelname, model, mock=mockID).loglike()
    return H0_chi2

def lnprob_joint(modelname, model, rd, mockID, sigma8=None, CMB=False, BAO=False, fsig8=False, SNe=False, H0=False):
    
    lnprob = 0.0
    
    if CMB == True and fsig8 == True:
        lnprob = lnprob+lnlike_CMBS(modelname, model, sigma8, rd, mockID)+lnlike_fsig8(modelname, model, sigma8, mockID)

    elif CMB == True and fsig8 == False:
        lnprob = lnprob + lnlike_CMB(modelname, model, rd, mockID)

    elif CMB == False and fsig8 == True:
        lnprob = lnprob+lnlike_fsig8(modelname, model, sigma8, mockID)

    elif CMB == False and fsig8 == False:
        lnprob = lnprob

    if BAO == True:
        lnprob = lnprob+lnlike_Gala(modelname, model, rd, mockID)+lnlike_LyaF(modelname, model, rd, mockID)
    
    if SNe == True:
        lnprob = lnprob+lnlike_SNe(modelname, model, mockID)
    
    if H0 == True:
        lnprob = lnprob+lnlike_H0(modelname, model, mockID)
    
    if CMB == False and BAO == False and fsig8 == False and SNe == False and H0 == False:
        print("No probability function was found, terminating...")
        exit()
    return lnprob

