from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
from mcmc import *
from CL_plot import Con_Reg


################################################
'''
aa = np.loadtxt("./output/Flat_LCDM_MCMC2.dat", unpack = True)
dim = len(aa)-3-2
acc = aa[0]
trial = aa[1]
chi2 = aa[2]
p1, p2 = aa[3], aa[4]

p = np.vstack((p1,p2))

cl = [2.30, 6.18, 11.83]
IFdot = 0
ll = ['$\Omega_{m}$', 'h']
Con_Reg(2, chi2, p, cl, IFdot, ll)

################################################

aa = np.loadtxt("./output/LCDM_MCMC1.dat", unpack = True)
dim = len(aa)-3-2
acc = aa[0]
trial = aa[1]
chi2 = aa[2]
p1, p2, p3= aa[3], aa[4], aa[5]

p = np.vstack((p1,p2,p3))

cl = [2.30, 6.18, 11.83]
IFdot = 0
ll = ['$\Omega_{m}$', '$\Omega_{k}$', 'h']
Con_Reg(3, chi2, p, cl, IFdot, ll)

################################################

aa = np.loadtxt("./output/XCDM_MCMC1.dat", unpack = True)
dim = len(aa)-3-2
acc = aa[0]
trial = aa[1]
chi2 = aa[2]
p1, p2, p3, p4 = aa[3], aa[4], aa[5], aa[6]

p = np.vstack((p1,p2,p3,p4))

cl = [2.30, 6.18, 11.83]
IFdot = 0
ll = ['$\Omega_{m}$', '$\Omega_{k}$', 'w', 'h']
Con_Reg(4, chi2, p, cl, IFdot, ll)
'''
################################################

aa = np.loadtxt("./output/W_CPL_MCMC2.dat", unpack = True)
dim = len(aa)-3-2
acc = aa[0]
trial = aa[1]
chi2 = aa[2]
p1, p2, p3, p4, p5 = aa[3], aa[4], aa[5], aa[6], aa[7]

p = np.vstack((p1,p2,p3,p4,p5))

cl = [2.30, 6.18, 11.83]
IFdot = 1
ll = ['$\Omega_{m}$', '$\Omega_{k}$', '$w_{0}$', '$w_{1}$',' h']
Con_Reg(5, chi2, p, cl, IFdot, ll)

################################################


zh, Dh, sigh = np.loadtxt('./data/BAO_Dh.dat', usecols=(0,1,2), unpack=True)
zm, Dm, sigm = np.loadtxt('./data/BAO_Dm.dat', usecols=(0,1,2), unpack=True)
zv, Dv, sigv = np.loadtxt('./data/BAO_Dv.dat', usecols=(0,1,2), unpack=True)

z = np.linspace(0, 3, 100)

plt.show()
