from __future__ import division
import numpy as np
import scipy as sp
import math
import de_hh
import scalar_hh
from mcmc import *

zh, Dh, sigh = np.loadtxt('./data/BAO_Dh.dat', usecols=(0,1,2), unpack=True)
zm, Dm, sigm = np.loadtxt('./data/BAO_Dm.dat', usecols=(0,1,2), unpack=True)
zv, Dv, sigv = np.loadtxt('./data/BAO_Dv.dat', usecols=(0,1,2), unpack=True)

z = np.hstack((zh, zm, zv))
Ddis = np.hstack((Dh, Dm, Dv))
sig = np.hstack((sigh, sigm, sigv))
cov_chi2 = np.diag(1.0/sig)


NW = 100
NStep = 10000
#################################################

cov_prop = np.diag([0.00002, 0.00002, 0.0, 0.00002])  # covariance matrix of proposal
dim = len(cov_prop)

### construct the initial positions in the parameter space ###

p00 = np.zeros(dim)
for i in range(NW):
    p0 = np.hstack((np.random.random(1), np.random.random(1), 0.0, 0.04))
    p00 = np.vstack((p00, p0))
p00 = p00[1:len(p00)]

aa = MCMC(cov_prop).Nsample('Flat_LCDM', NW, p00, NStep , z, Ddis, cov_chi2)

a = aa[1][1]

for j in range(NW):
    ac = aa[j]
    a = np.vstack((a,ac))
print(len(a))
np.savetxt('./output/Flat_LCDM_MCMC2.dat', a[1:len(a)])


################################################
'''
cov_prop = np.diag([0.00002, 0.00002, 0.00002, 0.00002, 0.00002])
dim = len(cov_prop)

p00 = np.zeros(dim)
for i in range(NW):
    p0 = np.hstack((np.random.random(1), -0.5+np.random.random(1), np.random.random(1), 0.01, 0.04))
    p00 = np.vstack((p00, p0))
p00 = p00[1:len(p00)]


aa = MCMC(cov_prop).Nsample('LCDM', NW, p00, NStep, z, Ddis, cov_chi2)

a = aa[1][1]

for j in range(NW):
    ac = aa[j]
    a = np.vstack((a,ac))
print(len(a))
np.savetxt('./output/LCDM_MCMC1.dat', a[1:len(a)])


################################################

cov_prop = np.diag([0.00002, 0.00002, 0.00006, 0.00002, 0.00002, 0.00002])
dim = len(cov_prop)

p00 = np.zeros(dim)
for i in range(NW):
    p0 = np.hstack((np.random.random(1), -0.5+np.random.random(1), -3.0*np.random.random(1), np.random.random(1), 0.01, 0.04))
    p00 = np.vstack((p00, p0))
p00 = p00[1:len(p00)]


aa = MCMC(cov_prop).Nsample('XCDM', NW, p00, NStep, z, Ddis, cov_chi2)

a = aa[1][1]

for j in range(NW):
    ac = aa[j]
    a = np.vstack((a,ac))
print(len(a))
np.savetxt('./output/XCDM_MCMC1.dat', a[1:len(a)])
'''

################################################

cov_prop = np.diag([0.00002, 0.00002, 0.002, 0.004, 0.00002, 0.00002, 0.00002])
dim = len(cov_prop)

p00 = np.zeros(dim)
for i in range(NW):
    p0 = np.hstack((np.random.random(1), -0.5+np.random.random(1), -10.0*np.random.random(1), -20*np.random.random(1)+10, np.random.random(1), 0.01, 0.04))
    p00 = np.vstack((p00, p0))
p00 = p00[1:len(p00)]


aa = MCMC(cov_prop).Nsample('W_CPL', NW, p00, NStep, z, Ddis, cov_chi2)

a = aa[1][1]

for j in range(NW):
    ac = aa[j]
    a = np.vstack((a,ac))
print(len(a))
np.savetxt('./output/W_CPL_MCMC2.dat', a[1:len(a)])


################################################
'''
aa = np.loadtxt("./output/Flat_LCDM_MCMC1.dat", unpack = True)
dim = len(aa)-3
acc = aa[0]
trial = aa[1]
chi2 = aa[2]
p1, p2 = aa[3], aa[4]

p = np.vstack((p1,p2))

cl = [2.30, 6.18, 11.83]
IFdot = 1
ll = ['$\Omega_{m}$', 'h']
Con_Reg(2, chi2, p, cl, IFdot, ll)
#plt.figure(2)
#plt.scatter(p1, p2, s = 0.1)
'''