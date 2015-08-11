import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
from de_plot import DE_vis

####################################################
#  Define your model by specifying the parameters

LCDM = de_hh.LCDM(0.3183, 0.0, 0.0005, 0.6704)
#TOPO = de_hh.Topo_defc_2D(0.3, 0.0001, 0.7)
#Phan = de_hh.Phan_DE(0.3, 0.0001, 0.7)
XCDM = de_hh.XCDM(0.3, 0.0, -0.9, 0.7)
#CG = de_hh.CG(0.3, 0.0001, 2, 0.7)
GCG = de_hh.GCG(0.3, 0.0, 0.98, 0.93, 0.7)
#W_Linear = de_hh.W_Linear(0.3, 0.0001, -0.9, 0.2, 0.7)
W_CPL = de_hh.W_CPL(0.3, 0.0001, -1.0, 0.5, 0.7)
#DE_Casimir = de_hh.DE_Casimir(0.3, 0.0001, 0.02, 0.7)
DE_Card = de_hh.DE_Card(0.3, 0.0005, 0.38, 0.7)
DGP = de_hh.DGP(0.3, 0.14, 0.7)
#DDG = de_hh.DDG(0.3, 0.4, 0.7)
#RS = de_hh.RS(0.3, 0.0001, 0.05, 0.7)
RSL = de_hh.RSL(0.3, 0.0001, 0.05, 0.2, 0.7)
S_Brane1 = de_hh.S_Brane1(0.3, 0.0, 0.2, 1, 0.7)
#S_Brane2 = de_hh.S_Brane2(0.3, 0.0001, 0.02, 0.1, 0.7)
#q_Linear = de_hh.q_Linear(-0.4, 0.3, 0.7)
q_CPL = de_hh.q_CPL(-0.6, 1.5, 0.7)

Quintessence = scalar_hh.Quintessence(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Phantom = scalar_hh.Phantom(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Tachyon = scalar_hh.Tachyon(0.8, 0.0001, 1, 2, 0.0, 0.7)
DiGhCondensate = scalar_hh.DiGhCondensate(0.00085, 0.00085, 1, 0, 0.1, 0.7)

H0 = 70

z = np.linspace(0, 3, 100)


zh, Dh, sigh = np.loadtxt('BAO_Dh.dat', usecols=(0,1,2), unpack=True)
zm, Dm, sigm = np.loadtxt('BAO_Dm.dat', usecols=(0,1,2), unpack=True)
zv, Dv, sigv = np.loadtxt('BAO_Dv.dat', usecols=(0,1,2), unpack=True)

###################################################################################

plt.figure(1)
plt.xlabel('z')
plt.xscale('log')
plt.ylabel('distance')
plt.xlim([0.09,3])

rd = LCDM.rd(0.008412, 0.049)

DH = np.array(map(LCDM.D_Hz, z))/rd*np.sqrt(z)
DH_p, = plt.plot(z, DH, 'b', label='$\Lambda$CDM: D$_{H}$')

DM = np.array(map(LCDM.D_M, z))/rd/np.sqrt(z)
DM_p, = plt.plot(z, DM, 'g', label='$\Lambda$CDM: D$_{M}$')

DV = np.array(map(LCDM.D_V, z))/rd/np.sqrt(z)
DV_p, = plt.plot(z, DV, 'r', label='$\Lambda$CDM: D$_{V}$')

plt.errorbar(zh, np.sqrt(zh)*Dh, sigh, fmt='bo')
plt.errorbar(zm, Dm/np.sqrt(zm), sigm ,fmt='go')
plt.errorbar(zv, Dv/np.sqrt(zv), sigv, fmt='ro')

###############################################################################

plt.show()

