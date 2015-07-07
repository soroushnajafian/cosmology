import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import de_hh
import scalar_hh
import de_plot


Qui = scalar_hh.Quintessence(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Pha = scalar_hh.Phantom(0.000001, 0.00005, 1, 1, 0.0, 0.7)
Tac = scalar_hh.Tachyon(0.8, 0.0001, 1, 2, 0.0, 0.7)
DGC = scalar_hh.DiGhCondensate(0.00085, 0.00085, 1, 0, 0.1, 0.7)
#z = np.linspace(0,10,100)
z, D,= Qui.D_solu(2)[:2]
#w_de_q = map(Qui.q, z)
#w_de_p = map(Pha.q, z)
#w_de_t = map(Tac.q, z)
#w_de_d = map(DGC.q, z)
plt.figure(1)
plt.plot(z, D, 'k')
#plt.plot(z, w_de_p, 'r')
#plt.plot(z, w_de_t, 'g')
#plt.plot(z, w_de_d, 'y')

#plt.xlim([0, 10])


'''
LCDM = de_hh.LCDM(0.3, 0.001, 0.0005, 0.7)
z = np.linspace(0,2,100)
E = map(LCDM.E, z)
plt.plot(z,E, 'r')
'''

'''
weff = map(LCDM.weff, z)
q = map(LCDM.q, z)
qq = 1.0/np.array(q)
H = map(LCDM.E, z)
D_L = map(LCDM.D_L, z)
mu = map(LCDM.mu, z)
zz, D, D1, f= LCDM.D_solu(2)
plt.figure(1)
plt.plot(1/(1+zz),D)
f = -(1+zz)/D*D1
plt.figure(2)
plt.plot(zz,f)
#plt.xscale('log')
'''
'''
Topo = de_hh.Topo_defc_2D(0.25, 0.01)
weff1 = map(Topo.weff,z)
plt.figure(1)
plt.plot(z,weff1,'r')
plt.xscale('log')
'''
'''
XCDM = de_hh.XCDM(0.3, 0.001, -0.6, 0.7)
z = np.linspace(0,2,100)
w_de = map(XCDM.w_de, z)
zz, D, D1, f= XCDM.D_solu(2)
q = map(XCDM.q, z)
plt.plot(z, q)
'''
'''
GCG = de_hh.GCG(0.3, 0.001, 0.9, 1, 0.7)
z = np.linspace(0,3, 100)
w_de = map(GCG.w_de, z)
plt.plot(z, w_de)
CG = de_hh.CG(0.3, 0.001, 0.9, 0.7)
w_de1 = map(CG.w_de, z)
plt.plot(z, w_de1, 'r')
'''
'''
DGP = de_hh.DGP(0.235, 0.138, 0.7)
z = np.linspace(0,3,100)
w_de = map(DGP.w_de, z)
q = map(DGP.q, z)
zz, D, D1, f = DGP.D_solu(2)
plt.plot(zz, D)
'''
'''
DDG = de_hh.DDG(0.255, 0.13)
z = np.linspace(0,3,100)
w_de = map(DDG.w_de, z)
plt.plot(z, w_de)
plt.ylim(-1.5, 1.5)
'''
'''
QL = de_hh_q.q_Linear(0.5, 0.3, 0.7)
z = np.linspace(0,4, 100)
q = map(QL.q, z)
w_de = map(QL.weff, z)
#ss = QL.D_solu(2)
mu = map(QL.mu, z)
plt.figure(1)
plt.plot(z, q)
plt.figure(2)
plt.plot(z,w_de)
plt.figure(3)
plt.plot(z,mu)
'''

plt.show()
