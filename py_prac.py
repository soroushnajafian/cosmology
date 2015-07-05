import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
'''
def ss(x):
    return x**x+x

a = np.linspace(0,10,10)
b = map(ss,a)
print(a,b)
'''

import de_hh
import scalar_hh
import de_hh_q
'''
LCDM_nf = de_hh.LCDM_nf(0.3, 0, 0.00004)
print(map(LCDM_nf.hh, np.linspace(0,1,10)))


DE_Card = de_hh.DE_Card(0.3, 0.0004, 1)
print(map(DE_Card.hh, np.linspace(0,1,10)))

DDG = de_hh.DDG(0.3, 0.2)
print(map(DDG.hh, np.linspace(0,1,10)))
'''
'''
Qui = scalar_hh.Quintessence(0.01, 0.01, 2, 1, 0.0)
a, Z, x, y, z = Qui.solu(0.0)
#print(Z)

print(len(a))
w = (x**2-y**2)/(x**2+y**2)
Oo = x**2+y**2
plt.figure(1)
plt.plot(Z,Oo)
#plt.xlim([0, 10])
plt.show()
'''
'''
LCDM = de_hh.LCDM(0.3, 0.001, 0.0005, 0.7)
z = 10**(np.linspace(-3,1,10))
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


plt.show()
